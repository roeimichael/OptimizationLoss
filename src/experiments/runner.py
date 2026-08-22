# Single experiment runner. Loads config -> data -> warmup -> methodology train
# -> evaluation -> save. The only entry point for any of the four methodologies.

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

# cuBLAS reads this ONCE, when the CUDA context is created, so it has to be set
# before any torch CUDA call -- setting it inside seed_all was too late, and
# without it torch.use_deterministic_algorithms raises on every matmul, which on
# a transformer backbone is every layer. Measured cost of NOT having it: the
# same arm, same seed, same config produced macro-F1 0.6709 and 0.7015 on two
# runs -- 0.0306 apart, against a headline effect of 0.0017.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.pipeline.data import load_data
from src.utils.error_handler import logger, log_exception
from src.methodologies.tralo.train import train as train_tralo                      # was tralo_fioretto
from src.methodologies.fioretto_ldf.train import train as train_fioretto_ldf
from src.methodologies.hounie_rcl.train import train as train_hounie_rcl
from src.methodologies.heuristic.train import train as train_heuristic
from src.methodologies.danits_lp.train import train as train_danits_lp          # LP-LG clipper (Shifman)
from src.methodologies.fioretto_alm.train import train as train_fioretto_alm    # ALM dual
from src.methodologies.focal.train import train as train_focal                  # imbalanced + LP clip
from src.methodologies.class_balanced.train import train as train_class_balanced
from src.methodologies.logit_adjust.train import train as train_logit_adjust
from src.methodologies.select.train import train as train_select
from src.pipeline.contracts import TrainInputs
from src.pipeline.warmup import run_warmup
from src.pipeline.features import EMBEDDING_CHUNK, save_test_embeddings
from src.pipeline.eval import evaluate_with_posthoc, write_evaluation_outputs
from src.training.logging import save_evaluation_metrics
from src.utils.filesystem_manager import (load_config_from_path,
                                          save_config_to_path,
                                          update_experiment_status)
from src.utils.gitver import git_version
from src.pipeline.setup import seed_all, runtime_provenance
from src.pipeline.io import save_results_to_config

log = logging.getLogger(__name__)


# The dispatch table. Module level so scripts.smoke_arms can execute every
# arm without going through run_experiment -- a config audit cannot see a
# runtime crash, and three arms once shipped with an undefined name here.
TRAIN_FNS = {
    'tralo': train_tralo,
    'tralo_fioretto': train_tralo,              # ALIAS for backward-compat with completed configs
    'fioretto_ldf': train_fioretto_ldf,
    'hounie_rcl': train_hounie_rcl,
    'heuristic': train_heuristic,               # the post-hoc clippers: clip / focal_clip
    'danits_lp': train_danits_lp,               # LP-LG: the OTHER post-hoc clipper
    'fioretto_alm': train_fioretto_alm,         # ALM: third dual-ascent baseline
    'focal': train_focal,                       # imbalanced warm-up + LP clip
    'class_balanced': train_class_balanced,
    'logit_adjust': train_logit_adjust,
    # 1c, jointly-trained selection head. ⛔ REJECTED 2026-08-22 -- worst arm
    # measured here (-22 items vs `clip`, 0 of 2 cells on every metric) and it
    # destabilises training. Registered so `results/selectrun` stays readable
    # and so `select_null` (a live control) can run; NOT a direction to resume.
    # docs/FRAMEWORK.md section (12).
    'select': train_select,
}


@logger()
def run_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    if config.get('status', 'pending') == 'completed':
        log.info("Skipping completed: %s", experiment_path)
        return None
    # THE COMMIT THAT PRODUCES THESE WEIGHTS, stamped at EXECUTION time.
    # `code_version` is written by configs/gen_campaign when the config is
    # created and is never revisited, so it describes the generator. Run half a
    # campaign, land a change to a training file, resume the rest, and every
    # config still carries the ORIGINAL stamp -- full_panel's provenance gate
    # then sees one value across both halves and scores them as one comparison,
    # which is precisely what that gate exists to refuse, and model_cache hands
    # the post-change runs the pre-change warm-up on the same false agreement.
    #
    # Written to disk BEFORE the status flips: update_experiment_status reloads
    # config.json from disk and rewrites it, so an in-memory key that has not
    # landed yet would be dropped and a crashed run would carry no runner
    # stamp at all.
    config['run_code_version'] = git_version()
    save_config_to_path(config, experiment_path)
    update_experiment_status(experiment_path, 'running')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('hyperparams', {}).get('seed', None)
    seed_all(seed)
    log.info("Running %s on %s (model=%s)", config_path, device, config['model_name'])
    if torch.cuda.is_available():
        # Report the regime actually SELECTED, not a capability probe.
        # This line used to print torch.cuda.is_bf16_supported(), which returns
        # True on the Quadro RTX 6000 (Turing, sm_75) because PyTorch emulates
        # bf16 there -- so dsisco01's log said "BF16: True" on every run while
        # setup_runtime, which gates on gpu_arch >= 8, correctly selected
        # FP16 + GradScaler. The numbers were right and the log contradicted
        # them. The FP16 path SKIPS an overflowing optimizer step and the BF16
        # path does not, so which regime ran is exactly the thing an analysis
        # has to know, and the one thing this line got wrong.
        _prov = runtime_provenance(device)
        log.info("GPU: %s | CUDA: %s | AMP: %s%s",
                 torch.cuda.get_device_name(0), torch.version.cuda,
                 _prov["amp_dtype"],
                 " + GradScaler" if _prov["grad_scaler"] else " (no scaler)")
    data = load_data(config)
    X_train_tensor = data.X_train
    y_train_tensor = data.y_train
    X_test_tensor = data.X_test.to(device)
    groups_test = data.groups_test
    global_con = data.global_con
    local_con = data.local_con
    num_classes = data.num_classes
    constrained_classes = data.constrained_classes
    csv_log_path = experiment_path / 'training_log.csv'
    warmup_start = time.time()
    model, from_cache = run_warmup(
        config, num_classes, X_train_tensor, y_train_tensor, device,
        csv_log_path=str(csv_log_path),
    )
    warmup_time = time.time() - warmup_start
    log.info("TIMING warmup=%.2fs (%d epochs, cached=%s)",
             warmup_time, config['hyperparams']['warmup_epochs'], from_cache)
    # Re-seed AFTER the warm-up. run_warmup either trains (drawing RNG for init,
    # dropout and shuffling) or returns a cached model (drawing none), so
    # otherwise the constraint phase starts from a different RNG state depending
    # on whether the cache happened to be warm. The four trained arms share one
    # base_model_id, so exactly one of them trains it and the other three load
    # it -- same config, different batch order, and method effects here are
    # ~0.1 pp.
    seed_all(seed)
    constraint_start = time.time()
    train_inputs = TrainInputs(
        model=model,
        X_train=X_train_tensor, y_train=y_train_tensor,
        X_test=X_test_tensor, y_test=data.y_test,
        group_ids=groups_test,
        global_con=global_con, local_con=local_con,
        constrained_classes=constrained_classes,
        num_classes=num_classes,
        config=config, hyperparams=config['hyperparams'],
        device=device,
        experiment_path=experiment_path,
        csv_log_path=csv_log_path,
    )
    methodology = config.get('methodology', 'tralo')
    train_fns = TRAIN_FNS
    if methodology not in train_fns:
        raise ValueError(f"Unknown methodology for run_experiment: {methodology!r}")
    train_outputs = train_fns[methodology](train_inputs)
    model = train_outputs.model
    constraint_train_time = time.time() - constraint_start
    training_time = warmup_time + constraint_train_time
    y_true = data.y_test
    group_ids = groups_test
    posthoc_start = time.time()

    result = evaluate_with_posthoc(
        model, X_test_tensor, y_true, group_ids,
        global_con, local_con, constrained_classes,
        skip_targeted_correction=train_outputs.skip_targeted_correction,
        precomputed_predictions=train_outputs.precomputed_predictions,
    )
    best_metrics = result['metrics']
    best_adj = result['adj']
    best_meta = result['posthoc_meta']
    best_source = train_outputs.summary.get('checkpoint_source', 'final')

    write_evaluation_outputs(experiment_path, y_true, group_ids, result, num_classes,
                             global_con, local_con)
    # Penultimate features, beside the predictions. Until now a finished run
    # stored a 2014x7 score matrix and nothing else, so every question that
    # needs the REPRESENTATION rather than the scores meant retraining. Never
    # fatal: see save_test_embeddings.
    save_test_embeddings(experiment_path, model, X_test_tensor,
                         EMBEDDING_CHUNK, device=device)
    best_metrics['satisfaction_epoch'] = train_outputs.summary.get('satisfaction_epoch')
    best_metrics['soft_hard_gap'] = train_outputs.summary.get('soft_hard_gap', {})
    best_metrics['best_sat_epoch'] = train_outputs.summary.get('best_sat_epoch')
    best_metrics['restored_from_epoch'] = train_outputs.summary.get('restored_from_epoch')
    best_metrics['min_excess_epoch'] = train_outputs.summary.get('min_excess_epoch')
    best_metrics['min_total_excess'] = train_outputs.summary.get('min_total_excess')
    best_metrics['restore_kind'] = train_outputs.summary.get('restore_kind')
    if 'checkpoint_source' in train_outputs.summary:
        best_metrics['checkpoint_source'] = train_outputs.summary['checkpoint_source']
    # Deliberately OUTSIDE config['results']: save_results_to_config runs
    # _non_finite over that block recursively and marks any run carrying a NaN
    # `diverged`. kendalltau of a CONSTANT score column is legitimately NaN --
    # exactly what a degenerate arm produces -- so nesting the diagnostic under
    # results would have manufactured false divergences on the very runs it
    # exists to describe. Until now the field was computed, logged once, and
    # dropped: it never reached disk, so no scorer could ever read it.
    config['reordering'] = train_outputs.summary.get('reordering', {})
    if 'results_comparison' in train_outputs.summary:
        config['results_comparison'] = train_outputs.summary['results_comparison']
    log.info("sat_epoch=%s", best_metrics['satisfaction_epoch'] or 'N/A')
    posthoc_time = time.time() - posthoc_start
    best_metrics['warmup_time'] = float(warmup_time)
    best_metrics['constraint_train_time'] = float(constraint_train_time)
    best_metrics['posthoc_time'] = float(posthoc_time)
    save_evaluation_metrics(experiment_path / 'evaluation_metrics.csv', best_metrics)
    save_results_to_config(config, experiment_path, {
        'accuracy': float(best_metrics['accuracy']),
        'precision_macro': float(best_metrics['precision_macro']),
        'recall_macro': float(best_metrics['recall_macro']),
        'f1_macro': float(best_metrics['f1_macro']),
        'training_time': float(training_time),
        'warmup_time': float(warmup_time),
        'constraint_train_time': float(constraint_train_time),
        'posthoc_time': float(posthoc_time),
        'used_cached_model': from_cache,
        'samples_adjusted': int(best_adj),
        'lp_fallback_used': best_meta.get('lp_fallback_used', False),
        # How many constraint steps LANDED, against how many were attempted.
        # finish_constraint_step drops a step whose gradient is non-finite and
        # says so in its return value; every trainer used to discard that, so
        # an arm could lose a third of its constraint phase and still write
        # `status: completed`. None for the post-hoc arms, which attempt none.
        'constraint_steps_applied':
            train_outputs.summary.get('constraint_steps_applied'),
        'constraint_steps_attempted':
            train_outputs.summary.get('constraint_steps_attempted'),
        'lp_fallback_candidates': best_meta.get('lp_fallback_candidates', 0),
        # which GPU and which AMP regime: FP16+scaler SKIPS an overflowing
        # optimizer step and BF16 does not, so the same config applies a
        # different number of steps on the two servers
        'runtime': runtime_provenance(device),
    })
    log.info("Done: accuracy=%.4f source=%s time=%.2fs path=%s",
             best_metrics['accuracy'], best_source, training_time, experiment_path)
    return config['results']


def main() -> None:
    parser = argparse.ArgumentParser(description='Run single experiment')
    parser.add_argument('config_path', type=str, help='Path to config.json')
    args = parser.parse_args()
    experiment_path = Path(args.config_path).parent
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    try:
        run_experiment(args.config_path)
    except Exception as e:
        # experiment_path= was missing, so _save_error_to_file never ran and a
        # failure left NO trace on disk -- the run just reappeared as pending on
        # the next dispatch, forever, with the reason only in a lost stdout.
        log_exception(e, context=f"Experiment: {experiment_path}",
                      experiment_path=experiment_path)
        update_experiment_status(str(experiment_path), 'pending',
                                 count_failure=True)
        exit(1)


if __name__ == "__main__":
    main()
