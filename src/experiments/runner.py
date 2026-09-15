from src.pipeline.config import validate_hyperparams
import argparse
import logging
import os
import time
import copy
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from src.pipeline.data import load_data
from src.utils.error_handler import logger, log_exception
from src.methodologies.tralo.train import train as train_tralo
from src.methodologies.fioretto_ldf.train import train as train_fioretto_ldf
from src.methodologies.hounie_rcl.train import train as train_hounie_rcl
from src.methodologies.heuristic.train import train as train_heuristic
from src.methodologies.fioretto_alm.train import train as train_fioretto_alm
from src.pipeline.contracts import TrainInputs
from src.pipeline.warmup import run_warmup
from src.pipeline.features import EMBEDDING_CHUNK, save_test_embeddings
from src.pipeline.eval import evaluate_with_posthoc, write_evaluation_outputs
from src.training.logging import save_evaluation_metrics
from src.utils.filesystem_manager import (
    load_config_from_path,
    save_config_to_path,
    update_experiment_status,
)
from src.utils.gitver import git_version
from src.pipeline.setup import seed_all, runtime_provenance
from src.pipeline.io import save_results_to_config
from src.pipeline.campaign import campaign_for_config, run_identity, write_receipt

log = logging.getLogger(__name__)
TRAIN_FNS = {
    "tralo": train_tralo,
    "fioretto_ldf": train_fioretto_ldf,
    "hounie_rcl": train_hounie_rcl,
    "heuristic": train_heuristic,
    "fioretto_alm": train_fioretto_alm,
}


@logger()
def run_experiment(config_path: str) -> Optional[Dict[str, Any]]:
    campaign_root, manifest = campaign_for_config(config_path)
    experiment_path = Path(config_path).parent
    config = load_config_from_path(experiment_path)
    validate_hyperparams(config["methodology"], config.get("hyperparams", {}))
    if config.get("status", "pending") == "completed":
        log.info("Skipping completed: %s", experiment_path)
        return None
    config["run_code_version"] = git_version()
    identity = run_identity(campaign_root, manifest, config_path)
    config['campaign_id'] = identity['campaign_id']
    config['release_id'] = identity['release_id']
    config['cache_identity'] = {k: identity[k] for k in ('release_id', 'data_id')}
    save_config_to_path(config, experiment_path)
    update_experiment_status(experiment_path, "running")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("hyperparams", {}).get("seed", None)
    seed_all(seed)
    log.info("Running %s on %s (model=%s)", config_path, device, config["model_name"])
    if torch.cuda.is_available():
        _prov = runtime_provenance(device)
        log.info(
            "GPU: %s | CUDA: %s | AMP: %s%s",
            torch.cuda.get_device_name(0),
            torch.version.cuda,
            _prov["amp_dtype"],
            " + GradScaler" if _prov["grad_scaler"] else " (no scaler)",
        )
    rel = Path(config_path).absolute().relative_to(campaign_root).as_posix()
    frozen_data = manifest['data'][manifest['runs'][rel]['data_id']]
    loading_config = copy.deepcopy(config)
    loading_config['dataset_config']['data_dir'] = str(Path(frozen_data['files']['train_images.npy']['logical']).parent)
    data = load_data(loading_config)
    config['data_fingerprint'] = loading_config['data_fingerprint']
    data.global_con = frozen_data['quotas']['global']
    data.local_con = {int(k): values for k, values in frozen_data['quotas']['local'].items()}
    X_train_tensor = data.X_train
    y_train_tensor = data.y_train
    X_test_tensor = data.X_test.to(device)
    groups_test = data.groups_test
    global_con = data.global_con
    local_con = data.local_con
    num_classes = data.num_classes
    constrained_classes = data.constrained_classes
    csv_log_path = experiment_path / "training_log.csv"
    warmup_start = time.time()
    (model, from_cache) = run_warmup(
        config,
        num_classes,
        X_train_tensor,
        y_train_tensor,
        device,
        csv_log_path=str(csv_log_path),
        groups_train=data.groups_train,
    )
    warmup_time = time.time() - warmup_start
    log.info(
        "TIMING warmup=%.2fs (%d epochs, cached=%s)",
        warmup_time,
        config["hyperparams"]["warmup_epochs"],
        from_cache,
    )
    seed_all(seed)
    constraint_start = time.time()
    train_inputs = TrainInputs(
        model=model,
        X_train=X_train_tensor,
        y_train=y_train_tensor,
        X_test=X_test_tensor,
        group_ids=groups_test,
        global_con=global_con,
        local_con=local_con,
        constrained_classes=constrained_classes,
        num_classes=num_classes,
        config=config,
        hyperparams=config["hyperparams"],
        device=device,
        experiment_path=experiment_path,
        csv_log_path=csv_log_path,
    )
    methodology = config.get("methodology", "tralo")
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
        model,
        X_test_tensor,
        y_true,
        group_ids,
        global_con,
        local_con,
        constrained_classes,
    )
    best_metrics = result["metrics"]
    best_adj = result["adj"]
    best_meta = result["posthoc_meta"]
    best_source = train_outputs.summary.get("checkpoint_source", "final")
    write_evaluation_outputs(
        experiment_path, y_true, group_ids, result, num_classes, global_con, local_con
    )
    save_test_embeddings(
        experiment_path, model, X_test_tensor, EMBEDDING_CHUNK, device=device
    )
    best_metrics["satisfaction_epoch"] = train_outputs.summary.get("satisfaction_epoch")
    best_metrics["soft_hard_gap"] = train_outputs.summary.get("soft_hard_gap", {})
    best_metrics["best_sat_epoch"] = train_outputs.summary.get("best_sat_epoch")
    best_metrics["restored_from_epoch"] = train_outputs.summary.get(
        "restored_from_epoch"
    )
    best_metrics["min_excess_epoch"] = train_outputs.summary.get("min_excess_epoch")
    best_metrics["min_total_excess"] = train_outputs.summary.get("min_total_excess")
    best_metrics["restore_kind"] = train_outputs.summary.get("restore_kind")
    if "checkpoint_source" in train_outputs.summary:
        best_metrics["checkpoint_source"] = train_outputs.summary["checkpoint_source"]
    config["reordering"] = train_outputs.summary.get("reordering", {})
    if "results_comparison" in train_outputs.summary:
        config["results_comparison"] = train_outputs.summary["results_comparison"]
    log.info("sat_epoch=%s", best_metrics["satisfaction_epoch"] or "N/A")
    posthoc_time = time.time() - posthoc_start
    best_metrics["warmup_time"] = float(warmup_time)
    best_metrics["constraint_train_time"] = float(constraint_train_time)
    best_metrics["posthoc_time"] = float(posthoc_time)
    save_evaluation_metrics(experiment_path / "evaluation_metrics.csv", best_metrics)
    save_results_to_config(
        config,
        experiment_path,
        {
            "accuracy": float(best_metrics["accuracy"]),
            "precision_macro": float(best_metrics["precision_macro"]),
            "recall_macro": float(best_metrics["recall_macro"]),
            "f1_macro": float(best_metrics["f1_macro"]),
            "training_time": float(training_time),
            "warmup_time": float(warmup_time),
            "constraint_train_time": float(constraint_train_time),
            "posthoc_time": float(posthoc_time),
            "used_cached_model": from_cache,
            "samples_adjusted": int(best_adj),
            "deployment_protocol": best_meta['deployment_protocol'],
            "inference_chunk_size": best_meta['inference_chunk_size'],
            "constraint_steps_applied": train_outputs.summary.get(
                "constraint_steps_applied"
            ),
            "constraint_steps_attempted": train_outputs.summary.get(
                "constraint_steps_attempted"
            ),
            "runtime": runtime_provenance(device),
        },
    )
    write_receipt(campaign_root, config_path, config.get('warmup_checkpoint'))
    log.info(
        "Done: accuracy=%.4f source=%s time=%.2fs path=%s",
        best_metrics["accuracy"],
        best_source,
        training_time,
        experiment_path,
    )
    return config["results"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run single experiment")
    parser.add_argument("config_path", type=str, help="Path to config.json")
    args = parser.parse_args()
    experiment_path = Path(args.config_path).parent
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    try:
        campaign_for_config(args.config_path)
    except (ValueError, OSError, KeyError) as e:
        log.error('REFUSED before run writes: %s', e)
        raise SystemExit(1)
    try:
        run_experiment(args.config_path)
    except Exception as e:
        try:
            campaign_for_config(args.config_path)
        except (ValueError, OSError, KeyError):
            log.error('Identity changed; refusing error/status writes: %s', e)
            raise SystemExit(1)
        log_exception(
            e, context=f"Experiment: {experiment_path}", experiment_path=experiment_path
        )
        update_experiment_status(str(experiment_path), "pending", count_failure=True)
        exit(1)


if __name__ == "__main__":
    main()
