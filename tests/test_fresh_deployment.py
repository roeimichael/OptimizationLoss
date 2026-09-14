"""Shared deployment probability provenance and fixed-class scoring."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.pipeline.eval import evaluate_with_posthoc, write_evaluation_outputs
from src.methodologies.heuristic.train import apply_allocation_heuristic
from src.training.metrics import compute_metrics
from src.utils.constants import UNLIMITED
from src.pipeline.config import validate_hyperparams


def test_weighted_ce_option_refused():
    with pytest.raises(ValueError, match='Unknown hyperparameter'):
        validate_hyperparams('tralo', {'class_weighted_ce': False})


def test_dispatch_header_uses_only_live_runtime_hyperparameters(capsys):
    from main import print_experiment_header
    from configs.gen_campaign import load_protocol, build_hyperparams
    protocol = load_protocol()
    hp = build_hyperparams(protocol, protocol['arms']['tralo'], 1)
    validate_hyperparams('tralo', hp)
    print_experiment_header(1, 1, 'fixture', {'methodology': 'tralo', 'hyperparams': hp}, 0, 0)
    text = capsys.readouterr().out
    assert 'lr_con=' in text and 'pretrained=' in text
    assert 'weighted_ce=' not in text


def test_fixed_class_metrics_include_absent_classes_and_map_class_ids():
    result = compute_metrics(np.array([0, 1]), np.array([0, 1]), np.eye(3)[:2])
    assert result['f1_macro'] == pytest.approx(2 / 3)
    assert result['classes'].tolist() == [0, 1, 2]
    result = compute_metrics(np.array([0, 0, 1, 1, 2, 2]),
                             np.array([0, 1, 1, 2, 2, 2]),
                             classes=[0, 1, 2], constrained_classes=[1, 2])
    assert result['f1_per_class'] == pytest.approx([2/3, 1/2, 4/5])
    assert result['cc_f1'] == pytest.approx(.65)
    assert result['f1_macro'] == pytest.approx(59/90)
    assert result['constrained_precision'] == pytest.approx(7/12)
    assert result['constrained_recall'] == pytest.approx(3/4)
    assert result['collateral_f1'] == pytest.approx(2/3)
    result = compute_metrics(np.array([2, 5]), np.array([2, 5]), classes=[2, 5, 7],
                             constrained_classes=[5, 7])
    assert result['cc_f1'] == .5
    result = compute_metrics(np.array([2, 5]), np.array([2, 5]), np.eye(3)[:2],
                             classes=[2, 5, 7], constrained_classes=[5, 7])
    assert result['brier_score'] == 0
    assert result['ece'] == 0


def test_shared_deployment_saves_exact_allocator_probabilities(tmp_path):
    class TraceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, x):
            self.batch_sizes.append(len(x))
            return torch.stack((x[:, 0], -x[:, 0], x[:, 0] * .2), dim=1)

    model = TraceModel()
    x = torch.linspace(-1, 1, 601).reshape(-1, 1)
    truth = np.arange(601) % 3
    groups = np.arange(601) % 2
    global_con = [UNLIMITED, 75, 90]
    local_con = {0: [UNLIMITED, 40, 45], 1: [UNLIMITED, 35, 45]}
    result = evaluate_with_posthoc(model, x, truth, groups, global_con, local_con, [1, 2])
    assert model.batch_sizes == [256, 256, 89]
    write_evaluation_outputs(tmp_path, truth, groups, result, 3, global_con, local_con)
    saved = pd.read_csv(tmp_path / 'final_predictions.csv', float_precision='round_trip')
    probability_columns = [c for c in saved if c.startswith('Prob_Class_')]
    probs = saved[probability_columns].to_numpy()
    np.testing.assert_array_equal(probs, result['y_proba'])
    deployed, _ = apply_allocation_heuristic(probs, groups, [1, 2, 0], global_con,
                                            local_con, 3)
    np.testing.assert_array_equal(deployed, result['y_pred'])
    changed_labels = evaluate_with_posthoc(model, x, truth[::-1], groups,
                                           global_con, local_con, [1, 2])
    np.testing.assert_array_equal(changed_labels['y_pred'], deployed)


@pytest.mark.parametrize('arm', ['clip', 'focal_clip', 'tralo', 'tralo_null', 'fioretto', 'hounie', 'alm'])
def test_labels_cannot_change_training_or_deployment_with_frozen_quotas(tmp_path, arm):
    from configs.gen_campaign import load_protocol
    from scripts.smoke_arms import make_inputs, fixture_metadata
    from src.experiments.runner import TRAIN_FNS
    from src.training.model_cache import state_digest
    p = load_protocol()
    labels, _ = fixture_metadata()
    states, deployments = [], []
    for index, scoring_labels in enumerate((labels, labels[::-1])):
        inputs, global_con, local_con = make_inputs(p, arm, tmp_path/str(index))
        out = TRAIN_FNS[p['arms'][arm]['methodology']](inputs)
        states.append(state_digest(out.model.state_dict()))
        result = evaluate_with_posthoc(out.model, inputs.X_test, scoring_labels,
                                       inputs.group_ids, global_con, local_con, [1])
        deployments.append(result['y_pred'])
    assert states[0] == states[1]
    np.testing.assert_array_equal(*deployments)
    zero_global = list(global_con)
    zero_global[1] = 0
    zero_local = {g: [0 if c == 1 else value for c, value in enumerate(bounds)]
                  for g, bounds in local_con.items()}
    changed_quota = evaluate_with_posthoc(out.model, inputs.X_test, labels,
                                          inputs.group_ids, zero_global, zero_local, [1])
    assert (deployments[0] == 1).sum() > 0
    assert (changed_quota['y_pred'] == 1).sum() == 0


def test_runner_replaces_label_derived_quotas_before_training(tmp_path, monkeypatch):
    import copy
    import shutil
    from test_fresh_campaign import tiny_data, config_for, load_protocol
    from src.experiments import runner
    from src.pipeline.campaign import stage_campaign, freeze_campaign, validate_receipts
    from src.training.constraints import compute_global_constraints, compute_local_constraints
    from src.training.model_cache import state_digest

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    data_dir = tmp_path/'data'
    tiny_data(data_dir)
    cfg = config_for(data_dir, 'tralo')
    cfg['hyperparams']['constraint_epochs'] = 1
    root, rel = tmp_path/'one', 'tralo/config.json'
    stage_campaign(root, {rel: cfg}, load_protocol())
    freeze_campaign(root)
    second = tmp_path/'two'
    shutil.copytree(root, second)
    real_load, real_train = runner.load_data, runner.TRAIN_FNS['tralo']
    loaded_quotas, received_quotas, states, frames = [], [], [], []
    scoring_labels = None

    def varied_loader(config):
        data = real_load(config)
        data.y_test = scoring_labels.copy()
        frame = pd.DataFrame({'label': data.y_test, 'group': data.groups_test})
        data.global_con = compute_global_constraints(frame, 'label', .5, [1], 2)
        data.local_con = compute_local_constraints(frame, 'label', .5, 'group', [1], 2)
        loaded_quotas.append(copy.deepcopy((data.global_con, data.local_con)))
        return data

    def observed_train(inputs):
        received_quotas.append(copy.deepcopy((inputs.global_con, inputs.local_con)))
        output = real_train(inputs)
        states.append(state_digest(output.model.state_dict()))
        return output

    monkeypatch.setattr(runner, 'load_data', varied_loader)
    monkeypatch.setitem(runner.TRAIN_FNS, 'tralo', observed_train)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for index, campaign in enumerate((root, second)):
            scoring_labels = np.full(4, index, dtype=np.int64)
            monkeypatch.setenv('OPTLOSS_MODEL_CACHE', str(tmp_path/f'cold_cache_{index}'))
            result = runner.run_experiment(str(campaign/rel))
            assert result is not None and result['used_cached_model'] is False
            validate_receipts(campaign)
            frames.append(pd.read_csv((campaign/rel).with_name('final_predictions.csv'),
                                      float_precision='round_trip'))
    finally:
        torch.set_num_threads(threads)
    assert [quota[0][1] for quota in loaded_quotas] == [0, 2]
    assert received_quotas == [([UNLIMITED, 1], {20: [UNLIMITED, 1]})] * 2
    assert states[0] == states[1]
    assert frames[0].True_Label.tolist() == [0] * 4
    assert frames[1].True_Label.tolist() == [1] * 4
    columns = ['Predicted_Label', 'Prob_Class_0', 'Prob_Class_1']
    np.testing.assert_array_equal(frames[0][columns].to_numpy(), frames[1][columns].to_numpy())
