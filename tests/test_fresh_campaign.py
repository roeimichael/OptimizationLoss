"""Fresh inventories fail closed before writes, through production boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from configs.gen_campaign import load_protocol, build_hyperparams, compute_base_model_id

REPO = Path(__file__).resolve().parents[1]


def tiny_data(path):
    path.mkdir()
    rng = np.random.default_rng(47)
    for split, groups in [('train', [10]*4), ('test', [20]*4)]:
        labels = np.array([0, 1, 0, 1], dtype=np.int64)
        np.save(path / (split + '_images.npy'), rng.integers(0, 256, (4, 3, 32, 32), dtype=np.uint8))
        np.save(path / (split + '_labels.npy'), labels)
        pd.DataFrame({'label': labels, 'location': groups}).to_csv(path / (split + '_meta.csv'), index=False)


def config_for(data_dir, arm='clip'):
    p = load_protocol()
    p['protocol'].update(total_epochs=1, trained_warmup=1)
    hp = build_hyperparams(p, p['arms'][arm], 1, pretrained=False)
    hp.update(batch_size=4)
    dc = dict(data_dir=str(data_dir), num_classes=2, group_column='location',
              constrained_class=[1], disjoint_groups=True)
    return dict(methodology=p['arms'][arm]['methodology'], model_name='MobileNetV2',
                constraint=[.5, .5], constraint_tag='L50_G50', dataset_mode='iwildcam',
                dataset_config=dc, hyperparams=hp,
                base_model_id=compute_base_model_id(p, 'MobileNetV2', hp, 'iwildcam', dc),
                arm=arm, exp_name='fixture', status='pending', code_version='fixture')


def staged(tmp_path):
    from src.pipeline.campaign import stage_campaign
    data = tmp_path / 'data'
    tiny_data(data)
    root = tmp_path / 'fresh'
    rel = 'MobileNetV2/iwildcam/L50_G50/clip/seed_1/config.json'
    stage_campaign(root, {rel: config_for(data)}, load_protocol())
    return root, rel


def test_unmarked_direct_runner_refuses_without_mutation(tmp_path):
    data = tmp_path / 'data'
    tiny_data(data)
    run = tmp_path / 'run'
    run.mkdir()
    config = run / 'config.json'
    config.write_text(json.dumps(config_for(data)))
    before = config.read_bytes()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='0', OPTLOSS_MODEL_CACHE=str(tmp_path/'cache'))
    proc = subprocess.run([sys.executable, '-m', 'src.experiments.runner', str(config)],
                          cwd=REPO, env=env, capture_output=True, text=True)
    assert proc.returncode != 0
    assert config.read_bytes() == before
    assert sorted(p.name for p in run.iterdir()) == ['config.json']
    assert not (tmp_path/'cache').exists()


def test_stage_requires_explicit_freeze_and_exact_inventory(tmp_path):
    from src.pipeline.campaign import freeze_campaign, validate_campaign
    root, rel = staged(tmp_path)
    with pytest.raises(ValueError, match='frozen'):
        validate_campaign(root)
    manifest = freeze_campaign(root)
    assert len(next(iter(manifest['data'].values()))['files']) == 6
    assert validate_campaign(root)['campaign_id'] == manifest['campaign_id']
    cfg = json.loads((root/rel).read_text())
    cfg['hyperparams']['lr'] = .9
    (root/rel).write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='config'):
        validate_campaign(root)


def test_extra_unmarked_config_and_missing_receipt_refused(tmp_path):
    from src.pipeline.campaign import freeze_campaign, validate_campaign, validate_receipts
    root, rel = staged(tmp_path)
    freeze_campaign(root)
    with pytest.raises(ValueError, match='incomplete.*pending'):
        validate_receipts(root)
    extra = root/'foreign'
    extra.mkdir()
    (extra/'config.json').write_bytes((root/rel).read_bytes())
    with pytest.raises(ValueError, match='inventory'):
        validate_campaign(root)


def test_frozen_data_replacement_refused(tmp_path):
    from src.pipeline.campaign import freeze_campaign, validate_campaign
    root, rel = staged(tmp_path)
    freeze_campaign(root)
    cfg = json.loads((root/rel).read_text())
    data = Path(cfg['dataset_config']['data_dir'])/'test_images.npy'
    with data.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='data'):
        validate_campaign(root)


def test_output_symlink_refused_before_receipt(tmp_path):
    from src.pipeline.campaign import freeze_campaign, validate_campaign
    root, rel = staged(tmp_path)
    freeze_campaign(root)
    outside = tmp_path/'outside.csv'
    outside.write_text('outside')
    try:
        (root/rel).with_name('final_predictions.csv').symlink_to(outside)
    except OSError:
        pytest.skip('OS does not permit symlinks')
    with pytest.raises(ValueError, match='symlink'):
        validate_campaign(root)


@pytest.mark.parametrize('module,args', [
    ('scripts.reset_crashed', ['--apply']),
    ('scripts.run_campaign', ['--step', 'launch']),
])
def test_operational_mutation_boundaries_refuse_unmarked_root(tmp_path, module, args):
    root = tmp_path/'foreign'
    root.mkdir()
    if module.endswith('run_campaign'):
        args = ['--root', str(root), *args]
    else:
        args = [str(root), *args]
    proc = subprocess.run([sys.executable, '-m', module, *args], cwd=REPO,
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert 'unmarked' in proc.stdout + proc.stderr
    assert not list(root.iterdir())


def test_rig_campaign_observation_refuses_unmarked_root(tmp_path):
    from scripts.rig_status import campaign_configs
    with pytest.raises(ValueError, match='unmarked'):
        campaign_configs(tmp_path)


def test_source_inventory_hashes_changed_and_new_executable_bytes(tmp_path):
    from src.pipeline.campaign import source_inventory
    (tmp_path/'src').mkdir()
    (tmp_path/'src'/'module.py').write_text('value = 1\n')
    (tmp_path/'main.py').write_text('from src import module\n')
    (tmp_path/'requirements.txt').write_text('torch\n')
    before = source_inventory(tmp_path)
    (tmp_path/'src'/'module.py').write_text('value = 2\n')
    assert before != source_inventory(tmp_path)
    (tmp_path/'src'/'new_import.py').write_text('value = 3\n')
    assert 'src/new_import.py' in source_inventory(tmp_path)


@pytest.mark.parametrize('mutation', ['missing', 'extra-field', 'manifest', 'plan'])
def test_campaign_mutations_fail_before_new_outputs(tmp_path, mutation):
    from src.pipeline.campaign import freeze_campaign, validate_campaign
    root, rel = staged(tmp_path)
    freeze_campaign(root)
    path = root/rel
    if mutation == 'missing':
        path.rename(path.with_name('preserved_config.json'))
    elif mutation == 'extra-field':
        cfg = json.loads(path.read_text())
        cfg['unknown'] = 1
        path.write_text(json.dumps(cfg))
    else:
        target = root/('campaign_manifest.json' if mutation == 'manifest' else 'campaign_plan.json')
        obj = json.loads(target.read_text())
        obj['campaign_id'] = 'foreign'
        target.write_text(json.dumps(obj))
    before = sorted(str(p) for p in root.rglob('*'))
    with pytest.raises(ValueError):
        validate_campaign(root)
    assert before == sorted(str(p) for p in root.rglob('*'))


def test_report_refuses_mixed_roots_before_output(tmp_path):
    from src.pipeline.campaign import freeze_campaign
    roots = []
    for name in ('one', 'two'):
        directory = tmp_path/name
        directory.mkdir()
        root, _ = staged(directory)
        freeze_campaign(root)
        roots.append(str(root))
    output = tmp_path/'report.json'
    proc = subprocess.run([sys.executable, '-m', 'scripts.deployed_h2h', '--campaign', *roots,
                           '--json', str(output)], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 1
    assert not output.exists()


def test_report_rejects_duplicate_observations_before_scoring(tmp_path):
    from src.pipeline.campaign import stage_campaign, freeze_campaign, write_receipt
    data = tmp_path/'data'
    tiny_data(data)
    root = tmp_path/'duplicate_observations'
    cfg = config_for(data)
    cfg['status'] = 'completed'
    paths = ['one/config.json', 'two/config.json']
    stage_campaign(root, {rel: cfg for rel in paths}, load_protocol())
    freeze_campaign(root)
    for rel in paths:
        # Receipt admission succeeds, but any scoring attempt would hit empty CSVs.
        for name in ('final_predictions.csv', 'final_predictions_raw.csv', 'evaluation_metrics.csv'):
            (root/rel).with_name(name).write_text('')
        write_receipt(root, root/rel, None)
    output, markdown = tmp_path/'refused.json', tmp_path/'refused.md'
    proc = subprocess.run([sys.executable, '-m', 'scripts.deployed_h2h', '--campaign', str(root),
                           '--json', str(output), '--markdown', str(markdown)], cwd=REPO,
                          capture_output=True, text=True)
    assert proc.returncode == 1 and 'duplicate observation' in proc.stdout
    assert not output.exists() and not markdown.exists()


def test_real_imagery_cli_generates_freezes_trains_receipts_and_reports(tmp_path):
    import yaml
    import shutil
    from src.pipeline.campaign import validate_receipts
    data = tmp_path/'data'
    tiny_data(data)
    protocol = load_protocol()
    protocol['protocol'].update(total_epochs=2, trained_warmup=1, seeds=[1])
    protocol['core'].update(batch_size=4, pretrained=False)
    protocol['datasets']['iwildcam'] = config_for(data)['dataset_config']
    protocol_path = tmp_path/'protocol.yml'
    protocol_path.write_text(yaml.safe_dump(protocol))
    root = tmp_path/'fresh_cli'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTORCH_NVML_BASED_CUDA_CHECK='0', OPTLOSS_MODEL_CACHE=str(tmp_path/'cache'),
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def run(module, *args):
        proc = subprocess.run([sys.executable, '-m', module, *map(str,args)], cwd=REPO,
                              env=env, text=True, capture_output=True, timeout=90)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        return proc

    run('configs.gen_campaign', '--protocol', protocol_path, '--root', root,
        '--datasets', 'iwildcam', '--models', 'MobileNetV2', '--arms', 'all',
        '--caps', 'L50_G50', 'L100_G100', '--pretrained', 'false')
    run('src.pipeline.campaign', 'freeze', '--root', root)
    configs = sorted(root.rglob('config.json'))
    assert len(configs) == 14
    for config in configs:
        run('src.experiments.runner', config)
    active = [json.loads(path.read_text()) for path in configs
              if json.loads(path.read_text())['arm'] == 'tralo']
    assert all(c['hyperparams']['constraint_epochs'] == 1 for c in active)
    assert any(c['results']['constraint_steps_applied'] > 0 for c in active)
    manifest, inventory = validate_receipts(root)
    assert len(inventory['completed']) == 14
    output = tmp_path/'report.json'
    scored = run('scripts.deployed_h2h', '--campaign', root, '--json', output)
    assert 'TraLO - clip' in scored.stdout
    assert len(json.loads(output.read_text())) == 14
    copied = tmp_path/'copied_complete_campaign'
    shutil.copytree(root, copied)
    # Either complete copy alone is usable, but copies are not new seed evidence.
    copied_output = tmp_path/'copied_report.json'
    run('scripts.deployed_h2h', '--campaign', copied, '--json', copied_output)
    assert len(json.loads(copied_output.read_text())) == 14
    refused_json, refused_md = tmp_path/'duplicate.json', tmp_path/'duplicate.md'
    duplicate = subprocess.run([sys.executable, '-m', 'scripts.deployed_h2h', '--campaign',
                               str(root), str(copied), '--json', str(refused_json),
                               '--markdown', str(refused_md)], cwd=REPO, env=env,
                              capture_output=True, text=True)
    assert duplicate.returncode != 0
    assert not refused_json.exists() and not refused_md.exists()
    # A local reporter can use copied receipts with server arrays absent.
    data.rename(tmp_path/'server_arrays_unavailable')
    run('scripts.deployed_h2h', '--campaign', root)
    first_receipt = configs[0].with_name('completion_receipt.json')
    preserved_receipt = first_receipt.read_bytes()
    first_receipt.write_bytes(configs[1].with_name('completion_receipt.json').read_bytes())
    mixed = subprocess.run([sys.executable, '-m', 'scripts.deployed_h2h', '--campaign', str(root)],
                            cwd=REPO, env=env, capture_output=True, text=True)
    assert mixed.returncode == 1 and 'foreign/mixed' in mixed.stdout
    first_receipt.write_bytes(preserved_receipt)
    cfg = json.loads(configs[0].read_text())
    cfg['status'] = 'failed'
    configs[0].write_text(json.dumps(cfg))
    refused = tmp_path/'must-not-exist.json'
    proc = subprocess.run([sys.executable, '-m', 'scripts.deployed_h2h', '--campaign', str(root),
                           '--json', str(refused)], cwd=REPO, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 1 and 'incomplete' in proc.stdout
    assert not refused.exists()
