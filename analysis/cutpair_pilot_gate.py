"""Pilot gate for the CUTPAIR study (experiments/claude_cutpair_protocol_20260926.md, section Pilot gate).

python analysis/cutpair_pilot_gate.py SEED_DIR

Label-free: reads summaries and logs only, never development labels. Exit code 0 = PASS.
"""
import json
import sys
from pathlib import Path

ARMS = ('clipper', 'tralo_null', 'aug_clip', 'cutpair_aug', 'cutpair_aug_shift')
UPDATES = 1810
EPOCHS = 10


def main():
    d = Path(sys.argv[1])
    summary = {r['arm']: r for r in json.loads((d / 'summary.json').read_text())}
    checks = {}
    checks['all five arms complete'] = set(summary) == set(ARMS)
    ids = {(summary[a]['warmup_sha256'], summary[a]['batch_sha256'], summary[a]['tta_draws_sha256']) for a in summary}
    checks['warm-up, batch and TTA hashes identical'] = len(ids) == 1
    updates = {summary[a]['task_updates_applied'] for a in summary}
    checks['%d updates in every arm' % UPDATES] = updates == {UPDATES}
    batches = UPDATES // EPOCHS
    for arm in ('cutpair_aug', 'cutpair_aug_shift'):
        logs = summary.get(arm, {}).get('cut_logs', [])
        ok = bool(logs) and all(abs(r['realised_ratio_mean'] - 0.1) < 1e-6 and abs(r['realised_ratio_max'] - 0.1) < 1e-6
                                for r in logs if r['active_batches'])
        checks['%s dose ratio 0.1 where active' % arm] = ok
    logs = summary.get('cutpair_aug', {}).get('cut_logs', [])
    n_act = sum(r['n_act'] for r in logs) / max(len(logs), 1)
    p_act = sum(r['p_act'] for r in logs) / max(len(logs), 1)
    frac = sum(r['active_batches'] for r in logs) / max(batches * len(logs), 1)
    checks['cutpair_aug mean N_act >= 20 (%.1f)' % n_act] = n_act >= 20
    checks['cutpair_aug mean P_act >= 20 (%.1f)' % p_act] = p_act >= 20
    checks['cutpair_aug active-batch fraction >= 0.5 (%.3f)' % frac] = frac >= 0.5
    events = [json.loads(line) for f in d.rglob("events.jsonl") for line in f.read_text().splitlines() if line.strip()]
    stamps = [e['timestamp'] for e in events if 'timestamp' in e]
    from datetime import datetime
    t = lambda s: datetime.fromisoformat(s.replace('Z', '+00:00'))
    minutes = (t(max(stamps)) - t(min(stamps))).total_seconds() / 60
    checks['seed finished in <= 90 min (%.1f)' % minutes] = minutes <= 90
    for arm in ('cutpair_aug', 'cutpair_aug_shift'):
        for r in summary.get(arm, {}).get('cut_logs', []):
            print('  %-17s ep %2d rank %3d tau %+6.2f N_act %4d P_act %4d active %3d/%d hinge %.3f'
                  % (arm, r['epoch'], r['anchor_rank'], r['tau'], r['n_act'], r['p_act'], r['active_batches'],
                     batches, r['hinge_mean'] if r['hinge_mean'] is not None else float('nan')))
    for name, ok in checks.items():
        print('%s  %s' % ('PASS' if ok else 'FAIL', name))
    verdict = all(checks.values())
    print('GATE', 'PASS' if verdict else 'FAIL')
    sys.exit(0 if verdict else 1)


if __name__ == '__main__':
    main()
