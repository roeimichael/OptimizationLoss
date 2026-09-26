"""Pilot gate for the backbone replication (experiments/claude_backbone_replication_prereg_20260926.md).

python analysis/repl_pilot_gate.py SEED_DIR

Label-free: reads summaries and event logs only. Exit code 0 = PASS.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

ARMS = ('clipper', 'tralo_null', 'tralo_adam', 'tralo_target', 'sham_target')
CAP, UPDATES = 76, 1810


def main():
    d = Path(sys.argv[1])
    s = {r['arm']: r for r in json.loads((d / 'summary.json').read_text())}
    checks = {}
    checks['all five arms complete'] = set(s) == set(ARMS)
    checks['warm-up and batch hashes identical'] = len({(s[a]['warmup_sha256'], s[a]['batch_sha256']) for a in s}) == 1
    checks['%d updates in every arm' % UPDATES] = {s[a]['task_updates_applied'] for a in s} == {UPDATES}
    ok = True
    for arm in ('tralo_target', 'sham_target'):
        for r in s[arm]['targeted_steps']:
            if r['applied']:
                ok &= r['hard_before'] > CAP
                if arm == 'tralo_target':
                    ok &= r['hard_after'] <= CAP
    checks['applied steps start above the cap; target lands at or below it'] = ok
    t_steps = s['tralo_target']['targeted_steps']
    checks['tralo_target applies at least one step (%d of %d)' % (sum(r['applied'] for r in t_steps), len(t_steps))] = \
        any(r['applied'] for r in t_steps)
    binding = sum(r['hard_before'] > CAP for r in t_steps)
    print('  (amendment 1) checks with hard count > %d in tralo_target: %d of %d' % (CAP, binding, len(t_steps)))
    a, b = t_steps[0], s['sham_target']['targeted_steps'][0]
    checks['first-check radii equal when both applied'] = not (a['applied'] and b['applied']) or abs(a['radius'] - b['radius']) < 1e-12
    stamps = [json.loads(line)['timestamp'] for f in d.rglob('events.jsonl') for line in f.read_text().splitlines() if line.strip()]
    t = lambda x: datetime.fromisoformat(x.replace('Z', '+00:00'))
    minutes = (t(max(stamps)) - t(min(stamps))).total_seconds() / 60
    checks['seed finished in <= 60 min (%.1f)' % minutes] = minutes <= 60
    for name, passed in checks.items():
        print('%s  %s' % ('PASS' if passed else 'FAIL', name))
    for r in t_steps:
        print('  target step: applied=%s hard %s -> %s radius %s' % (r['applied'], r['hard_before'], r.get('hard_after'), r.get('radius')))
    verdict = all(checks.values())
    print('GATE', 'PASS' if verdict else 'FAIL')
    sys.exit(0 if verdict else 1)


if __name__ == '__main__':
    main()
