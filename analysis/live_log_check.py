"""Label-free health check of a live BANDCONS block: reads events.jsonl only, never reports or labels.

python analysis/live_log_check.py RUN_ROOT
Per seed and arm: epochs done, seconds per epoch, training CE, natural grade-3 count at epoch start,
and for bandcons* arms the band disagreement and realised dose ratio.
"""
import json
import sys
from datetime import datetime
from pathlib import Path


def ts(row):
    return datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))


def main():
    root = Path(sys.argv[1])
    flags = []
    for d in sorted(root.glob('seed*')):
        if not d.is_dir():
            continue
        for arm_dir in sorted(p for p in d.iterdir() if p.is_dir()):
            rows = [json.loads(line) for line in open(arm_dir / 'events.jsonl')]
            epochs = [r for r in rows if r.get('event') == 'epoch']
            if not epochs:
                continue
            secs = [(ts(b) - ts(a)).total_seconds() for a, b in zip(epochs, epochs[1:])]
            ce = [round(r['training_ce'], 3) for r in epochs]
            nat = [r.get('natural_count') for r in epochs if r.get('natural_count') is not None]
            dis = [round(r['band_disagreement_start'], 2) for r in epochs if r.get('band_disagreement_start') is not None]
            ratio = [r.get('realised_ratio_max') for r in epochs if r.get('realised_ratio_max') is not None]
            done = any(r.get('event') == 'training_completed' for r in rows)
            print('%s %-13s ep=%2d%s s/ep=%s ce=%s nat=%s dis=%s ratio_ok=%s' % (
                d.name, arm_dir.name, len(epochs), '*' if done else ' ',
                round(sum(secs) / len(secs)) if secs else '-', ce[-3:], nat, dis,
                all(abs(x - 0.1) < 1e-6 for x in ratio) if ratio else '-'))
            if any(x > 10 for x in dis):
                flags.append('%s %s disagreement > 10' % (d.name, arm_dir.name))
            if ratio and not all(abs(x - 0.1) < 1e-6 for x in ratio):
                flags.append('%s %s dose ratio off' % (d.name, arm_dir.name))
            if ce and ce[-1] != ce[-1]:
                flags.append('%s %s nonfinite CE' % (d.name, arm_dir.name))
    print('FLAGS:', flags or 'none')


if __name__ == '__main__':
    main()
