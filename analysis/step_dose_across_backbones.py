"""Does TraLO's step gain follow how far the model is over the cap, across studies? (post hoc, stored scores)

Usage (server): python analysis/step_dose_across_backbones.py RUNS_DIR

Per seed: excess = the final targeted step's hard_before - 76 (argmax grade-3 count before it), and
P2 = tralo - sham capped_first cc-F1 (tralo_target/sham_target in the v3-recipe studies, where a step
follows every epoch from 6 to 10; tralo_final/sham_final in the Yuval-pipeline study, one final step).
"""

import glob
import json
import sys

import numpy as np
from scipy import stats

STUDIES = (('ResNet18, v3 recipe, cap 76', 'claude-target-20260925'), ('MobileNetV3, v3 recipe', 'claude-repl-mn3'),
           ('RegNetY, v3 recipe', 'claude-repl-rgy'), ('ResNet18, Yuval pipeline', 'claude-yuval-r18'),
           ('EfficientNet-B5, Yuval pipeline', 'claude-yuval-b5'))


def main(runs):
    for name, root in STUDIES:
        ex, p2 = [], []
        for path in sorted(glob.glob(f'{runs}/{root}/seed*/summary.json')):
            s = json.load(open(path))
            if isinstance(s, list):
                a = {x['arm']: x for x in s}
                ex.append(a['tralo_target']['targeted_steps'][-1]['hard_before'] - 76)
                p2.append(100 * (a['tralo_target']['scores']['capped_first']['cc_f1'] - a['sham_target']['scores']['capped_first']['cc_f1']))
            else:
                ex.append(s['steps']['tralo_final']['hard_before'] - 76)
                p2.append(100 * (s['arms']['tralo_final']['scores']['capped_first']['cc_f1'] - s['arms']['sham_final']['scores']['capped_first']['cc_f1']))
        ex, p2 = np.array(ex), np.array(p2)
        b = ex > 0
        r = stats.spearmanr(ex[b], p2[b])
        print(f'{name:28s} n {len(ex)} binding {b.sum():2d}  excess median {np.median(ex):5.1f} mean {ex.mean():5.1f}  '
              f'P2 mean {p2.mean():+.2f}  Spearman over binding {r.statistic:+.2f} p {r.pvalue:.3f}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
