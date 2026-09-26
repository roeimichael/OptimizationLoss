import json, glob, sys, re
import numpy as np
from lab import paired, mean_ci
names = sys.argv[1:]
for name in names:
    files = sorted(f for f in glob.glob('rows_%s_*.json' % name) if re.fullmatch(r'rows_%s_[0-9]+\.json' % re.escape(name), f))
    rows = [r for f in files for r in json.load(open(f))['rows']]
    rows.sort(key=lambda r: r['seed'])
    get = lambda k: [r.get(k, float('nan')) if not isinstance(r.get(k), (list, dict)) else float('nan') for r in rows]
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    print('=== %s  n_seeds=%d' % (name, len(rows)))
    for k in keys:
        if k == 'seed': continue
        v = mean_ci(get(k))
        if 'mean' in v: print('   %-26s %9.4f [%9.4f,%9.4f]' % (k, v['mean'], v['lo'], v['hi']))
    if 'clip' in keys:
        pairs = [('target', 'clip'), ('sham', 'clip'), ('target', 'sham'), ('ce_more', 'clip'), ('bayes', 'clip'), ('clip_em', 'clip'), ('joint_w1', 'clip_em'), ('joint_w10', 'clip_em'), ('trep', 'srep'), ('trep', 'crep'), ('srep', 'crep'), ('trep_gain_per_step', 'srep_gain_per_step'), ('trep_decay_per_block', 'srep_decay_per_block')]
        for k in keys:
            if re.match(r'^(joint|jtrain)_w[0-9.]+$', k):
                pairs += [(k, 'ce_more'), (k, 'clip')]
            if re.match(r'^joint_w[0-9.]+$', k):
                pairs += [(k, k.replace('joint', 'jtrain'))]
    else:
        pairs = [(a + '_local', b + '_local') for a, b in [('target', 'clip'), ('sham', 'clip'), ('target', 'sham'), ('bayes', 'clip'), ('tseq', 'clip'), ('tseq', 'tseqsham')]]
        pairs += [('clip_local', 'clip_global'), ('target_global', 'clip_global'), ('bayes_global', 'clip_global')]
        pairs += [('joint_local_local', 'ce_more_local'), ('joint_local_local', 'ce_more_global'), ('joint_local_global', 'ce_more_global'),
                  ('joint_global_global', 'ce_more_global'), ('joint_global_local', 'ce_more_local'), ('ce_more_local', 'clip_local'), ('joint_local_local', 'clip_local'), ('joint_global_local', 'clip_local'), ('joint_global_global', 'clip_global')]
    for a, b in pairs:
        v = paired(get(a), get(b))
        if 'mean' in v:
            print('   PAIR %-22s - %-12s %+.4f [%+.4f,%+.4f] n=%d W/L/T %d/%d/%d' % (a, b, v['mean'], v['lo'], v['hi'], v['n'], v['wins'], v['losses'], v['ties']))
