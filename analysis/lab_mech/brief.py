"""Print the pooled lines of mechanism_out.json compactly."""
import json
import sys


def fmt(v):
    if isinstance(v, dict) and 'mean' in v:
        sd = v.get('sd')
        return '%.4f [%.4f,%.4f] n=%d sd=%s' % (v['mean'], v['ci95'][0], v['ci95'][1], v['n'],
                                                 'na' if sd is None else '%.4f' % sd)
    return json.dumps(v)


def walk(obj, prefix=''):
    if isinstance(obj, dict) and 'mean' in obj:
        print(prefix, fmt(obj))
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.startswith('epoch') and '--epochs' not in sys.argv:
                continue
            walk(v, prefix + '.' + k)
    else:
        print(prefix, json.dumps(obj))


for study in json.load(open(sys.argv[1])):
    print('=====', study['root'], 'cap', study['cap'])
    walk(study)
