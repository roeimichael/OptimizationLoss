"""Check receipt-backed prediction rows and integer labels in an explicit campaign."""
import argparse
import io
import json
from pathlib import Path
import sys
import tempfile

LABEL_COLS = ('True_Label', 'Predicted_Label')
PRED_FILES = ('final_predictions.csv', 'final_predictions_raw.csv')


def row_count(path):
    with Path(path).open(encoding='utf-8') as stream:
        return sum(1 for _ in stream)


def label_dtype_ok(path):
    with Path(path).open(encoding='utf-8') as stream:
        header = stream.readline().strip().split(',')
        if not set(LABEL_COLS).issubset(header):
            return False, 'required label columns missing'
        indices = [(c, header.index(c)) for c in LABEL_COLS]
        for lineno, line in enumerate(stream, 2):
            parts = line.strip().split(',')
            if len(parts) != len(header):
                return False, 'line %d is torn (%d fields, expected %d)' % (lineno, len(parts), len(header))
            for name, index in indices:
                if not parts[index].strip().lstrip('-').isdigit():
                    return False, 'line %d: %s is not an integer class index' % (lineno, name)
    return True, 'ok'


def audit(roots, out=sys.stdout, deep=True):
    """Check the completed subset explicitly, while naming every pending run."""
    from src.pipeline.campaign import validate_receipts, safe_path
    problems = []
    for root in roots:
        try:
            manifest, inventory = validate_receipts(root, complete=False)
            print('Run inventory: ' + json.dumps(inventory), file=out)
            if not inventory['completed']:
                raise ValueError('no receipt-backed completed runs')
            if inventory['missing']:
                raise ValueError('completed runs are missing receipts')
            for rel in inventory['completed']:
                expected = manifest['data'][manifest['runs'][rel]['data_id']]['test_rows']
                for name in PRED_FILES:
                    path = (safe_path(root)/rel).with_name(name)
                    if row_count(path) != expected + 1:
                        problems.append((str(path), 'rows differ from frozen inventory'))
                    if deep:
                        ok, why = label_dtype_ok(path)
                        if not ok:
                            problems.append((str(path), why))
        except (ValueError, OSError, KeyError) as exc:
            problems.append((str(root), str(exc)))
    for path, problem in problems:
        print('FAIL %s: %s' % (path, problem), file=out)
    return problems


def self_test():
    with tempfile.TemporaryDirectory(prefix='pred_integrity_') as directory:
        path = Path(directory)/'fixture.csv'
        path.write_text('True_Label,Predicted_Label\n0,1\n1,1\n')
        checks = [label_dtype_ok(path)[0], row_count(path) == 3]
        path.write_text('True_Label,Predicted_Label\n0.5,1\n')
        checks.append(not label_dtype_ok(path)[0])
        path.write_text('True_Label,Predicted_Label\n0,1,extra\n')
        checks.append(not label_dtype_ok(path)[0])
        checks.append(bool(audit([directory], out=io.StringIO())))
    print('ALL PASS' if all(checks) else 'FAIL')
    return 0 if all(checks) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='*')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--shallow', action='store_true', help='skip lexical checks; receipts/row counts still required')
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.roots:
        parser.error('give at least one explicit campaign root')
    if audit(args.roots, deep=not args.shallow):
        return 1
    print('all completed prediction files intact; pending runs are listed above')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
