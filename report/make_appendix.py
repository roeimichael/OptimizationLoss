"""Build report/appendix/A_outputs.tex: every scorer output in analysis/*.txt, verbatim, grouped by study.

Long lines are wrapped (continuation lines start with '  >> ') and non-ASCII characters are spelled out, so pdflatex
can set every byte. The wrapped copies go to report/appendix/raw/; the analysis files themselves are untouched."""
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'report' / 'appendix' / 'raw'
WIDTH = 145
GROUPS = [
    ('Step-ensemble studies (2026-09-27/28)', ['stepens_r18_score', 'stepens_rgy_score', 'stepens_mn3_score',
                                               'stepens_b5_score', 'fmow_stepens_score', 'stepens_r18_swaps',
                                               'stepens_rgy_swaps', 'stepens_mn3_swaps', 'stepens_b5_swaps']),
    ("Yuval's pipeline: ResNet18 and EfficientNet-B5 blocks (2026-09-27)",
     ['yuval_r18_score', 'yuval_b5_score', 'yuval_r18_ranking', 'yuval_b5_ranking', 'yuval_r18_swaps', 'yuval_b5_swaps',
      'yuval_r18_step_collateral', 'yuval_b5_step_collateral', 'yuval_r18_count_noise', 'yuval_b5_count_noise',
      'yuval_r18_retrain_speed', 'yuval_b5_retrain_speed', 'yuval_r18_ensemble', 'yuval_b5_ensemble',
      'yuval_r18_deep_ensemble', 'yuval_b5_deep_ensemble', 'yuval_r18_ens_vs_pao', 'yuval_b5_ens_vs_pao',
      'yuval_metrics_rescore']),
    ("Yuval's pipeline: the recipe factorial and the small backbones (2026-09-27)",
     ['recipe_factorial_score', 'yuval_smallbb_score', 'yuval_mn3_ensemble', 'yuval_rgy_ensemble', 'yuval_mn3_swaps',
      'yuval_rgy_swaps', 'step_dose_across_backbones']),
    ('Our recipe: the knee rebuild studies (2026-09-25/26)',
     ['step_probe_score_20260926', 'bandcons_a1cap50', 'ens_bandcons_a1cap50', 'cutpair_cap76', 'ens_cutpair_cap76',
      'repl_mn3', 'repl_rgy', 'ens_c_mn3', 'ens_c_rgy', 'ens_conf_mn3', 'ens_conf_rgy', 'eviction_repl', 'washout_repl',
      'ens_vs_ens_target_studies']),
]
ASCII = {'\u2212': '-', '\u2013': '-', '\u2014': '--', '\u2248': '~', '\u00b1': '+/-', '\u2264': '<=', '\u2265': '>=',
         '\u00d7': 'x', '\u2192': '->', '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2026': '...',
         '\u00b7': '.', '\u2713': 'ok', '\u2717': 'x', '\u03bb': 'lambda', '\u03c1': 'rho', '\u0394': 'Delta'}


def ascii_line(line):
    out = ''.join(ASCII.get(ch, ch) for ch in line)
    return ''.join(ch if ord(ch) < 128 else f'<U+{ord(ch):04X}>' for ch in out)


def wrap(line):
    if len(line) <= WIDTH:
        return [line]
    parts = textwrap.wrap(line, WIDTH, subsequent_indent='  >> ', break_long_words=True, break_on_hyphens=False,
                          drop_whitespace=False)
    return parts or [line]


def tex_escape(text):
    return (text.replace('\\', '\\textbackslash{}').replace('_', '\\_').replace('%', '\\%').replace('&', '\\&')
            .replace('#', '\\#').replace('$', '\\$'))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    listed = {name for _, names in GROUPS for name in names}
    extra = sorted(p.stem for p in (ROOT / 'analysis').glob('*.txt') if p.stem not in listed)
    groups = GROUPS + ([('Other scorer outputs', extra)] if extra else [])
    out = ['\\section{Every scorer output, verbatim}',
           'Each file below is the text output of a scorer or analysis script, as committed under '
           '\\texttt{analysis/}, changed only as follows. Lines longer than %d characters are wrapped; a wrapped continuation '
           'starts with \\texttt{>>}. Non-ASCII characters are spelled out in ASCII. Files for studies that had not '
           'finished when this report was built are marked as missing.' % WIDTH, '']
    for title, names in groups:
        out.append('\\subsection{%s}' % tex_escape(title))
        for name in names:
            src = ROOT / 'analysis' / f'{name}.txt'
            out.append('\\subsubsection*{\\texttt{analysis/%s.txt}}' % tex_escape(name))
            out.append('\\addcontentsline{toc}{subsubsection}{\\texttt{%s.txt}}' % tex_escape(name))
            if not src.exists():
                out.append('\\emph{Not produced: this study had not finished when the report was built.}\n')
                continue
            lines = []
            for line in src.read_text(encoding='utf-8', errors='replace').splitlines():
                lines.extend(wrap(ascii_line(line.rstrip())))
            (RAW / f'{name}.txt').write_text('\n'.join(lines) + '\n', encoding='ascii')
            out.append('\\VerbatimInput[fontsize=\\tiny]{appendix/raw/%s.txt}' % name)
            out.append('')
    (ROOT / 'report' / 'appendix' / 'A_outputs.tex').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print(f'{sum(len(n) for _, n in groups)} files listed, {len(extra)} ungrouped')


if __name__ == '__main__':
    main()
