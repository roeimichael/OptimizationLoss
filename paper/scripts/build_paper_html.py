"""Render paper/main.tex into paper/paper.html -- a polished
two-column academic-paper layout: full-width title block + abstract spanning both
columns, then two justified columns; numbered sections; figure floats (centered images
+ captions); LaTeX tables rendered as real HTML tables; inline $math$, \cite, \ref,
\textbf/\emph/\texttt; equation/align rendered readably; itemize/enumerate.

Reuses the battle-tested LaTeX parser from scripts/build_html_report.py
(latex_inline, _render_tabular, _math, _match_brace, _expand_cmd) so tables and inline
math render exactly as in the results report.
"""
import base64
import os
import re
import sys
import html as _html

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, 'paper')
FIGDIR = os.path.join(PAPER, 'figures')
TABDIR = os.path.join(PAPER, 'tables')

# Reuse the existing LaTeX parser from build_html_report.py WITHOUT triggering its
# module-level report build (that file has no __main__ guard and would write
# paper/REPORT.html and os.chdir on import). We exec only the parser prefix --
# everything up to the "assemble" section -- in an isolated namespace.
sys.path.insert(0, HERE)
with open(os.path.join(HERE, 'build_html_report.py'), encoding='utf-8') as _f:
    _src = _f.read()
# Cut before the markdown+figures section so we don't require the `markdown` package
# nor trigger the report assembly. All needed parser fns are defined before this point.
_cut = _src.find('# ----------------------------------------------------------------------------- markdown + figures')
if _cut < 0:
    _cut = _src.find('SECTIONS = []')
# the import of `markdown` sits at the top; strip it so the prefix has no hard dep
_prefix = _src[:_cut].replace('\nimport markdown\n', '\n')
_ns = {'__name__': '_bhr_parser', '__file__': os.path.join(HERE, 'build_html_report.py')}
exec(compile(_prefix, 'build_html_report.py', 'exec'), _ns)
latex_inline = _ns['latex_inline']
_render_tabular = _ns['_render_tabular']
_math = _ns['_math']
_match_brace = _ns['_match_brace']
_expand_cmd = _ns['_expand_cmd']
os.chdir(ROOT)  # build_html_report's prefix calls os.chdir(ROOT); keep it explicit

# ----------------------------------------------------------------------------- helpers


def read(p):
    with open(p, encoding='utf-8') as f:
        return f.read()


def strip_comments(tex):
    """Drop full-line and trailing LaTeX comments (respect escaped \\%)."""
    out = []
    for line in tex.split('\n'):
        # remove a % that is not escaped
        res, i = [], 0
        while i < len(line):
            c = line[i]
            if c == '\\' and i + 1 < len(line):
                res.append(line[i:i + 2])
                i += 2
                continue
            if c == '%':
                break
            res.append(c)
            i += 1
        out.append(''.join(res))
    return '\n'.join(out)


def fig_b64(name):
    # main.tex uses extensionless \includegraphics{fig_x} (pdflatex picks the .pdf);
    # the HTML preview always resolves the >=300 dpi .png sibling.
    base = os.path.splitext(os.path.basename(name))[0]
    p = os.path.join(FIGDIR, base + '.png')
    if not os.path.exists(p):
        return None
    with open(p, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()


# ----------------------------------------------------------------------------- citations

# Discover cite keys in order of first appearance so we can number the reference list.
CITES = []          # ordered unique keys
CITE_NUM = {}       # key -> int


def register_cite(keys):
    nums = []
    for k in keys.split(','):
        k = k.strip()
        if not k:
            continue
        if k not in CITE_NUM:
            CITES.append(k)
            CITE_NUM[k] = len(CITES)
        nums.append(str(CITE_NUM[k]))
    return nums


# ----------------------------------------------------------------------------- bib parse


def parse_bib(path):
    """Very small .bib parser -> {key: formatted-html-string}."""
    if not os.path.exists(path):
        return {}
    txt = read(path)
    entries = {}
    # find @type{key, ...}
    i = 0
    while True:
        m = re.search(r'@(\w+)\s*\{', txt[i:])
        if not m:
            break
        start = i + m.end() - 1            # at '{'
        end = _match_brace(txt, start)
        block = txt[start + 1:end]
        i = end + 1
        # key is up to first comma
        comma = block.find(',')
        if comma < 0:
            continue
        key = block[:comma].strip()
        fields_src = block[comma + 1:]
        fields = {}
        for fm in re.finditer(r'(\w+)\s*=\s*', fields_src):
            fname = fm.group(1).lower()
            j = fm.end()
            if j >= len(fields_src):
                break
            if fields_src[j] == '{':
                k = _match_brace(fields_src, j)
                val = fields_src[j + 1:k]
            elif fields_src[j] == '"':
                k = fields_src.find('"', j + 1)
                val = fields_src[j + 1:k] if k > 0 else ''
            else:
                k = re.search(r'[,\n]', fields_src[j:])
                val = fields_src[j:j + k.start()] if k else fields_src[j:]
            fields[fname] = re.sub(r'\s+', ' ',
                                   val.replace('{', '').replace('}', '')).strip()
        entries[key] = fields
    return entries


def fmt_bib(fields):
    auth = fields.get('author', '')
    auth = auth.replace(' and ', ', ')
    title = fields.get('title', '')
    year = fields.get('year', '')
    venue = (fields.get('booktitle') or fields.get('journal')
             or fields.get('publisher') or '')
    parts = []
    if auth:
        parts.append(auth + '.')
    if title:
        parts.append('<em>%s</em>.' % _html.escape(title))
    if venue:
        parts.append(_html.escape(venue) + ('.' if not venue.endswith('.') else ''))
    if year:
        parts.append(_html.escape(year) + '.')
    return ' '.join(parts)


# ----------------------------------------------------------------------------- labels / refs

LABELS = {}     # label -> display number ("3", "5.2", "1" for fig/tab/eq)
# We resolve refs in a second pass by scanning assigned numbers.


# ----------------------------------------------------------------------------- inline conversion for body text


def inline(s):
    """Inline LaTeX -> HTML for body prose.

    Handles \\cite (-> [n]), \\ref/\\eqref (-> resolved number placeholder),
    then defers to the shared latex_inline for textbf/emph/texttt/$math$ etc.
    Returns HTML with __REF:label__ placeholders to be filled after numbering.
    """
    if s is None:
        return ''
    # natbib \citep/\citet render the same as \cite in this numbered HTML preview
    s = s.replace('\\citep{', '\\cite{').replace('\\citet{', '\\cite{')
    # \cite{a,b} -> bracketed numbers
    def _cite(arg):
        nums = register_cite(arg)
        return '[' + ', '.join(nums) + ']'
    s = _expand_cmd(s, 'cite', _cite)
    # \eqref / \ref -> placeholder token (resolved later)
    s = _expand_cmd(s, 'eqref', lambda a: '(\x01REF:%s\x01)' % a.strip())
    s = _expand_cmd(s, 'ref', lambda a: '\x01REF:%s\x01' % a.strip())
    # \label{..} inside prose -> drop (handled at block level)
    s = _expand_cmd(s, 'label', lambda a: '')
    # \paragraph already handled at block level; defensively drop \itemsep etc.
    s = re.sub(r'\\itemsep[0-9.a-z]*', '', s)
    # hand to shared renderer (does $math$, textbf, emph, texttt, escaping)
    out = latex_inline(s)
    return out


def resolve_refs(html_text):
    def _rep(m):
        lab = m.group(1)
        return LABELS.get(lab, '?')
    return re.sub(r'\x01REF:([^\x01]+)\x01', _rep, html_text)


# ----------------------------------------------------------------------------- list rendering


def render_itemize(body, ordered):
    items = []
    # split on \item at top level
    parts = re.split(r'\\item\b', body)
    for p in parts[1:]:               # parts[0] is pre-\item whitespace
        p = p.strip()
        if not p:
            continue
        items.append('<li>%s</li>' % inline(p))
    tag = 'ol' if ordered else 'ul'
    return '<%s class="tex-list">%s</%s>' % (tag, '\n'.join(items), tag)


# ----------------------------------------------------------------------------- equation rendering


def render_equation(body):
    """Render an equation/align body as a readable centered math line."""
    # strip \label
    body = _expand_cmd(body, 'label', lambda a: '')
    # underbrace{X}_{text} -> X (text)
    def _ub(arg):
        return arg
    body = _expand_cmd(body, 'underbrace', _ub)
    # the _{...} subscript after an underbrace becomes an annotation; keep it small
    # Convert \frac{a}{b} -> (a)/(b)
    def _frac(s):
        out, i, tok = [], 0, '\\frac{'
        while True:
            j = s.find(tok, i)
            if j < 0:
                out.append(s[i:])
                break
            out.append(s[i:j])
            k1 = _match_brace(s, j + len(tok) - 1)
            num = s[j + len(tok):k1]
            # next char should be '{'
            m = k1 + 1
            while m < len(s) and s[m] in ' \t':
                m += 1
            if m < len(s) and s[m] == '{':
                k2 = _match_brace(s, m)
                den = s[m + 1:k2]
                out.append('(%s)/(%s)' % (_frac(num), _frac(den)))
                i = k2 + 1
            else:
                out.append('(%s)/' % _frac(num))
                i = k1 + 1
        return ''.join(out)
    body = _frac(body)
    # split align rows / multiple lines
    lines = re.split(r'\\\\', body)
    rendered = []
    for ln in lines:
        ln = ln.replace('&', ' ')
        ln = _math(ln)                # unicode-ize symbols
        # tidy spacing
        ln = re.sub(r'\s+', ' ', ln).strip()
        if ln:
            rendered.append(_html.escape(ln, quote=False))
    return '<div class="equation">%s</div>' % '<br/>'.join(rendered)


# ----------------------------------------------------------------------------- table float rendering (from \input fragment)


def render_table_fragment(tex):
    """Render all table floats in a fragment (.tex) as HTML, in source order.

    Mirrors build_html_report.tex_to_html but emits a numbered <table> with a
    real <caption>, and registers \\label -> table number.
    """
    tex = strip_comments(tex)
    floats = list(re.finditer(r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}', tex, re.S))
    blocks = [(f.group(0).find('*') != -1 and '*' in f.group(0)[:14], f.group(1))
              for f in floats] if floats else [(False, tex)]
    out = []
    for is_wide, blk in blocks:
        global TABLE_COUNTER
        TABLE_COUNTER += 1
        tnum = TABLE_COUNTER
        # register label(s)
        for lm in re.finditer(r'\\label\{([^}]*)\}', blk):
            LABELS[lm.group(1)] = str(tnum)
        # gather captions / tabulars / notes in position order
        items = []
        for mm in re.finditer(r'\\caption\*?\{', blk):
            star = blk[mm.start():mm.end()].endswith('*{')
            k = _match_brace(blk, mm.end() - 1)
            items.append((mm.start(), 'cap*' if star else 'cap', blk[mm.end():k]))
        for mm in re.finditer(r'\\begin\{tabular\}\{[^}]*\}', blk):
            end = blk.find('\\end{tabular}', mm.end())
            items.append((mm.start(), 'tab', blk[mm.end():end]))
        for mm in re.finditer(r'\{\\footnotesize', blk):
            k = _match_brace(blk, mm.start())
            note = blk[mm.start() + 1:k]
            note = re.sub(r'^\s*\\footnotesize', '', note)
            items.append((mm.start(), 'note', note))
        items.sort(key=lambda t: t[0])
        cls = 'tblfloat wide' if is_wide else 'tblfloat'
        out.append('<div class="%s">' % cls)
        for _, kind, payload in items:
            if kind == 'cap':
                out.append('<div class="tcaption"><span class="tnum">Table %d.</span> %s</div>'
                           % (tnum, _inline_caption(payload)))
            elif kind == 'cap*':
                out.append('<div class="subcaption">%s</div>' % _inline_caption(payload))
            elif kind == 'tab':
                out.append('<div class="tscroll">%s</div>' % _render_tabular(payload))
            elif kind == 'note':
                out.append('<div class="note">%s</div>' % _inline_caption(payload))
        out.append('</div>')
    return '\n'.join(out)


def _inline_caption(payload):
    """Captions may contain \\cite and \\ref; resolve cites now, defer refs."""
    payload = payload.replace('\\citep{', '\\cite{').replace('\\citet{', '\\cite{')
    def _cite(arg):
        nums = register_cite(arg)
        return '[' + ', '.join(nums) + ']'
    payload = _expand_cmd(payload, 'cite', _cite)
    payload = _expand_cmd(payload, 'eqref', lambda a: '(\x01REF:%s\x01)' % a.strip())
    payload = _expand_cmd(payload, 'ref', lambda a: '\x01REF:%s\x01' % a.strip())
    payload = _expand_cmd(payload, 'label', lambda a: '')
    return latex_inline(payload)


# ----------------------------------------------------------------------------- figure rendering


def render_figure(block):
    """block = content between \\begin{figure}..\\end{figure} (any star)."""
    global FIG_COUNTER
    FIG_COUNTER += 1
    fnum = FIG_COUNTER
    for lm in re.finditer(r'\\label\{([^}]*)\}', block):
        LABELS[lm.group(1)] = str(fnum)
    # filename
    img_html = ''
    gm = re.search(r'\\includegraphics(\[[^\]]*\])?\{([^}]*)\}', block)
    if gm:
        name = gm.group(2)
        uri = fig_b64(name)
        if uri:
            img_html = '<img src="%s" alt="%s"/>' % (uri, _html.escape(os.path.basename(name)))
        else:
            img_html = '<p class="missing">[missing figure: %s]</p>' % _html.escape(name)
    cap_html = ''
    cm = re.search(r'\\caption\*?\{', block)
    if cm:
        k = _match_brace(block, cm.end() - 1)
        cap_html = ('<figcaption><span class="fnum">Figure %d.</span> %s</figcaption>'
                    % (fnum, _inline_caption(block[cm.end():k])))
    return '<figure class="figfloat">%s%s</figure>' % (img_html, cap_html)


# ----------------------------------------------------------------------------- main body parser

SECTION_COUNTER = 0
SUBSECTION_COUNTER = 0
FIG_COUNTER = 0
TABLE_COUNTER = 0
EQ_COUNTER = 0


def parse_body(body):
    """Walk the document body, emitting HTML blocks. Returns html string."""
    global SECTION_COUNTER, SUBSECTION_COUNTER, EQ_COUNTER
    html_out = []
    i = 0
    n = len(body)

    # We tokenize by scanning for environment/command starts; everything between is prose.
    # Build a list of "events" (start_index, kind, payload, end_index).
    pos = 0
    text_buf = []

    def flush_text():
        raw = ''.join(text_buf)
        text_buf.clear()
        # split into paragraphs on blank lines
        for para in re.split(r'\n\s*\n', raw):
            para = para.strip()
            if not para:
                continue
            html_para = inline(para)
            if html_para.strip():
                html_out.append('<p>%s</p>' % html_para)

    while pos < n:
        # find the next interesting token
        m = re.compile(
            r'\\section\{|\\subsection\{|\\paragraph\{'
            r'|\\begin\{abstract\}|\\begin\{itemize\}|\\begin\{enumerate\}'
            r'|\\begin\{figure\*?\}|\\begin\{equation\*?\}|\\begin\{align\*?\}'
            r'|\\input\{|\\label\{|\\maketitle\b|\\bibliographystyle\{|\\bibliography\{'
        ).search(body, pos)
        if not m:
            text_buf.append(body[pos:])
            break
        text_buf.append(body[pos:m.start()])
        tok = m.group(0)

        if tok.startswith('\\section{'):
            flush_text()
            k = _match_brace(body, m.end() - 1)
            title = inline(body[m.end():k])
            SECTION_COUNTER += 1
            SUBSECTION_COUNTER = 0
            html_out.append('<h2><span class="secnum">%d</span> %s</h2>'
                            % (SECTION_COUNTER, title))
            pos = k + 1
        elif tok.startswith('\\subsection{'):
            flush_text()
            k = _match_brace(body, m.end() - 1)
            title = inline(body[m.end():k])
            SUBSECTION_COUNTER += 1
            html_out.append('<h3><span class="secnum">%d.%d</span> %s</h3>'
                            % (SECTION_COUNTER, SUBSECTION_COUNTER, title))
            pos = k + 1
        elif tok.startswith('\\paragraph{'):
            flush_text()
            k = _match_brace(body, m.end() - 1)
            title = inline(body[m.end():k])
            html_out.append('<p class="runin"><strong>%s</strong> ' % title)
            # the following prose belongs to this paragraph; we keep it as normal text
            # but need to close the <p>. Simplest: emit run-in as its own opener and let
            # the next text flush continue. Instead, capture until next token's prose.
            pos = k + 1
            # capture immediate following prose up to next blank-line OR next token
            nxt = re.compile(
                r'\\section\{|\\subsection\{|\\paragraph\{'
                r'|\\begin\{abstract\}|\\begin\{itemize\}|\\begin\{enumerate\}'
                r'|\\begin\{figure\*?\}|\\begin\{equation\*?\}|\\begin\{align\*?\}'
                r'|\\input\{|\\label\{|\\bibliographystyle\{|\\bibliography\{'
                r'|\n\s*\n'
            ).search(body, pos)
            end = nxt.start() if nxt else n
            chunk = body[pos:end].strip()
            html_out[-1] += inline(chunk) + '</p>'
            pos = end
        elif tok.startswith('\\begin{abstract}'):
            flush_text()
            end = body.find('\\end{abstract}', m.end())
            abs = body[m.end():end]
            html_out.append('<div class="abstract"><div class="abshead">Abstract</div>'
                            '<p>%s</p></div>' % inline(abs.strip()))
            pos = end + len('\\end{abstract}')
        elif tok.startswith('\\begin{itemize}'):
            flush_text()
            end = body.find('\\end{itemize}', m.end())
            html_out.append(render_itemize(body[m.end():end], ordered=False))
            pos = end + len('\\end{itemize}')
        elif tok.startswith('\\begin{enumerate}'):
            flush_text()
            end = body.find('\\end{enumerate}', m.end())
            html_out.append(render_itemize(body[m.end():end], ordered=True))
            pos = end + len('\\end{enumerate}')
        elif tok.startswith('\\begin{figure'):
            flush_text()
            em = re.search(r'\\end\{figure\*?\}', body[m.end():])
            end = m.end() + em.start()
            html_out.append(render_figure(body[m.end():end]))
            pos = m.end() + em.end()
        elif tok.startswith('\\begin{equation') or tok.startswith('\\begin{align'):
            flush_text()
            envname = re.match(r'\\begin\{(\w+\*?)\}', tok).group(1)
            em = re.search(r'\\end\{%s\}' % re.escape(envname), body[m.end():])
            end = m.end() + em.start()
            eqbody = body[m.end():end]
            EQ_COUNTER += 1
            for lm in re.finditer(r'\\label\{([^}]*)\}', eqbody):
                LABELS[lm.group(1)] = str(EQ_COUNTER)
            eq_html = render_equation(eqbody)
            html_out.append('<div class="eqwrap">%s<span class="eqnum">(%d)</span></div>'
                            % (eq_html, EQ_COUNTER))
            pos = m.end() + em.end()
        elif tok.startswith('\\input{'):
            flush_text()
            k = _match_brace(body, m.end() - 1)
            fname = body[m.end():k].strip()
            if not fname.endswith('.tex'):
                fname += '.tex'
            # path relative to PAPER
            tpath = os.path.join(PAPER, fname.replace('/', os.sep))
            if os.path.exists(tpath):
                html_out.append(render_table_fragment(read(tpath)))
            else:
                html_out.append('<p class="missing">[missing input: %s]</p>'
                                % _html.escape(fname))
            pos = k + 1
        elif tok.startswith('\\label{'):
            # bare label in prose: bind to current section number
            k = _match_brace(body, m.end() - 1)
            lab = body[m.end():k].strip()
            if SUBSECTION_COUNTER:
                LABELS[lab] = '%d.%d' % (SECTION_COUNTER, SUBSECTION_COUNTER)
            else:
                LABELS[lab] = str(SECTION_COUNTER)
            pos = k + 1
        elif tok.startswith('\\maketitle'):
            pos = m.end()
        elif tok.startswith('\\bibliographystyle{'):
            k = _match_brace(body, m.end() - 1)
            pos = k + 1
        elif tok.startswith('\\bibliography{'):
            flush_text()
            html_out.append('\x02BIBLIO\x02')   # placeholder, filled after full pass
            k = _match_brace(body, m.end() - 1)
            pos = k + 1
        else:
            text_buf.append(tok)
            pos = m.end()

    flush_text()
    return '\n'.join(html_out)


# ----------------------------------------------------------------------------- title block


def parse_title(tex):
    tm = re.search(r'\\title\{', tex)
    title = ''
    if tm:
        k = _match_brace(tex, tm.end() - 1)
        raw = tex[tm.end():k]
        raw = raw.replace('\\\\', '\x03')           # line breaks
        raw = inline(raw)
        title = raw.replace('\x03', '<br/>')
    am = re.search(r'\\author\{', tex)
    author = ''
    if am:
        k = _match_brace(tex, am.end() - 1)
        author = inline(tex[am.end():k].replace('\\\\', '\x03')).replace('\x03', '<br/>')
    return title, author


# ----------------------------------------------------------------------------- assemble


def main():
    tex = read(os.path.join(PAPER, 'main.tex'))
    tex = strip_comments(tex)

    title, author = parse_title(tex)

    # body = between \begin{document} and \end{document}
    dm = re.search(r'\\begin\{document\}(.*)\\end\{document\}', tex, re.S)
    body_src = dm.group(1) if dm else tex

    body_html = parse_body(body_src)

    # build bibliography from references.bib in citation order
    bib = parse_bib(os.path.join(PAPER, 'references.bib'))
    ref_items = []
    for k in CITES:
        fields = bib.get(k)
        label = '[%d]' % CITE_NUM[k]
        if fields:
            ref_items.append('<li id="ref-%d"><span class="rnum">%s</span> %s</li>'
                             % (CITE_NUM[k], label, fmt_bib(fields)))
        else:
            ref_items.append('<li id="ref-%d"><span class="rnum">%s</span> <code>%s</code></li>'
                             % (CITE_NUM[k], label, _html.escape(k)))
    biblio = ('<h2><span class="secnum"></span>References</h2>'
              '<ol class="refs">%s</ol>' % '\n'.join(ref_items))
    body_html = body_html.replace('\x02BIBLIO\x02', biblio)

    # resolve \ref placeholders now that all numbers are assigned
    body_html = resolve_refs(body_html)

    # split abstract+title (full width) from the rest (two columns)
    # abstract block is rendered inline in body_html; we want it full width, so we
    # pull it out and place it in the header band.
    abs_html = ''
    am = re.search(r'<div class="abstract">.*?</div>\s*</div>', body_html, re.S)
    if am:
        abs_html = am.group(0)
        body_html = body_html[:am.start()] + body_html[am.end():]

    page = DOC_TEMPLATE % {
        'css': CSS,
        'title': title,
        'author': author,
        'abstract': abs_html,
        'columns': body_html,
    }
    out_path = os.path.join(PAPER, 'paper.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(page)
    size = os.path.getsize(out_path)
    nimg = page.count('<img ')
    ntab = page.count('<table')
    print('wrote %s  (%.1f KB)' % (out_path, size / 1024))
    print('imgs=%d  tables=%d  figs_numbered=%d  sections=%d  cites=%d'
          % (nimg, ntab, FIG_COUNTER, SECTION_COUNTER, len(CITES)))
    # leak scan
    leaks = {}
    for pat in (r'\\toprule', r'\\textbf', r'\\section', r'\\includegraphics',
                r'\\begin', r'\\cite', r'\\ref', r'\\caption', r'\\midrule',
                r'\\bottomrule', r'\\emph', r'\\texttt', r'\\multicolumn', r'\\input'):
        c = len(re.findall(pat, page))
        if c:
            leaks[pat] = c
    print('leaks=%s' % (leaks if leaks else 'NONE'))


# ----------------------------------------------------------------------------- HTML template

CSS = """
:root{--fg:#15171a;--muted:#5c636b;--line:#dfe3e8;--rule:#222;--bg:#fff;--code:#f4f5f7;--accent:#0b6;}
*{box-sizing:border-box}
html{font-size:15px}
body{margin:0;background:#e9ebee;color:var(--fg);
 font-family:"Times New Roman",Times,Georgia,serif;line-height:1.5}
.sheet{max-width:1000px;margin:28px auto;background:var(--bg);padding:54px 60px 80px;
 box-shadow:0 2px 18px rgba(0,0,0,.12);border-radius:2px}
/* full-width title band */
header.titleband{text-align:center;border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:18px}
header.titleband h1{font-size:25px;line-height:1.28;margin:0 0 14px;font-weight:700;letter-spacing:-.005em}
header.titleband .author{font-size:15px;color:var(--muted);font-style:italic}
/* abstract spans both columns */
.abstract{margin:0 auto 8px;max-width:84%;font-size:14px;line-height:1.5}
.abstract .abshead{text-align:center;font-weight:700;font-size:15px;margin-bottom:6px;font-variant:small-caps;letter-spacing:.03em}
.abstract p{margin:0;text-align:justify}
.divider{border:none;border-top:1px solid var(--line);margin:18px 0 0}
/* two-column body */
.columns{column-count:2;column-gap:34px;column-fill:balance;margin-top:20px;
 text-align:justify;hyphens:auto;font-size:14px}
.columns p{margin:0 0 .62em}
.columns p.runin{margin-top:.7em}
h2{font-size:16px;margin:1.1em 0 .45em;font-weight:700;break-after:avoid}
h2 .secnum{margin-right:.5em}
h3{font-size:14.5px;margin:1.0em 0 .35em;font-weight:700;font-style:italic;break-after:avoid}
h3 .secnum{font-style:normal;margin-right:.5em}
.secnum{font-weight:700}
/* lists */
ul.tex-list,ol.tex-list{margin:.4em 0 .7em;padding-left:1.3em}
ul.tex-list li,ol.tex-list li{margin:.28em 0}
/* code / mono */
code{background:var(--code);padding:.05em .3em;border-radius:3px;
 font-family:"SF Mono",Consolas,Menlo,monospace;font-size:.86em}
strong{font-weight:700}
em{font-style:italic}
a{color:#0a5bd6;text-decoration:none}
a:hover{text-decoration:underline}
/* equations */
.eqwrap{display:flex;align-items:center;justify-content:center;gap:.8em;
 margin:.7em 0;break-inside:avoid}
.equation{text-align:center;font-family:"Cambria Math","Times New Roman",serif;
 font-size:14.5px;font-style:italic}
.eqnum{font-style:normal;color:var(--muted);font-size:13px}
/* figures (float to top of column; here flow but kept inside a column) */
figure.figfloat{margin:14px 0 16px;text-align:center;break-inside:avoid}
figure.figfloat img{max-width:100%;height:auto;border:1px solid var(--line)}
figcaption{font-size:12px;color:var(--fg);margin-top:6px;text-align:justify;line-height:1.42}
.fnum{font-weight:700}
.missing{color:#b00;font-style:italic}
/* tables */
.tblfloat{margin:14px 0 16px;break-inside:avoid}
.tblfloat.wide{column-span:all;margin:18px 0 20px}
.tscroll{overflow-x:auto}
table.ltx{border-collapse:collapse;margin:6px auto;font-size:12px;width:100%;
 border-top:1.4px solid var(--rule);border-bottom:1.4px solid var(--rule);
 font-family:"Helvetica Neue",Arial,sans-serif}
table.ltx th,table.ltx td{padding:3.5px 9px;text-align:left;border:none}
table.ltx tr.hdr th{border-bottom:1px solid var(--rule);font-weight:700;
 vertical-align:bottom;text-align:center}
table.ltx tr.hdr th:first-child{text-align:left}
table.ltx td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
table.ltx tr.sec td{border-top:1px solid #b9bdc2}
.tcaption{font-size:12px;line-height:1.42;margin:0 0 5px;text-align:justify}
.tnum{font-weight:700}
.subcaption{font-size:11.5px;font-style:italic;color:#333;margin:8px 0 4px;text-align:center}
.note{font-size:11px;color:var(--muted);margin-top:5px;line-height:1.4;text-align:justify}
/* references */
ol.refs{column-span:none;list-style:none;padding-left:0;margin:.5em 0;font-size:12px}
ol.refs li{margin:.3em 0;padding-left:2.2em;text-indent:-2.2em;line-height:1.4}
.rnum{font-weight:700;margin-right:.35em}
@media print{body{background:#fff}.sheet{box-shadow:none;margin:0;max-width:none}}
@media (max-width:820px){.columns{column-count:1}.sheet{padding:28px 22px}}
"""

DOC_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TraLO — AAAI paper preview</title>
<style>%(css)s</style></head>
<body>
<article class="sheet">
<header class="titleband">
<h1>%(title)s</h1>
<div class="author">%(author)s</div>
</header>
%(abstract)s
<hr class="divider"/>
<div class="columns">
%(columns)s
</div>
</article>
</body></html>
"""


if __name__ == '__main__':
    main()
