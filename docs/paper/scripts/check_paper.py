import re
tex = open('paper/main.tex', encoding='utf-8').read()
bib = open('paper/references.bib', encoding='utf-8').read()
lines = []
BS = chr(92)
for ln in tex.splitlines():
    if '%' in ln:
        idx = 0
        while idx < len(ln):
            i = ln.find('%', idx)
            if i == -1: break
            if i > 0 and ln[i-1] == BS:
                idx = i + 1; continue
            ln = ln[:i]; break
    lines.append(ln)
clean = '\n'.join(lines)
opens = clean.count('{'); closes = clean.count('}')
cites = set()
for m in re.finditer(r'\\cite\{([^}]+)\}', clean):
    for k in m.group(1).split(','):
        cites.add(k.strip())
bibkeys = set(re.findall(r'@\w+\{([^,]+),', bib))
orphans = cites - bibkeys
print(f'Braces: {opens} open / {closes} close (delta={opens-closes})')
print(f'Unique cite keys: {len(cites)}')
print(f'Bib entries: {len(bibkeys)}')
print(f'Orphan cites: {sorted(orphans) if orphans else "NONE"}')
refs = set(re.findall(r'\\ref\{([^}]+)\}', clean))
labels = set(re.findall(r'\\label\{([^}]+)\}', clean))
ref_orphans = refs - labels
print(f'ref orphans: {sorted(ref_orphans) if ref_orphans else "NONE"}')
print(f'Total .tex lines: {len(tex.splitlines())}')
