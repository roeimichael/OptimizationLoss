"""Audit Chen v1's original train/val/test PNG folders before using images."""
import hashlib
import re
from pathlib import Path


def audit(root):
    from PIL import Image
    root = Path(root).resolve()
    rows, subjects, pixels, counts = [], {}, {}, {}
    for split in ('train', 'val', 'test'):
        subjects[split], pixels[split] = set(), set()
        paths = sorted((root/split).rglob('*.png'))
        if not paths: raise ValueError('empty split: '+split)
        seen = set()
        for p in paths:
            relative = p.relative_to(root)
            if len(relative.parts) != 3 or relative.parts[1] not in ('0','1','2','3','4'):
                raise ValueError('unexpected class/path: '+str(relative))
            match = re.fullmatch(r'(\d{7})([LR])', p.stem)
            if not match: raise ValueError('unrecognized subject/knee ID: '+p.name)
            if p.stem in seen: raise ValueError('repeated knee ID within split')
            seen.add(p.stem)
            with Image.open(p) as image:
                image.load()
                if image.width < 2 or image.height < 2: raise ValueError('invalid image size')
                rgb = image.convert('RGB')
                digest = hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
                size = list(image.size)
            subjects[split].add(match[1]); pixels[split].add(digest)
            rows.append(dict(path=relative.as_posix(), split=split, label=int(relative.parts[1]),
                             subject=match[1], sample_id=p.stem, size=size,
                             sha256=hashlib.sha256(p.read_bytes()).hexdigest(), pixel_sha256=digest))
        counts[split] = len(paths)
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if subjects[a]&subjects[b]: raise ValueError('subject overlap: '+a+'/'+b)
        if pixels[a]&pixels[b]: raise ValueError('pixel overlap: '+a+'/'+b)
    return dict(root=str(root), rows=rows, counts=counts,
                subjects={k:len(v) for k,v in subjects.items()},
                within_split_pixel_duplicates={k:counts[k]-len(v) for k,v in pixels.items()},
                cross_split_subject_overlap=0, cross_split_pixel_overlap=0,
                limitation='Filename-based subject identity; no independent clinical metadata verification or near-duplicate guarantee.')
