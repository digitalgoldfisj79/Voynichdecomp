"""Makes the small WebP textures and assets.json used by the journey page.   python tools/make_tex.py
Reads ../plants, ../nymphs and ../animals; writes tex/<size>/ and assets.json."""
import os, json
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
out = {'plants': [], 'nymphs': [], 'animals': []}
for kind in out:
    src = os.path.join(ROOT, '..', kind)
    for f in sorted(os.listdir(src)):
        if not f.endswith('.png') or f.startswith('guide_'): continue
        im = Image.open(os.path.join(src, f)).convert('RGBA')
        box = im.getbbox(); im = im.crop(box) if box else im
        for size in (512, 1024):
            d = os.path.join(ROOT, 'tex', str(size)); os.makedirs(d, exist_ok=True)
            s = im.copy(); s.thumbnail((size, size), Image.LANCZOS)
            s.save(os.path.join(d, f[:-4] + '.webp'), quality=86, method=6)
        out[kind].append({'name': f[:-4], 'aspect': round(im.width / im.height, 4)})
json.dump(out, open(os.path.join(ROOT, 'assets.json'), 'w'))
print({k: len(v) for k, v in out.items()})
