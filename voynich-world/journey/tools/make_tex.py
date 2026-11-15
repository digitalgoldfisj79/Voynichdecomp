"""Makes the WebP textures and assets.json used by the journey page.   python tools/make_tex.py
Reads ../plants, ../nymphs and ../animals; writes tex/512 and tex/1024 and assets.json.
Nothing is redrawn: edges are trimmed by one pixel (removes the parchment fringe), the colour is spread
outwards under the transparent area (stops pale halos when the picture is shrunk on screen), and the small
figure cut-outs are enlarged with ordinary resampling so their edges are smooth close up."""
import os, json
import numpy as np
from PIL import Image, ImageFilter
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')

def clean(im):
    a = im.getchannel('A').filter(ImageFilter.MinFilter(3))                    # trim one pixel off the edge
    rgb = np.asarray(im.convert('RGB')).astype(np.float32); al = (np.asarray(a) > 128)
    cur, known = rgb.copy(), al.copy()
    for _ in range(12):                                                        # spread edge colour outwards
        w = known.astype(np.float32)[..., None]
        s = sum(np.roll(cur * w, d, ax) for ax in (0, 1) for d in (-1, 1)); n = sum(np.roll(w, d, ax) for ax in (0, 1) for d in (-1, 1))
        fill = (~known) & (n[..., 0] > 0)
        cur[fill] = (s / np.maximum(n, 1))[fill]; known |= fill
    cur[~known] = cur[known].mean(0) if known.any() else 200
    out = Image.fromarray(cur.astype(np.uint8)).convert('RGBA'); out.putalpha(a); return out

out = {'plants': [], 'nymphs': [], 'animals': []}
for kind in out:
    src = os.path.join(ROOT, '..', kind)
    for f in sorted(os.listdir(src)):
        if not f.endswith('.png') or f.startswith('guide_'): continue
        im = Image.open(os.path.join(src, f)).convert('RGBA')
        box = im.getbbox(); im = clean(im.crop(box) if box else im)
        for size in (512, 1024):
            d = os.path.join(ROOT, 'tex', str(size)); os.makedirs(d, exist_ok=True)
            target = 1024 if kind != 'plants' else size                        # figures are seen close: always the large size
            k = target / max(im.size); s = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
            if k < 1: s = s.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))
            s.save(os.path.join(d, f[:-4] + '.webp'), quality=90, method=4, exact=True)
        out[kind].append({'name': f[:-4], 'aspect': round(im.width / im.height, 4)})
json.dump(out, open(os.path.join(ROOT, 'assets.json'), 'w'))
print({k: len(v) for k, v in out.items()})
