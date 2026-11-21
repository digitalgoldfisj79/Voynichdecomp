"""Round-trip check: draw the SHIPPED path data (out/glyphs/<folio>.json) back onto the full-resolution image.
Usage: python3 -I k4_roundtrip.py <dl_dir> <out_dir> <folio> <x> <y> <w> <h>"""
import json, os, re, sys
import numpy as np
from PIL import Image, ImageDraw, ImageChops
DL, OUT, FO = sys.argv[1], sys.argv[2], sys.argv[3]; X, Y, Wd, Ht = map(int, sys.argv[4:8])
G = json.load(open(os.path.join(OUT, os.environ.get("GLYPH_DIR", "glyphs"), FO + ".json"))); U = G["unit"]
im = Image.open(os.path.join(DL, "full", "%d.jpg" % G["service"])).convert("RGB").crop((X, Y, X + Wd, Y + Ht))
CH = "oeyadchiklqrstnpmfgxvz"
PAL = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180), (0, 160, 160), (240, 50, 230), (128, 128, 0),
       (0, 0, 128), (170, 110, 40), (128, 0, 0), (0, 0, 0), (184, 134, 11), (70, 70, 255), (255, 105, 180), (0, 200, 255),
       (120, 60, 160), (90, 90, 90), (255, 160, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
def subpaths(d):
    for sp in d.split("z"):
        if not sp: continue
        head, rel = sp[1:].split("l"); x, y = map(int, head.split()); pts = [(x, y)]; v = list(map(int, rel.split()))
        for dx, dy in zip(v[0::2], v[1::2]): x += dx; y += dy; pts.append((x, y))
        yield [(px * U - X, py * U - Y) for px, py in pts]
out = np.asarray(im).astype(np.float32)
for i, variants in G["words"].items():
    word, paths, est = variants[0]
    for ch, d in zip(word, paths):
        if not d: continue
        m = Image.new("1", im.size, 0)
        for pts in subpaths(d):
            t = Image.new("1", im.size, 0); ImageDraw.Draw(t).polygon(pts, fill=1); m = ImageChops.logical_xor(m, t)
        mm = np.asarray(m); col = np.array(PAL[CH.index(ch)] if ch in CH else (90, 90, 90), np.float32)
        out[mm] = 0.35 * out[mm] + 0.65 * col
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(os.path.join(OUT, "trial", "%s_roundtrip_%s.jpg" % (FO, os.environ.get("GLYPH_DIR", "glyphs"))), quality=90); print("saved")
