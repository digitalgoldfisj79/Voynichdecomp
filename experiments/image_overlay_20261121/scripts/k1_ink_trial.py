"""Ink glyph trial: render one folio (full-resolution Yale image) with each glyph's ink tinted by character.
Usage: python3 -I k1_ink_trial.py <dl_dir> <out_dir> <folio> [x y w h]   (crop in full-res px; omit for auto text block)
Writes out/trial/<folio>_ink.jpg (crop) and prints page diagnostics.
"""
import colorsys, io, json, os, pickle, sys, urllib.request
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkglyph as IG

DL, OUT, FO = sys.argv[1], sys.argv[2], sys.argv[3]
CROP = list(map(int, sys.argv[4:8])) if len(sys.argv) >= 8 else None
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
data = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"][FO]
os.makedirs(os.path.join(DL, "full"), exist_ok=True); os.makedirs(os.path.join(OUT, "trial"), exist_ok=True)
p = os.path.join(DL, "full", "%d.jpg" % data["service"])
if not os.path.exists(p):
    req = urllib.request.Request("https://collections.library.yale.edu/iiif/2/%d/full/full/0/default.jpg" % data["service"], headers={"User-Agent": "voynichstats-overlay/1.0"})
    open(p, "wb").write(urllib.request.urlopen(req, timeout=300).read())
im = Image.open(p).convert("RGB"); W, H = im.size
cw, ch = data["cw"], data["ch"]; kx, ky = W / cw, H / ch
words = [(b[5], (b[0] / 1e4 * cw * kx, b[1] / 1e4 * ch * ky, (b[0] + b[2]) / 1e4 * cw * kx, (b[1] + b[3]) / 1e4 * ch * ky)) for b in data["boxes"]]

CH = "oeyadchiklqrstnpmfgxvz"      # frequent characters first so they get well-separated hues
PAL = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180), (0, 160, 160), (240, 50, 230), (128, 128, 0),
       (0, 0, 128), (170, 110, 40), (128, 0, 0), (0, 0, 0), (184, 134, 11), (70, 70, 255), (255, 105, 180), (0, 200, 255),
       (120, 60, 160), (90, 90, 90), (255, 160, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
colour_of = lambda c: PAL[CH.index(c)] if c in CH else (90, 90, 90)

ctx, groups = IG.page_glyphs(im.convert("L"), words, J["widths"], J["default"])
out = Image.fromarray(IG.paint(np.asarray(im), groups, words, colour_of))
d = ImageDraw.Draw(out)
n_glyph = sum(len(w) for w, _ in words); empty = sum(1 for (w, _), g in zip(words, groups) for grp in g if grp is None)
shared = sum(1 for (w, _), g in zip(words, groups) for grp in g if grp is not None and grp["est"])
noink = sum(1 for (w, _), g in zip(words, groups) if not any(x is not None for x in g))
for (wd, bx), g in zip(words, groups):
    for c, grp in zip(wd, g):
        if grp is not None:
            bb = grp["bbox"]; d.text(((bb[0] + bb[2]) / 2 - 3, bx[3] + 1), c, fill=colour_of(c))
print(FO, "image", im.size, "xh %.1f px  stroke %.1f px  threshold %.3f" % (ctx["xh"], ctx["t"], ctx["thr"]),
      "| words", len(words), "glyphs", n_glyph, "| glyphs with no ink %d (%.1f%%)" % (empty, 100 * empty / max(n_glyph, 1)), "| glyphs cut by column (fallback) %d (%.1f%%)" % (shared, 100 * shared / max(n_glyph, 1)), "| words with no ink", noink)
if CROP is None:
    cx = np.array([(b[0] + b[2]) / 2 for _, b in words]); cy = np.array([(b[1] + b[3]) / 2 for _, b in words]); best = (-1, 0, 0)
    for yy in range(0, max(1, H - 950), 80):
        for xx in range(0, max(1, W - 1500), 80):
            c = int(((cx >= xx) & (cx < xx + 1500) & (cy >= yy) & (cy < yy + 950)).sum())
            if c > best[0]: best = (c, xx, yy)
    CROP = [best[1], best[2], 1500, 950]
x, y, w, h = CROP
out.crop((x, y, min(W, x + w), min(H, y + h))).save(os.path.join(OUT, "trial", "%s_ink.jpg" % FO), quality=90)
pickle.dump(dict(xh=ctx["xh"], t=ctx["t"], thr=ctx["thr"], empty=empty, shared=shared, glyphs=n_glyph, noink=noink, crop=CROP), open(os.path.join(OUT, "trial", "%s_diag.pkl" % FO), "wb"))
