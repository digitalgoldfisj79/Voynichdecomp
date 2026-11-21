"""v4 debug view (DEV folios): flagged words, v3 vs v4, each glyph's ink in its own colour, band limits marked.
Usage: python3 -I v3_look.py <dl_dir> <out_dir> <folio> <n> <tag> [param=value ...] -> out/v4/look_<folio>_<tag>.jpg"""
import json, os, pickle, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkglyph as IG
from k3_lib import readings_for
DL, OUT, FO, NW, TAG = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
P = dict(IG.P0)
for kv in sys.argv[6:]:
    k, v = kv.split("="); P[k] = float(v) if "." in v else int(v)
T = pickle.load(open(os.path.join(OUT, "v4", "templates.pkl"), "rb")); J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
words, im = readings_for(DL, OUT, FO); ctx = IG.analyse_page(im, [b for _, b in words], P); rgb = np.asarray(im.convert("RGB"))
PAL = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48), (145, 30, 180), (0, 160, 160), (240, 50, 230), (128, 128, 0), (0, 0, 128), (170, 110, 40)]
cells = []
for (wd, bx), mem in zip(words, ctx["members"]):
    g3 = IG.word_glyphs(ctx, mem, wd, J["widths"], J["default"], P)
    if not any(g is not None and g["est"] for g in g3): continue
    g4 = IG.word_glyphs(ctx, mem, wd, J["widths"], J["default"], P, templates=T)
    x0, y0, x1, y1 = [int(v) for v in bx]; pad = int(0.3 * (y1 - y0)); X, Y = max(0, x0 - pad), max(0, y0 - pad)
    crop = rgb[Y:y1 + pad, X:x1 + pad]; panels = [Image.fromarray(crop)]
    for gl in (g3, g4):
        out = crop.astype(np.float32).copy()
        for k, g in enumerate(gl):
            if g is None: continue
            m = g["mask"]; ys, xs = np.nonzero(m); ys = ys + g["y"] - Y; xs = xs + g["x"] - X
            ok = (ys >= 0) & (ys < out.shape[0]) & (xs >= 0) & (xs < out.shape[1])
            out[ys[ok], xs[ok]] = 0.15 * out[ys[ok], xs[ok]] + 0.85 * np.array(PAL[k % len(PAL)], np.float32)
        panels.append(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)))
    w = sum(p.size[0] for p in panels) + 20; h = max(p.size[1] for p in panels) + 16
    c = Image.new("RGB", (w, h), (255, 255, 255)); x = 0
    for p in panels: c.paste(p, (x, 0)); x += p.size[0] + 10
    ImageDraw.Draw(c).text((4, h - 13), "%s   flagged v3: %s" % (wd, "".join("1" if (g is not None and g["est"]) else "0" for g in g3)), fill=(0, 0, 0))
    cells.append(c)
    if len(cells) >= NW: break
H = sum(c.size[1] for c in cells) + 4 * len(cells); W = max(c.size[0] for c in cells)
sheet = Image.new("RGB", (W, H), (200, 200, 200)); y = 0
for c in cells: sheet.paste(c, (0, y)); y += c.size[1] + 4
sc = min(1.0, 1500 / W); sheet = sheet.resize((int(W * sc), int(H * sc)))
sheet.save(os.path.join(OUT, "v4", "look_%s_%s.jpg" % (FO, TAG)), quality=88); print(sheet.size, len(cells))
