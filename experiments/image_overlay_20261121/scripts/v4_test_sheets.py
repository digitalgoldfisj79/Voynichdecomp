"""v4 Phase 2: sealed paired test on TEST folios. 150 v3-flagged glyphs (seed 20261125); for each, original crop +
two panels (v3, v4) in random left/right order; target glyph red, rest of word grey. Key sealed in a separate file.
Usage: python3 -I v4_test_sheets.py <dl_dir> <out_dir>"""
import json, os, pickle, random, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkglyph as IG
from k3_lib import readings_for
DL, OUT = sys.argv[1], sys.argv[2]; V = os.path.join(OUT, "v4", "test"); os.makedirs(V, exist_ok=True)
SP = json.load(open(os.path.join(OUT, "split_v4.json"))); T = pickle.load(open(os.path.join(OUT, "v4", "templates.pkl"), "rb"))
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
pool = []
for fo in SP["test"]:
    G = json.load(open(os.path.join(OUT, "glyphs", fo + ".json")))
    for bi, var in G["words"].items():
        r, paths, est = var[0]
        for k, e in enumerate(est):
            if e == "1": pool.append((fo, int(bi), r, k))
rng = random.Random(20261125); items = rng.sample(pool, 150); rng.shuffle(items)
key = []; cells = {}
byfo = {}
for n, it in enumerate(items): byfo.setdefault(it[0], []).append((n, it))
for fo, lst in byfo.items():
    words, im = readings_for(DL, OUT, fo); ctx = IG.analyse_page(im, [b for _, b in words]); rgb = np.asarray(im.convert("RGB"))
    for n, (fo_, bi, r, k) in lst:
        wd, bx = words[bi]; assert wd == r, (fo, bi, wd, r)
        g3 = IG.word_glyphs(ctx, ctx["members"][bi], wd, J["widths"], J["default"])
        g4 = IG.word_glyphs(ctx, ctx["members"][bi], wd, J["widths"], J["default"], templates=T)
        x0, y0, x1, y1 = [int(v) for v in bx]; pad = int(0.3 * (y1 - y0)); X, Y = max(0, x0 - pad), max(0, y0 - pad)
        crop = rgb[Y:y1 + pad, X:x1 + pad]
        def panel(gl):
            out = crop.astype(np.float32).copy()
            for q, g in enumerate(gl):
                if g is None: continue
                ys, xs = np.nonzero(g["mask"]); ys = ys + g["y"] - Y; xs = xs + g["x"] - X
                ok = (ys >= 0) & (ys < out.shape[0]) & (xs >= 0) & (xs < out.shape[1])
                col = np.array((230, 0, 0) if q == k else (70, 70, 70), np.float32)
                out[ys[ok], xs[ok]] = 0.15 * out[ys[ok], xs[ok]] + 0.85 * col
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        left_is_v4 = rng.random() < 0.5
        L, R = (panel(g4), panel(g3)) if left_is_v4 else (panel(g3), panel(g4))
        key.append(dict(id=n, folio=fo, box=bi, word=wd, pos=k, char=wd[k], left_is_v4=left_is_v4))
        a = Image.fromarray(crop); sc = min(3.0, 300 / a.size[0], 140 / a.size[1])
        sz = (max(1, int(a.size[0] * sc)), max(1, int(a.size[1] * sc)))
        c = Image.new("RGB", (960, 170), (255, 255, 255))
        for j, p in enumerate((a, L, R)): c.paste(p.resize(sz), (5 + j * 318, 5))
        ImageDraw.Draw(c).text((8, 152), "#%d  %s   target '%s' at position %d of %d      original | A | B" % (n, wd, wd[k], k + 1, len(wd)), fill=(0, 0, 0))
        cells[n] = c
    print(fo, len(lst), flush=True)
key.sort(key=lambda d: d["id"])
json.dump(dict(seed=20261125, pool=len(pool), key=key), open(os.path.join(V, "SEALED_key.json"), "w"), indent=0)
for s in range(0, 150, 6):
    sheet = Image.new("RGB", (960, 6 * 175), (200, 200, 200))
    for j in range(6):
        if s + j in cells: sheet.paste(cells[s + j], (0, j * 175))
    sheet.save(os.path.join(V, "pair_%02d.jpg" % (s // 6)), quality=88)
print("pool", len(pool), "sheets", 25)
