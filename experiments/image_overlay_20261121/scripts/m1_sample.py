"""Step 2: draw the sealed glyph sample and render one contact sheet per 12 glyphs (target ink red, word grey).
Usage: python3 -I m1_sample.py <dl_dir> <out_dir>  -> out/step2/step2_sample.json, out/step2/sheet_XX.jpg
Sheets show only the index number, word and target position; stratum and est flags are kept out of the images."""
import json, os, random, sys
import numpy as np
from PIL import Image, ImageDraw, ImageChops
DL, OUT = sys.argv[1], sys.argv[2]; S = os.path.join(OUT, "step2"); os.makedirs(S, exist_ok=True)
OFF = set("f67r1 f67r2 f67v1 f67v2 f68r1 f68r2 f68r3 f68v1 f68v2 f68v3 f69r f69v f70r1 f70r2 f70v1 f70v2 f71r f71v f72r1 f72r2 f72r3 f72v1 f72v2 f72v3 f73r f73v".split())
NODRAW = set("f100v f101r1 f101v2 f102r1 f102r2 f102v1 f102v2 f85r1 f85r2 f86v3 f86v4 f86v5 f86v6 f88v f89r1 f89r2 f89v1 f89v2 f90v1 f90v2 f94v f95r1 f95r2 f90r1 f90r2 f95v2".split())
BOX = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"]
ZL = {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in json.load(open(os.path.join(DL, "ZLZI_folios.json")))["data"]}
pool = []
for fo in sorted(BOX):
    if fo in OFF or fo in NODRAW: continue
    G = json.load(open(os.path.join(OUT, "glyphs", fo + ".json"))); zl = {t for _, toks in ZL.get(fo, []) for t in toks}
    for bi, variants in G["words"].items():
        r, paths, est = variants[0]
        if r not in zl: continue
        for k, (c, p) in enumerate(zip(r, paths)): pool.append((fo, int(bi), r, k, c, int(est[k]), int(bool(p))))
rng = random.Random(20261122)
A = rng.sample(pool, 160); seen = set(A); narrow = [g for g in pool if g[4] in "oaeiy" and g not in seen]; B = rng.sample(narrow, 80)
items = [dict(fo=g[0], box=g[1], word=g[2], pos=g[3], char=g[4], est=g[5], has_ink=g[6], stratum=s) for s, grp in (("A", A), ("B", B)) for g in grp]
rng.shuffle(items)
for i, it in enumerate(items): it["id"] = i
json.dump(dict(seed=20261122, pool=len(pool), items=items), open(os.path.join(S, "step2_sample.json"), "w"), indent=0)
print("pool", len(pool), "sample", len(items))

def mask_of(d, X, Y, size, U):
    m = Image.new("1", size, 0)
    for sp in d.split("z"):
        if not sp: continue
        head, rel = sp[1:].split("l"); x, y = map(int, head.split()); pts = [(x * U - X, y * U - Y)]; v = list(map(int, rel.split()))
        for dx, dy in zip(v[0::2], v[1::2]): x += dx; y += dy; pts.append((x * U - X, y * U - Y))
        t = Image.new("1", size, 0); ImageDraw.Draw(t).polygon(pts, fill=1); m = ImageChops.logical_xor(m, t)
    return np.asarray(m)
cache = {}
def cell(it):
    fo = it["fo"]; G = json.load(open(os.path.join(OUT, "glyphs", fo + ".json"))); U = G["unit"]
    if fo not in cache: cache.clear(); cache[fo] = Image.open(os.path.join(DL, "full", "%d.jpg" % G["service"])).convert("RGB")
    im = cache[fo]; W, H = im.size; b = BOX[fo]["boxes"][it["box"]]; cw, ch = BOX[fo]["cw"], BOX[fo]["ch"]
    x0, y0 = b[0] / 1e4 * cw, b[1] / 1e4 * ch; x1, y1 = x0 + b[2] / 1e4 * cw, y0 + b[3] / 1e4 * ch
    pad = 0.25 * (y1 - y0); X, Y = int(max(0, x0 - pad)), int(max(0, y0 - pad)); X1, Y1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
    crop = np.asarray(im.crop((X, Y, X1, Y1))).astype(np.float32); orig = crop.copy()
    r, paths, est = G["words"][str(it["box"])][0]
    for k, d in enumerate(paths):
        if not d: continue
        m = mask_of(d, X, Y, (X1 - X, Y1 - Y), U)
        col = np.array((230, 0, 0) if k == it["pos"] else (70, 70, 70), np.float32)
        crop[m] = 0.15 * crop[m] + 0.85 * col
    a = Image.fromarray(orig.astype(np.uint8)); bimg = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8))
    sc = min(1.0, 330 / a.size[0], 150 / a.size[1]) if a.size[0] > 330 or a.size[1] > 150 else min(3.0, 330 / a.size[0], 150 / a.size[1])
    a = a.resize((int(a.size[0] * sc), int(a.size[1] * sc))); bimg = bimg.resize(a.size)
    c = Image.new("RGB", (700, 190), (255, 255, 255)); c.paste(a, (5, 5)); c.paste(bimg, (355, 5))
    lab = "#%d  %s   target: '%s' at position %d of %d" % (it["id"], r, it["char"], it["pos"] + 1, len(r))
    ImageDraw.Draw(c).text((8, 168), lab, fill=(0, 0, 0)); return c
for s in range(0, len(items), 8):
    sheet = Image.new("RGB", (1400, 760), (200, 200, 200))
    for j, it in enumerate(items[s:s + 8]): sheet.paste(cell(it), ((j % 2) * 700, (j // 2) * 190))
    sheet.save(os.path.join(S, "sheet_%02d.jpg" % (s // 8)), quality=88)
print("sheets", (len(items) + 7) // 8)
