"""Stage 8: independent simulation of the site's overlay (file + runtime rule + ZLZI) drawn on a Yale image.
Usage: python3 -I s8_sim_overlay.py <dl_dir> <out_dir> <folio> <long_px>"""
import colorsys, io, json, os, sys, urllib.request
from collections import Counter
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DL, OUT, FO, LONG = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
data = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"][FO]
ZL = {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in json.load(open(os.path.join(DL, "ZLZI_folios.json")))["data"]}[FO]
def lcs_pairs(a, b):
    n, m = len(a), len(b); L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if a[i] == b[j] else max(L[i + 1][j], L[i][j + 1])
    out = []; i = j = 0
    while i < n and j < m:
        if a[i] == b[j]: out.append((i, j)); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]: i += 1
        else: j += 1
    return out
def pair_line(bw, toks):
    n, m = len(bw), len(toks)
    if n == m: return [(i, i) for i in range(n)]
    anchors = lcs_pairs(bw, toks); out = list(anchors); pi, pj = -1, -1
    for ai, aj in anchors + [(n, m)]:
        if ai - pi - 1 == aj - pj - 1 and ai - pi - 1 > 0: out += [(pi + 1 + k, pj + 1 + k) for k in range(ai - pi - 1)]
        pi, pj = ai, aj
    return out
p = os.path.join(DL, "sim_%s_%d.jpg" % (FO, LONG))
if not os.path.exists(p):
    req = urllib.request.Request("https://collections.library.yale.edu/iiif/2/%d/full/!%d,%d/0/default.jpg" % (data["service"], LONG, LONG), headers={"User-Agent": "voynichstats-overlay/1.0"})
    open(p, "wb").write(urllib.request.urlopen(req, timeout=120).read())
im = Image.open(p).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im, "RGBA")
counts = Counter(t for _, toks in ZL for t in toks); by = {}
for b in data["boxes"]:
    if b[4]: by.setdefault(b[4], []).append(b)
n = 0
for ln, toks in ZL:
    bx = by.get(ln, [])
    for i, j in pair_line([b[5] for b in bx], toks):
        t = toks[j]
        if counts[t] < 2: continue
        h = (hash(t) % 360) / 360; r, g, bl = [int(255 * c) for c in colorsys.hsv_to_rgb(h, 0.9, 0.9)]
        x, y, w, hh = [v / 1e4 for v in bx[i][:4]]
        d.rectangle([x * W, y * H, (x + w) * W, (y + hh) * H], fill=(r, g, bl, 70), outline=(r, g, bl, 255), width=2)
        d.text((x * W + 2, y * H - 11), t, fill=(0, 0, 0, 255)); n += 1
print("coloured boxes", n, "image", im.size)
xs = [b[0] for b in data["boxes"]]; ys = [b[1] for b in data["boxes"]]
x0, y0 = int(min(xs) / 1e4 * W) - 20, int(min(ys) / 1e4 * H) - 20
im.crop((max(0, x0), max(0, y0), min(W, x0 + 1100), min(H, y0 + 620))).save(os.path.join(OUT, "sim_overlay_%s.png" % FO))
