"""Glyph stage 7: draw the shipped glyph bands on a FULL-resolution Yale region to judge them by eye.
Usage: python3 -I g7_fullres_look.py <dl_dir> <out_dir> <folio> <x> <y> <w> <h> <tag>"""
import colorsys, json, os, sys, urllib.request, io
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DL, OUT, FO = sys.argv[1], sys.argv[2], sys.argv[3]; X, Y, Wd, Ht = map(int, sys.argv[4:8]); TAG = sys.argv[8]
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
data = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"][FO]
ZL = {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in json.load(open(os.path.join(DL, "ZLZI_folios.json")))["data"]}[FO]
import importlib.util
def lcs_pairs(a, b):
    n, m = len(a), len(b); L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1): L[i][j] = L[i + 1][j + 1] + 1 if a[i] == b[j] else max(L[i + 1][j], L[i][j + 1])
    out = []; i = j = 0
    while i < n and j < m:
        if a[i] == b[j]: out.append((i, j)); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]: i += 1
        else: j += 1
    return out
def pair_line(bw, toks):
    n, m = len(bw), len(toks)
    if n == m: return [(i, i) for i in range(n)]
    an = lcs_pairs(bw, toks); out = list(an); pi, pj = -1, -1
    for ai, aj in an + [(n, m)]:
        if ai - pi - 1 == aj - pj - 1 and ai - pi - 1 > 0: out += [(pi + 1 + k, pj + 1 + k) for k in range(ai - pi - 1)]
        pi, pj = ai, aj
    return out
def edges(word):
    w = [max(J["widths"].get(ch, J["default"]), 0.001) for ch in word]; tot = sum(w) + J["pad"]; e = [0.0]; acc = J["pad"] / 2
    for x in w: acc += x; e.append(acc / tot)
    e[0] = 0.0; e[-1] = 1.0; return e
url = "https://collections.library.yale.edu/iiif/2/%d/%d,%d,%d,%d/full/0/default.jpg" % (data["service"], X, Y, Wd, Ht)
im = Image.open(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "voynichstats-overlay/1.0"}), timeout=120).read())).convert("RGB")
d = ImageDraw.Draw(im, "RGBA"); cw, ch = data["cw"], data["ch"]
by = {}
for b in data["boxes"]:
    if b[4]: by.setdefault(b[4], []).append(b)
chars = "acdefghiklmnopqrstvxyz"; col = {c: tuple(int(255 * v) for v in colorsys.hsv_to_rgb(i / len(chars), 0.95, 0.95)) for i, c in enumerate(chars)}
for ln, toks in ZL:
    bx = by.get(ln, [])
    for i, j in pair_line([b[5] for b in bx], toks):
        b, t = bx[i], toks[j]; x, y, w, h = b[0] / 1e4 * cw - X, b[1] / 1e4 * ch - Y, b[2] / 1e4 * cw, b[3] / 1e4 * ch
        if x + w < 0 or x > Wd or y + h < 0 or y > Ht: continue
        e = edges(t)
        d.rectangle([x, y, x + w, y + h], outline=(0, 0, 0, 255), width=2)
        for k, c in enumerate(t):
            r, g, bl = col.get(c, (120, 120, 120)); d.rectangle([x + e[k] * w, y, x + e[k + 1] * w, y + h], fill=(r, g, bl, 60), outline=(r, g, bl, 255))
            d.text((x + (e[k] + e[k + 1]) / 2 * w - 3, y + h + 2), c, fill=(0, 0, 0, 255))
im.save(os.path.join(OUT, "look_%s_%s.png" % (FO, TAG))); print(im.size)
