"""Glyph stage 6: independent simulation of the site's glyph bands (box file + pairing rule + width table).
Prints expected band counts and saves a crop with Timm's hand-drawn boxes outlined in black.
Usage: python3 -I g6_sim_glyphs.py <dl_dir> <out_dir> [folio=f1r]   (needs dl/sim_<folio>_2000.jpg from s8)"""
import colorsys, json, os, pickle, sys
from PIL import Image, ImageDraw
DL, OUT = sys.argv[1], sys.argv[2]; FO = sys.argv[3] if len(sys.argv) > 3 else "f1r"
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
data = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"][FO]
ZL = {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in json.load(open(os.path.join(DL, "ZLZI_folios.json")))["data"]}[FO]
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
GR = ["cth", "ckh", "cph", "cfh", "ch", "sh"]
def units(s):
    out = []; i = 0
    while i < len(s):
        g = next((x for x in GR if s.startswith(x, i)), None); out.append(g or s[i]); i += len(g) if g else 1
    return out
by = {}
for b in data["boxes"]:
    if b[4]: by.setdefault(b[4], []).append(b)
pairs = []
for ln, toks in ZL:
    bx = by.get(ln, [])
    for i, j in pair_line([b[5] for b in bx], toks): pairs.append((bx[i], toks[j]))
print("paired tokens", len(pairs), "raw-character bands", sum(len(t) for _, t in pairs), "EVA-grouped bands", sum(len(units(t)) for _, t in pairs))
im = Image.open(os.path.join(DL, "sim_%s_2000.jpg" % FO)).convert("RGB"); W, H = im.size; d = ImageDraw.Draw(im, "RGBA")
chars = sorted({c for _, t in pairs for c in t}); col = {c: tuple(int(255 * v) for v in colorsys.hsv_to_rgb(i / len(chars), 0.95, 0.95)) for i, c in enumerate(chars)}
for b, t in pairs:
    x, y, w, h = [v / 1e4 for v in b[:4]]; e = edges(t)
    for i, ch in enumerate(t):
        r, g, bl = col[ch]; d.rectangle([(x + e[i] * w) * W, y * H, (x + e[i + 1] * w) * W, (y + h) * H], fill=(r, g, bl, 80), outline=(r, g, bl, 255))
        d.text(((x + (e[i] + e[i + 1]) / 2 * w) * W - 3, (y + h) * H + 1), ch, fill=(0, 0, 0, 255))
for g in pickle.load(open(os.path.join(OUT, "g1_timm.pkl"), "rb")):
    if "f" + g["locus"] == FO:
        sx, sy = W / g["W"], H / g["H"]; d.rectangle([g["x0"] * sx, g["y0"] * sy, g["x1"] * sx, g["y1"] * sy], outline=(0, 0, 0, 255), width=2)
xs = [b[0] for b in data["boxes"]]; ys = [b[1] for b in data["boxes"]]
x0, y0 = int(min(xs) / 1e4 * W) - 20, int(min(ys) / 1e4 * H) - 20
im.crop((x0, y0, x0 + 1100, y0 + 420)).save(os.path.join(OUT, "sim_glyphs_%s.png" % FO)); print("saved")
