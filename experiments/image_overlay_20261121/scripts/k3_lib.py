"""Shared helper: for a folio, the first reading per placed box (ZLZI token paired by the site's line rule,
else the voynichese word) with its full-resolution pixel box, plus the greyscale page image."""
import json, os
from PIL import Image
_C = {}
def _load(DL, OUT):
    if "box" not in _C:
        _C["box"] = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"]
        _C["zl"] = {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in json.load(open(os.path.join(DL, "ZLZI_folios.json")))["data"]}
    return _C["box"], _C["zl"]
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
def readings_for(DL, OUT, fo):
    BOX, ZL = _load(DL, OUT); rec = BOX[fo]
    im = Image.open(os.path.join(DL, "full", "%d.jpg" % rec["service"])).convert("L"); W, H = im.size; cw, ch = rec["cw"], rec["ch"]
    kx, ky = W / cw, H / ch; boxes = rec["boxes"]; by = {}
    for i, b in enumerate(boxes):
        if b[4]: by.setdefault(b[4], []).append(i)
    tok = {}
    for ln, toks in ZL.get(fo, []):
        idx = by.get(ln, [])
        for a, j in pair_line([boxes[i][5] for i in idx], toks): tok[idx[a]] = toks[j]
    words = [(tok.get(i, b[5]), (b[0] / 1e4 * cw * kx, b[1] / 1e4 * ch * ky, (b[0] + b[2]) / 1e4 * cw * kx, (b[1] + b[3]) / 1e4 * ch * ky)) for i, b in enumerate(boxes)]
    return words, im
