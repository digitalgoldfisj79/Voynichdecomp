"""Corpus run: ink glyph outlines for every folio, as compact SVG path data for the site.

For each word box, up to two readings are segmented: the ZLZI token paired to the box by the site's line rule,
and the voynichese word (only if different). Each glyph's ink mask is traced to polygons.
Coordinates: integers in units of 2 full-resolution canvas pixels (canvas = cw x ch from the box file).

Usage: python3 -I k3_corpus_glyphs.py <dl_dir> <out_dir> [part k of n: k n]
Writes out/glyphs/<folio>.json (one file per folio = the checkpoint; existing files are skipped).
"""
import json, os, sys, time
import numpy as np
import cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkglyph as IG

DL, OUT = sys.argv[1], sys.argv[2]
K, N = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
GD = os.path.join(OUT, os.environ.get("GLYPH_DIR", "glyphs")); os.makedirs(GD, exist_ok=True)
import pickle
TM = pickle.load(open(os.environ["TEMPLATES"], "rb")) if os.environ.get("TEMPLATES") else None
J = json.load(open(os.path.join(OUT, "glyph_width_table.json")))
BOX = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"]
ZL = {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in json.load(open(os.path.join(DL, "ZLZI_folios.json")))["data"]}
UNIT = 2


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


def path_of(g, sx, sy):
    """glyph dict -> SVG path string (relative, integer, evenodd), in UNIT canvas pixels."""
    if g is None: return ""
    m = g["mask"].astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE); parts = []
    for c in cs:
        if len(c) < 3 or cv2.contourArea(c) < 6: continue
        c = cv2.approxPolyDP(c, 1.8, True).reshape(-1, 2).astype(float)
        pts = np.rint(np.c_[(c[:, 0] + g["x"]) * sx / UNIT, (c[:, 1] + g["y"]) * sy / UNIT]).astype(int)
        keep = [pts[0]]
        for p in pts[1:]:
            if (p != keep[-1]).any(): keep.append(p)
        if len(keep) < 3: continue
        d = "M%d %d" % (keep[0][0], keep[0][1]); rel = []
        for a, b in zip(keep[:-1], keep[1:]): rel.append("%d %d" % (b[0] - a[0], b[1] - a[1]))
        parts.append(d + "l" + " ".join(rel) + "z")
    return "".join(parts)


folios = os.environ["ONLY"].split(",") if os.environ.get("ONLY") else sorted(BOX)[K::N]
for fo in folios:
    fp = os.path.join(GD, fo + ".json")
    if os.path.exists(fp): continue
    t0 = time.time(); rec = BOX[fo]; ip = os.path.join(DL, "full", "%d.jpg" % rec["service"])
    if not os.path.exists(ip): print("NO IMAGE", fo, flush=True); continue
    im = Image.open(ip).convert("L"); W, H = im.size; cw, ch = rec["cw"], rec["ch"]; kx, ky = W / cw, H / ch
    boxes = rec["boxes"]
    px = [(b[0] / 1e4 * cw * kx, b[1] / 1e4 * ch * ky, (b[0] + b[2]) / 1e4 * cw * kx, (b[1] + b[3]) / 1e4 * ch * ky) for b in boxes]
    ctx = IG.analyse_page(im, px)
    by = {}
    for i, b in enumerate(boxes):
        if b[4]: by.setdefault(b[4], []).append(i)
    tok = {}
    for ln, toks in ZL.get(fo, []):
        idx = by.get(ln, [])
        for a, j in pair_line([boxes[i][5] for i in idx], toks): tok[idx[a]] = toks[j]
    words = {}; n_gl = n_est = n_empty = 0
    for i, b in enumerate(boxes):
        readings = [r for r in dict.fromkeys([tok.get(i), b[5]]) if r]
        out = []
        for r in readings:
            gl = IG.word_glyphs(ctx, ctx["members"][i], r, J["widths"], J["default"], templates=TM)
            paths = [path_of(g, 1 / kx, 1 / ky) for g in gl]
            est = "".join("1" if (g is not None and g["est"]) else "0" for g in gl)
            out.append([r, paths, est])
            if r == readings[0]:
                n_gl += len(gl); n_est += est.count("1"); n_empty += sum(1 for p in paths if not p)
        if out: words[str(i)] = out
    json.dump(dict(v=1, folio=fo, unit=UNIT, cw=cw, ch=ch, service=rec["service"], xh=round(ctx["xh"] / kx, 1), thr=round(ctx["thr"], 3),
                   glyphs=n_gl, shared=n_est, empty=n_empty, words=words), open(fp + ".tmp", "w"), separators=(",", ":"))
    os.replace(fp + ".tmp", fp)
    print(fo, "boxes", len(boxes), "glyphs", n_gl, "shared %.1f%%" % (100 * n_est / max(n_gl, 1)), "empty", n_empty, "bytes", os.path.getsize(fp), "%.1fs" % (time.time() - t0), flush=True)
print("part", K, "of", N, "done")
