"""Summary of the corpus glyph run + fixtures for the site tests.
Usage: python3 -I k5_corpus_summary.py <dl_dir> <out_dir> -> out/glyph_corpus_by_folio.csv"""
import csv, json, os, sys
import numpy as np
DL, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BOX = json.load(open(os.path.join(OUT, "voynichese-boxes.json")))["folios"]
def load(t): return {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in json.load(open(os.path.join(DL, t + "_folios.json")))["data"]}
T = {"ZLZI": load("ZLZI"), "TTLI": load("TTLI")}
GR = ["cth", "ckh", "cph", "cfh", "ch", "sh"]
def units(s):
    out = []; i = 0
    while i < len(s):
        g = next((x for x in GR if s.startswith(x, i)), None); out.append(g or s[i]); i += len(g) if g else 1
    return out
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
def site_counts(fo, tid):
    G = json.load(open(os.path.join(OUT, os.environ.get("GLYPH_DIR", "glyphs"), fo + ".json"))); boxes = BOX[fo]["boxes"]; by = {}
    for i, b in enumerate(boxes):
        if b[4]: by.setdefault(b[4], []).append(i)
    paired = shaped = raw = grp = tokens = 0
    for ln, toks in T[tid].get(fo, []):
        tokens += len(toks); idx = by.get(ln, [])
        for a, j in pair_line([boxes[i][5] for i in idx], toks):
            paired += 1; v = next((v for v in G["words"].get(str(idx[a]), []) if v[0] == toks[j]), None)
            if v is None: continue
            shaped += 1; raw += sum(1 for p in v[1] if p)
            k = 0
            for u in units(toks[j]):
                if any(v[1][k:k + len(u)]): grp += 1
                k += len(u)
    return dict(tokens=tokens, paired=paired, words_with_shapes=shaped, raw_paths=raw, grouped_paths=grp)
rows = []; tot = dict(glyphs=0, shared=0, empty=0, bytes=0)
for fo in sorted(BOX):
    G = json.load(open(os.path.join(OUT, os.environ.get("GLYPH_DIR", "glyphs"), fo + ".json"))); sz = os.path.getsize(os.path.join(OUT, os.environ.get("GLYPH_DIR", "glyphs"), fo + ".json"))
    z = site_counts(fo, "ZLZI")
    rows.append(dict(folio=fo, boxes=len(BOX[fo]["boxes"]), glyphs=G["glyphs"], shared_pct=round(100 * G["shared"] / max(G["glyphs"], 1), 1), no_ink=G["empty"],
                     threshold=G["thr"], x_height_px=G["xh"], file_bytes=sz, zl_tokens=z["tokens"], zl_words_with_shapes=z["words_with_shapes"], zl_raw_paths=z["raw_paths"]))
    for k in ("glyphs", "shared", "empty"): tot[k] += G[k]
    tot["bytes"] += sz
with open(os.path.join(OUT, "glyph_corpus_by_folio_%s.csv" % os.environ.get("GLYPH_DIR", "glyphs")), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("folios", len(rows), "| glyphs (first reading)", tot["glyphs"], "| shared by column %.1f%%" % (100 * tot["shared"] / tot["glyphs"]), "| no ink %d (%.2f%%)" % (tot["empty"], 100 * tot["empty"] / tot["glyphs"]), "| bytes", tot["bytes"])
sp = np.array([r["shared_pct"] for r in rows]); print("shared%% per folio: median %.1f p90 %.1f max %.1f (%s)" % (np.median(sp), np.quantile(sp, .9), sp.max(), max(rows, key=lambda r: r["shared_pct"])["folio"]))
print("largest files:", sorted([(r["file_bytes"], r["folio"]) for r in rows])[-3:], "median bytes", int(np.median([r["file_bytes"] for r in rows])))
print("ZLZI corpus: tokens", sum(r["zl_tokens"] for r in rows), "words with shapes", sum(r["zl_words_with_shapes"] for r in rows), "glyph paths", sum(r["zl_raw_paths"] for r in rows))
for fo, tid in (("f1r", "ZLZI"), ("f1r", "TTLI"), ("f78r", "ZLZI"), ("f103r", "ZLZI")): print(fo, tid, site_counts(fo, tid))
