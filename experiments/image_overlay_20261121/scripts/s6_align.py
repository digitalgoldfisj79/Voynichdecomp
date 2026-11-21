"""Stage 6: give each box a transliteration line number and an order within the line by aligning the
voynichese word sequence to ZLZI per folio; then simulate the runtime matching rule for ZLZI and TTLI.

Usage: python3 -I s6_align.py <dl_dir> <out_dir>
Writes out/s6_align.pkl, out/align_by_folio.csv
"""
import csv, json, os, pickle, sys
from difflib import SequenceMatcher
import numpy as np

DL, OUT = sys.argv[1], sys.argv[2]
S1 = pickle.load(open(os.path.join(OUT, "s1_boxes.pkl"), "rb"))
ARMS = pickle.load(open(os.path.join(OUT, "s3_arms.pkl"), "rb"))
def load(t):
    d = json.load(open(os.path.join(DL, t + "_folios.json")))
    return {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in d["data"]}
ZL, TT = load("ZLZI"), load("TTLI")
ALIAS = {"f101r1": "f101r", "f101v2": "f101v"}
for _a, _b in ALIAS.items():
    for _T in (ZL, TT):
        if _b in _T: _T[_a] = _T[_b]

def sim(a, b):
    if a == b: return 1.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()

def nw(A, B, gap=-0.35, floor=0.45):
    """global alignment; returns list of (i, j) pairs with similarity >= floor."""
    n, m = len(A), len(B)
    S = np.zeros((n + 1, m + 1)); T = np.zeros((n + 1, m + 1), np.int8)
    S[:, 0] = gap * np.arange(n + 1); S[0, :] = gap * np.arange(m + 1); T[1:, 0] = 1; T[0, 1:] = 2
    for i in range(1, n + 1):
        ai = A[i - 1]
        for j in range(1, m + 1):
            s = sim(ai, B[j - 1]); d = S[i - 1, j - 1] + (s - 0.5 if s >= floor else -0.6)
            u = S[i - 1, j] + gap; l = S[i, j - 1] + gap
            if d >= u and d >= l: S[i, j] = d; T[i, j] = 0
            elif u >= l: S[i, j] = u; T[i, j] = 1
            else: S[i, j] = l; T[i, j] = 2
    out = []; i, j = n, m
    while i > 0 or j > 0:
        t = T[i, j]
        if t == 0 and i > 0 and j > 0:
            if sim(A[i - 1], B[j - 1]) >= floor: out.append((i - 1, j - 1))
            i -= 1; j -= 1
        elif t == 1: i -= 1
        else: j -= 1
    return out[::-1]

align, rows = {}, []
missing = [fo for fo in S1["folios"] if fo not in ZL]
for fo in S1["folios"]:
    X = S1["xml_boxes"][fo]; words = [x[1] for x in X]
    fit = ARMS[fo]["fit"]
    # placed = centre inside the runtime viewport (homography is only trusted there)
    cx = np.array([(x[2] + x[4] / 2) * fit["sx"] + fit["tx"] for x in X]); cy = np.array([(x[3] + x[5] / 2) * fit["sy"] + fit["ty"] for x in X])
    placed = (cx >= 0) & (cx <= 636) & (cy >= 0) & (cy <= 900)
    line = [None] * len(X); exact = 0; pairs = []
    if fo in ZL:
        flat = [(ln, k, t) for ln, toks in ZL[fo] for k, t in enumerate(toks)]
        pairs = nw(words, [t for _, _, t in flat])
        for i, j in pairs:
            line[i] = flat[j][0]; exact += words[i] == flat[j][2]
        # unaligned boxes inherit the line of the nearest aligned neighbour in sequence that overlaps vertically
        B = ARMS[fo]["B"]
        for i in range(len(X)):
            if line[i] is not None: continue
            for d in range(1, 6):
                for k in (i - d, i + d):
                    if 0 <= k < len(X) and line[k] is not None and any((a == k) for a, _ in pairs):
                        ov = min(B[i, 3], B[k, 3]) - max(B[i, 1], B[k, 1])
                        if ov > 0.3 * min(B[i, 3] - B[i, 1], B[k, 3] - B[k, 1]):
                            line[i] = line[k]; break
                if line[i] is not None: break
    align[fo] = dict(line=line, placed=placed.tolist(), n_pairs=len(pairs), exact=exact)

    # ---- simulate the runtime rule for a transliteration: per line, boxes (placed, ordered left-to-right) vs tokens
    def simulate(T):
        if fo not in T: return (0, 0, 0)
        B = ARMS[fo]["B"]; tot = col = exact_ = 0
        by = {}
        for i, ln in enumerate(line):
            if ln is not None and placed[i]: by.setdefault(ln, []).append(i)
        for ln, toks in T[fo]:
            tot += len(toks); bx = by.get(ln, [])
            if not bx: continue
            bw = [words[i] for i in bx]          # sequence order kept (reading order of the source)
            if len(bx) == len(toks):
                col += len(toks); exact_ += sum(a == b for a, b in zip(bw, toks))
            else:
                p = nw(bw, toks)
                col += len(p); exact_ += sum(bw[a] == toks[b] for a, b in p)
        return (tot, col, exact_)
    z, t = simulate(ZL), simulate(TT)
    rows.append(dict(folio=fo, n_boxes=len(X), placed=int(placed.sum()), with_line=sum(l is not None for l in line),
                     zl_tokens=z[0], zl_coloured=z[1], zl_same_string=z[2], tt_tokens=t[0], tt_coloured=t[1], tt_same_string=t[2]))
pickle.dump(align, open(os.path.join(OUT, "s6_align.pkl"), "wb"))
with open(os.path.join(OUT, "align_by_folio.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
S = lambda k: sum(r[k] for r in rows)
print("folios without ZLZI text:", missing, "| ZLZI folios without boxes:", sorted(set(ZL) - set(S1["folios"])))
print("boxes %d | placed %d (unplaced %d) | with a line number %d" % (S("n_boxes"), S("placed"), S("n_boxes") - S("placed"), S("with_line")))
print("ZLZI: tokens %d | would be coloured-capable (have a box) %d = %.1f%% | box word identical to token %d = %.1f%% of matched" % (
    S("zl_tokens"), S("zl_coloured"), 100 * S("zl_coloured") / S("zl_tokens"), S("zl_same_string"), 100 * S("zl_same_string") / S("zl_coloured")))
print("TTLI: tokens %d | have a box %d = %.1f%% | identical %d = %.1f%% of matched" % (
    S("tt_tokens"), S("tt_coloured"), 100 * S("tt_coloured") / S("tt_tokens"), S("tt_same_string"), 100 * S("tt_same_string") / S("tt_coloured")))
worst = sorted(rows, key=lambda r: r["zl_coloured"] / max(r["zl_tokens"], 1))[:12]
print("lowest ZLZI coverage:", [(r["folio"], r["zl_tokens"], r["zl_coloured"]) for r in worst])
print("unplaced by folio:", [(r["folio"], r["n_boxes"] - r["placed"]) for r in rows if r["n_boxes"] > r["placed"]])
