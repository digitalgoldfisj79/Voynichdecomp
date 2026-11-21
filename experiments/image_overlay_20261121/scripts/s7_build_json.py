"""Stage 7: build the overlay data file + expected-number fixtures for the runtime matching rule.

Runtime rule (deterministic, simulated here exactly as the site must implement it), per folio and line:
  boxes b_1..b_n in stored order, tokens t_1..t_m in position order.
  1. n == m  -> pair by position.
  2. else, EVA-family alphabets only: longest common subsequence on exact string equality gives anchors;
     between consecutive anchors (and before the first / after the last), if the number of unpaired boxes
     equals the number of unpaired tokens, pair those by position. Everything else stays unpaired.
Usage: python3 -I s7_build_json.py <dl_dir> <out_dir>
"""
import csv, hashlib, json, os, pickle, sys
import numpy as np

DL, OUT = sys.argv[1], sys.argv[2]
S1 = pickle.load(open(os.path.join(OUT, "s1_boxes.pkl"), "rb"))
S2 = pickle.load(open(os.path.join(OUT, "s2_mapped.pkl"), "rb"))["mapped"]
ARMS = pickle.load(open(os.path.join(OUT, "s3_arms.pkl"), "rb"))
AL = pickle.load(open(os.path.join(OUT, "s6_align.pkl"), "rb"))
PEAK = {r["folio"]: r for r in csv.DictReader(open(os.path.join(OUT, "g2b_peak_by_folio_1000.csv")))}
ALIAS = {"f101r1": "f101r", "f101v2": "f101v"}       # voynichese name -> site (ZL) name; see RESULTS for the check
def load(t):
    d = json.load(open(os.path.join(DL, t + "_folios.json")))
    return {f["folio"]: [(l["line"], l["tokens"]) for l in f["lines"]] for f in d["data"]}
ZL, TT = load("ZLZI"), load("TTLI")

def lcs_pairs(a, b):
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            L[i][j] = L[i + 1][j + 1] + 1 if a[i] == b[j] else max(L[i + 1][j], L[i][j + 1])
    out = []; i = j = 0
    while i < n and j < m:
        if a[i] == b[j]: out.append((i, j)); i += 1; j += 1
        elif L[i + 1][j] >= L[i][j + 1]: i += 1
        else: j += 1
    return out

def pair_line(bw, toks, eva=True):
    n, m = len(bw), len(toks)
    if n == m: return [(i, i) for i in range(n)]
    if not eva: return []
    anchors = lcs_pairs(bw, toks); out = list(anchors)
    pi, pj = -1, -1
    for ai, aj in anchors + [(n, m)]:
        if ai - pi - 1 == aj - pj - 1 and ai - pi - 1 > 0:
            out += [(pi + 1 + k, pj + 1 + k) for k in range(ai - pi - 1)]
        pi, pj = ai, aj
    return sorted(out)

data = {"version": "20261121-1", "source": "voynichese_data.zip (Apache-2.0) mapped to Yale IIIF canvases",
        "units": "x,y,w,h are fractions of the canvas width/height x 10000", "folios": {}}
rows = []
for fo in S1["folios"]:
    site = ALIAS.get(fo, fo); m = S2[fo]; fw, fh = m["full_wh"]; B = ARMS[fo]["B"]
    al = AL[fo]; X = S1["xml_boxes"][fo]
    boxes = []; keep_idx = []
    for i, x in enumerate(X):
        if not al["placed"][i]: continue
        x0, y0, x1, y1 = B[i]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
        if x1 - x0 < 1 or y1 - y0 < 1: continue
        boxes.append([int(round(1e4 * x0 / fw)), int(round(1e4 * y0 / fh)), int(round(1e4 * (x1 - x0) / fw)), int(round(1e4 * (y1 - y0) / fh)),
                      al["line"][i] if al["line"][i] is not None else 0, x[1]])
        keep_idx.append(i)
    pk = PEAK[fo]
    data["folios"][site] = {"service": int(m["service_id"].rsplit("/", 1)[1]), "cw": fw, "ch": fh,
                            "check": pk["verdict_B"].split(" (")[0].replace("Inconclusive at this power", "unchecked"),
                            "boxes": boxes}
    for name, T in (("ZLZI", ZL), ("TTLI", TT)):
        tot = col = 0
        by = {}
        for b in boxes:
            if b[4]: by.setdefault(b[4], []).append(b[5])
        for ln, toks in T.get(site, []):
            tot += len(toks); col += len(pair_line(by.get(ln, []), toks))
        rows.append(dict(folio=site, transliteration=name, tokens=tot, tokens_with_box=col, boxes=len(boxes),
                         boxes_with_line=sum(1 for b in boxes if b[4])))
p = os.path.join(OUT, "voynichese-boxes.json")
json.dump(data, open(p, "w"), separators=(",", ":"), ensure_ascii=True)
with open(os.path.join(OUT, "overlay_expected_counts.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
raw = open(p, "rb").read()
print("file bytes", len(raw), "sha256", hashlib.sha256(raw).hexdigest()[:16], "folios", len(data["folios"]), "boxes", sum(len(v["boxes"]) for v in data["folios"].values()))
for name in ("ZLZI", "TTLI"):
    r = [x for x in rows if x["transliteration"] == name]
    t, c = sum(x["tokens"] for x in r), sum(x["tokens_with_box"] for x in r)
    print(name, "tokens", t, "with box", c, "= %.1f%%" % (100 * c / t))
for fo in ("f1r", "f1v", "f101r", "f101v", "f76r", "f68r1"):
    print([ (x["folio"], x["transliteration"], x["tokens"], x["tokens_with_box"], x["boxes"]) for x in rows if x["folio"] == fo])
import collections
print("check flags:", collections.Counter(v["check"] for v in data["folios"].values()))
print("sample f1r:", data["folios"]["f1r"]["boxes"][:3], {k: v for k, v in data["folios"]["f1r"].items() if k != "boxes"})
