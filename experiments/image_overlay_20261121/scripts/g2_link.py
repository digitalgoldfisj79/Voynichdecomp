"""Glyph stage 2: link Timm graphs to word boxes, cut ink profiles, build DEV / HELD-OUT sets.
Usage: python3 -I g2_link.py <dl_dir> <out_dir>  -> out/g2_cases.pkl, out/g2_corpus_profiles.pkl
"""
import json, os, pickle, sys, collections
import numpy as np
from PIL import Image, ImageFilter
DL, OUT = sys.argv[1], sys.argv[2]
S1 = pickle.load(open(os.path.join(OUT, "s1_boxes.pkl"), "rb"))
S2 = pickle.load(open(os.path.join(OUT, "s2_mapped.pkl"), "rb"))["mapped"]
ARMS = pickle.load(open(os.path.join(OUT, "s3_arms.pkl"), "rb"))
AL = pickle.load(open(os.path.join(OUT, "s6_align.pkl"), "rb"))
TIMM = pickle.load(open(os.path.join(OUT, "g1_timm.pkl"), "rb"))
reg = {json.loads(l)["selected"]["folio"]: json.loads(l)["selected"] for l in open(os.path.join(DL, "reg_full_225.jsonl")) if l.strip()}
IMG = os.path.join(DL, "img2000")

_cache = {}
def dark(sid):
    if sid not in _cache:
        _cache.clear()
        im = Image.open(os.path.join(IMG, sid.rsplit("/", 1)[1] + ".jpg")).convert("L")
        g = np.asarray(im, np.float32); bg = np.asarray(im.filter(ImageFilter.GaussianBlur(radius=50)), np.float32)
        _cache[sid] = np.clip(1.0 - g / np.maximum(bg, 1.0), 0, 1)
    return _cache[sid]

def profile(fo, box):
    """column ink profile of a full-res box, at derivative resolution. returns (profile, x_origin_full, px_per_col_full)."""
    d = dark(reg[fo]["service_id"]); fw, fh = S2[fo]["full_wh"]; k = max(d.shape) / max(fw, fh)
    x0, y0, x1, y1 = box
    c0, c1 = int(np.floor(x0 * k)), int(np.ceil(x1 * k)); r0, r1 = int(np.floor(y0 * k)), int(np.ceil(y1 * k))
    c0, r0 = max(0, c0), max(0, r0); c1, r1 = min(d.shape[1], c1), min(d.shape[0], r1)
    if c1 - c0 < 2 or r1 - r0 < 2: return None
    return d[r0:r1, c0:c1].sum(axis=0).astype(np.float32), c0 / k, 1.0 / k

# ---- corpus profiles for every placed box (used later for the shipped bands and for the width table)
corpus = {}
for fo in S1["folios"]:
    if not os.path.exists(os.path.join(IMG, reg[fo]["service_id"].rsplit("/", 1)[1] + ".jpg")): continue
    B = ARMS[fo]["B"]; X = S1["xml_boxes"][fo]; rows = []
    for i, x in enumerate(X):
        if not AL[fo]["placed"][i]: rows.append(None); continue
        p = profile(fo, B[i]); rows.append(None if p is None else (x[1], B[i].tolist(), p[0], p[1], p[2]))
    corpus[fo] = rows
pickle.dump(corpus, open(os.path.join(OUT, "g2_corpus_profiles.pkl"), "wb"))

# ---- link Timm graphs
def target_span(word, allo):
    if word.count(allo) != 1: return None
    i = word.index(allo)
    if allo == "iin" and i > 0 and word[i - 1] == "i": return None
    if allo == "in" and i > 0 and word[i - 1] == "i": return None
    return i, i + len(allo)
cases = []; why = collections.Counter()
for g in TIMM:
    if g["allograph"] not in ("iin", "k", "sh"): continue
    fo = "f" + g["locus"]
    if fo not in corpus: fo = {"f101r": "f101r1", "f101v": "f101v2"}.get(fo, fo)
    if fo not in corpus: why["folio not in box set"] += 1; continue
    fw, fh = S2[fo]["full_wh"]
    if abs(g["W"] / g["H"] - fw / fh) > 0.01: why["image aspect differs from Yale canvas"] += 1; continue
    sx, sy = fw / g["W"], fh / g["H"]
    gx0, gx1, gy0, gy1 = g["x0"] * sx, g["x1"] * sx, g["y0"] * sy, g["y1"] * sy
    cx, cy = (gx0 + gx1) / 2, (gy0 + gy1) / 2
    hit = [(i, r) for i, r in enumerate(corpus[fo]) if r and r[1][0] <= cx <= r[1][2] and r[1][1] <= cy <= r[1][3]]
    if not hit: why["centre in no word box"] += 1; continue
    i, r = min(hit, key=lambda t: (t[1][1][2] - t[1][1][0]) * (t[1][1][3] - t[1][1][1]))
    span = target_span(r[0], g["allograph"])
    if span is None: why["allograph absent or repeated in the box word"] += 1; continue
    cases.append(dict(folio=fo, allograph=g["allograph"], word=r[0], span=span, box=r[1], profile=r[2], x_origin=r[3], px_per_col=r[4],
                      gold=(gx0, gx1), gold_y=(gy0, gy1), box_index=i, created=g["created"]))
    why["linked"] += 1
rng = np.random.default_rng(20261121)
fos = sorted({c["folio"] for c in cases}); rng.shuffle(fos); dev = set(fos[:len(fos) // 2])
for c in cases:
    c["set"] = "HELD_B" if c["allograph"] == "sh" else ("DEV" if c["folio"] in dev else "HELD_A")
pickle.dump(cases, open(os.path.join(OUT, "g2_cases.pkl"), "wb"))
print("folios with images:", len(corpus), "| corpus boxes with a profile:", sum(1 for v in corpus.values() for r in v if r))
print("linkage:", dict(why))
print("sets:", collections.Counter((c["set"], c["allograph"]) for c in cases))
print("word length of linked cases: median", np.median([len(c["word"]) for c in cases]))
