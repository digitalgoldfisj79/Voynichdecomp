#!/usr/bin/env python3
"""PREREG 34b: line-block bootstrap of npcompl33t's pool50 ordering. Per-line ZL counts rebuilt with his toolkit; then the s49 pipeline.
usage: python3 s50_line_bootstrap_order.py REPLICATION_ROOT OUT [B]
"""
import importlib.util, json, os, pickle, sys
from collections import Counter
import numpy as np
from multiprocessing import Pool
ROOT, OUT = sys.argv[1], sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 100
TEAM = os.path.join(ROOT, "solve/botanical_enrichment/team_golf")
sys.argv = [sys.argv[0], TEAM, OUT]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s49_text_resampled_order.py")).read().split("if __name__ == \"__main__\":\n    init()")[0])
SEED = 20261018
LINES = None
def build_lines():
    sys.path.insert(0, ROOT)
    from voynich_toolkit.source_lattice import SourceLattice
    from voynich_toolkit.glyph_units import units
    alphabet = CD["alphabet"]; ix = {g: i for i, g in enumerate(alphabet)}
    design = json.load(open(os.path.join(TEAM, "golf_published_stability_20261016/ensemble_paragraph207_point_l010.json")))["design"]
    page_block = {m["id"]: b for m, b in zip(design["meta"], design["blocks"])}
    def glyphs(w): return tuple("other" if u.rare else u.text for u in units(w, "zattera"))
    doc = SourceLattice(os.path.join(ROOT, "ZL_ivtff_raw.txt"), verify=False)
    per_block = {m["bifolium"]: [] for m in META}
    for line in doc.stream("all_spaces", "preferred_first"):
        if line.locus.category != "para" or line.locus.page not in page_block: continue
        ts = line.tokens; good = [not t.has_unreadable and "?" not in t.text for t in ts]
        cw, cl = Counter(), Counter()
        for t, ok in zip(ts, good):
            if ok:
                seq = glyphs(t.text)
                if seq: cw.update(zip(("^",) + seq, seq + ("$",)))
        start = None
        for i in range(len(ts) + 1):
            ok = i < len(ts) and good[i]
            if ok and start is None: start = i
            if not ok and start is not None:
                seq = glyphs("".join(t.text for t in ts[start:i]))
                if seq:
                    cl.update(zip(seq, seq[1:]))
                    if start == 0: cl[("^", seq[0])] += 1
                    if i == len(ts): cl[(seq[-1], "$")] += 1
                start = None
        a = {}
        for mode, c in (("word", cw), ("line", cl)):
            m_ = np.zeros((len(alphabet), len(alphabet)), np.int64)
            for (g, h), n in c.items(): m_[ix[g], ix[h]] = n
            a[mode] = m_
        per_block[page_block[line.locus.page]].append(a)
    return [per_block[m["bifolium"]] for m in META]
def init2():
    global LINES
    init()
    p = os.path.join(OUT, "lines.pkl")
    if os.path.exists(p): LINES = pickle.load(open(p, "rb")); return
    LINES = build_lines()
    for mode in ("word", "line"):
        summed = np.array([sum(l[mode] for l in ls) for ls in LINES])
        assert np.array_equal(summed, CD["pool_counts"][mode]), "GATE FAILED: per-line sums != his pool counts (" + mode + ")"
    pickle.dump(LINES, open(p, "wb"))
def resample_lines(rng):
    out = {"word": [], "line": []}
    for ls in LINES:
        pick = rng.integers(len(ls), size=len(ls))
        for mode in out: out[mode].append(sum(ls[k][mode] for k in pick))
    return {m: np.array(v) for m, v in out.items()}
def one_line(b):
    p = os.path.join(OUT, "stages", f"rep_{b:03d}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    global resample
    resample = lambda counts, rng: resample_lines(rng)
    return one(b)
# ==MAIN==
if __name__ == "__main__":
    os.makedirs(os.path.join(OUT, "stages"), exist_ok=True)
    init2(); print("GATE PASSED: per-line sums reproduce his pool counts exactly;", sum(len(l) for l in LINES), "lines", flush=True)
    with Pool(2, initializer=init2) as pool:
        for r in pool.imap_unordered(one_line, range(1, B + 1)):
            print("rep", r["b"], "best start", None if r["best"] is None else r["best"]["start"], flush=True)
