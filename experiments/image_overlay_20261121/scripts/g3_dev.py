"""Glyph stage 3: development on DEV only (iin + k, half the folios). Logs every variant.
Usage: python3 -I g3_dev.py <out_dir>   -> out/g3_dev_variants.csv, out/g3_width_table.pkl
"""
import csv, os, pickle, sys, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyphsplit as G
OUT = sys.argv[1]
cases = pickle.load(open(os.path.join(OUT, "g2_cases.pkl"), "rb"))
corpus = pickle.load(open(os.path.join(OUT, "g2_corpus_profiles.pkl"), "rb"))

# width table from word-box widths over the whole corpus (no Timm labels). scale = folio median box height.
words, widths, scales = [], [], []
for fo, rows in corpus.items():
    hs = [r[1][3] - r[1][1] for r in rows if r]
    if not hs: continue
    sc = float(np.median(hs))
    for r in rows:
        if r and r[0] and all(ch in G.CHARS for ch in r[0]):
            words.append(r[0]); widths.append(r[1][2] - r[1][0]); scales.append(sc)
tabs = {True: G.fit_width_table(words, widths, scales, True), False: G.fit_width_table(words, widths, scales, False)}
pickle.dump(tabs, open(os.path.join(OUT, "g3_width_table.pkl"), "wb"))
print("width table (with pad), units of folio median box height:", {k: round(v, 3) for k, v in sorted(tabs[True][0].items(), key=lambda kv: -kv[1])}, "pad", round(tabs[True][1], 3), "n words", len(words))

def score(c, edges):
    i0, i1 = c["span"]; p0 = c["x_origin"] + edges[i0] * c["px_per_col"]; p1 = c["x_origin"] + edges[i1] * c["px_per_col"]
    g0, g1 = c["gold"]; inter = max(0.0, min(p1, g1) - max(p0, g0)); union = max(p1, g1) - min(p0, g0)
    gc = (g0 + g1) / 2
    return inter / union if union > 0 else 0.0, float(p0 <= gc <= p1), abs((p0 + p1) / 2 - gc) / max(g1 - g0, 1e-9)

def run(cs, mode, pad, tau, lam):
    tab, pd_ = tabs[pad]
    out = np.array([score(c, G.split(c["word"], c["profile"], mode=mode, table=tab, pad=pd_, tau=tau, lam=lam)) for c in cs])
    return out

if __name__ == "__main__":
    dev = [c for c in cases if c["set"] == "DEV"]
    rows = []
    grid = [("equal", False, t, 0.0) for t in (None, 0.05, 0.15, 0.3)]
    grid += [("table", p, t, 0.0) for p in (True, False) for t in (None, 0.05, 0.15, 0.3)]
    grid += [("snap", p, t, l) for p in (True, False) for t in (None, 0.05, 0.15) for l in (0.5, 2.0, 8.0)]
    for mode, pad, tau, lam in grid:
        r = run(dev, mode, pad, tau, lam)
        row = dict(mode=mode, pad=int(pad), tau=tau, lam=lam, n=len(dev), mean_iou=round(float(r[:, 0].mean()), 4),
                   median_iou=round(float(np.median(r[:, 0])), 4), hit=round(float(r[:, 1].mean()), 4), centre_err_med=round(float(np.median(r[:, 2])), 4))
        for a in ("iin", "k"):
            m = np.array([c["allograph"] == a for c in dev]); row["iou_" + a] = round(float(r[m, 0].mean()), 4); row["hit_" + a] = round(float(r[m, 1].mean()), 4)
        rows.append(row)
    rows.sort(key=lambda r: -r["mean_iou"])
    with open(os.path.join(OUT, "g3_dev_variants.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows[:8]: print(r)
    print("...")
    for r in rows:
        if r["mode"] == "equal": print(r)
    print("variants logged:", len(rows))
