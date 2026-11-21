"""Glyph stage 5: descriptive diagnostics on the held-out sets (post hoc, not part of acceptance) and
fixtures for the site's JS port of the frozen splitter.
Usage: python3 -I g5_diag_fixtures.py <out_dir> -> out/glyph_width_table.json, out/g5_diag.csv"""
import csv, json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glyphsplit as G, g3_dev as D
OUT = sys.argv[1]
tab, pad = D.tabs[True]
held = [c for c in D.cases if c["set"] in ("HELD_A", "HELD_B")]
rows = []
for c in held:
    e = G.split(c["word"], c["profile"], mode="table", table=tab, pad=pad)
    i0, i1 = c["span"]; p0 = c["x_origin"] + e[i0] * c["px_per_col"]; p1 = c["x_origin"] + e[i1] * c["px_per_col"]
    g0, g1 = c["gold"]; bw = c["box"][2] - c["box"][0]; mcw = bw / len(c["word"])
    rows.append(dict(allograph=c["allograph"], n_chars=len(c["word"]), start=i0, at_end=int(i1 == len(c["word"])),
                     centre_err_px=abs((p0 + p1) / 2 - (g0 + g1) / 2), centre_err_in_mean_char_w=abs((p0 + p1) / 2 - (g0 + g1) / 2) / mcw,
                     left_edge_err_px=abs(p0 - g0), right_edge_err_px=abs(p1 - g1), gold_w=g1 - g0, pred_w=p1 - p0))
with open(os.path.join(OUT, "g5_diag.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
A = lambda k, sel=None: np.array([r[k] for r in rows if sel is None or sel(r)])
print("held-out n", len(rows))
print("centre error, full-res px: median %.1f  p90 %.1f" % (np.median(A("centre_err_px")), np.quantile(A("centre_err_px"), .9)))
print("centre error in mean character widths: median %.2f  p90 %.2f  share > 0.5: %.3f" % (np.median(A("centre_err_in_mean_char_w")), np.quantile(A("centre_err_in_mean_char_w"), .9), (A("centre_err_in_mean_char_w") > 0.5).mean()))
print("edge error px: left median %.1f, right median %.1f" % (np.median(A("left_edge_err_px")), np.median(A("right_edge_err_px"))))
print("gold width median %.0f px vs predicted %.0f px" % (np.median(A("gold_w")), np.median(A("pred_w"))))
for lo, hi in ((1, 3), (4, 5), (6, 7), (8, 30)):
    s = lambda r: lo <= r["n_chars"] <= hi
    if len(A("n_chars", s)): print("word length %d-%d: n %d, centre error in char widths median %.2f p90 %.2f" % (lo, hi, len(A("n_chars", s)), np.median(A("centre_err_in_mean_char_w", s)), np.quantile(A("centre_err_in_mean_char_w", s), .9)))
s = lambda r: r["allograph"] == "k"
print("k only: centre error in char widths median %.2f p90 %.2f share>0.5 %.3f" % (np.median(A("centre_err_in_mean_char_w", s)), np.quantile(A("centre_err_in_mean_char_w", s), .9), (A("centre_err_in_mean_char_w", s) > 0.5).mean()))

mean_w = float(np.mean(list(tab.values())))
J = {"version": "20261121-1", "units": "relative character widths; only ratios matter", "pad": round(pad, 4), "default": round(mean_w, 4),
     "widths": {k: round(v, 4) for k, v in sorted(tab.items())}}
json.dump(J, open(os.path.join(OUT, "glyph_width_table.json"), "w"), indent=1)
def edges(word):
    w = [max(J["widths"].get(ch, J["default"]), 0.001) for ch in word]; tot = sum(w) + J["pad"]; e = [0.0]; acc = J["pad"] / 2
    for x in w: acc += x; e.append(acc / tot)
    e[0] = 0.0; e[-1] = 1.0
    return [round(v, 4) for v in e]
for wd in ("daiin", "qokeedy", "shol", "y", "chol8"):
    print(wd, edges(wd))
# check the JSON port equals the frozen library on real cases
mx = 0
for c in held[:300]:
    e = G.prior_cuts(c["word"], 0, 1, tab, pad); mx = max(mx, float(np.abs(np.array(edges(c["word"])) - e).max()))
print("max |port - library| edge difference over 300 words:", round(mx, 5), "| table:", J)
