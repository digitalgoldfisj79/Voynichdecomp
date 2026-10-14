#!/usr/bin/env python3
"""Forum figures from the results files in ROOT. usage: python3 make_figures.py ROOT  -> ROOT/figures/*.png"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = sys.argv[1]; OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a9a8a2"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlecolor": INK})
J = lambda f: json.load(open(os.path.join(ROOT, f)))
def note(fig, t): fig.text(0.01, 0.01, t, fontsize=8.5, color=MUTED, ha="left", va="bottom")
def save(fig, name): fig.savefig(os.path.join(OUT, name), dpi=160); plt.close(fig); print("wrote", name)

# ---- Fig 1: per-sheet conjugate vs same-quire same-scribe other sheets
P = pd.read_csv(os.path.join(ROOT, "per_sheet_conjugate.csv"))
P["diff"] = P.same_stratum_other_sheet_JS - P.conjugate_mean_JS; P = P.sort_values("diff").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 10.5))
for i, r in P.iterrows(): ax.plot([r.conjugate_mean_JS, r.same_stratum_other_sheet_JS], [i, i], color=AXIS, lw=1.2, zorder=1)
ax.scatter(P.same_stratum_other_sheet_JS, P.index, s=46, color=ORANGE, zorder=2, edgecolor=SURF, lw=1.5, label="other sheets, same quire and same scribe")
ax.scatter(P.conjugate_mean_JS, P.index, s=46, color=BLUE, zorder=3, edgecolor=SURF, lw=1.5, label="the other half of the same sheet")
ax.set_yticks(P.index); ax.set_yticklabels(P.sheet, fontsize=8.5); ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
ax.set_xlabel("Text distance between pages (Jensen-Shannon on glyph transitions; lower = more alike)")
ax.set_title(f"In {int((P['diff'] > 0).sum())} of {len(P)} sheets, a page is closer to the other half of its own sheet", pad=28)
ax.legend(loc="upper center", bbox_to_anchor=(.45, 1.045), ncol=2, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, .04, 1, 1)); note(fig, "One row per bifolium (sheet codes from eo777/voynich-clusters). Comparison pages restricted to the same quire and the same Davis scribe.")
save(fig, "fig1_conjugate_per_sheet.png")

# ---- Fig 2: sheet vs opening, real vs shuffled control
real = J("results_t1b_t2_t5.json")["T2"]; R = J("results_posted_rarefied.json"); shuf = R["SHUF_RAW"]["T2"]
rows = [("Same sheet\n(conjugate leaves)", real["betas"]["conjugate"], real["ci_conj"], shuf["conj_beta"], shuf["ci_conj"]),
        ("Facing pages\n(same opening)", real["betas"]["facing"], real["ci_facing"], shuf["facing_beta"], shuf["ci_facing"])]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (lab, b, ci, bs, cis) in enumerate(rows):
    y = 1 - i
    ax.errorbar(bs, y - .14, xerr=[[bs - cis[0]], [cis[1] - bs]], fmt="o", color=GREY, ms=8, lw=2, label="control: words shuffled between pages of the same scribe and quire" if i == 0 else None)
    ax.errorbar(b, y + .14, xerr=[[b - ci[0]], [ci[1] - b]], fmt="o", color=BLUE if i == 0 else ORANGE, ms=9, lw=2.4, label="real text" if i == 0 else None)
    ax.text(max(ci[1], cis[1]) + .004, y, f"real - control = {b - bs:+.3f}", va="center", color=INK2, fontsize=10)
ax.axvline(0, color=AXIS, lw=1); ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-.6, 1.6); ax.set_xlim(-.085, .075)
ax.set_xlabel("Effect on text distance (standardised regression coefficient; further left = more alike)")
ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True); ax.set_title("Text similarity follows the sheet, not the opening", pad=34)
ax.legend(loc="upper center", bbox_to_anchor=(.45, 1.16), ncol=2, frameon=False, fontsize=9.5); fig.tight_layout(rect=(0, .1, 1, 1))
note(fig, "Regression on all page pairs, controlling for leaf, quire, scribe, section, Currier language, distance in book order and page size.\n"
          f"Bars = delete-one-sheet jackknife intervals; compare each point with its grey control. Same-sheet vs leaf-shuffle null: z = {real['conj_z']:.1f} "
          f"(z = {R['REAL_R60']['T2']['conj_z']:.1f} with every page cut to equal length).")
save(fig, "fig2_sheet_vs_opening.png")

# ---- Fig 3: robustness of the sheet effect
T1 = J("results_t1b_t2_t5.json")["T1b"]; TR = J("results_translits.json")["per_transliteration"]
tz = [v["F5"]["z"] for v in TR.values() if "F5" in v]
items = [("Sheet halves closer (leaf-shuffle test)", [("as measured", R["REAL_RAW"]["T_conj"]["z"], BLUE), ("every page cut to equal length", R["REAL_R60"]["T_conj"]["z"], BLUE),
          ("control: words shuffled within scribe x quire", R["SHUF_RAW"]["T_conj"]["z"], GREY), ("control, equal length", R["SHUF_R60"]["T_conj"]["z"], GREY)]),
         ("Sheet term after Currier, scribe, section, quire (PERMANOVA)", [("as measured", T1["z"], BLUE), ("every page cut to equal length", R["REAL_R60"]["T1b"]["z"], BLUE),
          ("control: words shuffled within scribe x quire", R["SHUF_RAW"]["T1b"]["z"], GREY)])]
fig, ax = plt.subplots(figsize=(10, 6)); y = 0; yt, yl = [], []
for head, rs in items:
    ax.text(-2.4, y + .55, head, fontsize=10.5, color=INK, fontweight="bold", va="center")
    for lab, z, c in rs:
        ax.plot([0, z], [y, y], color=c, lw=2.2, solid_capstyle="round"); ax.scatter([z], [y], s=60, color=c, zorder=3, edgecolor=SURF, lw=1.5)
        ax.text(max(z, 0) + .15, y, f"z {z:.1f}", va="center", fontsize=9.5, color=INK2); yt.append(y); yl.append(lab); y -= 1
    y -= 1
ax.text(-2.4, y + .55, f"Sheet halves closer, rebuilt in each of {len(tz)} transliterations", fontsize=10.5, color=INK, fontweight="bold", va="center")
ax.scatter(tz, [y] * len(tz), s=46, color=BLUE, alpha=.8, edgecolor=SURF, lw=1.2, zorder=3)
ax.text(max(tz) + .15, y, f"z {min(tz):.1f} to {max(tz):.1f}", va="center", fontsize=9.5, color=INK2); yt.append(y); yl.append("12 transliterations, 4 alphabets")
ax.axvline(2, color=MUTED, lw=1, ls=(0, (4, 3))); ax.text(2.05, 1.1, "z = 2", color=MUTED, fontsize=9)
ax.axvline(0, color=AXIS, lw=1); ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9.5); ax.set_xlim(-2.5, 9); ax.set_ylim(y - .8, 1.6)
ax.set_xlabel("z against the null (standard deviations)"); ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
ax.set_title("The sheet effect is robust; shuffled text gives nothing", pad=12)
fig.tight_layout(rect=(0, .05, 1, 1)); note(fig, "Equal length: every page (>= 60 words) subsampled to the same number of glyph pairs, 5 draws. Shuffled control keeps each page's size and its scribe/quire stratum.")
save(fig, "fig3_robustness.png")

# ---- Fig 4: what the text can order (known-order calibration) + VMS gradient reproducibility
O = J("results_order_methods.json"); ctrl = ["CI_start", "CI_end", "Galen_w1", "Galen_w2", "Galen_w3"]
meths = [("B1", "shortest path"), ("B0", "spectral\nseriation"), ("M1", "smoothness\n(TempSeq-like)"), ("M2", "incidence\n(biostratigraphy)")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
for i, (m, lab) in enumerate(meths):
    re_ = [100 * O[c]["all"][m]["adj"] for c in ctrl]; sh = [100 * O["SHUF_" + c]["all"][m]["adj"] for c in ctrl]
    a1.bar(i - .19, np.mean(re_), .34, color=ORANGE); a1.bar(i + .19, np.mean(sh), .34, color=GREY)
    a1.scatter([i - .19] * 5, re_, s=16, color=INK2, zorder=3); a1.text(i - .19, max(re_) + 1.2, f"{np.mean(re_):.0f}%", ha="center", fontsize=9.5, color=INK2)
a1.axhline(100 * 2 / 49, color=MUTED, ls=(0, (4, 3)), lw=1); a1.text(3.45, 100 * 2 / 49 + .6, "chance", color=MUTED, fontsize=9, ha="right")
a1.set_xticks(range(4)); a1.set_xticklabels([l for _, l in meths], fontsize=9.5); a1.set_ylabel("true neighbours found (%)")
from matplotlib.patches import Patch
a1.legend(handles=[Patch(color=ORANGE, label="known-order texts (Circa instans, Galen)"), Patch(color=GREY, label="same texts, words shuffled")], frameon=False, fontsize=9, loc="upper right"); a1.grid(axis="y", color=GRID, lw=.8); a1.set_axisbelow(True)
a1.set_title("Texts of KNOWN order, cut to VMS sheet sizes:\nat best about 1 neighbour in 5 is recovered", fontsize=11.5)
for i, (m, lab) in enumerate([("B0", "spectral seriation"), ("M2", "incidence")]):
    a2.scatter([O["VMS"]["reliability_z"][m]], [i + .2], s=80, color=BLUE, zorder=3, edgecolor=SURF, lw=1.5, label="VMS Herbal A, scribe 1 (21 sheets)" if i == 0 else None)
    a2.scatter([O[c]["reliability_z"][m] for c in ctrl], [i] * 5, s=34, color=ORANGE, zorder=3, label="known-order texts" if i == 0 else None)
    a2.scatter([O["SHUF_" + c]["reliability_z"][m] for c in ctrl], [i - .2] * 5, s=34, color=GREY, zorder=3, label="shuffled" if i == 0 else None)
a2.axvline(2, color=MUTED, lw=1, ls=(0, (4, 3))); a2.set_yticks([0, 1]); a2.set_yticklabels(["spectral\nseriation", "incidence"])
a2.set_xlabel("leaf-A order vs leaf-B order agreement (z)"); a2.set_ylim(-.6, 1.8)
a2.legend(frameon=False, fontsize=9, loc="upper left"); a2.grid(axis="x", color=GRID, lw=.8); a2.set_axisbelow(True)
a2.set_title("VMS: the order built from one leaf of each sheet\nagrees with the order from the other leaf", fontsize=11.5)
fig.tight_layout(rect=(0, .07, 1, 1)); note(fig, "49 sheets per text; every sheet given the page sizes of a VMS sheet; equal-size subsamples, 3 draws. Right: 21-sheet group, "
                                             "order rebuilt independently from each sheet's two leaves.")
save(fig, "fig4_what_text_can_order.png")

# ---- Fig 5: Echo chain, interior stages vs a mixture of earlier and later pages
C = J("results_chain_mixture.json")["full_chain"]; E = J("results_chain_rule_enrichment.json")
pages = pd.read_csv(os.path.join(ROOT, "data", "eo777_chain_pages.csv"))
st = list(C); x = np.arange(len(st))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.6), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 1]))
for i, k in enumerate(st):
    m, s = C[k]["null_mean"], C[k]["null_sd"]
    a1.add_patch(plt.Rectangle((i - .25, max(m - 2 * s, 0)), .5, 4 * s, color=GRID, zorder=1))
    a1.plot([i - .25, i + .25], [m, m], color=MUTED, lw=1.5, zorder=2)
    a1.scatter([i], [C[k]["N_peaks"]], s=70, color=BLUE, zorder=3, edgecolor=SURF, lw=1.5)
    a1.text(i + .3, C[k]["N_peaks"], f"z {C[k]['z']:.1f}", va="center", fontsize=9.5, color=INK2)
a1.set_ylabel("word forms concentrated\nin this stage"); a1.grid(axis="y", color=GRID, lw=.8); a1.set_axisbelow(True)
a1.scatter([], [], s=60, color=BLUE, label="observed"); a1.add_patch(plt.Rectangle((0, 0), 0, 0, color=GRID, label="expected if the stage were a mix of earlier and later pages (mean +/- 2 SD)"))
a1.legend(frameon=False, fontsize=9, loc="upper left"); a1.set_title("Echo chain: which stages are more than a mix of their neighbours?", pad=10)
sh = [100 * E[k]["peaks_rule_outputs"] / max(E[k]["peaks"], 1) for k in st]; base = 100 * E[st[0]]["tested_rule_outputs"] / E[st[0]]["tested"]
a2.bar(x, sh, .5, color=BLUE); a2.axhline(base, color=MUTED, ls=(0, (4, 3)), lw=1); a2.text(-.45, base + 2, "share among all forms tested", color=MUTED, fontsize=9, ha="left")
for i, k in enumerate(st): a2.text(i, sh[i] + 2, f"{E[k]['peaks_rule_outputs']}/{E[k]['peaks']}", ha="center", fontsize=9, color=INK2)
a2.set_ylabel("% of those forms that are\noutputs of Echo's rules"); a2.set_ylim(0, 100); a2.grid(axis="y", color=GRID, lw=.8); a2.set_axisbelow(True)
lab = []
for k in st:
    sc = pages[pages.group == k].scribe.astype(str).str.replace(".0", "", regex=False).value_counts()
    lab.append(f"{k}\nscribe {'+'.join(s for s in sc.index if s not in ('missing', 'nan'))}")
a2.set_xticks(x); a2.set_xticklabels(lab, fontsize=9.5)
fig.tight_layout(rect=(0, .07, 1, 1)); note(fig, "Interior stages of eo777's Echo chain (1a and 4c are the ends). A form counts if its rate in the stage beats both all earlier and all later stages (binomial p < .01 each).\n"
                                             "Null: whole pages reshuffled among earlier / this / later stages, 2,000 times. Lower panel is exploratory. Scribes = Davis's hands on the chain's pages.")
save(fig, "fig5_echo_chain_mixture.png")
