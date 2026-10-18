#!/usr/bin/env python3
"""Fig 3 v2: real text vs within-stratum word-shuffle control, MRM betas with jackknife intervals (R3 correction)."""
import json, os, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = sys.argv[1]; OUT = os.path.join(ROOT, "post")
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlecolor": INK})
real = json.load(open(os.path.join(ROOT, "results_t1b_t2_t5.json")))["T2"]
shuf = json.load(open(os.path.join(ROOT, "results_posted_rarefied.json")))["SHUF_RAW"]["T2"]
rows = [("Same sheet\n(conjugate leaves)", real["betas"]["conjugate"], real["ci_conj"], shuf["conj_beta"], shuf["ci_conj"]),
        ("Facing pages\n(same opening)", real["betas"]["facing"], real["ci_facing"], shuf["facing_beta"], shuf["ci_facing"])]
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for i, (lab, b, ci, bs, cis) in enumerate(rows):
    y = 1 - i
    ax.errorbar(bs, y - 0.14, xerr=[[bs - cis[0]], [cis[1] - bs]], fmt="o", color=MUTED, ms=8, lw=2,
                label="control: words shuffled between pages of the same scribe and quire" if i == 0 else None)
    ax.errorbar(b, y + 0.14, xerr=[[b - ci[0]], [ci[1] - b]], fmt="o", color=BLUE if i == 0 else ORANGE, ms=9, lw=2.4,
                label="real text" if i == 0 else None)
    ax.text(max(ci[1], cis[1]) + 0.004, y, f"real - control = {b - bs:+.3f}", va="center", color=INK2, fontsize=10)
ax.axvline(0, color=AXIS, lw=1)
ax.set_yticks([1, 0]); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-0.6, 1.6); ax.set_xlim(-0.085, 0.075)
ax.set_xlabel("Effect on text distance (standardised; further left = more alike)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Text similarity follows the sheet, not the opening", pad=34)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=2, frameon=False, fontsize=9.5)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.text(0.01, 0.01, "Regression on all page-pair distances, controlling for leaf, quire, scribe, section, Currier language, "
         "distance in book order and page size.\nBars = delete-one-sheet jackknife intervals. Compare each point with its grey "
         "control, not with zero. Same-sheet effect vs leaf-shuffle null:\nz = 7.0 (z = 6.4 with every page cut to equal length).",
         fontsize=8.5, color=MUTED, ha="left", va="bottom")
p = os.path.join(OUT, "fig3_sheet_vs_opening_v2.png"); fig.savefig(p, dpi=160); print("wrote", p)
