#!/usr/bin/env python3
"""Figures for the voynich.ninja post (thread 6182). Light surface, palette validated (blue/orange)."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = sys.argv[1]; OUT = os.path.join(ROOT, "post"); os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlecolor": INK})
def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8.5, color=MUTED, ha="left", va="bottom")
def save(fig, name):
    p = os.path.join(OUT, name); fig.savefig(p, dpi=160); plt.close(fig); print("wrote", p)

# ---------------------------------------------------------------- Fig 1: schematic
fig = plt.figure(figsize=(11, 5.2))
ax = fig.add_axes([0.03, 0.15, 0.52, 0.73]); ax.set_axis_off()
ax.set_xlim(-5, 5); ax.set_ylim(-3.3, 3.2)
xs = {1: -4, 2: -3, 3: -2, 4: -1, 5: 1, 6: 2, 7: 3, 8: 4}
for k, (a, b) in enumerate([(1, 8), (2, 7), (3, 6), (4, 5)]):
    col = BLUE if (a, b) == (1, 8) else AXIS; lw = 3 if (a, b) == (1, 8) else 1.6
    for leaf in (a, b):
        ax.plot([xs[leaf], xs[leaf]], [0, 2.2], color=col, lw=lw, solid_capstyle="round")
    t = np.linspace(np.pi, 2 * np.pi, 60); cx = (xs[a] + xs[b]) / 2; r = (xs[b] - xs[a]) / 2
    ax.plot(cx + r * np.cos(t), r * 0.55 * np.sin(t), color=col, lw=lw)
for leaf, x in xs.items():
    ax.text(x, 2.45, str(leaf), ha="center", color=INK, fontsize=12, fontweight="bold")
ax.annotate("", xy=(xs[3] + 0.08, 2.95), xytext=(xs[4] - 0.08, 2.95),
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=3))
ax.text(-1.5, 3.1, "opening: 3v faces 4r", ha="center", color=ORANGE, fontsize=11, fontweight="bold")
ax.text(0, -2.75, "sheet (bifolium): leaves 1 and 8 are one folded piece of parchment",
        ha="center", color=BLUE, fontsize=11, fontweight="bold")
ax.text(-5, -3.3, "A quire of four nested sheets, leaves 1-8 in reading order", color=INK2, fontsize=10)
ax2 = fig.add_axes([0.60, 0.14, 0.38, 0.72]); ax2.set_axis_off(); ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
for y0, left, right, lab in ((5.6, "8v", "1r", "outer face"), (0.8, "1v", "8r", "inner face")):
    ax2.add_patch(Rectangle((0.5, y0), 9, 3.2, fill=False, ec=BLUE, lw=2.2))
    ax2.plot([5, 5], [y0, y0 + 3.2], color=BLUE, lw=1, ls=(0, (3, 3)))
    ax2.text(2.75, y0 + 1.6, left, ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
    ax2.text(7.25, y0 + 1.6, right, ha="center", va="center", fontsize=18, color=INK, fontweight="bold")
    ax2.text(0.5, y0 + 3.45, lab, color=INK2, fontsize=11)
ax2.text(0.5, 9.6, "The same sheet laid flat", color=INK, fontsize=12, fontweight="bold")
ax2.text(5, 4.9, "fold", ha="center", color=MUTED, fontsize=9)
fig.suptitle("Three kinds of neighbour", x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)
note(fig, "Written in the bound book, front to back: facing pages (orange) should resemble each other most.\n"
          "Written on the loose sheet: the two halves of a sheet (blue), and the two pages on one face, should.")
save(fig, "fig1_sheet_opening_face.png")

# ---------------------------------------------------------------- Fig 2: per-sheet conjugate vs control
P = pd.read_csv(os.path.join(ROOT, "per_sheet_conjugate.csv"))
P["diff"] = P.same_stratum_other_sheet_JS - P.conjugate_mean_JS
P = P.sort_values("diff").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 10.5))
for i, r in P.iterrows():
    ax.plot([r.conjugate_mean_JS, r.same_stratum_other_sheet_JS], [i, i], color=AXIS, lw=1.2, zorder=1)
ax.scatter(P.same_stratum_other_sheet_JS, P.index, s=46, color=ORANGE, zorder=2, edgecolor=SURF, linewidth=1.5,
           label="other sheets, same quire and same scribe")
ax.scatter(P.conjugate_mean_JS, P.index, s=46, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.5,
           label="the other half of the same sheet")
ax.set_yticks(P.index); ax.set_yticklabels(P.sheet, fontsize=8.5)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Text distance between pages (Jensen-Shannon, glyph transitions) - lower = more alike")
wins = int((P["diff"] > 0).sum())
ax.set_title(f"In {wins} of {len(P)} sheets, a page's text is closer to the other half of its own sheet", pad=28)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.045), ncol=2, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0.05, 1, 1))
note(fig, "Each row is one bifolium (sheet codes from eo777/voynich-clusters). Comparison pages are restricted to the same quire "
          "and the same Davis scribe.\nThe 3 rows at the bottom go the other way. Sign test p = 3e-8.")
save(fig, "fig2_conjugate_per_sheet.png")

# ---------------------------------------------------------------- Fig 3: sheet vs opening
B = json.load(open(os.path.join(ROOT, "results_bounding5.json")))["T2_bound"][:3]
labels = ["All page pairs", "Facing pairs that are not\nalso the same sheet", "Pages with 60+ words only"]
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for i, (r, lab) in enumerate(zip(B, labels)):
    y = len(B) - 1 - i
    ax.errorbar(r["beta_conj"], y + 0.13, xerr=[[r["beta_conj"] - r["ci_conj"][0]], [r["ci_conj"][1] - r["beta_conj"]]],
                fmt="o", color=BLUE, ms=8, capsize=0, lw=2, label="same sheet (conjugate leaves)" if i == 0 else None)
    ax.errorbar(r["beta_facing"], y - 0.13, xerr=[[r["beta_facing"] - r["ci_facing"][0]], [r["ci_facing"][1] - r["beta_facing"]]],
                fmt="o", color=ORANGE, ms=8, capsize=0, lw=2, label="facing pages (same opening)" if i == 0 else None)
ax.axvline(0, color=INK2, lw=1)
ax.set_yticks(range(len(B))); ax.set_yticklabels(labels[::-1])
ax.set_xlabel("Effect on text distance (standardised; left of zero = more alike than chance)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Text similarity follows the sheet, not the opening", pad=30)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.13), ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.08, 1, 1))
note(fig, "Regression on all page-pair distances, controlling for leaf, quire, scribe, section, Currier language, distance in book order\n"
          "and page size. Bars = 95% delete-one-sheet jackknife intervals. Conjugate effect vs leaf-permutation null: z = 7.0.")
save(fig, "fig3_sheet_vs_opening.png")

# ---------------------------------------------------------------- Fig 4: same face, text and image
Fc = pd.read_csv(os.path.join(ROOT, "faces_per_sheet.csv")); Fc = Fc[~Fc.central]
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
for ax, col, title, color in ((axes[0], "D_text_JS", "Text (glyph transitions)", BLUE),
                              (axes[1], "D_image_raw", "Images (control: skin side)", ORANGE)):
    d = Fc[col].sort_values().values
    ax.bar(range(len(d)), d, color=color, width=0.78)
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(d.mean(), color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0, d.mean(), f"mean {d.mean():.4f}", ha="left", va="bottom", fontsize=9, color=INK2, bbox=dict(fc=SURF, ec="none", pad=1))
    ax.set_xticks([]); ax.set_xlabel(f"{len(d)} bifolia, sorted ({int((d > 0).sum())} above zero)")
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("opposite-face distance minus same-face distance\n(above zero = same face more alike)")
fig.suptitle("Pages on the same face of the flat sheet are more alike", x=0.01, ha="left", fontsize=14,
             fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
note(fig, "Within each bifolium (leaves a<b): same face = (ar, bv) and (av, br); opposite face = (ar, br) and (av, bv). Every page sits in one pair of each kind,\n"
          "so scribe, section and page size cancel. Centre bifolia excluded. Text: z = 2.39, p = 0.007 (sign-flip). Images: z = 4.3, the face mapping checks out.")
save(fig, "fig4_same_face.png")

# ---------------------------------------------------------------- Fig 5: across transliterations
R = json.load(open(os.path.join(ROOT, "results_translits.json")))["per_transliteration"]
order = [("TTVE", "Takahashi (TTVE)"), ("TTIA", "Takahashi (IT)"), ("TTLI", "Takahashi (LSI)"),
         ("ZLZI", "Zandbergen-Landini (ZL ivtff)"), ("ZLZB", "Zandbergen-Landini (ZL3b)"),
         ("VDRB-1", "VMS Database (RF1b-er)"), ("VDRB", "VMS Database (RF1b-e)"),
         ("FFSG", "Friedman group (FG)"), ("FFSG-2", "Friedman group (FSG)"),
         ("GCGA", "Glen Claston (v101)"), ("PCCA", "Currier (CD)"), ("RGVN", "Reeds/Gillogly")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw=dict(width_ratios=[1.25, 1]), sharey=True)
for i, (k, lab) in enumerate(order):
    y = len(order) - 1 - i; f1, f5 = R[k]["F1"], R[k]["F5"]
    se = f1["mean_D"] / f1["z"]
    a1.errorbar(f1["mean_D"], y, xerr=1.96 * se, fmt="o", color=BLUE, ms=7, lw=2, capsize=0)
    a2.barh(y, f5["pct_lower"], color=ORANGE, height=0.62)
    a2.text(f5["pct_lower"] + 0.1, y, f"z {f5['z']:.1f}", va="center", fontsize=9, color=INK2)
a1.axvline(0, color=INK2, lw=1)
a1.set_yticks(range(len(order))); a1.set_yticklabels([lab for _, lab in order][::-1])
a1.set_xlabel("same-face effect (with 95% interval)")
a1.set_title("Same face of the flat sheet", fontsize=12)
a2.set_xlabel("% closer than other sheets\n(same quire, same scribe)")
a2.set_title("Two halves of one sheet", fontsize=12); a2.set_xlim(0, 7.5)
for a in (a1, a2): a.grid(axis="x", color=GRID, lw=0.8); a.set_axisbelow(True)
fig.suptitle("Same result in 12 transliterations and four alphabet families", x=0.01, ha="left", fontsize=14,
             fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
note(fig, "Each transliteration rebuilt from scratch in its own alphabet (EVA, Currier, FSG, v101). Exact duplicates removed. "
          "Currier and Reeds/Gillogly cover 25 of 33 sheets, the rest 31-32.")
save(fig, "fig5_transliterations.png")
