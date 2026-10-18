#!/usr/bin/env python3
"""R3 corrections: fig1 without the flat-face panel; fig transliterations with the sheet panel only."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = sys.argv[1]; OUT = os.path.join(ROOT, "post")
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlelocation": "left", "axes.titlecolor": INK})
def note(fig, t): fig.text(0.01, 0.01, t, fontsize=8.5, color=MUTED, ha="left", va="bottom")
fig = plt.figure(figsize=(8.5, 5.4)); ax = fig.add_axes([0.04, 0.16, 0.92, 0.72]); ax.set_axis_off()
ax.set_xlim(-5, 5); ax.set_ylim(-3.3, 3.2)
xs = {1: -4, 2: -3, 3: -2, 4: -1, 5: 1, 6: 2, 7: 3, 8: 4}
for a, b in [(1, 8), (2, 7), (3, 6), (4, 5)]:
    col = BLUE if (a, b) == (1, 8) else AXIS; lw = 3 if (a, b) == (1, 8) else 1.6
    for leaf in (a, b): ax.plot([xs[leaf]] * 2, [0, 2.2], color=col, lw=lw, solid_capstyle="round")
    t = np.linspace(np.pi, 2 * np.pi, 60); cx = (xs[a] + xs[b]) / 2; r = (xs[b] - xs[a]) / 2
    ax.plot(cx + r * np.cos(t), r * 0.55 * np.sin(t), color=col, lw=lw)
for leaf, x in xs.items(): ax.text(x, 2.45, str(leaf), ha="center", color=INK, fontsize=12, fontweight="bold")
ax.annotate("", xy=(xs[3] + 0.08, 2.95), xytext=(xs[4] - 0.08, 2.95), arrowprops=dict(arrowstyle="-", color=ORANGE, lw=3))
ax.text(-1.5, 3.1, "opening: 3v faces 4r", ha="center", color=ORANGE, fontsize=11, fontweight="bold")
ax.text(0, -2.75, "sheet (bifolium): leaves 1 and 8 are one folded piece of parchment", ha="center", color=BLUE,
        fontsize=11, fontweight="bold")
ax.text(-5, -3.3, "A quire of four nested sheets, leaves 1-8 in reading order", color=INK2, fontsize=10)
fig.suptitle("Two kinds of neighbour", x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)
note(fig, "Written in the bound book, front to back: facing pages (orange) should resemble each other most.\n"
          "Written on the loose sheet before nesting: the two halves of a sheet (blue) should.")
fig.savefig(os.path.join(OUT, "fig1_sheet_opening_v2.png"), dpi=160); plt.close(fig)
R = json.load(open(os.path.join(ROOT, "results_translits.json")))["per_transliteration"]
order = [("TTVE", "Takahashi (TTVE)"), ("TTIA", "Takahashi (IT)"), ("TTLI", "Takahashi (LSI)"),
         ("ZLZI", "Zandbergen-Landini (ZL ivtff)"), ("ZLZB", "Zandbergen-Landini (ZL3b)"),
         ("VDRB-1", "VMS Database (RF1b-er)"), ("VDRB", "VMS Database (RF1b-e)"),
         ("FFSG", "Friedman group (FG)"), ("FFSG-2", "Friedman group (FSG)"),
         ("GCGA", "Glen Claston (v101)"), ("PCCA", "Currier (CD)"), ("RGVN", "Reeds/Gillogly")]
fig, ax = plt.subplots(figsize=(9, 6.2))
for i, (k, lab) in enumerate(order):
    y = len(order) - 1 - i; f5 = R[k]["F5"]
    ax.barh(y, f5["pct_lower"], color=BLUE, height=0.62)
    ax.text(f5["pct_lower"] + 0.1, y, f"z {f5['z']:.1f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([l for _, l in order][::-1]); ax.set_xlim(0, 7.5)
ax.set_xlabel("% closer than other sheets (same quire, same scribe)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Two halves of one sheet: same result in 12 transliterations", pad=14)
fig.tight_layout(rect=(0, 0.06, 1, 1))
note(fig, "Each transliteration rebuilt from scratch in its own alphabet (EVA, Currier, FSG, v101). Exact duplicates removed.\n"
          "Distances not length-equalised here; the length-equalised rerun on the main text used throughout gives z = 6.6.")
fig.savefig(os.path.join(OUT, "fig4_transliterations_v2.png"), dpi=160); print("ok")
