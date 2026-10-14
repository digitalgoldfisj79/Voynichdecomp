#!/usr/bin/env python3
"""Per-sheet table for Figure 2: mean text distance between the two halves of each sheet vs the mean distance from
its pages to pages of OTHER sheets in the same quire and by the same Davis scribe. -> per_sheet_conjugate.csv"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib_methods as L
ROOT = sys.argv[1]; S = pickle.load(open(os.path.join(ROOT, "data", "s14_features.pkl"), "rb")); F = S["F"]
Dm = L.js_matrix(L.profiles(S["M"], S["wmask"])); sheet, strat = F.sheet.values, F.stratum.values
leaf = F.leaf.astype(str).values; m = len(F); rows = []
for s in np.unique(sheet):
    a = np.where(sheet == s)[0]
    conj = [Dm[i, j] for i in a for j in a if i < j and leaf[i] != leaf[j]]
    oth = [Dm[i, j] for i in a for j in range(m) if sheet[j] != s and strat[j] == strat[i]]
    if conj and oth: rows.append((s, np.mean(conj), np.mean(oth)))
P = pd.DataFrame(rows, columns=["sheet", "conjugate_mean_JS", "same_stratum_other_sheet_JS"])
P.to_csv(os.path.join(ROOT, "per_sheet_conjugate.csv"), index=False)
print(f"{int((P.conjugate_mean_JS < P.same_stratum_other_sheet_JS).sum())}/{len(P)} sheets: conjugate halves closer")
