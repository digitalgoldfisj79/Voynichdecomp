#!/usr/bin/env python3
"""PREREG 8: is the same-face effect a within-sheet reading-order (singulion sequence) effect?
inner face (av,br) = reading distance 1; cross (ar,br),(av,bv) = distance 2; outer face (ar,bv) = distance 3.
SEQUENCE predicts D_outer = cross - outer < 0; FACE predicts D_outer > 0. Verdict at |z| >= 2."""
import json, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/horseshoe/scripts"); import lib_methods as L
R = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/horseshoe"
rng = np.random.default_rng(20261012)
S = pickle.load(open(f"{R}/data/s14_features.pkl", "rb"))
D = L.js_matrix(L.profiles(S["M"], S["wmask"]))
T = pd.read_csv(f"{R}/faces_per_sheet.csv"); T = T[~T.central]
ar, av, br, bv = [T[c].values for c in ("ar", "av", "br", "bv")]
inner, outer, cross = D[av, br], D[ar, bv], (D[ar, br] + D[av, bv]) / 2
def sf(x):
    null = (x[None, :] * rng.choice([1, -1], size=(100000, len(x)))).mean(1)
    return dict(mean=round(float(x.mean()), 5), null_sd=round(float(null.std()), 5),
                z=round(float(x.mean() / null.std()), 2), positive=f"{int((x > 0).sum())}/{len(x)}")
res = dict(n=len(T), mean_inner=float(inner.mean()), mean_cross=float(cross.mean()), mean_outer=float(outer.mean()),
           D_inner=sf(cross - inner), D_outer=sf(cross - outer), inner_vs_outer=sf(outer - inner))
z = res["D_outer"]["z"]; res["verdict"] = "FACE" if z >= 2 else ("SEQUENCE" if z <= -2 else "NOT RESOLVED")
json.dump(res, open(f"{R}/results_face_vs_sequence.json", "w"), indent=1); print(json.dumps(res, indent=1))
