#!/usr/bin/env python3
"""Stage 18 (PREREG 6): same-face vs cross-face pairs within flat bifolia."""
import json, os, pickle, re, sys
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
sys.path.insert(0, "/home/claude/horseshoe/scripts")
import lib_methods as L
ROOT = sys.argv[1]; D = os.path.join(ROOT, "data"); rng = np.random.default_rng(20261012)
S = pickle.load(open(os.path.join(D, "s14_features.pkl"), "rb")); F = S["F"]
Dtext = L.js_matrix(L.profiles(S["M"], S["wmask"]))
C = pickle.load(open(os.path.join(D, "s2_folio.pkl"), "rb"))[2]
geo = ["logw", "logh", "logasp", "wlen", "n_words"]
G = F[geo].values.astype(float); G = (G - G.mean(0)) / G.std(0); A = np.c_[np.ones(len(G)), G]
Ci = C - A @ np.linalg.lstsq(A, C, rcond=None)[0]
dists = {"text_JS": Dtext, "image_raw": squareform(pdist(C, "cosine")),
         "image_geo_resid": squareform(pdist(Ci, "cosine")), "geometry_only": squareform(pdist(G))}
idx = {f: i for i, f in enumerate(F.index)}
rows = []
for sh, sub in F.groupby("sheet"):
    m = re.match(r".*:(\d+)-(\d+)$", sh)
    if not m: continue
    a, b = sorted((int(m.group(1)), int(m.group(2))))
    if a == b: continue
    want = {("a", "r"): f"f{a}r", ("a", "v"): f"f{a}v", ("b", "r"): f"f{b}r", ("b", "v"): f"f{b}v"}
    if not all(w in idx for w in want.values()): continue
    rows.append(dict(sheet=sh, a=a, b=b, central=(b - a == 1),
                     ar=idx[want[("a", "r")]], av=idx[want[("a", "v")]], br=idx[want[("b", "r")]], bv=idx[want[("b", "v")]]))
T = pd.DataFrame(rows)
def per_sheet(Dm):
    same = (Dm[T.ar, T.bv] + Dm[T.av, T.br]) / 2
    cross = (Dm[T.ar, T.br] + Dm[T.av, T.bv]) / 2
    return np.asarray(cross - same)
def signflip(d, n=100000):
    obs = d.mean()
    if len(d) <= 20:
        import itertools
        null = np.array([(d * np.array(s)).mean() for s in itertools.product([1, -1], repeat=len(d))])
    else:
        null = (d[None, :] * rng.choice([1, -1], size=(n, len(d)))).mean(1)
    return dict(n_sheets=int(len(d)), mean_D=float(obs), null_sd=float(null.std()), z=float(obs / null.std()),
                p_one_sided=float((np.sum(null >= obs) + 1) / (len(null) + 1)),
                sheets_same_face_closer=int((d > 0).sum()), MDE_2sd=float(2 * null.std()))
res = {"n_standard_bifolia": int(len(T)), "n_central": int(T.central.sum()), "sheets": T.sheet.tolist()}
for tag, mask in (("F1_noncentral", ~T.central.values), ("F2_all", np.ones(len(T), bool))):
    for name, Dm in dists.items():
        res[f"{tag}__{name}"] = signflip(per_sheet(Dm)[mask])
r = res["F1_noncentral__text_JS"]
if r["mean_D"] > 0 and r["p_one_sided"] < 0.01 and r["z"] >= 2: v = "SAME-FACE CLOSER"
elif r["z"] < 1: v = "UNDERPOWERED" if r["MDE_2sd"] > 0.019 else "NO FACE EFFECT"
else: v = "NOT RESOLVED"
res["F1_verdict"] = v
T.assign(**{f"D_{k}": per_sheet(v_) for k, v_ in dists.items()}).to_csv(os.path.join(ROOT, "faces_per_sheet.csv"), index=False)
json.dump(res, open(os.path.join(ROOT, "results_faces.json"), "w"), indent=1)
for k, v_ in res.items():
    if k != "sheets": print(k, v_)
