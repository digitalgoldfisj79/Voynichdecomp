#!/usr/bin/env python3
"""PREREG 15: outer-face exposure wear from top/bottom edge soiling. -> results_wear.json, wear/page_S.csv"""
import json, os, pickle, sys
import numpy as np, pandas as pd
from PIL import Image
ROOT = sys.argv[1]; W = os.path.join(ROOT, "wear"); rng = np.random.default_rng(20261022)
F = pickle.load(open(os.path.join(ROOT, "data", "s14_features.pkl"), "rb"))["F"]; folios = list(F.index)
T = pd.read_csv(os.path.join(ROOT, "faces_per_sheet.csv"))
def S_index(p):
    im = np.asarray(Image.open(os.path.join(W, "img", f"{p}.jpg")).convert("L"), float); h, w = im.shape
    c = im[:, int(w * 0.25):int(w * 0.75)]; prof = np.median(c, 1); rows = np.where(prof > 60)[0]
    if len(rows) < 0.5 * h: return None
    top, bot = rows.min(), rows.max(); H = bot - top
    if H < 0.6 * h: return None
    def band(a, b): return np.percentile(c[a:b], 60)
    e = (band(top + int(.01 * H), top + int(.05 * H)) + band(bot - int(.05 * H), bot - int(.01 * H))) / 2
    i = (band(top + int(.20 * H), top + int(.30 * H)) + band(bot - int(.30 * H), bot - int(.20 * H))) / 2
    return dict(page=p, S=i - e, edge=e, inner=i, H=int(H))
pages = sorted(json.load(open(os.path.join(W, "pages_to_fetch.json"))))
rows = [r for r in (S_index(p) for p in pages) if r]
PS = pd.DataFrame(rows).set_index("page"); PS.to_csv(os.path.join(W, "page_S.csv"))
res = dict(n_pages=len(pages), n_ok=len(PS), detect_rate=len(PS) / len(pages))
pct = PS.S.rank(pct=True); res["gate_f1r_pct"] = float(pct.get("f1r", np.nan)); res["gate_f116v_pct"] = float(pct.get("f116v", np.nan))
res["gate_pass"] = bool(res["detect_rate"] >= 0.9 and res["gate_f1r_pct"] >= 0.9 and res["gate_f116v_pct"] >= 0.9)
T["quire"] = T.sheet.str.split(":").str[0]
T["outermost"] = T.groupby("quire").a.transform("min") == T.a
def D_of(r):
    pg = {k: folios[r[k]] for k in ("ar", "av", "br", "bv")}
    if not all(v in PS.index for v in pg.values()): return np.nan
    return (PS.S[pg["ar"]] + PS.S[pg["bv"]]) / 2 - (PS.S[pg["av"]] + PS.S[pg["br"]]) / 2
T["D"] = T.apply(D_of, axis=1)
def signflip(d):
    d = d[~np.isnan(d)]; null = (d[None, :] * rng.choice([1, -1], size=(100000, len(d)))).mean(1)
    return dict(n=int(len(d)), mean_D=float(d.mean()), null_sd=float(null.std()), z=float(d.mean() / null.std()),
                p_one=float((np.sum(null >= d.mean()) + 1) / (len(null) + 1)), positive=f"{int((d > 0).sum())}/{len(d)}")
res["PRIMARY_interior_sheets"] = signflip(T.loc[~T.outermost, "D"].values)
res["SECONDARY_outermost_sheets"] = signflip(T.loc[T.outermost, "D"].values)
res["all_sheets"] = signflip(T.D.values)
res["per_sheet"] = T[["sheet", "outermost", "central", "D"]].round(3).to_dict("records")
json.dump(res, open(os.path.join(ROOT, "results_wear.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "per_sheet"}, indent=1, default=float))
print(PS.S.describe().round(2).to_dict()); print(PS.sort_values("S", ascending=False).head(8).round(1))
