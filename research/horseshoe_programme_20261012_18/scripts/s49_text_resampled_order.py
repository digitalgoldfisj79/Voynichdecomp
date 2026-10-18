#!/usr/bin/env python3
"""PREREG 34 (RUNNING_RESULTS_horseshoe.md, 2026-10-17): npcompl33t's pool50 ElPiGraph ordering with TEXT (transition-count) resampling
instead of fixed MDS coordinates. His code is imported unchanged from the cloned repo; only the input counts are resampled.
usage: python3 s49_text_resampled_order.py TEAM_GOLF_DIR OUT [B]
"""
import importlib.util, json, os, pickle, sys
import numpy as np
from multiprocessing import Pool
TEAM, OUT = sys.argv[1], sys.argv[2]; B = int(sys.argv[3]) if len(sys.argv) > 3 else 100
SEED, LAM = 20261017, 0.01
def load_core():
    spec = importlib.util.spec_from_file_location("golf_core", os.path.join(TEAM, "golf_stability_20261016/core.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
core = None; CD = None; META = None; LABELS = None; ANCHORS = None
def init():
    global core, CD, META, LABELS, ANCHORS
    os.environ["OMP_NUM_THREADS"] = "1"
    core = load_core(); CD = core.load_count_data(); META = CD["meta"]; LABELS = np.array([m["group"] for m in META])
    design = json.load(open(os.path.join(TEAM, "golf_published_stability_20261016/ensemble_pool50_point_l010.json")))["design"]
    ANCHORS = [s["anchor_order"] for s in design["starts"]]
def resample(counts, rng):
    out = {}
    for mode, x in counts.items():
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            for r in range(x.shape[1]):
                n = int(x[i, r].sum())
                if n: y[i, r] = rng.multinomial(n, x[i, r] / n)
        out[mode] = y
    return out
def one(b):
    p = os.path.join(OUT, "stages", f"rep_{b:03d}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    counts = CD["pool_counts"] if b == 0 else resample(CD["pool_counts"], np.random.default_rng(SEED + b))
    d = core.distance_from_counts(counts); x, _ = core.COMMON.embed(d)
    w = np.ones(len(x)) / len(x)
    centers = {g: x[LABELS == g].mean(0) for g in core.ORDER}
    fits = []
    for k, anchor in enumerate(ANCHORS):
        init_nodes = np.array([centers[g] for g in anchor])
        grown = core.orient_curve(core.grow_curve(x, w, init_nodes, lam=LAM), x, META)
        model = core.refine_curve(x, w, grown["nodes"], grown["edges"], lam=LAM)
        t = core.project_curve(x, model)["positions"]
        med = lambda g: float(np.median(t[LABELS == g]))
        fits.append(dict(start=k, objective=float(model["objective"]), positions=np.asarray(t), early_ok=med("1a") < med("1b") - 1e-7))
    elig = [f for f in fits if f["early_ok"]]
    best = min(elig, key=lambda f: (f["objective"], f["start"])) if elig else None
    res = dict(b=b, best=best, fits=fits)
    pickle.dump(res, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    return res
if __name__ == "__main__":
    init()
    with Pool(2, initializer=init) as pool:
        for r in pool.imap_unordered(one, range(B + 1)):
            print("rep", r["b"], "best start", None if r["best"] is None else r["best"]["start"], flush=True)
