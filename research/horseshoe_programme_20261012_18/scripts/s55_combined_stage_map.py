#!/usr/bin/env python3
"""Combined-uncertainty refits of npcompl33t's pool50 order: line-bootstrap text + his point masks + all 8 starts, lambda .01/.05.
usage: python3 s55_combined_stage_map.py REPLICATION_ROOT OUT [B]
"""
import json, os, pickle, sys
import numpy as np
from multiprocessing import Pool
ROOT, OUT55 = sys.argv[1], sys.argv[2]; B55 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
__file__ = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "s50_line_bootstrap_order.py")
sys.argv = [sys.argv[0], ROOT, "/home/claude/horseshoe/s50_out"]
exec(open(__file__).read().split("# ==MAIN==")[0])     # init2, LINES, resample_lines, core, META, LABELS, ANCHORS
TEAM_ = os.path.join(ROOT, "solve/botanical_enrichment/team_golf")
MASKS = np.array(json.load(open(os.path.join(TEAM_, "golf_published_stability_20261016/ensemble_pool50_point_l010.json")))["design"]["masks"], bool)
def job(task):
    b, lam = task
    p = os.path.join(OUT55, "stages", f"rep_{b:03d}_l{int(lam*1000):03d}.pkl")
    if os.path.exists(p): return b, lam
    rng = np.random.default_rng(20261021 + b)            # same text draw for both lambdas
    counts = resample_lines(rng); d = core.distance_from_counts(counts); x, _ = core.COMMON.embed(d)
    mask = MASKS[(b - 1) % len(MASKS)]; w = np.ones(int(mask.sum())) / mask.sum()
    xm, Lm = x[mask], LABELS[mask]
    centers = {g: xm[Lm == g].mean(0) if np.any(Lm == g) else x[LABELS == g].mean(0) for g in core.ORDER}
    fits = []
    for k, anchor in enumerate(ANCHORS):
        init_nodes = np.array([centers[g] for g in anchor])
        grown = core.orient_curve(core.grow_curve(xm, w, init_nodes, lam=lam), x, META)
        model = core.refine_curve(xm, w, grown["nodes"], grown["edges"], lam=lam)
        t = np.asarray(core.project_curve(x, model)["positions"])
        med = lambda g: float(np.median(t[LABELS == g]))
        fits.append(dict(start=k, objective=float(model["objective"]), positions=t, early_ok=med("1a") < med("1b") - 1e-7))
    pickle.dump(dict(b=b, lam=lam, mask=mask, fits=fits), open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    return b, lam
if __name__ == "__main__":
    os.makedirs(os.path.join(OUT55, "stages"), exist_ok=True); init2()
    tasks = [(b, lam) for b in range(1, B55 + 1) for lam in (0.01, 0.05)]
    with Pool(2, initializer=init2) as pool:
        for b, lam in pool.imap_unordered(job, tasks): print("done", b, lam, flush=True)
