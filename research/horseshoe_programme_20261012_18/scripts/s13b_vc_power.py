#!/usr/bin/env python3
"""Stage 13b (diagnostic, after G6 failed): power and bias of the variance-component LRT at a gate-sized design."""
import json, os, sys
import numpy as np
from scipy.stats import chi2
sys.path.insert(0, "/home/claude/horseshoe/scripts")
import lib_methods as L
ROOT = sys.argv[1]; rng = np.random.default_rng(777)
nq, sps, lps = 12, 4, 2
quire = np.repeat(np.arange(nq), sps * lps * 2); sh = np.repeat(np.arange(nq * sps), lps * 2); lf = np.repeat(np.arange(nq * sps * lps), 2)
out = {}
for share_true in (0.10, 0.20):
    vs = share_true; vq, vl, ve = 0.3, 0.1, 1 - 0.3 - 0.1 - vs
    shares, ps = [], []
    for r in range(30):
        y = (rng.normal(0, np.sqrt(vq), nq)[quire] + rng.normal(0, np.sqrt(vs), nq * sps)[sh]
             + rng.normal(0, np.sqrt(vl), nq * sps * lps)[lf] + rng.normal(0, np.sqrt(ve), len(quire)))
        f, sf, lf_ = L.fit_vc(y, {"quire": quire, "sheet": sh, "leaf": lf})
        rd, sr, lr_ = L.fit_vc(y, {"quire": quire, "leaf": lf})
        shares.append(f["sheet"] / (sum(f.values()) + sf)); ps.append(0.5 * chi2.sf(max(2 * (lf_ - lr_), 0), 1))
    out[f"true_{share_true}"] = dict(mean_share=float(np.mean(shares)), sd_share=float(np.std(shares)),
                                     power_p01=float(np.mean(np.array(ps) < 0.01)), n_sheets=nq * sps, n_pages=len(quire))
json.dump(out, open(os.path.join(ROOT, "results_vc_power.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
