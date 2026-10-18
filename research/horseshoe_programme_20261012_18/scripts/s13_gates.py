#!/usr/bin/env python3
"""Stage 13: known-answer gates for every from-scratch method (PREREG 5). Writes results_gates.json."""
import json, os, sys, itertools
import numpy as np
from scipy.stats import f_oneway
sys.path.insert(0, "/home/claude/horseshoe/scripts")
import lib_methods as L
ROOT = sys.argv[1]; rng = np.random.default_rng(20261012); res = {}

# G1 PERMANOVA == one-way ANOVA F on Euclidean distances
g = np.repeat(np.arange(5), 12); y = rng.normal(size=60) + 0.6 * g
D = np.abs(y[:, None] - y[None, :]); G = L.gower(D)
o = L.permanova_seq(G, [("A", g)])
F_perm = (o["A"][0] / o["A"][1]) / (o["resid"][0] / o["resid"][1])
F_anova = f_oneway(*[y[g == k] for k in range(5)]).statistic
res["G1_permanova"] = dict(F_perm=F_perm, F_anova=float(F_anova), pass_=bool(abs(F_perm - F_anova) < 1e-8))

# G2 DCA removes the arch on a simulated 1-D coenocline
sites, spp = 60, 80
grad = np.linspace(0, 10, sites); opt = rng.uniform(-1, 11, spp); tol = rng.uniform(0.8, 1.5, spp)
mu = 30 * np.exp(-0.5 * ((grad[:, None] - opt[None, :]) / tol[None, :]) ** 2)
N = rng.poisson(mu) + 0.0; N = N[:, N.sum(0) > 0]; N = N[N.sum(1) > 0]
rows, _, eig = L.ca(N)
arch_ca = L.quad_r2(rows[:, 0], rows[:, 1])
sc, lam, glen = L.dca(N)
arch_dca = L.quad_r2(sc[:, 0], sc[:, 1])
res["G2_dca"] = dict(arch_ca=arch_ca, arch_dca=arch_dca, ca_eig=eig[:2].tolist(), dca_lam=lam.tolist(),
                     grad_len=glen, rho_axis1_true=float(abs(np.corrcoef(sc[:, 0], grad[:len(sc)])[0, 1])),
                     pass_=bool(arch_ca >= 0.5 and arch_dca <= 0.5 * arch_ca))

# G3 delta score: 0 on additive tree metric, high on random distances
n = 14
# random binary tree by sequential attachment; path distances
parent = {}; length = {}; nodes = [0]; nxt = 1; leaves = []
import networkx as nx
T = nx.Graph(); T.add_node(0)
for i in range(n):
    # attach leaf i to a random existing edge-midpoint (new internal node) or root
    if T.number_of_edges() == 0:
        T.add_edge(0, ("L", i), w=rng.uniform(0.1, 1)); leaves.append(("L", i)); continue
    u, v = list(T.edges())[rng.integers(T.number_of_edges())]
    w = T[u][v]["w"]; T.remove_edge(u, v); m = ("I", i)
    T.add_edge(u, m, w=w / 2); T.add_edge(m, v, w=w / 2); T.add_edge(m, ("L", i), w=rng.uniform(0.1, 1)); leaves.append(("L", i))
Dt = np.array([[nx.shortest_path_length(T, a, b, weight="w") for b in leaves] for a in leaves])
Dr = rng.uniform(size=(n, n)); Dr = (Dr + Dr.T) / 2; np.fill_diagonal(Dr, 0)
res["G3_delta"] = dict(tree=L.delta_mean(Dt), random=L.delta_mean(Dr),
                       pass_=bool(L.delta_mean(Dt) < 1e-9 and L.delta_mean(Dr) >= 0.25))

# G4 NCD
x = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz "), 2000)); z = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz "), 2000))
Dn = L.ncd_matrix([x, x, z])
res["G4_ncd"] = dict(self=float(Dn[0, 1]), random=float(Dn[0, 2]), pass_=bool(Dn[0, 1] < 0.2 and Dn[0, 2] > 0.9))

# G5 MRM recovers a planted conjugate effect
m = 80; sheet = np.repeat(np.arange(20), 4); leaf = np.repeat(np.arange(40), 2)
iu = np.triu_indices(m, 1)
conj = ((sheet[iu[0]] == sheet[iu[1]]) & (leaf[iu[0]] != leaf[iu[1]])).astype(float)
same_leaf = (leaf[iu[0]] == leaf[iu[1]]).astype(float)
nuis = rng.normal(size=len(conj))
beta_true = -0.15
ycomb = beta_true * (conj - conj.mean()) / conj.std() + 0.3 * nuis + rng.normal(size=len(conj))
ycomb = (ycomb - ycomb.mean()) / ycomb.std()
b, r2 = L.mrm_fit(ycomb, np.c_[conj, same_leaf, nuis])
res["G5_mrm"] = dict(beta_conj=float(b[0]), planted_standardised=beta_true / np.sqrt(1 + 0.09 + beta_true ** 2),
                     pass_=bool(b[0] < 0 and abs(b[0] - beta_true / np.sqrt(1.09 + beta_true ** 2)) <= 0.2 * abs(beta_true)))

# G6 variance components recover a 10% sheet share and LRT detects it
nq, sps, lps = 12, 4, 2   # quires, sheets per quire, leaves per sheet; 2 pages per leaf
quire = np.repeat(np.arange(nq), sps * lps * 2); sh = np.repeat(np.arange(nq * sps), lps * 2); lf = np.repeat(np.arange(nq * sps * lps), 2)
vq, vs, vl, ve = 0.3, 0.1, 0.1, 0.5
yv = (rng.normal(0, np.sqrt(vq), nq)[quire] + rng.normal(0, np.sqrt(vs), nq * sps)[sh]
      + rng.normal(0, np.sqrt(vl), nq * sps * lps)[lf] + rng.normal(0, np.sqrt(ve), len(quire)))
full, s_full, ll_full = L.fit_vc(yv, {"quire": quire, "sheet": sh, "leaf": lf})
red, s_red, ll_red = L.fit_vc(yv, {"quire": quire, "leaf": lf})
tot = sum(full.values()) + s_full; share = full["sheet"] / tot
lrt = 2 * (ll_full - ll_red)
from scipy.stats import chi2
p = 0.5 * chi2.sf(max(lrt, 0), 1)
res["G6_vc"] = dict(est=full, resid=s_full, sheet_share=share, true_share=0.1, lrt=lrt, p=p,
                    pass_=bool(abs(share - 0.1) <= 0.05 and p < 0.01))
json.dump(res, open(os.path.join(ROOT, "results_gates.json"), "w"), indent=1, default=float)
for k, v in res.items(): print(k, "PASS" if v["pass_"] else "FAIL", {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "pass_"})
