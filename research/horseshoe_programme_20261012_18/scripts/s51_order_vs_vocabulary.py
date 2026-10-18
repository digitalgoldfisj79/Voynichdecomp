#!/usr/bin/env python3
"""PREREG 35: does npcompl33t's glyph-transition order predict shared word vocabulary beyond section x scribe (and beyond the transition distance)?
usage: python3 s51_order_vs_vocabulary.py REPLICATION_ROOT VOYNICHDECOMP OUT
"""
import csv, json, os, pickle, sys
from collections import Counter
import numpy as np
from scipy.stats import rankdata
ROOT, REPO, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True)
TEAM = os.path.join(ROOT, "solve/botanical_enrichment/team_golf"); DATA = os.path.join(ROOT, "apps/clustermatrix/research-data")
SEED, NPERM = 20261019, 10000
dj = json.load(open(os.path.join(TEAM, "golf_rf1b_stability_20261016/individuals/data.json")))
s = [s for s in dj["settings"] if s["rep"] == "ZL" and s["cohort"] == "pool50" and s["penalty"] == 0.01][0]
ids = [m["id"] for m in s["meta"]]; pos = np.array([q[1] for q in s["scopes"]["best"]["quantiles"]])
P = {r["profile_id"]: r for r in csv.DictReader(open(os.path.join(DATA, "profiles.csv")))}
pool_rows = {r["shared_mds_id"]: r for r in P.values() if r["profile_id"].startswith("pool|") and r["shared_mds_id"]}
strata, members = [], []
for i in ids:
    r = pool_rows[i]; mem = r["constituent_units"].split(";"); members.append([P[m]["page"] for m in mem])
    h = Counter(P[m]["lfd_hand"] for m in mem).most_common(1)[0][0]; strata.append(r["section"] + "|" + h)
strata = np.array(strata); page2b = {p: k for k, ps in enumerate(members) for p in ps}
# transition distance (his combined JS between pools)
tab = {r[0]: r[1:] for r in csv.reader(open(os.path.join(DATA, "pairwise/shared-combined-js.csv")))}
hdr = tab.pop(next(k for k in tab if not k.startswith(("f", "pool"))) if False else "") if "" in tab else None
with open(os.path.join(DATA, "pairwise/shared-combined-js.csv")) as f:
    rr = list(csv.reader(f))
cols = rr[0][1:]; rowsd = {r[0]: r[1:] for r in rr[1:]}
pid = ["pool|" + i if "pool|" + i in rowsd else pool_rows[i]["profile_id"] for i in ids]
TD = np.array([[float(rowsd[a][cols.index(b)]) for b in pid] for a in pid])
def tokens_zl():
    p = os.path.join(OUT, "tokens_zl.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    sys.path.insert(0, ROOT)
    from voynich_toolkit.source_lattice import SourceLattice
    doc = SourceLattice(os.path.join(ROOT, "ZL_ivtff_raw.txt"), verify=False); toks = [Counter() for _ in ids]
    for line in doc.stream("all_spaces", "preferred_first"):
        if line.locus.category != "para" or line.locus.page not in page2b: continue
        for t in line.tokens:
            if not t.has_unreadable and "?" not in t.text: toks[page2b[line.locus.page]][t.text] += 1
    pickle.dump(toks, open(p, "wb")); return toks
def tokens_tt():
    sys.argv = [sys.argv[0], REPO, OUT]
    g = {}; exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s41_prefix.py")).read().split("perm = list(")[0], g)
    pg = g["load_pages"]("TTLI"); toks = [Counter() for _ in ids]; miss = set()
    for f, ls in pg.items():
        key = f if f in page2b else None
        if key is None: miss.add(f); continue
        for l in ls: toks[page2b[key]].update(l)
    return toks
def similarity(toks):
    tot = Counter(); [tot.update(c) for c in toks]; V = sorted(w for w, n in tot.items() if n >= 3); ix = {w: k for k, w in enumerate(V)}
    M = np.zeros((len(toks), len(V)))
    for i, c in enumerate(toks):
        for w, n in c.items():
            if w in ix: M[i, ix[w]] = n
    M /= np.linalg.norm(M, axis=1, keepdims=True); return M @ M.T, np.array([sum(c.values()) for c in toks])
def analyse(S, size, label, with_td):
    iu = np.triu_indices(len(ids), 1); same = strata[iu[0]] == strata[iu[1]]
    a, b = iu[0][same], iu[1][same]; y = S[a, b]
    X = [np.ones(len(a)), np.log(size[a]) + np.log(size[b])] + ([TD[a, b]] if with_td else [])
    X = np.column_stack(X); res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]; rr = rankdata(res)
    def stat(p_):
        d = rankdata(np.abs(p_[a] - p_[b])); return float(np.corrcoef(d, rr)[0, 1])
    obs = stat(pos); rng = np.random.default_rng(SEED); null = []
    for _ in range(NPERM):
        q = pos.copy()
        for g in np.unique(strata):
            k = np.flatnonzero(strata == g)
            if len(k) > 1: q[k] = rng.permutation(q[k])
        null.append(stat(q))
    null = np.array(null); z = (obs - null.mean()) / null.std(ddof=1); p = float(np.mean(null <= obs))
    out = dict(label=label, with_transition_distance=with_td, n_pairs=int(len(a)), rho=obs, null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)), z=float(z), p_one_sided=p)
    # per stratum (descriptive)
    out["by_stratum"] = {}
    for g in np.unique(strata):
        m = (strata[a] == g)
        if m.sum() >= 3: out["by_stratum"][g] = dict(pairs=int(m.sum()), rho=float(np.corrcoef(rankdata(np.abs(pos[a[m]] - pos[b[m]])), rankdata(res[m]))[0, 1]))
    print(label, "TD" if with_td else "  ", "pairs %d rho %.3f null %.3f sd %.3f z %.2f p %.4f" % (len(a), obs, null.mean(), null.std(ddof=1), z, p), {k: round(v["rho"], 2) for k, v in out["by_stratum"].items()}, flush=True)
    return out
R = {}
for label, fn in (("ZL", tokens_zl), ("TTLI", tokens_tt)):
    toks = fn(); S, size = similarity(toks)
    print(label, "tokens", int(size.sum()), "min bifolium", int(size.min()), flush=True)
    R[label] = [analyse(S, size, label, False), analyse(S, size, label, True)]
prim = R["ZL"][0]; rep = R["TTLI"][0]
v = "CONFIRMS" if all(x["z"] <= -2 and x["p_one_sided"] < .05 for x in (prim, rep)) else ("FALSIFIES" if abs(prim["z"]) < 1 else "UNRESOLVED")
sec = "CONFIRMS" if all(x[1]["z"] <= -2 and x[1]["p_one_sided"] < .05 for x in R.values()) else ("FALSIFIES" if abs(R["ZL"][1]["z"]) < 1 else "UNRESOLVED")
R["verdict_primary"], R["verdict_secondary"] = v, sec
json.dump(R, open(os.path.join(OUT, "results_order_vocabulary.json"), "w"), indent=1)
print("PRIMARY", v, "| SECONDARY (beyond transition distance)", sec)
