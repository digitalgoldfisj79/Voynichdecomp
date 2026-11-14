"""M1: maths-only cross-quire test. Are BOTTOM-N6 bifolia textually a group beyond depth/Currier/hand?
Usage: python3 -I text_cross_quire.py <voynich_transcriptions_slim.json> <profiles.csv> <outdir>"""
import sys, json, re, pickle, numpy as np, pandas as pd
from collections import Counter
js, prof, out = sys.argv[1:4]
rng = np.random.default_rng(20261115); N = 20000
d = pickle.load(open(f"{out}/null_depth.pkl","rb")); depth = d["depth"]; U = list(depth)
pages = json.load(open(js))["pages"]
def toks(folio):
    w = []
    for ln, L in pages.get(folio, {}).items():
        t = re.sub(r"<[^>]*>|\{[^}]*\}", " ", L.get("t", {}).get("ZLZI", "") or "")
        w += [x for x in re.split(r"[.,\s]+", t) if re.fullmatch(r"[a-z]+", x)]
    return w
def leaf_pages(n): return [p for p in pages if re.fullmatch(rf"f{n}[rv]\d*", p)]
B = {}
for s in U:
    a, b = map(int, s.split(":")[1].split("-"))
    B[s] = sum((toks(p) for n in (a, b) for p in leaf_pages(n)), [])
print("words per bifolium min/median:", min(map(len, B.values())), int(np.median(list(map(len, B.values())))))
def vecs(unit):
    C = {s: Counter(w if unit == "word" else [x[i:i+2] for x in w for i in range(len(x)-1)]) for s, w in B.items()}
    V = sorted(set().union(*C.values())); ix = {v: i for i, v in enumerate(V)}
    M = np.zeros((len(U), len(V)))
    for i, s in enumerate(U):
        for k, c in C[s].items(): M[i, ix[k]] = c
    M /= np.linalg.norm(M, axis=1, keepdims=True); return M @ M.T
P = pd.read_csv(prof)
lab = P.groupby("bifolium").agg(cur=("currier", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
                                hand=("lfd_hand", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))))
pos = {s: i for i, s in enumerate(U)}
SETS = {"BOTTOM-N6": ["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"],
        "BOTTOM-L9": ["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37","B:9-16","D:27-30","G:52-53"]}
def mps(S, st): ii = [pos[s] for s in st]; sub = S[np.ix_(ii, ii)]; k = len(ii); return (sub.sum()-k)/(k*(k-1))
rows = []
for unit in ["word", "bigram"]:
    S = vecs(unit)
    for sname, st in SETS.items():
        obs = mps(S, st)
        for nname, key in [("D", None), ("DL", "cur"), ("DH", "hand")]:
            strat = {s: (depth[s], None if key is None else lab.at[s, key]) for s in U}
            pools = {}
            for s in U: pools.setdefault(strat[s], []).append(s)
            nd = []
            for _ in range(N):
                pick = []
                for g in set(strat[s] for s in st):
                    m = sum(strat[s] == g for s in st); pick += list(rng.choice(pools[g], m, replace=False))
                nd.append(mps(S, pick))
            nd = np.array(nd)
            rows.append(dict(unit=unit, set=sname, null=nname, obs=round(obs,4), null_mean=round(nd.mean(),4),
                             null_sd=round(nd.std(),4), z=round((obs-nd.mean())/nd.std(),2),
                             p=((nd >= obs).sum()+1)/(N+1)))
R = pd.DataFrame(rows); R.to_csv(f"{out}/text_cross_quire_results.csv", index=False); print(R.to_string(index=False))
