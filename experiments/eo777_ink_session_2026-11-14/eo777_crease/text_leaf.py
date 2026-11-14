"""M2: leaf-level word-cosine test. Usage: python3 -I text_leaf.py <slim.json> <profiles.csv> <outdir>"""
import sys, json, re, pickle, numpy as np, pandas as pd
from collections import Counter
js, prof, out = sys.argv[1:4]
rng = np.random.default_rng(20261116); N = 20000
depth_b = pickle.load(open(f"{out}/null_depth.pkl","rb"))["depth"]
pages = json.load(open(js))["pages"]; P = pd.read_csv(prof)
P["leaf"] = P.page.str.extract(r"f(\d+)")[0].astype(float)
def toks(f):
    w = []
    for L in pages.get(f, {}).values():
        t = re.sub(r"<[^>]*>|\{[^}]*\}", " ", L.get("t", {}).get("ZLZI", "") or "")
        w += [x for x in re.split(r"[.,\s]+", t) if re.fullmatch(r"[a-z]+", x)]
    return w
info = {}
for s, dep in depth_b.items():
    a, b = map(int, s.split(":")[1].split("-"))
    for n, side in ((a, "A"), (b, "B")):
        pg = [p for p in pages if re.fullmatch(rf"f{n}[rv]\d*", p)]
        w = sum((toks(p) for p in pg), [])
        if len(w) < 30: continue
        sub = P[P.leaf == n]
        info[n] = dict(sheet=s, depth=dep, side=side, words=w,
                       cur="|".join(sorted({str(v) for v in sub.currier.dropna()})),
                       hand="|".join(sorted({str(v) for v in sub.lfd_hand.dropna()})))
L = sorted(info); pos = {n: i for i, n in enumerate(L)}
print("leaves", len(L), "missing/short:", sorted(set(range(1,67)) - set(L)))
def sim(unit):
    C = {n: Counter(info[n]["words"] if unit == "word" else [x[i:i+2] for x in info[n]["words"] for i in range(len(x)-1)]) for n in L}
    V = sorted(set().union(*C.values())); ix = {v: i for i, v in enumerate(V)}
    M = np.zeros((len(L), len(V)))
    for i, n in enumerate(L):
        for k, c in C[n].items(): M[i, ix[k]] = c
    M /= np.linalg.norm(M, axis=1, keepdims=True); return M @ M.T
SETS = {"DAMAGED-6": [5,15,21,29,32,37], "CONJUGATE-6": [4,10,20,25,28,36],
        "DAMAGED-5": [5,15,21,29,37], "LISTED-9": [5,9,15,21,29,30,32,37,52]}
def mps(S, st): ii = [pos[n] for n in st]; sub = S[np.ix_(ii, ii)]; k = len(ii); return (sub.sum()-k)/(k*(k-1))
rows = []
for unit in ["word", "bigram"]:
    S = sim(unit)
    for sname, st in SETS.items():
        st = [n for n in st if n in pos]; obs = mps(S, st)
        for nname, key in [("depth x side x cur", "cur"), ("depth x side x hand", "hand")]:
            strat = {n: (info[n]["depth"], info[n]["side"], info[n][key]) for n in L}
            pools = {}
            for n in L: pools.setdefault(strat[n], []).append(n)
            nd = []
            for _ in range(N):
                pick = []
                for g in set(strat[n] for n in st):
                    m = sum(strat[n] == g for n in st); pick += list(rng.choice(pools[g], m, replace=False))
                nd.append(mps(S, pick))
            nd = np.array(nd)
            rows.append(dict(unit=unit, set=sname, k=len(st), null=nname, obs=round(obs,4), null_mean=round(nd.mean(),4),
                             null_sd=round(nd.std(),4), z=round((obs-nd.mean())/nd.std(),2), p=((nd >= obs).sum()+1)/(N+1),
                             min_pool=min(len(pools[strat[n]]) for n in st)))
R = pd.DataFrame(rows); R.to_csv(f"{out}/text_leaf_results.csv", index=False); print(R.to_string(index=False))
