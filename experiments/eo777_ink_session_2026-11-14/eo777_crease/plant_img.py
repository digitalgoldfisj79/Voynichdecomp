"""M3: plant-image coherence of crease sets. Usage: python3 -I plant_img.py <objs.json> <profiles.csv> <null_depth.pkl> <outdir>"""
import sys, json, re, pickle, numpy as np, pandas as pd
objs, prof, pk, out = sys.argv[1:5]
rng = np.random.default_rng(20261117); N = 20000
O = pd.DataFrame(json.load(open(objs)))
O = O[O.crop_qa.isna() & O.embed_variant.isna()].copy()
O["leaf"] = O.slug.str.extract(r"^(\d+)")[0].astype(int)
O["v"] = O.embedding.map(lambda s: (lambda a: a/np.linalg.norm(a))(np.array(json.loads(s), dtype=float)))
depth = pickle.load(open(pk,"rb"))["depth"]; P = pd.read_csv(prof)
P["leaf"] = P.page.str.extract(r"f(\d+)")[0].astype(float)
labB = P.groupby("bifolium").agg(cur=("currier", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))),
                                 hand=("lfd_hand", lambda s: "|".join(sorted({str(v) for v in s.dropna()}))))
def leaves(s): a, b = map(int, s.split(":")[1].split("-")); return a, b
rows = []
for part in ["plant", "root"]:
    Q = O[O.part == part]
    # bifolium level
    BV = {}
    for s in depth:
        vs = list(Q[Q.leaf.isin(leaves(s))].v)
        if vs: m = np.mean(vs, 0); BV[s] = m/np.linalg.norm(m)
    U = list(BV); X = np.array([BV[s] for s in U]); S = X @ X.T; pos = {s: i for i, s in enumerate(U)}
    if part == "plant": print("bifolia with plant crops:", len(U), "of", len(depth), "| missing:", sorted(set(depth)-set(U)))
    SETS = {"BOTTOM-N6": ["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"], "FORE-F4": ["E:33-40","E:34-39","G:50-55","H:57-66"]}
    def mps(st): ii=[pos[s] for s in st]; sub=S[np.ix_(ii,ii)]; k=len(ii); return (sub.sum()-k)/(k*(k-1))
    for sname, st in SETS.items():
        st = [s for s in st if s in pos]; obs = mps(st)
        for nname, key in [("depth", None), ("depth x cur", "cur"), ("depth x hand", "hand")]:
            strat = {s: (depth[s], None if key is None else labB.at[s, key]) for s in U}
            pools = {}
            for s in U: pools.setdefault(strat[s], []).append(s)
            nd = []
            for _ in range(N):
                pick = []
                for g in set(strat[s] for s in st):
                    m = sum(strat[s]==g for s in st); pick += list(rng.choice(pools[g], m, replace=False))
                nd.append(mps(pick))
            nd = np.array(nd)
            rows.append(dict(part=part, level="bifolium", set=sname, k=len(st), null=nname, obs=round(obs,4), null_mean=round(nd.mean(),4),
                             null_sd=round(nd.std(),4), z=round((obs-nd.mean())/nd.std(),2) if nd.std() else None, p=((nd>=obs).sum()+1)/(N+1)))
    # leaf level
    LV = {}
    for n, g in Q.groupby("leaf"):
        m = np.mean(list(g.v), 0); LV[n] = m/np.linalg.norm(m)
    info = {}
    for s, d in depth.items():
        a, b = leaves(s)
        for n, side in ((a,"A"),(b,"B")):
            if n in LV:
                sub = P[P.leaf==n]; info[n] = (d, side, "|".join(sorted({str(v) for v in sub.currier.dropna()})))
    L = sorted(info); XL = np.array([LV[n] for n in L]); SL = XL @ XL.T; pl = {n:i for i,n in enumerate(L)}
    for sname, st in {"DAMAGED-6":[5,15,21,29,32,37], "CONJUGATE-6":[4,10,20,25,28,36]}.items():
        st = [n for n in st if n in pl]; ii=[pl[n] for n in st]; k=len(ii)
        obs = (SL[np.ix_(ii,ii)].sum()-k)/(k*(k-1))
        pools = {}
        for n in L: pools.setdefault(info[n], []).append(n)
        nd = []
        for _ in range(N):
            pick = []
            for g in set(info[n] for n in st):
                m = sum(info[n]==g for n in st); pick += list(rng.choice(pools[g], m, replace=False))
            jj=[pl[n] for n in pick]; nd.append((SL[np.ix_(jj,jj)].sum()-k)/(k*(k-1)))
        nd = np.array(nd)
        rows.append(dict(part=part, level="leaf", set=sname, k=k, null="depth x side x cur", obs=round(obs,4), null_mean=round(nd.mean(),4),
                         null_sd=round(nd.std(),4), z=round((obs-nd.mean())/nd.std(),2), p=((nd>=obs).sum()+1)/(N+1)))
R = pd.DataFrame(rows); R.to_csv(f"{out}/plant_img_results.csv", index=False); print(R.to_string(index=False))
