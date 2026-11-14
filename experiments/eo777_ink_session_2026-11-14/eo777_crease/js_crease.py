"""J1: crease sets on eo777's published JS distances. Usage: python3 -I js_crease.py <research-data dir> <profiles(research-data)/profiles.csv for labels> <null_depth.pkl> <outdir>"""
import sys, re, pickle, numpy as np, pandas as pd
rd, prof, pk, out = sys.argv[1:5]
rng = np.random.default_rng(20261119); N = 20000
depth = pickle.load(open(pk,"rb"))["depth"]; P = pd.read_csv(prof)
P["leaf"] = P.page.str.extract(r"f(\d+)")[0].astype(float)
pg = P[P.text_kind=="para"]
labB = pg.groupby("bifolium").agg(cur=("currier", lambda s:"|".join(sorted({str(v) for v in s.dropna()}))),
                                  hand=("lfd_hand", lambda s:"|".join(sorted({str(v) for v in s.dropna()}))))
U = [s for s in depth]
def lv(s): a,b = map(int, s.split(":")[1].split("-")); return a,b
rows=[]
def run(S, units, strat_of, sets, level, feat, nulls):
    pos={u:i for i,u in enumerate(units)}
    def mp(st): ii=[pos[u] for u in st]; sub=S[np.ix_(ii,ii)]; k=len(ii); return sub.sum()/(k*(k-1))
    for sname, st in sets.items():
        st=[u for u in st if u in pos]; obs=mp(st)
        for nname in nulls:
            strat={u:strat_of(u,nname) for u in units}; pools={}
            for u in units: pools.setdefault(strat[u],[]).append(u)
            nd=[]
            for _ in range(N):
                pick=[]
                for g in set(strat[u] for u in st):
                    m=sum(strat[u]==g for u in st); pick+=list(rng.choice(pools[g],m,replace=False))
                nd.append(mp(pick))
            nd=np.array(nd)
            rows.append(dict(level=level,feature=feat,set=sname,k=len(st),null=nname,obs=round(obs,5),null_mean=round(nd.mean(),5),
                null_sd=round(nd.std(),5),z=round((nd.mean()-obs)/nd.std(),2) if nd.std() else None,p=((nd<=obs).sum()+1)/(N+1)))
for feat in ["combined","word","line"]:
    D = pd.read_csv(f"{rd}/pairwise/shared-{feat}-js.csv", index_col=0)
    ids=[f"pool|{s}" for s in U]; S=D.loc[ids,ids].values.copy(); np.fill_diagonal(S,0)
    def sb(u,n): return (depth[u],) if n=="depth" else (depth[u], labB.at[u,"cur" if n=="depth x cur" else "hand"])
    run(S,U,sb,{"BOTTOM-N6":["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"],
                "BOTTOM-L9":["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37","B:9-16","D:27-30","G:52-53"],
                "FORE-F4":["E:33-40","E:34-39","G:50-55","H:57-66"]},"bifolium",feat,["depth","depth x cur","depth x hand"])
    # leaves
    info={}
    for s,d in depth.items():
        a,b=lv(s)
        for n,side in ((a,"A"),(b,"B")):
            pages=[i for i in D.index if re.fullmatch(rf"f{n}[rv]\d*\|para",i)]
            if pages:
                sub=pg[pg.leaf==n]; info[n]=(d,side,"|".join(sorted({str(v) for v in sub.currier.dropna()})),pages)
    L=sorted(info); SL=np.zeros((len(L),len(L)))
    for i,a in enumerate(L):
        for j,b in enumerate(L):
            if i!=j: SL[i,j]=D.loc[info[a][3],info[b][3]].values.mean()
    run(SL,L,lambda u,n:info[u][:3],{"DAMAGED-6":[5,15,21,29,32,37],"PARTNER-6":[4,10,20,25,28,36]},"leaf",feat,["depth x side x cur"])
R=pd.DataFrame(rows); R.to_csv(f"{out}/js_crease_results.csv",index=False); pd.set_option("display.width",220); print(R.to_string(index=False))
