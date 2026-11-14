"""S1 chain overlay, S1b whole orders, S4 layout tests. python3 -I s1_s4_pairs.py"""
import sys, json, pickle, re, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261123); res={}
P=pd.DataFrame(pickle.load(open("/home/claude/spatial/pairs_cut3.pkl","rb"))["rows"])
res["n_pairs"]=len(P); res["n_pages"]=len(set(P.a)|set(P.b))
METS={"dice_current":False,"spatial":False,"js_joined":True}   # lower_is_signal
# sheet-pair means
P["key"]=[tuple(sorted(x)) for x in zip(P.node_a,P.node_b)]
SP={m:P[P.node_a!=P.node_b].groupby("key")[m].mean().to_dict() for m in METS}
def f(m): return lambda a,b: SP[m].get(tuple(sorted((a,b))),float("nan"))
pool=[s for s in HERB_A1]
for m,low in METS.items():
    ob=adj_mean(CHAIN,f(m))
    res[f"S1_chain_{m}_nullA"]=summ(ob,[adj_mean(list(rng.permutation(CHAIN)),f(m)) for _ in range(10000)],low)
    res[f"S1_chain_{m}_nullB"]=summ(ob,[adj_mean(list(rng.choice(pool,10,replace=False)),f(m)) for _ in range(10000)],low)
    for name,o in ORDERS.items():
        res[f"S1b_{name}_{m}"]=summ(adj_mean(o,f(m)),[adj_mean(perm_within(o,rng),f(m)) for _ in range(5000)],low)
# S4a same-sheet page pairs vs same-quire same-hand other-sheet pairs
qa=P.a.map(PM.quire); qb=P.b.map(PM.quire); same_q=(qa==qb)&(P.hand_a==P.hand_b); Q=P[same_q].copy(); Q["same_sheet"]=Q.node_a==Q.node_b
Q["same_leaf"]=Q.a.map(PM.leaf)==Q.b.map(PM.leaf)
for m,low in METS.items():
    x=Q[m].values; lab=(Q.same_sheet&~Q.same_leaf).values; pool_mask=(~Q.same_leaf).values
    ob=x[lab].mean(); idx=np.where(pool_mask)[0]; k=lab.sum()
    # permutation: shuffle sheet labels of pages within quire -> approximate by resampling k pairs from same-quire non-same-leaf pool
    nd=[x[rng.choice(idx,k,replace=False)].mean() for _ in range(10000)]
    res[f"S4a_conjugate_pages_{m}"]={"k":int(k),"pool":int(len(idx))}|summ(ob,nd,low)
    lab2=Q.same_leaf.values; k2=lab2.sum(); idx2=np.arange(len(Q))
    res[f"S4a_recto_verso_{m}"]={"k":int(k2)}|summ(x[lab2].mean(),[x[rng.choice(idx2,k2,replace=False)].mean() for _ in range(10000)],low)
# S4b Mantel dice vs his combined JS (page level), herbal hand-1 Currier A pages
D=pd.read_csv("/home/claude/ext/voynich-clusters/research-data/pairwise/shared-combined-js.csv",index_col=0)
hp=[p for p in sorted(set(P.a)|set(P.b)) if PM.at[p,"bifolium"] in set(HERB_A1) and f"{p}|para" in D.index]
ix={p:i for i,p in enumerate(hp)}; n=len(hp); M=np.full((n,n),np.nan); T=np.full((n,n),np.nan); J=np.zeros((n,n))
for a,b,dc,jj in zip(P.a,P.b,P.dice_current,P.js_joined):
    if a in ix and b in ix: M[ix[a],ix[b]]=M[ix[b],ix[a]]=dc
ids=[f"{p}|para" for p in hp]; J=D.loc[ids,ids].values; iu=np.triu_indices(n,1); ok=~np.isnan(M[iu])
r=float(np.corrcoef(M[iu][ok],J[iu][ok])[0,1]); nr=[]
for _ in range(5000):
    pm=rng.permutation(n); Jp=J[np.ix_(pm,pm)]; nr.append(np.corrcoef(M[iu][ok],Jp[iu][ok])[0,1])
nr=np.array(nr); res["S4b_mantel_dice_vs_JS"]={"n_pages":n,"n_pairs":int(ok.sum()),"r":round(r,4),"null_sd":round(float(nr.std()),4),"p_two_sided":round(float(((np.abs(nr)>=abs(r)).sum()+1)/5001),5)}
# S4c crease leaves layout
depth=pickle.load(open(PK,"rb"))["depth"]; info={}
for s,d in depth.items():
    a,b=map(int,s.split(":")[1].split("-"))
    for leaf,side in ((a,"A"),(b,"B")):
        pg=[p for p in PM.index[PM.leaf==leaf] if p in ix or p in set(P.a)|set(P.b)]
        if pg: info[leaf]=(d,side,maj(PM.loc[pg,"currier"]),pg)
L=sorted(info); pp={}
for a,b,dc in zip(P.a,P.b,P.dice_current): pp[(a,b)]=pp[(b,a)]=dc
def leafsim(x,y):
    v=[pp[(p,q)] for p in info[x][3] for q in info[y][3] if (p,q) in pp]; return np.mean(v) if v else np.nan
LS={(x,y):leafsim(x,y) for x in L for y in L if x<y}
def mp(st): 
    v=[LS[(min(x,y),max(x,y))] for i,x in enumerate(st) for y in st[i+1:]]; return np.nanmean(v)
pools={}
for l in L: pools.setdefault(info[l][:3],[]).append(l)
for nm,st in {"DAMAGED-6":[5,15,21,29,32,37],"PARTNER-6":[4,10,20,25,28,36]}.items():
    st=[l for l in st if l in info]; nd=[]
    for _ in range(20000):
        pick=[]
        for g in set(info[l][:3] for l in st):
            m_=sum(info[l][:3]==g for l in st); pick+=list(rng.choice(pools[g],m_,replace=False))
        nd.append(mp(pick))
    res[f"S4c_{nm}_dice"]={"k":len(st)}|summ(mp(st),nd,False)
json.dump(res,open("/home/claude/spatial/s1_s4_results.json","w"),indent=1); print(json.dumps(res,indent=1))
