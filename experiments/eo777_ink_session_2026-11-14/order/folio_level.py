"""Folio-level checks on eo777 Golf page paths. Usage: python3 -I folio_level.py <golf csv> <slim json> <outdir>"""
import sys, json, re, numpy as np, pandas as pd
from collections import Counter
from scipy.stats import spearmanr
csv, js, out = sys.argv[1:4]; rng = np.random.default_rng(20261118); NP = 10000
e = pd.read_csv(csv)
P = e.groupby("page")[["page_path_rank","page_path_2_rank","page_path_3_rank","page_path_4_rank","sheet"]].first()
paths = ["page_path_rank","page_path_2_rank","page_path_3_rank","page_path_4_rank"]
P = P.dropna(subset=paths); P = P[P.index.str.match(r"^f\d+")]  # excludes fRos (Rosettes foldout, no leaf number)
pages = list(P.index); n = len(pages)
leaf = {p: re.match(r"f(\d+)", p).group(1) for p in pages}
side = {p: ("r" if re.match(r"f\d+r", p) else "v") for p in pages}
orders = {k: list(P[k].sort_values().index) for k in paths}
def adj(o): return {frozenset(x) for x in zip(o, o[1:])}
# F1a: r/v adjacency
rv_leaves = [l for l in set(leaf.values()) if {side[p] for p in pages if leaf[p]==l} == {"r","v"}]
def rv_share(o):
    pos = {p:i for i,p in enumerate(o)}; hit = 0; tot = 0
    for l in rv_leaves:
        r = [p for p in pages if leaf[p]==l and side[p]=="r"]; v = [p for p in pages if leaf[p]==l and side[p]=="v"]
        tot += 1; hit += any(abs(pos[a]-pos[b])==1 for a in r for b in v)
    return hit/tot
null_rv = np.array([rv_share(list(rng.permutation(pages))) for _ in range(2000)])
res = {"n_pages": n, "n_leaves_rv": len(rv_leaves),
       "F1a": {k: round(rv_share(o),4) for k,o in orders.items()} | {"null_mean": round(null_rv.mean(),4), "null_sd": round(null_rv.std(),4)}}
# text-only page path
pg = json.load(open(js))["pages"]
def toks(f):
    w=[]
    for L in pg.get(f,{}).values():
        t=re.sub(r"<[^>]*>|\{[^}]*\}"," ",L.get("t",{}).get("ZLZI","") or "")
        w+=[x for x in re.split(r"[.,\s]+",t) if re.fullmatch(r"[a-z]+",x)]
    return w
def page_words(p): return sum((toks(k) for k in pg if k==p or re.fullmatch(re.escape(p)+r"\d+",k)), [])
W = {p: page_words(p) for p in pages}
def seriate(units, words):
    C={u:Counter(words[u]) for u in units}; V=sorted(set().union(*C.values())); ix={v:i for i,v in enumerate(V)}
    M=np.zeros((len(units),len(V)))
    for i,u in enumerate(units):
        for k,c in C[u].items(): M[i,ix[k]]=c
    nr=np.linalg.norm(M,axis=1,keepdims=True); nr[nr==0]=1; M/=nr; D=1-M@M.T; N=len(units)
    def plen(o): return D[o[:-1],o[1:]].sum()
    best=None
    for start in range(0,N,max(1,N//25)):
        o=[start]; left=set(range(N))-{start}
        while left: nx=min(left,key=lambda j:D[o[-1],j]); o.append(nx); left.remove(nx)
        o=np.array(o); improved=True
        while improved:
            improved=False
            for i in range(1,N-1):
                a,b=o[i-1],o[i]
                gains=D[a,b]+D[o[i:-1],o[i+1:]]-D[a,o[i:-1]]-D[b,o[i+1:]]
                j=np.argmax(gains) if len(gains) else 0
                if len(gains) and gains[j]>1e-12: o[i:i+j+1]=o[i:i+j+1][::-1]; improved=True
        if best is None or plen(best)>plen(o): best=o.copy()
    return [units[i] for i in best]
text_pages = seriate(pages, W)
g1 = adj(orders[paths[0]]); T = adj(text_pages)
tp = {p:i for i,p in enumerate(text_pages)}
res["F1b"] = {"shared_adj_path1_vs_text": len(g1&T), "of": n-1, "chance": round(2*(n-1)/n,2),
              "abs_spearman": round(abs(spearmanr([tp[p] for p in orders[paths[0]]], range(n)).statistic),3),
              "golf_paths_shared_with_path1": {k: len(adj(o)&g1) for k,o in orders.items()},
              "text_path_rv_share": round(rv_share(text_pages),4)}
# F1c: conjugate leaves gap in page path (leaf rank = mean of its pages' ranks)
pos1 = {p:i for i,p in enumerate(orders[paths[0]])}
lrank = pd.Series({l: np.mean([pos1[p] for p in pages if leaf[p]==l]) for l in set(leaf.values())})
sheet_of = {leaf[p]: P.at[p,"sheet"] for p in pages}
sheets = {}
for l,s in sheet_of.items(): sheets.setdefault(s,set()).add(l)
pairs = [tuple(sorted(v)) for v in sheets.values() if len(v)==2]
gap = np.mean([abs(lrank[a]-lrank[b]) for a,b in pairs]); L = list(lrank.index)
ng = np.array([np.mean([abs(x-y) for x,y in rng.permutation(lrank.values)[:2*len(pairs)].reshape(-1,2)]) for _ in range(NP)])
res["F1c"] = {"n_sheets": len(pairs), "mean_gap": round(gap,2), "null_mean": round(ng.mean(),2), "null_sd": round(ng.std(),2),
              "z": round((ng.mean()-gap)/ng.std(),2), "p": float(((ng<=gap).sum()+1)/(NP+1))}
# F1d: crease leaves vs partner leaves, span of leaf ranks, null = random leaves from quires A-H (f1-f66)
herb = [l for l in L if int(l) <= 66]
def span(ls): r=[lrank[l] for l in ls]; return max(r)-min(r)
for name, st in {"DAMAGED-6":["5","15","21","29","32","37"], "PARTNER-6":["4","10","20","25","28","36"]}.items():
    st=[l for l in st if l in lrank.index]; ob=span(st)
    nd=np.array([span(rng.choice(herb,len(st),replace=False)) for _ in range(NP)])
    res["F1d_"+name]={"leaf_ranks":{l:round(lrank[l],1) for l in st},"span":round(ob,1),"null_mean":round(nd.mean(),1),
                      "null_sd":round(nd.std(),1),"z":round((nd.mean()-ob)/nd.std(),2),"p":float(((nd<=ob).sum()+1)/(NP+1))}
json.dump(res, open(f"{out}/folio_level_results.json","w"), indent=1)
pd.DataFrame(dict(rank=range(1,n+1), golf_path1=orders[paths[0]], text_only=text_pages)).to_csv(f"{out}/page_orders.csv", index=False)
print(json.dumps(res, indent=1))
