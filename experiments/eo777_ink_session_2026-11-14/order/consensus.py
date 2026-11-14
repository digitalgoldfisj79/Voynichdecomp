"""Descriptive consensus over eo777 golf-v21 bifolium paths + independent text seriation (word cosine, shortest path).
Usage: python3 -I consensus.py <golf-v21-assignments.csv> <voynich_transcriptions_slim.json> <outdir>"""
import sys, json, re, itertools, numpy as np, pandas as pd
from collections import Counter
csv, js, out = sys.argv[1:4]
e = pd.read_csv(csv)
paths = ["bifolium_path_rank","bifolium_path_2_rank","bifolium_path_3_rank","bifolium_path_4_rank"]
S = e.groupby("sheet")[paths].first().dropna()
sheets = list(S.index); n = len(sheets)
orders = {p: list(S[p].sort_values().index) for p in paths}
def adj(order): return {frozenset(x) for x in zip(order, order[1:])}
A = Counter(); [A.update(adj(o)) for o in orders.values()]
# text seriation: word cosine on whole bifolia, greedy nearest-neighbour + 2-opt on path length
pages = json.load(open(js))["pages"]
def toks(f):
    w=[]
    for L in pages.get(f,{}).values():
        t=re.sub(r"<[^>]*>|\{[^}]*\}"," ",L.get("t",{}).get("ZLZI","") or "")
        w+=[x for x in re.split(r"[.,\s]+",t) if re.fullmatch(r"[a-z]+",x)]
    return w
def sheet_pages(s):
    pg = e[e.sheet==s].page.unique().tolist(); return pg
C = {s: Counter(sum((toks(p) for p in sheet_pages(s)), [])) for s in sheets}
V = sorted(set().union(*C.values())); ix={v:i for i,v in enumerate(V)}
M = np.zeros((n,len(V)))
for i,s in enumerate(sheets):
    for k,c in C[s].items(): M[i,ix[k]]=c
M/=np.linalg.norm(M,axis=1,keepdims=True); D = 1-M@M.T
def plen(o): return sum(D[a,b] for a,b in zip(o,o[1:]))
best=None
for start in range(n):
    o=[start]; left=set(range(n))-{start}
    while left: nx=min(left,key=lambda j:D[o[-1],j]); o.append(nx); left.remove(nx)
    improved=True
    while improved:
        improved=False
        for i in range(1,n-1):
            for j in range(i+1,n):
                c=o[:i]+o[i:j+1][::-1]+o[j+1:]
                if plen(c)<plen(o)-1e-12: o=c; improved=True
    if best is None or plen(o)<plen(best): best=o
text_order=[sheets[i] for i in best]; T=adj(text_order)
golf_mean = S[paths].mean(1).sort_values()
cons = list(golf_mean.index)
rows=[]
for a,b in sorted(A, key=lambda x:-A[x]):
    a_,b_=sorted(a for a in (a,b)) if False else sorted([a,b])
    rows.append(dict(pair=f"{a_} | {b_}", golf_paths=A[frozenset([a_,b_])], text_path=frozenset([a_,b_]) in T,
                     cosine=round(1-D[sheets.index(a_),sheets.index(b_)],3)))
P=pd.DataFrame(rows).sort_values(["golf_paths","text_path"],ascending=False)
P.to_csv(f"{out}/adjacency_support.csv",index=False)
pd.DataFrame(dict(consensus_rank=range(1,n+1),sheet=cons,mean_golf_rank=golf_mean.round(2).values)).to_csv(f"{out}/consensus_order.csv",index=False)
pd.DataFrame(dict(rank=range(1,n+1),sheet=text_order)).to_csv(f"{out}/text_seriation_order.csv",index=False)
# agreement stats
g1=adj(orders[paths[0]])
pair_agree={f"{p} vs path1": len(adj(orders[p])&g1) for p in paths[1:]}
all4=[k for k,v in A.items() if v==4]
print("sheets",n,"| adjacencies per path",n-1)
print("overlap with path1:",pair_agree)
print("adjacent in all 4 golf paths:",len(all4),"| of those also in text path:",len([k for k in all4 if k in T]))
print("golf path1 adjacencies also in text path:",len(g1&T),"of",n-1,"| chance ~",round((n-1)*2/n,1))
from scipy.stats import spearmanr
tr={s:i for i,s in enumerate(text_order)}
rho=spearmanr([tr[s] for s in cons],range(n)).statistic
print("Spearman text order vs golf consensus:",round(rho,3))
print(P[P.golf_paths>=3].to_string(index=False))
