import re, json, gzip, pickle, numpy as np, pandas as pd
VS="/home/claude/ext/vms-spatial"; GOLF="/home/claude/ext/voynich-clusters/golf-v21-assignments.csv"
LF="/home/claude/order/zl-unlabelled-bifolia/scores.csv"; PK="/home/claude/eo777_crease/out/null_depth.pkl"
CHAIN="A:1-8 D:27-30 F:42-47 G:49-56 A:2-7 D:25-32 C:20-21 E:35-38 A:4-5 D:28-29".split()
PM=pd.read_csv(f"{VS}/data/manuscript/physical_map.csv").set_index("folio")
def maj(s): 
    s=s.dropna().astype(str); return s.value_counts().index[0] if len(s) else "NA"
SH=PM.groupby("bifolium").agg(section=("section",maj),hand=("hand",maj),currier=("currier",maj),quire=("quire","first"),first_leaf=("leaf","min"))
SH["stratum"]=SH.section+"|"+SH.hand
e=pd.read_csv(GOLF); _g=e.groupby("sheet").bifolium_path_rank.first().dropna().sort_values()
ORDERS={"path1":list(_g.index),"labelfree":list(pd.read_csv(LF).set_index("id").score.sort_values().index),
        "binding":list(SH.loc[[s for s in _g.index]].sort_values("first_leaf").index)}
HERB_A1=[s for s in SH.index if SH.at[s,"quire"] in list("ABCDEFGH") and SH.at[s,"currier"]=="A" and SH.at[s,"hand"]=="1" and s in set(_g.index)]
def adj_mean(order, f):
    v=[f(a,b) for a,b in zip(order,order[1:])]; v=[x for x in v if x==x]; return float(np.mean(v))
def perm_within(order, rng):
    o=np.array(order,dtype=object); st=SH.loc[order,"stratum"].values
    for g in set(st):
        idx=np.where(st==g)[0]; o[idx]=rng.permutation(o[idx])
    return list(o)
def summ(obs, null, lower_is_signal):
    null=np.asarray(null); p=((null<=obs).sum()+1)/(len(null)+1) if lower_is_signal else ((null>=obs).sum()+1)/(len(null)+1)
    z=(null.mean()-obs)/null.std() if lower_is_signal else (obs-null.mean())/null.std()
    return dict(obs=round(float(obs),4),null_mean=round(float(null.mean()),4),null_sd=round(float(null.std()),4),z=round(float(z),2),p=round(float(p),5))
