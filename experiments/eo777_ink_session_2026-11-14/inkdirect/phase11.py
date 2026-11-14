import sys, json, pickle, itertools, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261204); res={}; e=pd.read_csv(GOLF); para=e[e.text_kind=="para"]
cls=para.groupby("sheet").assignment_code.agg(lambda s: s.value_counts().index[0]); clp=para.groupby("page").assignment_code.first()
d=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb"))["SETS"]
def pages(U):
    U=U[U.darkness>0].copy(); ch=sorted(set("".join(U.w))); GX=np.c_[np.ones(len(U)),[[len(w)]+[w.count(c) for c in ch] for w in U.w]].astype(float); y=np.log(U.darkness.values); U["D"]=y-GX@np.linalg.lstsq(GX,y,rcond=None)[0]
    P=U.groupby("page").agg(D=("D","mean"),W=("W_net","mean"),N=("N_net","mean"),sig=("sigma_hat","median"),B=("B","median")); P["logB"]=np.log(P.B); P["bif"]=PM.loc[P.index,"bifolium"].values; return P
P14,P04=pages(d["y2014"]),pages(d["y2004"]); S14=P14.groupby("bif")[["D","W","N","sig","logB"]].mean(); S04=P04.groupby("bif")[["D","W","N","sig","logB"]].mean()
H=[s for s in S14.index if s in SH.index and SH.at[s,"stratum"]=="H|1" and s in cls.index]; lab=cls.reindex(H).values; big=[c for c in pd.Series(lab).value_counts().index if (lab==c).sum()>=4]; res["herbal_hand1_cluster_sizes"]=pd.Series(lab).value_counts().to_dict()
def diff(v,l,a="1b",b="1a"): return float(v[l==a].mean()-v[l==b].mean())
def maxpair(v,l,cl): return max(abs(v[l==a].mean()-v[l==b].mean()) for a,b in itertools.combinations(cl,2))
v=S14.loc[H,"D"].values
# K1
nl=np.array([maxpair(v,rng.permutation(lab),big) for _ in range(20000)]); o=abs(diff(v,lab)); res["K1_familywise_herbal_hand1"]={"clusters":big,"obs_1b_minus_1a":round(diff(v,lab),4),"largest_pair_obs":round(maxpair(v,lab,big),4),"null_sd_of_max":round(float(nl.std()),4),"p_fw":round(float(((nl>=o).sum()+1)/(len(nl)+1)),5)}
allS=[s for s in S14.index if s in SH.index and s in cls.index]; st=SH.loc[allS,"stratum"].values; la=cls.reindex(allS).values; va=S14.loc[allS,"D"].values
prs=[(g,a,b) for g in set(st) for a,b in itertools.combinations([c for c in set(la[st==g]) if ((la==c)&(st==g)).sum()>=4],2)]
def sperm(l):
    o_=l.copy()
    for g in set(st): ix=np.where(st==g)[0]; o_[ix]=rng.permutation(o_[ix])
    return o_
def mx(l): return max(abs(va[(st==g)&(l==a)].mean()-va[(st==g)&(l==b)].mean()) for g,a,b in prs)
nl=np.array([mx(sperm(la)) for _ in range(10000)]); res["K1_familywise_all_pairs_manuscript"]={"n_pairs":len(prs),"pairs":[f"{g}:{a}-{b}" for g,a,b in prs],"p_fw_for_1b_1a":round(float(((nl>=o).sum()+1)/(len(nl)+1)),5)}
res["K1_power_plant_0.15"]=float(np.mean([abs(diff(v+np.where((l:=rng.permutation(lab))=="1b",0.15,0),l))>=np.quantile(np.array([maxpair(v,rng.permutation(lab),big) for _ in range(400)]),0.99) for _ in range(300)]))
# K2
def bss(l):
    t=0
    for g in set(st):
        m=st==g; mu=va[m].mean()
        for c in set(l[m]): k=m&(l==c); t+=k.sum()*(va[k].mean()-mu)**2
    return t
nl=np.array([bss(sperm(la)) for _ in range(10000)]); res["K2_omnibus_clusters_within_section_hand"]={"n_sheets":len(allS),"obs":round(bss(la),4),"null_mean":round(float(nl.mean()),4),"null_sd":round(float(nl.std()),4),"z":round(float((bss(la)-nl.mean())/nl.std()),2),"p":round(float(((nl>=bss(la)).sum()+1)/(len(nl)+1)),5)}
# K3
m2=np.isin(lab,["1a","1b"]); Hs=np.array(H)[m2]; l2=lab[m2]
def ratio(x,n=20000):
    nl=np.array([diff(x,rng.permutation(l2)) for _ in range(n)]); return dict(diff=round(diff(x,l2),4),null_sd=round(float(nl.std()),4),ratio=round(float(diff(x,l2)/nl.std()),2))
def adjv(cols):
    X=np.c_[np.ones(len(allS))]+0
    for c in cols: X=np.c_[X,S14.loc[allS,c].values]
    r=pd.Series(va-X@np.linalg.lstsq(X,va,rcond=None)[0],index=allS); return r.loc[Hs].values
x0=S14.loc[Hs,"D"].values; res["K3_raw"]=ratio(x0); res["K3_focus_adjusted_sheet_level"]=ratio(adjv(["sig"])); res["K3_parchment_adjusted"]=ratio(adjv(["logB"])); res["K3_focus_and_parchment_adjusted"]=ratio(adjv(["sig","logB"]))
Xh=np.c_[np.ones(len(Hs)),S14.loc[Hs,"sig"].values]; res["K3_focus_adjusted_within_1a1b_only"]=ratio(x0-Xh@np.linalg.lstsq(Xh,x0,rcond=None)[0])
loo=[]
for i in range(len(Hs)):
    k=np.arange(len(Hs))!=i; nl=np.array([diff(x0[k],rng.permutation(l2[k])) for _ in range(3000)]); loo.append((round(float(diff(x0[k],l2[k])/nl.std()),2),Hs[i]))
res["K3_leave_one_out"]={"min":min(loo),"max":max(loo)}; res["K3_focus_reading_1b_minus_1a"]=ratio(S14.loc[Hs,"sig"].values)
# K4
for tag,S,cols in (("2014",S14,["W","N"]),("2004",S04,["D","W","N"])):
    hs=[s for s in Hs if s in S.index]; ll=cls.reindex(hs).values
    for c in cols:
        x=S.loc[hs,c].values; nl=np.array([diff(x,rng.permutation(ll)) for _ in range(20000)]); res[f"K4_{tag}_{ {'D':'darkness','W':'stroke_width','N':'ink_amount'}[c]}"]=dict(n=len(hs),diff=round(diff(x,ll),4),null_sd=round(float(nl.std()),4),ratio=round(float(diff(x,ll)/nl.std()),2))
# K5 / K6
P=P14[P14.bif.isin(Hs)].copy(); P["cl"]=clp.reindex(P.index).values; mix=P[P.cl.isin(["1a","1b"])].pivot_table(index="bif",columns="cl",values="D").dropna(); res["K5_mixed_sheets"]={"n":len(mix)}
if len(mix)>=5:
    dd=(mix["1b"]-mix["1a"]).values; nl=np.array([(dd*rng.choice([-1,1],len(dd))).mean() for _ in range(20000)]); res["K5_mixed_sheets"].update(mean_diff=round(float(dd.mean()),4),null_sd=round(float(nl.std()),4),ratio=round(float(dd.mean()/nl.std()),2))
res["K5_page_cluster_counts_in_1a1b_sheets"]=P.cl.value_counts(dropna=False).to_dict()
Pp=P[P.cl.isin(["1a","1b"])]; shl=dict(zip(Hs,l2)); f=lambda m: float(Pp.D[Pp.bif.map(m)=="1b"].mean()-Pp.D[Pp.bif.map(m)=="1a"].mean()); nl=np.array([f(dict(zip(Hs,rng.permutation(l2)))) for _ in range(10000)])
res["K6_page_level_sheet_label_shuffle"]=dict(n_pages=len(Pp),diff=round(f(shl),4),null_sd=round(float(nl.std()),4),ratio=round(float(f(shl)/nl.std()),2))
res["sheets"]={"1a":list(Hs[l2=="1a"]),"1b":list(Hs[l2=="1b"])}; res["quires"]={k:pd.Series([SH.at[s,"quire"] for s in Hs[l2==k]]).value_counts().to_dict() for k in ("1a","1b")}
json.dump(res,open("/home/claude/inkdirect/phase11_results.json","w"),indent=1,default=str)
for k,x in res.items(): print(k,x)
