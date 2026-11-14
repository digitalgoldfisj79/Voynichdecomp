import sys, pickle, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/spatial"); sys.path.insert(0,"/home/claude/inkdirect"); from common import *; import reg
rng=np.random.default_rng(20261117); d=pickle.load(open("/home/claude/inkdirect/phase2.pkl","rb")); U=d["U"].copy(); res={}
B,_=reg.bundle(); t2l={}
for li,ln in enumerate(B["lines"]):
    for i in range(ln["start"],ln["end"]): t2l[i]=li
U["line"]=U.i.map(t2l); U=U[U.line.notna()&(U.n_frac>0)].copy(); U["x"]=[B["geometry"][int(i)][0] for i in U.i]
X=np.c_[np.ones(len(U)),U.logH,np.log(U.box_w/U.mm)]; y=U.N_frac.values; b=np.linalg.lstsq(X,y,rcond=None)[0]; U["N_net"]=y-X@b
U=U.sort_values(["page","line","x"]).reset_index(drop=True); M=["W_net","N_net"]
def z(obs,nl): nl=np.asarray(nl); return dict(obs=round(float(obs),5),null_mean=round(float(nl.mean()),5),null_sd=round(float(nl.std()),5),z=round(float((obs-nl.mean())/nl.std()),2))
def grp_perm(v,gid,starts):  # shuffle within groups via random keys
    k=rng.random(len(v)); o=np.lexsort((k,gid)); return v[o]
def slope_test(D,g,m,plants=(0.02,0.05),n=2000,npl=100):
    D=D.sort_values(g).reset_index(drop=True); gid=D[g].values; cnt=D.groupby(g)[g].transform("size").values
    pos=(D.groupby(g).cumcount()/(cnt-1)).values; pc=pos-D.assign(p=pos).groupby(g).p.transform("mean").values; v=D[m].values-D.groupby(g)[m].transform("mean").values
    st=lambda vv: float((vv*pc).sum()/(pc**2).sum()); nl=np.array([st(grp_perm(v,gid,None)) for _ in range(n)]); out={}
    for pz in plants:
        det=0
        for _ in range(npl): vv=grp_perm(v,gid,None)+np.log(1+pz)*pc; det+=abs((st(vv)-nl.mean())/nl.std())>=3
        out[f"power_{int(pz*100)}pct"]=det/npl
    r=z(st(v),nl); r["pct_end_to_end"]=round(100*(np.exp(r["obs"])-1),2); r.update(out); r["n"]=len(D); r["groups"]=int(D[g].nunique()); return r
# G-A
A=U[U.groupby("line").line.transform("size")>=3]
for m in M: res[f"GA_along_line_{m}"]=slope_test(A,"line",m)
# G-B
Lm=U.groupby(["page","line"])[M].mean().reset_index(); Lm=Lm[Lm.groupby("page").page.transform("size")>=5]
for m in M: res[f"GB_down_page_{m}"]=slope_test(Lm.sort_values(["page","line"]),"page",m)
# G-C
A2=A.sort_values(["line","x"]).reset_index(drop=True); gid=A2.line.values; same=np.r_[gid[1:]==gid[:-1],False]
for m in M:
    v=A2[m].values-A2.groupby("line")[m].transform("mean").values; st=lambda vv: float((vv[:-1]*vv[1:])[same[:-1]].mean()/vv.var())
    nl=[st(grp_perm(v,gid,None)) for _ in range(2000)]; res[f"GC_neighbour_words_{m}"]=z(st(v),nl)
# page table
PG=U.groupby("page")[M].mean(); PG=PG[U.groupby("page").size()>=20]
for c in ("leaf","side","quire","bifolium","partner_leaf","panel"): PG[c]=PM.loc[PG.index,c].values
PG=PG[PG.panel.isna()]; key={(r.leaf,r.side):p for p,r in PG.iterrows()}
pairs={"1_recto_verso":[],"2_facing":[],"3_conj_same_face":[],"4_conj_opp_face":[]}
for (lf,sd),p in key.items():
    if sd=="r" and (lf,"v") in key: pairs["1_recto_verso"].append((p,key[(lf,"v")]))
    if sd=="v" and (lf+1,"r") in key and PG.at[key[(lf+1,"r")],"quire"]==PG.at[p,"quire"]: pairs["2_facing"].append((p,key[(lf+1,"r")]))
    pl=PG.at[p,"partner_leaf"]
    if pd.notna(pl) and lf<pl:
        pl=int(pl)
        for s2 in ("r","v"):
            if (pl,s2) in key: pairs["3_conj_same_face" if s2!=sd else "4_conj_opp_face"].append((p,key[(pl,s2)]))
byq={q:list(g.index) for q,g in PG.groupby("quire") if len(g)>=2}
for m in M:
    for cl,pp in pairs.items():
        pp=[(a,b) for a,b in pp if PG.at[a,"quire"] in byq]; obs=np.mean([abs(PG.at[a,m]-PG.at[b,m]) for a,b in pp]); qs=[PG.at[a,"quire"] for a,b in pp]; nl=[]
        for _ in range(5000):
            dd=[]
            for q in qs: a,b=rng.choice(byq[q],2,replace=False); dd.append(abs(PG.at[a,m]-PG.at[b,m]))
            nl.append(np.mean(dd))
        r=z(obs,nl); r["z"]=-r["z"]; r["n_pairs"]=len(pp); res[f"GD_{cl}_{m}"]=r
# G-E
LF=PG.groupby("leaf")[M].mean(); LF["st"]=[SH.at[PM[PM.leaf==l].bifolium.iloc[0],"stratum"] if PM[PM.leaf==l].bifolium.iloc[0] in SH.index else "NA" for l in LF.index]
LF=LF[LF.groupby("st").st.transform("size")>=4].sort_index(); st=LF.st.values; pos=np.arange(len(LF))
for m in M:
    v=LF[m].values-LF.groupby("st")[m].transform("mean").values; f=lambda vv: float(spearmanr(pos,vv).statistic)
    def sh(vv):
        o=vv.copy()
        for g in set(st): ix=np.where(st==g)[0]; o[ix]=rng.permutation(o[ix])
        return o
    nl=np.array([f(sh(v)) for _ in range(5000)]); r=z(f(v),nl); r["n_leaves"]=len(LF); wsd=v.std()
    for amp in (1,2):
        det=0
        for _ in range(200):
            vv=sh(v)+amp*wsd*np.linspace(0,1,len(v))
            for g in set(st): vv[st==g]-=vv[st==g].mean()
            det+=abs((f(vv)-nl.mean())/nl.std())>=3
        r[f"power_{amp}SD"]=det/200
    res[f"GE_through_book_{m}"]=r
    # per-stratum trend (descriptive)
    res[f"GE_by_stratum_{m}"]={g:[int((st==g).sum()),round(float(spearmanr(pos[st==g],v[st==g]).statistic),2)] for g in sorted(set(st))}
json.dump(res,open("/home/claude/inkdirect/phase4_results.json","w"),indent=1,default=float); U.to_pickle("/home/claude/inkdirect/phase4_words.pkl")
for k,v in res.items(): print(k,v)
