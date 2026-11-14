"""K1-K6 ink coverage deep dive. Run with golfenv python from vms-spatial dir."""
import sys, json, gzip, numpy as np, pandas as pd
from scipy.stats import spearmanr, norm
sys.path.insert(0,"/home/claude/spatial"); from common import *
from vmsspatial import ink, load_bundle
rng=np.random.default_rng(20261126); res={}
I=ink.load_ink(); U=pd.DataFrame(I["units"]); U["tid"]=I["token_ids"]; U["page"]=[t.split(".")[0] for t in U.tid]
B=load_bundle("sensitivity", warp="recorded"); W=B["words"]; G=B["geometry"]
U["word"]=["".join(W[a:b+1]) for a,b in zip(U["first"],U["last"])]
def boxh(a,b):
    g=[G[i] for i in range(a,b+1) if G[i]]; return (max(x[3] for x in g)-min(x[1] for x in g)) if g else np.nan
U["Hpx"]=[boxh(a,b) for a,b in zip(U["first"],U["last"])]
LV=json.load(gzip.open(f"{VS}/data/manuscript/leaves.json.gz"))["pages"]
U["mm"]=U.page.map(lambda p:(LV.get(p) or {}).get("mm")); U["scale"]=U.page.map(lambda p:(I["pages"].get(p) or {}).get("scale")); U["img"]=U.page.map(lambda p:(I["pages"].get(p) or {}).get("image_id"))
bad=1|2|4|8|16|256|512
U=U[((U["flags"].astype(int)&bad)==0)&U.page.isin(PM.index)].copy(); U["w"]=U.word.str.replace(r"[^a-z]","",regex=True); U["nc"]=U.w.str.len()
U=U[(U.nc>0)&U.mm.notna()&U.scale.notna()&U.Hpx.notna()&(U.coverage>0)&(U.n_ink>0)].copy()
U["H"]=U.Hpx/U.mm; U["Wc"]=(U.x1-U.x0)/U.mm/U.nc; U["A"]=U.n_ink/(U.scale**2)/(U.mm**2)/U.nc; U["Xh"]=U.xheight/U.mm
U=U[(U.Wc>0)&(U.H>0)].copy()
chars=sorted(set("".join(U.w))); GX=np.array([[len(w)]+[w.count(c) for c in chars] for w in U.w],float)
def resid(y, extra):
    X=np.c_[np.ones(len(U)),GX]+0; X=np.c_[X]+0
    if extra is not None: X=np.c_[X,extra]
    b=np.linalg.lstsq(X,y,rcond=None)[0]; r=y-X@b; return r, 1-(r**2).sum()/((y-y.mean())**2).sum()
size=np.c_[np.log(U.Wc.values),np.log(U.H.values)]
U["NET"],r2a=resid(np.log(U.coverage.values),size); U["NET_A"],r2b=resid(np.log(U.A.values),size)
U["logH"]=np.log(U.H); U["logWc"]=np.log(U.Wc); U["logA"]=np.log(U.A)
res["n_words"]=len(U); res["model_r2"]={"NET":round(r2a,3),"NET_A":round(r2b,3)}
res["word_level_corr"]={"NET_vs_NET_A":round(float(np.corrcoef(U.NET,U.NET_A)[0,1]),3),"NET_vs_logXh":round(float(np.corrcoef(U.NET,np.log(U.Xh))[0,1]),3),"NET_A_vs_logXh":round(float(np.corrcoef(U.NET_A,np.log(U.Xh))[0,1]),3)}
FE=["logH","logWc","logA","NET","NET_A"]
g=U.groupby("page"); T=pd.DataFrame({"n":g.size()})
for k in FE:
    T[k]=g[k].median(); T[k+"_o"]=g[k].apply(lambda s:np.median(s.values[0::2])); T[k+"_e"]=g[k].apply(lambda s:np.median(s.values[1::2]))
T=T[T.n>=30].join(PM[["leaf","side","quire","bifolium","hand","section","currier"]]); T["hand"]=T.hand.astype(str); T["img"]=[ (I["pages"].get(p) or {}).get("image_id") for p in T.index]
T["leaf"]=T.leaf.astype(int); res["n_pages"]=len(T)
def tests(k):
    out={}; out["splithalf_r"]=round(float(np.corrcoef(T[k+"_o"],T[k+"_e"])[0,1]),3)
    out["recto_minus_verso"]=round(float(T[T.side=="r"][k].median()-T[T.side=="v"][k].median()),4)
    HH=T[(T.section=="H")&T.hand.isin(["1","2"])]; lab=HH.hand.values; x=HH[k].values
    d=x[lab=="2"].mean()-x[lab=="1"].mean(); nd=[]
    for _ in range(10000):
        p=rng.permutation(lab); nd.append(x[p=="2"].mean()-x[p=="1"].mean())
    nd=np.array(nd); out["herbal_hand2_minus_hand1"]={"diff":round(float(d),4),"null_sd":round(float(nd.std()),4),"z":round(float(d/nd.std()),2),"p_two_sided":round(float(((np.abs(nd)>=abs(d)).sum()+1)/10001),5)}
    LZ=T.groupby("leaf").agg(v=(k,"mean"),quire=("quire","first"),bifolium=("bifolium","first"),hand=("hand","first"))
    pairs=[tuple(gg.index) for _,gg in LZ.groupby("bifolium") if len(gg)==2 and gg.hand.nunique()==1]; grp={kk:gg.index.values for kk,gg in LZ.groupby(["quire","hand"])}
    ob=np.mean([abs(LZ.at[a,"v"]-LZ.at[b,"v"]) for a,b in pairs]); nd=[]
    for _ in range(5000):
        dd=[]
        for a,b in pairs:
            q=grp[(LZ.at[a,"quire"],LZ.at[a,"hand"])]
            if len(q)>2: xx,yy=rng.choice(q,2,replace=False); dd.append(abs(LZ.at[xx,"v"]-LZ.at[yy,"v"]))
        nd.append(np.mean(dd))
    out["conjugates"]={"n":len(pairs)}|summ(ob,nd,True)
    SZ=T.groupby("bifolium")[k].mean(); f=lambda a,b: abs(SZ[a]-SZ[b]) if a in SZ.index and b in SZ.index else float("nan")
    for name,o in ORDERS.items():
        o2=[s for s in o if s in SZ.index]; out["smooth_"+name]=summ(adj_mean(o2,f),[adj_mean(perm_within(o2,rng),f) for _ in range(5000)],True)
    return out
res["K1"]={k:tests(k) for k in FE}
# K2 photograph
def k2(k):
    prs=[]; 
    for im,gg in T.groupby("img"):
        if len(gg)==2: prs.append(tuple(gg.index))
    if len(prs)<8: return {"n_pairs":len(prs)}
    ob=np.mean([abs(T.at[a,k]-T.at[b,k]) for a,b in prs]); grp={kk:gg.index.values for kk,gg in T.groupby(["quire","hand"])}; nd=[]
    for _ in range(5000):
        dd=[]
        for a,b in prs:
            q=grp[(T.at[a,"quire"],T.at[a,"hand"])]
            if len(q)>2: xx,yy=rng.choice(q,2,replace=False); dd.append(abs(T.at[xx,k]-T.at[yy,k]))
        nd.append(np.mean(dd))
    return {"n_pairs":len(prs)}|summ(ob,nd,True)
res["K2_same_photo"]={k:k2(k) for k in ["NET","NET_A","logH"]}
res["K2_images"]={"pages":len(T),"distinct_images":int(T.img.nunique())}
# K3 pen physics: lag-1 within line, on page-centred NET_A / NET
U=U[U.page.isin(T.index)].copy()
for k in ["NET","NET_A"]:
    U["c"]=U[k]-U.groupby("line")[k].transform("mean"); lines=[gg.sort_values("x0")["c"].values for _,gg in U.groupby("line") if len(gg)>=5]
    def lag1(ls): a=np.concatenate([l[:-1] for l in ls]); b=np.concatenate([l[1:] for l in ls]); return float(np.corrcoef(a,b)[0,1])
    ob=lag1(lines); nd=np.array([lag1([rng.permutation(l) for l in lines]) for _ in range(1000)])
    lag2=lambda ls: float(np.corrcoef(np.concatenate([l[:-2] for l in ls]),np.concatenate([l[2:] for l in ls]))[0,1])
    res["K3_lag1_"+k]={"n_lines":len(lines),"lag2":round(lag2(lines),4)}|summ(ob,nd,False)
    # across-line: line mean centred on page, consecutive lines
    Lm=U.groupby(["page","line"])[k].mean().reset_index(); Lm["c"]=Lm[k]-Lm.groupby("page")[k].transform("mean")
    pl=[gg.sort_values("line")["c"].values for _,gg in Lm.groupby("page") if len(gg)>=6]
    ob=lag1(pl); nd=np.array([lag1([rng.permutation(l) for l in pl]) for _ in range(1000)]); res["K3_lineseq_lag1_"+k]={"n_pages":len(pl)}|summ(ob,nd,False)
U["pos"]=U.groupby("line").x0.rank(pct=True); U["c"]=U["NET_A"]-U.groupby("page")["NET_A"].transform("mean")
res["K3_position_in_line_NET_A"]=U.groupby(pd.cut(U.pos,[0,.2,.4,.6,.8,1])).c.mean().round(4).astype(float).to_dict().__str__()
U["lrank"]=U.groupby("page").line.rank(method="dense",pct=True); res["K3_down_the_page_NET_A"]=U.groupby(pd.cut(U.lrank,[0,.2,.4,.6,.8,1])).c.mean().round(4).astype(float).to_dict().__str__()
# K4 cells
for k in ["NET","NET_A","logH","logWc"]:
    cc=T.groupby(["hand","section"])[k].agg(["median","count"]); cc=cc[cc["count"]>=5]; res["K4_cells_"+k]={f"hand{h}|{s}":[round(float(m),4),int(n)] for (h,s),(m,n) in zip(cc.index,cc.values)}
# K5 gradient within strata
e=pd.read_csv(GOLF); sc=e.groupby("sheet").bifolium_path_score.first(); lf=pd.read_csv(LF).set_index("id").score
for k in ["NET","NET_A"]:
    SZ=T.groupby("bifolium")[k].mean().to_frame("v").join(SH[["stratum"]]).join(sc.rename("p1")).join(lf.rename("lf")).dropna()
    for oc in ["p1","lf"]:
        zs=[];det={}
        for st,gg in SZ.groupby("stratum"):
            if len(gg)>=6:
                r=spearmanr(gg.v,gg[oc]); det[st]=[round(float(r.statistic),3),len(gg)]; zs.append(np.arctanh(np.clip(r.statistic,-.999,.999))*np.sqrt(len(gg)-3))
        res[f"K5_{k}_{oc}"]={"strata":det,"stouffer_z_signed":round(float(np.sum(zs)/np.sqrt(len(zs))),2) if zs else None,"pooled_spearman_all_sheets":round(float(spearmanr(SZ.v,SZ[oc]).statistic),3),"n":len(SZ)}
# K6 crease six
depth=pickle.load(open(PK,"rb"))["depth"]; N6=["A:4-5","B:10-15","C:20-21","D:25-32","D:28-29","E:36-37"]
for k in ["NET","NET_A"]:
    SZ=T.groupby("bifolium")[k].mean(); Uu=[s for s in SZ.index if s in depth]; st=[s for s in N6 if s in SZ.index]
    pools={}
    for s in Uu: pools.setdefault((depth[s],SH.at[s,"currier"]),[]).append(s)
    nd=[]
    for _ in range(20000):
        pick=[]
        for gk in set((depth[s],SH.at[s,"currier"]) for s in st):
            m=sum((depth[s],SH.at[s,"currier"])==gk for s in st); pick+=list(rng.choice(pools[gk],m,replace=False))
        nd.append(SZ[pick].std())
    res["K6_N6_"+k]={"k":len(st),"values":SZ[st].round(3).to_dict()}|summ(float(SZ[st].std()),nd,True)
T.to_csv("/home/claude/spatial/page_ink_decomposed.csv"); json.dump(res,open("/home/claude/spatial/k_ink_results.json","w"),indent=1,default=str); print(json.dumps(res,indent=1,default=str))
