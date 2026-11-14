import sys, pickle, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/spatial"); from common import *
rng=np.random.default_rng(20261127); res={}
U=pickle.load(open("/home/claude/inkdirect/phase5.pkl","rb"))["SETS"]["y2014"]; PG=pd.read_csv("/home/claude/inkdirect/phase2d_pages.csv",index_col=0)
for c in ("bifolium","leaf","side","partner_leaf","section","hand","panel"): PG[c]=PM.loc[PG.index,c].values
e=pd.read_csv(GOLF); PG["prank"]=e.dropna(subset=["page_path_rank"]).groupby("page").page_path_rank.first().reindex(PG.index)
# PL-0 noise
U=U[(U.darkness>0)].copy(); U["ld"]=np.log(U.darkness); U["k"]=U.groupby("page").cumcount(); a=U[U.k%2==0].groupby("page").ld.mean(); b=U[U.k%2==1].groupby("page").ld.mean()
noise=float(((a-b)/2).reindex(PG.index).std()); w=PG.D_net-PG.groupby("bifolium").D_net.transform("mean"); wsd=float(w[PG.groupby("bifolium").D_net.transform("size")>1].std())
res["PL0"]={"noise_sd":round(noise,4),"within_sheet_sd":round(wsd,4),"ratio":round(wsd/noise,2),"usable":bool(wsd/noise>=3)}
adj=lambda v: float(np.mean(np.abs(np.diff(v))))
def wperm(v,g):
    return v[np.lexsort((rng.random(len(v)),g))]
def pl1(D,key,m="D_net"):
    D=D.dropna(subset=[key]).sort_values("bifolium",kind="stable").reset_index(drop=True); g=D.bifolium.values; order=np.argsort(D[key].values,kind="stable"); v=D[m].values
    st=lambda x: adj(x[order]); nl=np.array([st(wperm(v,g)) for _ in range(5000)]); o=st(v); r=dict(n_pages=len(D),obs=round(o,4),null_mean=round(float(nl.mean()),4),null_sd=round(float(nl.std()),4),z=round(float((nl.mean()-o)/nl.std()),2))
    rank=np.empty(len(v),int); rank[order]=np.arange(len(v))
    for f in (0.5,1.0):
        det=0
        for _ in range(200):
            x=wperm(v,g); rw=np.cumsum(rng.normal(0,1,len(v))); rw=(rw-rw.mean())/rw.std()*f*wsd; x=x+rw[rank]; n2=np.array([st(wperm(x,g)) for _ in range(200)]); det+=(n2.mean()-st(x))/n2.std()>=3
        r[f"power_{f}x"]=det/200
    return r
PG["read"]=PG.leaf*2+(PG.side=="v")
res["PL1_page_path"]=pl1(PG[PG.panel.isna()],"prank"); res["PL1_reading_order"]=pl1(PG[PG.panel.isna()],"read")
H=PG[(PG.section=="H")&(PG.hand.astype(str)=="1")&PG.panel.isna()]
res["PL1_page_path_herbal_hand1"]=pl1(H,"prank"); res["PL1_reading_order_herbal_hand1"]=pl1(H,"read")
# PL-2
D=H.dropna(subset=["prank"]).copy(); D["sm"]=D.groupby("bifolium").D_net.transform("mean"); D["off"]=D.D_net-D.sm; sheets=D.bifolium.unique(); smean=D.groupby("bifolium").sm.first()
pos=D.prank.rank().values; f=lambda x: float(spearmanr(pos,x).statistic)
def shuf():
    m=dict(zip(sheets,rng.permutation(smean.loc[sheets].values))); return D.bifolium.map(m).values+D.off.values
nl=np.array([f(shuf()) for _ in range(5000)]); o=f(D.D_net.values); r=dict(n_pages=len(D),n_sheets=len(sheets),rho=round(o,3),null_sd=round(float(nl.std()),3),z=round(float((o-nl.mean())/nl.std()),2))
for amp in (2,3): r[f"power_{amp}SD"]=float(np.mean([abs((f(shuf()+amp*D.D_net.std()*(pos-pos.min())/(pos.max()-pos.min()))-nl.mean())/nl.std())>=3 for _ in range(300)]))
res["PL2_direction_herbal_hand1_page_path"]=r
# PL-3
T=PG[PG.panel.isna()].copy(); first=T.leaf<T.partner_leaf; T["leafpos"]=np.where(first,"first","second"); T["face"]=np.where((first&(T.side=="r"))|(~first&(T.side=="v")),"outer","inner")
def paired(col,a,b):
    x=T.pivot_table(index="bifolium",columns=col,values="D_net",aggfunc="mean").dropna(); d=(x[a]-x[b]).values; nl=np.array([(d*rng.choice([-1,1],len(d))).mean() for _ in range(10000)])
    return dict(n_sheets=len(d),mean_diff_pct=round(100*(np.exp(d.mean())-1),2),null_sd=round(float(nl.std()),4),z=round(float(d.mean()/nl.std()),2))
res["PL3_recto_minus_verso"]=paired("side","r","v"); res["PL3_first_minus_second_leaf"]=paired("leafpos","first","second"); res["PL3_outer_minus_inner_face"]=paired("face","outer","inner")
json.dump(res,open("/home/claude/inkdirect/phase7_results.json","w"),indent=1,default=float); PG.to_csv("/home/claude/inkdirect/phase7_pages.csv")
for k,v in res.items(): print(k,v)
