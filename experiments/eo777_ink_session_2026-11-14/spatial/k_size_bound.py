import sys, json, gzip, numpy as np, pandas as pd
sys.path.insert(0,"/home/claude/spatial"); from common import *
from vmsspatial import ink, load_bundle
rng=np.random.default_rng(20261127)
I=ink.load_ink(); U=pd.DataFrame(I["units"]); U["page"]=[t.split(".")[0] for t in I["token_ids"]]
B=load_bundle("sensitivity", warp="recorded"); W=B["words"]
U["w"]=pd.Series(["".join(W[a:b+1]) for a,b in zip(U["first"],U["last"])]).str.replace(r"[^a-z]","",regex=True).values
LV=json.load(gzip.open(f"{VS}/data/manuscript/leaves.json.gz"))["pages"]; U["mm"]=U.page.map(lambda p:(LV.get(p) or {}).get("mm"))
bad=1|2|4|8|16|256|512; U=U[((U["flags"].astype(int)&bad)==0)&U.page.isin(PM.index)&U.mm.notna()&(U.w.str.len()>0)].copy()
U["logW"]=np.log((U.x1-U.x0)/U.mm); U=U[np.isfinite(U.logW)]
chars=sorted(set("".join(U.w))); X=np.c_[np.ones(len(U)),np.log(U.w.str.len().values),np.array([[w.count(c) for c in chars] for w in U.w],float)]
b=np.linalg.lstsq(X,U.logW.values,rcond=None)[0]; U["r"]=U.logW-X@b
res={"model_r2":round(float(1-(U.r**2).sum()/((U.logW-U.logW.mean())**2).sum()),3)}
g=U.groupby("page"); T=pd.DataFrame({"n":g.size(),"r":g.r.median()}); T=T[T.n>=30].join(PM[["hand","section","quire","bifolium"]]); T["hand"]=T.hand.astype(str)
HH=T[(T.section=="H")&T.hand.isin(["1","2"])]; lab=HH.hand.values; x=HH.r.values; d=x[lab=="2"].mean()-x[lab=="1"].mean()
nd=np.array([ (lambda p: x[p=="2"].mean()-x[p=="1"].mean())(rng.permutation(lab)) for _ in range(10000)])
res["herbal_hand2_minus_hand1_width_glyph_adjusted"]={"log_diff":round(float(d),4),"percent":round(float(100*(np.exp(d)-1)),1),"null_sd":round(float(nd.std()),4),"z":round(float(d/nd.std()),2),"p_two_sided":round(float(((np.abs(nd)>=abs(d)).sum()+1)/10001),5)}
cc=T.groupby(["hand","section"]).r.agg(["median","count"]); cc=cc[cc["count"]>=5]
res["cells_percent_vs_overall"]={f"hand{h}|{s}":[round(float(100*(np.exp(m)-1)),1),int(n)] for (h,s),(m,n) in zip(cc.index,cc.values)}
# density proxy: words per page vs width residual
T["words"]=T.n; res["corr_width_resid_vs_clean_words_on_page"]=round(float(np.corrcoef(T.r,np.log(T.words))[0,1]),3)
json.dump(res,open("k_size_bound_results.json","w"),indent=1); print(json.dumps(res,indent=1))
