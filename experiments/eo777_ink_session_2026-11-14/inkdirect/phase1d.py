import sys, json, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/inkdirect"); import phase1 as p1, phase1c as c1, instrument as ins, sharp
MEAS=ins.measure
def run(tiles,extra=0.0,gamma=1.0,meas=None):
    meas=meas or MEAS; rows=[]; sg=[]
    for w in [5,6,7,8,9,10,12]:
        for d in [0.3,0.4,0.5,0.6,0.7,0.8]:
            for g in [0.8,1.0,1.2]:
                for rep in range(5):
                    im=c1.lowres(p1.synth(tiles[p1.rng.integers(len(tiles))],w,d,g,1.0,gamma),extra); r=meas(im); s=sharp.gauge(im)
                    if s: sg.append(s["sigma_hat"])
                    rows.append((w,d,g,r["darkness"] if r and r["darkness"]==r["darkness"] else np.nan))
    A=np.array(rows); cov=float(np.isfinite(A[:,3]).mean()); A=A[np.isfinite(A[:,3])]; o={"coverage":round(cov,3),"gauge_median":round(float(np.median(sg)),2)}
    o["coverage_by_width"]={int(w):round(float(np.mean([np.isfinite(x[3]) for x in rows if x[0]==w])),2) for w in [5,6,7,8,9,10,12]}
    o["spearman"]=round(float(spearmanr(A[:,1],A[:,3]).statistic),3); dr=np.ptp([A[A[:,1]==d][:,3].mean() for d in np.unique(A[:,1])])
    ws=[w for w in np.unique(A[:,0]) if all(((A[:,0]==w)&(A[:,1]==d)).sum()>=3 for d in np.unique(A[:,1]))]
    o["crosstalk"]=round(float(np.mean([np.ptp([A[(A[:,1]==d)&(A[:,0]==w)][:,3].mean() for w in ws]) for d in np.unique(A[:,1])])/dr),3); o["widths_used"]=[int(w) for w in ws]
    o["gain_pct"]=round(float(100*abs(A[A[:,2]==1.2][:,3].mean()/A[A[:,2]==0.8][:,3].mean()-1)),2)
    o["pass"]=bool(o["spearman"]>=.95 and o["crosstalk"]<.10 and o["gain_pct"]<5); return o
if __name__=="__main__":
    t,_=p1.blank_tiles("/home/claude/tif/f5r.sid.tif"); t2,_=p1.blank_tiles("/home/claude/tif/f58r.sid.tif")
    res={"dev_f5r":run(t),"held_f58r_blur0.5":run(t2,0.5),"held_f58r_gamma0.9":run(t2,0.0,0.9),"extra_f58r_blur1.0":run(t2,1.0)}
    json.dump(res,open("/home/claude/inkdirect/phase1d_v3.json","w"),indent=1)
    for k,v in res.items(): print(k,v)
