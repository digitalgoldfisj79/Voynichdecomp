import sys, json, hashlib, numpy as np
sys.path.insert(0,"/home/claude/inkdirect"); import phase1 as p1, sharp
from scipy.stats import spearmanr
def run(tiles,gamma=1.0,core_dt=0.0,nw=30):
    rows=[]
    for sg in [0.8,1.0,1.3,1.6,2.0,2.5]:
        for w in [6,8,10,12]:
            for d in [0.3,0.5,0.7]:
                for g in [0.8,1.2]:
                    v=[]
                    for k in range(nw//3):
                        r=sharp.gauge(p1.synth(tiles[p1.rng.integers(len(tiles))],w,d,g,sg,gamma),core_dt=core_dt)
                        if r: v.append(r["sigma_hat"])
                    if v: rows.append((sg,w,d,g,np.median(v)))
    A=np.array(rows); o={"n":len(A)}
    o["S1_spearman"]=round(float(spearmanr(A[:,0],A[:,4]).statistic),3)
    by=[A[A[:,0]==s][:,4].mean() for s in np.unique(A[:,0])]; rng_=np.ptp(by); o["mean_by_sigma"]=[round(float(x),2) for x in by]
    o["S2_width_frac"]=round(float(np.mean([np.ptp([A[(A[:,0]==s)&(A[:,1]==w)][:,4].mean() for w in np.unique(A[:,1])]) for s in np.unique(A[:,0])])/rng_),3)
    o["S2_dark_frac"]=round(float(np.mean([np.ptp([A[(A[:,0]==s)&(A[:,2]==d)][:,4].mean() for d in np.unique(A[:,2])]) for s in np.unique(A[:,0])])/rng_),3)
    o["S3_gain_pct"]=round(float(100*abs(A[A[:,3]==1.2][:,4].mean()/A[A[:,3]==0.8][:,4].mean()-1)),2)
    o["pass"]=bool(o["S1_spearman"]>=.95 and o["S2_width_frac"]<.15 and o["S2_dark_frac"]<.15 and o["S3_gain_pct"]<5); return o
if __name__=="__main__":
    mode=sys.argv[1]; cd=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
    res={"sharp_sha256":hashlib.sha256(open("/home/claude/inkdirect/sharp.py","rb").read()).hexdigest(),"core_dt":cd}
    if mode=="dev":
        t,_=p1.blank_tiles("/home/claude/tif/f5r.sid.tif"); res["dev_f5r"]=run(t,core_dt=cd)
        t,_=p1.blank_tiles("/home/claude/tif/f1r.sid.tif"); res["dev_f1r"]=run(t,core_dt=cd)
    else:
        t,_=p1.blank_tiles("/home/claude/tif/f58r.sid.tif"); res["held_f58r"]=run(t,core_dt=cd); res["held_gamma0.9"]=run(t,gamma=0.9,core_dt=cd)
    print(json.dumps(res,indent=1)); json.dump(res,open(f"/home/claude/inkdirect/phase1b_{sharp.VERSION}_{mode}_cd{cd}.json","w"),indent=1)
