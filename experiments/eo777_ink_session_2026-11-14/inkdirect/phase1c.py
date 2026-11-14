import sys, json, numpy as np
from scipy import ndimage as ndi
from scipy.stats import spearmanr
sys.path.insert(0,"/home/claude/inkdirect"); import phase1 as p1, instrument as ins
def lowres(L,extra=0.0):
    if extra: L=ndi.gaussian_filter(L,extra)
    s=ndi.zoom(ndi.gaussian_filter(L,0.6),2/3,order=1); return ndi.zoom(s,1.5,order=3)
def run(tiles,extra=0.0):
    rows=[]
    for w in [5,6,7,8,9,10,12]:
        for d in [0.3,0.5,0.7]:
            for g in [0.8,1.0,1.2]:
                for rep in range(8):
                    r=ins.measure(lowres(p1.synth(tiles[p1.rng.integers(len(tiles))],w,d,g,1.0),extra))
                    if r: rows.append((w,d,g,r["width_px"]))
    A=np.array(rows); o={"n":len(A),"spearman":round(float(spearmanr(A[:,0],A[:,3]).statistic),3)}
    wr=np.ptp([A[A[:,0]==w][:,3].mean() for w in np.unique(A[:,0])])
    o["crosstalk"]=round(float(np.mean([np.ptp([A[(A[:,0]==w)&(A[:,1]==d)][:,3].mean() for d in np.unique(A[:,1])]) for w in np.unique(A[:,0])])/wr),3)
    o["gain_pct"]=round(float(100*abs(A[A[:,2]==1.2][:,3].mean()/A[A[:,2]==0.8][:,3].mean()-1)),2); o["fit"]=[round(float(x),3) for x in np.polyfit(A[:,0],A[:,3],1)]
    o["mean_by_true"]={int(w):round(float(A[A[:,0]==w][:,3].mean()),2) for w in np.unique(A[:,0])}
    o["pass"]=bool(o["spearman"]>=.95 and o["crosstalk"]<.10 and o["gain_pct"]<5); return o
if __name__=="__main__":
  t,_=p1.blank_tiles("/home/claude/tif/f5r.sid.tif"); t2,_=p1.blank_tiles("/home/claude/tif/f58r.sid.tif")
  res={"dev_f5r":run(t),"held_f58r_blur1":run(t2,1.0)}; json.dump(res,open("/home/claude/inkdirect/phase1c.json","w"),indent=1); print(json.dumps(res))
