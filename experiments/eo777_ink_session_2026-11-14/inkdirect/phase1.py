"""Phase 1: known-answer calibration of instrument.py on synthetic strokes over real blank parchment."""
import sys, json, hashlib, numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from scipy import ndimage as ndi
sys.path.insert(0,"/home/claude/inkdirect"); import instrument as ins
rng=np.random.default_rng(20261128)
def blank_tiles(path, n=6, size=220):
    im=np.asarray(Image.open(path).convert("RGB"),float); L=ins.lum(im); H,W=L.shape; best=[]
    for y in range(200,H-size-200,size//2):
        for x in range(200,W-size-200,size//2):
            t=L[y:y+size,x:x+size]; m=t.mean()
            if m>110: best.append((t.std(),y,x,m))
    best.sort(); out=[]; used=[]
    for s,y,x,m in best:
        if all(abs(y-yy)>size or abs(x-xx)>size for yy,xx in used): out.append(L[y:y+size,x:x+size].copy()); used.append((y,x))
        if len(out)==n: break
    return out,[(float(b[0]),float(b[3])) for b in best[:n]]
def strokes(size,w):
    """random glyph-like strokes of full width w px: polylines + arcs; returns soft mask in [0,1]"""
    big=4; S=size*big; m=np.zeros((S,S),bool); yy,xx=np.mgrid[0:S,0:S]
    for k in range(5):
        x0,y0=rng.uniform(.15,.85,2)*S; ang=rng.uniform(0,np.pi); ln=rng.uniform(.15,.35)*S
        x1,y1=x0+ln*np.cos(ang),y0+ln*np.sin(ang)
        d=np.abs((yy-y0)*(x1-x0)-(xx-x0)*(y1-y0))/ln; t=((xx-x0)*(x1-x0)+(yy-y0)*(y1-y0))/ln**2
        m|=(d<=w*big/2)&(t>=0)&(t<=1)
    for k in range(3):
        cx,cy=rng.uniform(.25,.75,2)*S; r=rng.uniform(.06,.12)*S; rr=np.hypot(xx-cx,yy-cy)
        m|=np.abs(rr-r)<=w*big/2
    return m.reshape(size,big,size,big).mean((1,3))
def synth(P,w,d,g,sigma,gamma=1.0):
    a=ndi.gaussian_filter(strokes(P.shape[0],w),sigma); L=P*(1-d*a); L=L+rng.normal(0,1.5,L.shape); L=np.clip(L*g,0,255)
    return 255*(L/255)**gamma
def run(tiles,sigma,tag,gamma=1.0):
    rows=[]
    for w in [5,6,7,8,9,10,12]:
        for d in [0.3,0.4,0.5,0.6,0.7,0.8]:
            for g in [0.8,1.0,1.2]:
                for rep in range(6):
                    P=tiles[rng.integers(len(tiles))]; r=ins.measure(synth(P,w,d,g,sigma,gamma))
                    if r and r["darkness"]==r["darkness"]: rows.append((w,d,g,r["width_px"],r["darkness"]))
    A=np.array(rows); from scipy.stats import spearmanr
    out={"n":len(A)}
    out["A1_width_spearman"]=round(float(spearmanr(A[:,0],A[:,3]).statistic),3)
    out["A2_dark_spearman"]=round(float(spearmanr(A[:,1],A[:,4]).statistic),3)
    wr=np.ptp([A[A[:,0]==w][:,3].mean() for w in np.unique(A[:,0])]); dr=np.ptp([A[A[:,1]==d][:,4].mean() for d in np.unique(A[:,1])])
    out["A1_crosstalk_frac"]=round(float(np.mean([np.ptp([A[(A[:,0]==w)&(A[:,1]==d)][:,3].mean() for d in np.unique(A[:,1])]) for w in np.unique(A[:,0])])/wr),3)
    out["A2_crosstalk_frac"]=round(float(np.mean([np.ptp([A[(A[:,1]==d)&(A[:,0]==w)][:,4].mean() for w in np.unique(A[:,0])]) for d in np.unique(A[:,1])])/dr),3)
    out["A3_gain_width_pct"]=round(float(100*abs(A[A[:,2]==1.2][:,3].mean()/A[A[:,2]==0.8][:,3].mean()-1)),2)
    out["A3_gain_dark_pct"]=round(float(100*abs(A[A[:,2]==1.2][:,4].mean()/A[A[:,2]==0.8][:,4].mean()-1)),2)
    out["width_fit"]=[round(float(x),3) for x in np.polyfit(A[:,0],A[:,3],1)]; out["dark_fit"]=[round(float(x),3) for x in np.polyfit(A[:,1],A[:,4],1)]
    out["mean_width_by_true"]={int(w):round(float(A[A[:,0]==w][:,3].mean()),2) for w in np.unique(A[:,0])}
    out["mean_dark_by_true"]={float(d):round(float(A[A[:,1]==d][:,4].mean()),3) for d in np.unique(A[:,1])}
    out["pass"]=bool(out["A1_width_spearman"]>=.95 and out["A2_dark_spearman"]>=.95 and out["A1_crosstalk_frac"]<.10 and out["A2_crosstalk_frac"]<.10 and out["A3_gain_width_pct"]<5 and out["A3_gain_dark_pct"]<5)
    return out
if __name__=="__main__":
    mode=sys.argv[1]; res={"instrument_sha256":hashlib.sha256(open("/home/claude/inkdirect/instrument.py","rb").read()).hexdigest()}
    if mode=="dev":
        tiles,info=blank_tiles("/home/claude/tif/f5r.sid.tif"); res["dev_sigma1"]=run(tiles,1.0,"dev"); res["dev_sigma1.6"]=run(tiles,1.6,"dev2")
        t1,_=blank_tiles("/home/claude/tif/f1r.sid.tif"); res["dev_f1r"]=run(t1,1.0,"dev3")
    else:
        tiles,info=blank_tiles(sys.argv[2]); res["tiles"]=info; res["heldout_third_parchment"]=run(tiles,1.0,"h1")
        t2,_=blank_tiles("/home/claude/tif/f5r.sid.tif"); res["heldout_blur2_gamma0.9"]=run(t2,2.0,"h2",0.9)
    print(json.dumps(res,indent=1)); json.dump(res,open(f"/home/claude/inkdirect/phase1_{ins.VERSION}_{mode}.json","w"),indent=1)
