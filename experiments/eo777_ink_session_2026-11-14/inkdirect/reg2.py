"""Registration v2: thin-stroke map (paint suppressed) x line-structured template, scale constrained by the page's recorded px/mm."""
import sys, json, gzip, numpy as np
from scipy import ndimage as ndi
sys.path.insert(0,"/home/claude/inkdirect"); import reg
TIF_PXMM=23.62
_LV=None
def leaf(p):
    global _LV
    if _LV is None: _LV=json.load(gzip.open("/home/claude/ext/vms-spatial/data/manuscript/leaves.json.gz"))
    def find(o,d=0):
        if isinstance(o,dict):
            if p in o and isinstance(o[p],dict) and "mm" in o[p]: return o[p]
            for v in o.values():
                r=find(v,d+1) if d<3 else None
                if r: return r
    return find(_LV)
def thin_map(L,f=4):
    H,W=L.shape; s=L[:H//f*f,:W//f*f].reshape(H//f,f,W//f,f).mean((1,3))
    bg=ndi.median_filter(s,size=31); D=np.clip(1-s/np.maximum(bg,1),0,1)
    # white top-hat: keep dark features narrower than ~1.2 mm (7 px at 5.9 px/mm)
    T=D-ndi.grey_opening(D,size=(7,7)); return np.clip(T,0,None)
def template(bx,s,f,shape,x0,y0):
    m=np.zeros(shape,np.float32)
    b=np.round((bx*s)/f).astype(int)
    for a,t,c,d in b:
        h=max((d-t)//2,1)
        m[max(t-h-y0,0):max(d+h-y0,0),max(a-x0,0):max(c-x0,0)]-=0.5
    for a,t,c,d in b:
        m[max(t-y0,0):max(d-y0,0),max(a-x0,0):max(c-x0,0)]=1
    return m
def register(L,bx,mm,f=4,span=0.04,step=0.004):
    T=thin_map(L,f); T=T-T.mean(); s0=TIF_PXMM/mm; bx=np.asarray(bx,float); res=[]
    FT=np.fft.rfft2(T)
    for s in np.arange(s0*(1-span),s0*(1+span)+1e-9,s0*step):
        m=template(bx,s,f,T.shape,0,0)
        if (m>0).sum()==0: continue
        c=np.fft.irfft2(FT*np.conj(np.fft.rfft2(m)),T.shape)/(m>0).sum()
        iy,ix=np.unravel_index(np.argmax(c),c.shape)
        dy=iy if iy<T.shape[0]//2 else iy-T.shape[0]; dx=ix if ix<T.shape[1]//2 else ix-T.shape[1]
        # peak sharpness: peak vs the 99.9th percentile of the surface
        res.append((float(c.max()),float(s),dx*f,dy*f,float(c.max()/np.percentile(c,99.9))))
    res.sort(reverse=True); pk,s,dx,dy,sharp=res[0]
    return dict(scale=s,s0=s0,dx=dx,dy=dy,peak=pk,sharp=sharp)
def check(L,bx,r,f=4):
    """mean thin-ink inside placed boxes vs same boxes shifted by +-1 line / +-1 word width: ratio > 1 means boxes sit on text"""
    T=thin_map(L,f); bx=np.asarray(bx,float); h=np.median(bx[:,3]-bx[:,1])*r["scale"]
    def val(ox,oy):
        v=[];
        for a,t,c,d in bx:
            x0,y0,x1,y1=[int(round(q/f)) for q in (a*r["scale"]+r["dx"]+ox,t*r["scale"]+r["dy"]+oy,c*r["scale"]+r["dx"]+ox,d*r["scale"]+r["dy"]+oy)]
            if x0<0 or y0<0 or x1>T.shape[1] or y1>T.shape[0] or x1<=x0 or y1<=y0: continue
            v.append(T[y0:y1,x0:x1].mean())
        return float(np.mean(v)) if v else float("nan"), len(v)
    on,n=val(0,0); off=[val(ox,oy)[0] for ox,oy in ((0,-h*0.5),(0,h*0.5),(-h,0),(h,0),(0,-h*1.5),(0,h*1.5))]
    return dict(on=on,n_in=n,half_line=float(np.mean(off[:2])),side=float(np.mean(off[2:4])),ratio_halfline=on/np.mean(off[:2]),ratio_side=on/np.mean(off[2:4]))
if __name__=="__main__":
    B,pages=reg.bundle()
    for p in sys.argv[1:]:
        L=reg.load_L(f"/home/claude/tif/{p}.sid.tif"); bx=[g for _,_,g,_ in pages[p]]; lf=leaf(p)
        r=register(L,bx,lf["mm"]); c=check(L,bx,r)
        print(p,{k:round(v,3) for k,v in r.items()},{k:round(v,3) for k,v in c.items()})
