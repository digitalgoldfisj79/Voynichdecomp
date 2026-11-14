"""Place vmsspatial recorded word boxes on a SID TIFF: uniform scale + offset, found by maximising ink inside boxes."""
import sys, numpy as np
from PIL import Image; Image.MAX_IMAGE_PIXELS=None
from scipy import ndimage as ndi
sys.path.insert(0,"/home/claude/inkdirect"); import instrument as ins
_B=None
def bundle():
    global _B
    if _B is None:
        from vmsspatial import load_bundle
        B=load_bundle("sensitivity", warp="recorded"); pages={}
        for ln in B["lines"]:
            for i in range(ln["start"],ln["end"]):
                g=B["geometry"][i]
                if g: pages.setdefault(ln["page"],[]).append((i,B["words"][i],g,ln.get("locus")))
        _B=(B,pages)
    return _B
def load_L(path):
    return ins.lum(np.asarray(Image.open(path).convert("RGB"),float))
def register(L, boxes, s0=None):
    """returns (scale, dx, dy, score, score_ratio). Coarse search on a 1/8 darkness map."""
    f=8; H,W=L.shape; small=L[:H//f*f,:W//f*f].reshape(H//f,f,W//f,f).mean((1,3))
    bg=ndi.median_filter(small,size=25); D=np.clip(1-small/np.maximum(bg,1),0,1); D=ndi.gaussian_filter(D,1)
    bx=np.array(boxes,float); best=None; res=[]
    xmax=bx[:,2].max(); 
    for s in np.arange(3.2,4.3,0.02):
        m=np.zeros_like(D); b=(bx*s/f).astype(int)
        for x0,y0,x1,y1 in b:
            if 0<=y0<D.shape[0] and 0<=x0<D.shape[1]: m[max(y0,0):y1,max(x0,0):x1]=1
        if m.sum()==0: continue
        # offset by FFT cross-correlation
        c=np.fft.irfft2(np.fft.rfft2(D-D.mean())*np.conj(np.fft.rfft2(m-m.mean())),D.shape)
        iy,ix=np.unravel_index(np.argmax(c),c.shape); dy=iy if iy<D.shape[0]//2 else iy-D.shape[0]; dx=ix if ix<D.shape[1]//2 else ix-D.shape[1]
        res.append((c.max()/m.sum(),s,dx*f,dy*f))
    res.sort(reverse=True); sc,s,dx,dy=res[0]
    others=[r[0] for r in res if abs(r[1]-s)>0.15]
    return dict(scale=float(s),dx=float(dx),dy=float(dy),score=float(sc),ratio=float(sc/max(others)) if others else None)
def inbox_fraction(L, boxes, s, dx, dy, pad=0):
    """share of strong-ink pixels (page level) that fall inside the placed boxes, vs the boxes' area share"""
    f=4; H,W=L.shape; small=L[:H//f*f,:W//f*f].reshape(H//f,f,W//f,f).mean((1,3)); bg=ndi.median_filter(small,size=41); D=1-small/np.maximum(bg,1)
    m=np.zeros(D.shape,bool)
    for x0,y0,x1,y1 in boxes:
        a,b,c,d=[int(round(v)) for v in ((x0*s+dx)/f,(y0*s+dy)/f,(x1*s+dx)/f,(y1*s+dy)/f)]; m[max(b,0):d,max(a,0):c]=True
    ink=D>0.25; return float((ink&m).sum()/max(ink.sum(),1)), float(m.mean())
if __name__=="__main__":
    B,pages=bundle()
    for p in sys.argv[1:]:
        L=load_L(f"/home/claude/tif/{p}.sid.tif"); bx=[g for _,_,g,_ in pages[p]]; r=register(L,bx); fr=inbox_fraction(L,bx,r["scale"],r["dx"],r["dy"])
        print(p,L.shape,len(bx),r,"ink_in_boxes,box_area_share",fr)
