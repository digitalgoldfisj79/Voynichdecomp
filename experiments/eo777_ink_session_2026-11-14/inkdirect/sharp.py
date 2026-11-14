"""Edge-softness gauge. sigma_hat = local peak darkness / (sqrt(2*pi) * steepest edge gradient)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
VERSION="s3"
def gauge(L, peak_pct=98.0, half=0.5, min_ink=40, core_dt=0.0, min_core=8, smooth=1.0, mf=3):
    L=L.astype(float); B=np.median(L[L>=np.median(L)])
    if B<=0: return None
    D=1.0-L/B; peak=np.percentile(D,peak_pct)
    if peak<=0.05: return None
    mask=ndi.binary_opening(D>half*peak,iterations=1)
    if mask.sum()<min_ink: return None
    dt=ndi.distance_transform_edt(mask); sk=skeletonize(mask); core=sk&(dt>=core_dt)
    if core.sum()<min_core: return None
    Ds=ndi.gaussian_filter(D,smooth); gy,gx=np.gradient(Ds); g=np.hypot(gx,gy)
    gmax=ndi.maximum_filter(g,size=mf); edge=mask&~ndi.binary_erosion(mask)
    # nearest core pixel for each edge pixel -> local peak
    idx=ndi.distance_transform_edt(~core,return_distances=False,return_indices=True)
    ey,ex=np.nonzero(edge); cy,cx=idx[0][ey,ex],idx[1][ey,ex]
    near=np.hypot(ey-cy,ex-cx)<=dt[cy,cx]+1.5
    if near.sum()<20: return None
    R=float(np.median(Ds[cy,cx][near]/np.maximum(gmax[ey,ex][near],1e-6)))
    w=float(np.median(2*dt[cy,cx][near]-1))
    # blurred bar of width w: peak/steepest gradient = st*sqrt(2pi)*erf(w/(2*sqrt2*st))/(1-exp(-w^2/(2 st^2))); solve for total blur st
    from scipy.special import erf
    from scipy.optimize import brentq
    f=lambda st: st*2.5066*erf(w/(2.8284*st))/(1-np.exp(-w*w/(2*st*st)))-R
    try: st=brentq(f,0.2,8.0)
    except ValueError: return None
    return dict(sigma_hat=float(np.sqrt(max(st*st-smooth*smooth,0.0))),sigma_total=float(st),R=R,w=w,n_edge=int(near.sum()))
