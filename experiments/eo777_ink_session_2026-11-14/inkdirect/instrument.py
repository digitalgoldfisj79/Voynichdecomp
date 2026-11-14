"""Ink instrument v1: stroke width and ink darkness of a word crop. Pure function, no I/O."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
VERSION="v3"
def lum(rgb): return rgb[...,0]*0.299+rgb[...,1]*0.587+rgb[...,2]*0.114
def measure(L, peak_pct=98.0, half=0.5, min_ink=40, core_dt=4.0, min_core=8):
    """L: 2-D luminance array (float). Returns dict(width_px, darkness, n_ink, B) or None."""
    L=L.astype(float); B=np.median(L[L>=np.median(L)])
    if B<=0: return None
    D=1.0-L/B; peak=np.percentile(D,peak_pct)
    if peak<=0.05: return None
    mask=D>half*peak; mask=ndi.binary_opening(mask,iterations=1)
    if mask.sum()<min_ink: return None
    dt=ndi.distance_transform_edt(mask); sk=skeletonize(mask)
    if sk.sum()<10: return None
    core=sk&(dt>=core_dt)
    dark=float(np.median(D[core])) if core.sum()>=min_core else float("nan")
    return dict(width_px=float(2*np.median(dt[sk])-1), darkness=dark, n_core=int(core.sum()), n_ink=int(mask.sum()), B=float(B), peak=float(peak))
