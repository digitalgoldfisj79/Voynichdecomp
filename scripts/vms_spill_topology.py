#!/usr/bin/env python3
"""VMS upper-margin spill topology measurement v03.

Retractions:
- S1 retracted: scan-border/text/paint leakage.
- S2 ordering statistics retracted: recto/verso physical x coordinates were not
  mirrored before folio aggregation.

v03 fixes physical-sheet coordinates and adds an identically processed bottom-edge
negative control.  The upper spill is allowed to influence topology only if it shows
signal beyond this matched edge/scanner/parchment control.  No Voynich text metadata
enters scoring.
"""
from __future__ import annotations
import csv, hashlib, io, json, math, os, random, time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np, requests
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter, median_filter

OUT=Path(os.environ.get('SPILL_OUT','artifacts/vms_spill_v01')); OUT.mkdir(parents=True,exist_ok=True)
CACHE=Path(os.environ.get('SPILL_CACHE','.cache/vms_spill_v03')); CACHE.mkdir(parents=True,exist_ok=True)
PAGES=[]; iid=1006076
for f in range(1,12):
    for s in 'rv': PAGES.append((f'f{f}{s}',iid)); iid+=1
for f in range(13,57):
    for s in 'rv': PAGES.append((f'f{f}{s}',iid)); iid+=1
assert len(PAGES)==110 and iid==1006186
FOLIOS=list(range(1,12))+list(range(13,57))
BIFOLIA={'q01':[(1,8),(2,7),(3,6),(4,5)],'q02':[(9,16),(10,15),(11,14),(12,13)],
'q03':[(17,24),(18,23),(19,22),(20,21)],'q04':[(25,32),(26,31),(27,30),(28,29)],
'q05':[(33,40),(34,39),(35,38),(36,37)],'q06':[(41,48),(42,47),(43,46),(44,45)],
'q07':[(49,56),(50,55),(51,54),(52,53)]}
S=requests.Session(); S.headers.update({'User-Agent':'VoynichTopologyResearch/0.3'})

def fetch(label,iid,width=900):
    p=CACHE/f'{label}_{width}.jpg'
    if not p.exists():
        u=f'https://collections.library.yale.edu/iiif/2/{iid}/full/{width},/0/default.jpg'; err=None
        for a in range(5):
            try:
                r=S.get(u,timeout=60); r.raise_for_status(); p.write_bytes(r.content); break
            except Exception as e: err=e; time.sleep(2**a)
        else: raise RuntimeError(f'fetch failed {label}: {err}')
    b=p.read_bytes(); return Image.open(io.BytesIO(b)).convert('RGB'),hashlib.sha256(b).hexdigest()

def detect_edge(a):
    g=.2126*a[:,:,0]+.7152*a[:,:,1]+.0722*a[:,:,2]; h,w=g.shape
    gs=gaussian_filter(g,sigma=(2,1.5),mode='nearest'); sm=gs[:int(.32*h),int(.05*w):int(.95*w)]
    lo=float(np.quantile(sm,.08)); hi=float(np.quantile(sm,.82)); th=lo+.43*(hi-lo); e=np.full(w,np.nan,np.float32); maxy=int(.22*h)
    for x in range(w):
        hit=np.convolve((gs[:maxy,x]>th).astype(np.int8),np.ones(5,np.int8),mode='same')>=4; ys=np.where(hit)[0]
        if len(ys): e[x]=ys[0]
    v=np.where(np.isfinite(e))[0]
    if len(v)<.7*w: raise RuntimeError('parchment edge detection failed')
    e=np.interp(np.arange(w),v,e[v]); return median_filter(e,size=max(7,int(w/35))|1,mode='nearest')

def edge_feature(im,which='top'):
    a=np.asarray(im).astype(np.float32)/255.
    if which=='bottom': a=a[::-1,:,:]
    h,w,_=a.shape; a=a[:,int(.02*w):int(.98*w)]; w=a.shape[1]; e=detect_edge(a); dmax=max(40,int(.17*h))
    st=np.full((dmax,w,3),np.nan,np.float32)
    for x in range(w):
        y=int(round(float(e[x]))); n=min(dmax,h-y)
        if n>0: st[:n,x]=a[y:y+n,x]
    L=.2126*st[:,:,0]+.7152*st[:,:,1]+.0722*st[:,:,2]; mx=np.nanmax(st,2); mn=np.nanmin(st,2); sat=(mx-mn)/(mx+1e-6); Y=(st[:,:,0]+st[:,:,1])/2-st[:,:,2]
    d0,d1=int(.008*h),int(.105*h); r0,r1=int(.115*h),min(st.shape[0],int(.165*h))
    if r1-r0<8: r0=max(d1,int(.11*h)); r1=st.shape[0]
    bgl=np.zeros(w,np.float32); bgy=np.zeros(w,np.float32)
    for x in range(w):
        lv,sv,yv=L[r0:r1,x],sat[r0:r1,x],Y[r0:r1,x]; ok=np.isfinite(lv)&(lv>.42)&(sv<.28)
        if ok.sum()>=4: bgl[x]=np.quantile(lv[ok],.80); bgy[x]=np.median(yv[ok])
        else:
            ok=np.isfinite(lv); bgl[x]=np.quantile(lv[ok],.80) if ok.any() else .75; bgy[x]=np.median(yv[ok]) if ok.any() else .08
    bgl=gaussian_filter(bgl,max(2,w/100),mode='nearest'); bgy=gaussian_filter(bgy,max(2,w/100),mode='nearest')
    l,y,satb=L[d0:d1],Y[d0:d1],sat[d0:d1]; dark=np.clip(bgl[None,:]-l,0,.18); warm=np.clip(y-bgy[None,:],0,.12); score=.78*dark+.22*warm
    valid=np.isfinite(score)&np.isfinite(l)&(l>.38)&(satb<.30); lf=np.nan_to_num(l,nan=.8); gy,gx=np.gradient(gaussian_filter(lf,1)); valid&=np.hypot(gx,gy)<.055
    num=gaussian_filter(np.nan_to_num(np.where(valid,score,np.nan),nan=0.),sigma=(2.2,4),mode='nearest'); den=gaussian_filter(valid.astype(np.float32),sigma=(2.2,4),mode='nearest'); sm=np.where(den>.25,num/np.maximum(den,1e-6),np.nan)
    ny,nx=6,32; grid=np.zeros((ny,nx),np.float32)
    for iy in range(ny):
        ya,yb=int(iy*sm.shape[0]/ny),int((iy+1)*sm.shape[0]/ny)
        for ix in range(nx):
            xa,xb=int(ix*w/nx),int((ix+1)*w/nx); v=sm[ya:yb,xa:xb]; grid[iy,ix]=np.nanmedian(v) if np.isfinite(v).any() else 0
    prof=np.zeros(nx,np.float32)
    for ix in range(nx):
        xa,xb=int(ix*w/nx),int((ix+1)*w/nx); v=sm[:,xa:xb]; prof[ix]=np.nanmedian(v) if np.isfinite(v).any() else 0
    vals=sm[np.isfinite(sm)]
    return {'mean':float(vals.mean()),'p90':float(np.quantile(vals,.9)),'area015':float(np.mean(vals>.015)),'area025':float(np.mean(vals>.025)),
            'valid':float(valid.mean()),'edge_y_mean':float(e.mean()/h),'edge_y_sd':float(e.std()/h),'grid':grid,'profile':prof,'aligned':st,'score':sm,'h':h}

def vec(d): return np.concatenate([d['grid'].ravel(),d['profile']])
def stdz(X):
    m=np.median(X,0); mad=np.median(abs(X-m),0); s=np.where(mad>1e-6,1.4826*mad,np.std(X,0)+1e-6); return (X-m)/np.where(s>1e-6,s,1)
def dmat(X):
    Z=stdz(X); return np.sqrt(np.mean((Z[:,None]-Z[None,:])**2,2))
def energy(o,D): return float(np.mean([D[o[i],o[i+1]] for i in range(len(o)-1)]))
def perms(order,n,seed):
    rng=random.Random(seed); out=[]
    for _ in range(n): p=order[:]; rng.shuffle(p); out.append(p)
    return out
def summarize(order,D,ps):
    obs=energy(order,D); a=np.array([energy(p,D) for p in ps]); sd=float(a.std(ddof=1)); return {'observed':obs,'null_mean':float(a.mean()),'null_sd':sd,'z':float((obs-a.mean())/sd),'p_lower':float((1+(a<=obs).sum())/(len(a)+1)),'n_perm':len(a)}
def paired_specific(order,Dt,Db,ps):
    obs=energy(order,Dt)-energy(order,Db); a=np.array([energy(p,Dt)-energy(p,Db) for p in ps]); sd=float(a.std(ddof=1)); return {'observed_top_minus_bottom':obs,'null_mean':float(a.mean()),'null_sd':sd,'z':float((obs-a.mean())/sd),'p_lower':float((1+(a<=obs).sum())/(len(a)+1)),'n_perm':len(a)}

def mirror_corr(rows,edge):
    vals=[]
    for f in FOLIOS:
        a=rows[f'f{f}r'][edge]['profile']; b=rows[f'f{f}v'][edge]['profile'][::-1]
        vals.append(float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-8 and np.std(b)>1e-8 else 0)
    return {'median':float(np.median(vals)),'mean':float(np.mean(vals)),'n_mirror_positive':int(sum(x>0 for x in vals))}

def contact(rows,labels,path):
    W,H=420,110; block=4*H+24; sh=Image.new('RGB',(2*W,math.ceil(len(labels)/2)*block),'white'); dr=ImageDraw.Draw(sh)
    for k,l in enumerate(labels):
        rr,cc=divmod(k,2); x,y=cc*W,rr*block; dr.text((x+4,y+2),l,fill='black')
        for j,edge in enumerate(['top','bottom']):
            d=rows[l][edge]; raw=Image.fromarray(np.uint8(np.clip(d['aligned'][:int(.105*d['h'])],0,1)*255)).resize((W,H)); sh.paste(raw,(x,y+20+j*2*H))
            sc=d['score']; q=np.nan_to_num(np.clip(sc/.06,0,1),nan=0); heat=np.zeros((sc.shape[0],sc.shape[1],3),np.uint8)+190; heat[:,:,0]=255; heat[:,:,1]=np.uint8(255*(1-q)); heat[:,:,2]=np.uint8(255*(1-q)); heat[~np.isfinite(sc)]=180
            sh.paste(Image.fromarray(heat).resize((W,H)),(x,y+20+(j*2+1)*H))
    sh.save(path,quality=91)

def main():
    rows={}; hashes={}
    for label,i in PAGES:
        im,hh=fetch(label,i); hashes[label]=hh; rows[label]={'top':edge_feature(im,'top'),'bottom':edge_feature(im,'bottom')}; print(label,rows[label]['top']['mean'],rows[label]['bottom']['mean'],flush=True)
    cols=['mean','p90','area015','area025','valid','edge_y_mean','edge_y_sd']
    with (OUT/'spill_page_features.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['page_id','iiif_id','sha256',*[f'top_{c}' for c in cols],*[f'bottom_{c}' for c in cols],*[f'top_profile_{i:02d}' for i in range(32)],*[f'bottom_profile_{i:02d}' for i in range(32)]])
        for l,i in PAGES: w.writerow([l,i,hashes[l],*[rows[l]['top'][c] for c in cols],*[rows[l]['bottom'][c] for c in cols],*rows[l]['top']['profile'].tolist(),*rows[l]['bottom']['profile'].tolist()])
    Xt=[]; Xb=[]; Rt=[]; Rb=[]; Vt=[]; Vb=[]; topmean=[]; botmean=[]
    for f in FOLIOS:
        tr,tv=rows[f'f{f}r']['top'],rows[f'f{f}v']['top']; br,bv=rows[f'f{f}r']['bottom'],rows[f'f{f}v']['bottom']
        rtv,tvv=vec(tr),vec(tv); rbv,bvv=vec(br),vec(bv)
        # Opposite sides of one physical leaf are horizontal mirror views.
        tvv=np.concatenate([tv['grid'][:,::-1].ravel(),tv['profile'][::-1]]); bvv=np.concatenate([bv['grid'][:,::-1].ravel(),bv['profile'][::-1]])
        Rt.append(rtv); Vt.append(tvv); Rb.append(rbv); Vb.append(bvv); Xt.append((rtv+tvv)/2); Xb.append((rbv+bvv)/2)
        topmean.append((tr['mean']+tv['mean'])/2); botmean.append((br['mean']+bv['mean'])/2)
    Dt,Db=dmat(np.stack(Xt)),dmat(np.stack(Xb)); Drt,Drb=dmat(np.stack(Rt)),dmat(np.stack(Rb)); Dvt,Dvb=dmat(np.stack(Vt)),dmat(np.stack(Vb)); fi={f:i for i,f in enumerate(FOLIOS)}; order=list(range(len(FOLIOS))); ps=perms(order,20000,20260907)
    # scalar distance matrices are separately standardized; this is stain-intensity rather than contour/profile.
    tm=stdz(np.array(topmean)[:,None]).ravel(); bm=stdz(np.array(botmean)[:,None]).ravel(); Dts=np.abs(tm[:,None]-tm[None,:]); Dbs=np.abs(bm[:,None]-bm[None,:])
    summary={'protocol':'vms_spill_topology_s3_20260907','retracts':['vms_spill_topology_s1_20260907','vms_spill_topology_s2_ordering_20260907'],
      'source':'Beinecke IIIF','n_page_sides':110,'n_folios':55,'missing_folios':[12],
      'reliability':{'top_recto_mirrored_verso_profile':mirror_corr(rows,'top'),'bottom_recto_mirrored_verso_profile':mirror_corr(rows,'bottom')},
      'current_order':{'top_profile':summarize(order,Dt,ps),'bottom_profile':summarize(order,Db,ps),'top_specific_profile':paired_specific(order,Dt,Db,ps),
                       'top_scalar':summarize(order,Dts,ps),'bottom_scalar':summarize(order,Dbs,ps),'top_specific_scalar':paired_specific(order,Dts,Dbs,ps),
                       'top_recto':summarize(order,Drt,ps[:10000]),'bottom_recto':summarize(order,Drb,ps[:10000]),'top_specific_recto':paired_specific(order,Drt,Drb,ps[:10000]),
                       'top_verso':summarize(order,Dvt,ps[:10000]),'bottom_verso':summarize(order,Dvb,ps[:10000]),'top_specific_verso':paired_specific(order,Dvt,Dvb,ps[:10000])},
      'quire_top_specific':{},'known_discontinuity':{},'license':'SEALED_PENDING_MATCHED_CONTROL_AND_KNOWN_POSITIVE'}
    for q,pairs in BIFOLIA.items():
        leaves=sorted({z for p in pairs for z in p if z in fi}); o=[fi[z] for z in leaves]; qp=perms(o,10000,20260907+int(q[-2:])); summary['quire_top_specific'][q]=paired_specific(o,Dt,Db,qp)
    # Known Davis f32v/f33r: scalar upper discontinuity relative to all surviving current v->r facing pairs; bottom is matched control.
    def facing_metric(edge,key):
        vals=[]
        for a,b in zip(FOLIOS[:-1],FOLIOS[1:]):
            if b==a+1: vals.append(abs(rows[f'f{a}v'][edge][key]-rows[f'f{b}r'][edge][key]))
        tar=abs(rows['f32v'][edge][key]-rows['f33r'][edge][key]); a=np.array(vals); sd=float(a.std(ddof=1)); return {'target':tar,'reference_mean':float(a.mean()),'reference_sd':sd,'z':float((tar-a.mean())/sd),'percentile':float(np.mean(a<=tar)),'n_reference':len(a)}
    for key in ['mean','area025']: summary['known_discontinuity'][key]={'top':facing_metric('top',key),'bottom':facing_metric('bottom',key)}
    # Conjoint similarity is descriptive only.
    conj=[]
    for q,pairs in BIFOLIA.items():
        leaves=sorted({z for p in pairs for z in p if z in fi}); ap=[Dt[fi[leaves[a]],fi[leaves[b]]] for a in range(len(leaves)) for b in range(a+1,len(leaves))]; ob=[Dt[fi[x],fi[y]] for x,y in pairs if x in fi and y in fi]; sd=float(np.std(ap,ddof=1)); conj.append({'quire':q,'effect':float(np.mean(ob)-np.mean(ap)),'null_sd':sd,'effect_sd':float((np.mean(ob)-np.mean(ap))/sd)})
    summary['conjoint_top_profile']=conj
    summary['decision_rule']='Do not optimize historical order unless top-specific paired control is materially stronger than bottom, known f32v/f33r discontinuity is recovered, and signal replicates recto/verso. Developmental thresholds will be frozen before any later confirmation run.'
    (OUT/'spill_summary.json').write_text(json.dumps(summary,indent=2))
    with (OUT/'spill_folio_distance_top.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['folio',*FOLIOS]); [w.writerow([fol,*Dt[j].tolist()]) for j,fol in enumerate(FOLIOS)]
    with (OUT/'spill_folio_distance_bottom.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['folio',*FOLIOS]); [w.writerow([fol,*Db[j].tolist()]) for j,fol in enumerate(FOLIOS)]
    contact(rows,['f1r','f8v','f16v','f24v','f25r','f31v','f32v','f33r','f34v','f40v','f41r','f48v','f49r','f56v'],OUT/'spill_diagnostic_contact.jpg')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
