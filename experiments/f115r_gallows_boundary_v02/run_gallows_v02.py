#!/usr/bin/env python3
from __future__ import annotations
import itertools, json, math, re, sys
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

HERE=Path(__file__).resolve().parent
V01=HERE.parent/'f115r_local_hand_boundary_v01'
sys.path.insert(0,str(V01))
import run_assay as v1

OUT=HERE/'results'; OUT.mkdir(parents=True,exist_ok=True)
IMAGE_URL=v1.IMAGE_URL; TR_URL=v1.TR_URL
KFEATS=['aspect','occupancy','slant','top_span','mid_span','bottom_span','max_horiz_run_top','row_sd','hole_ratio','width_lineh']
INKFEATS=v1.INK_FEATURES
DEV_LINES=list(range(25,46))
TARGET_LINES=list(range(7,25))
PARA_BOUNDARIES=[8,10,13,19,22,26,29,35,37,40]


def gallows_seq(rec):
    out=[]
    # retain textual order and token context, but no hand labels
    for wi,tok in enumerate(rec['tokens']):
        for ci,ch in enumerate(tok):
            if ch in 'ktpf':
                out.append({'type':ch,'token':tok,'word_index':wi,'char_index':ci})
    return out


def structures(line_mask, upper_frac, join_px):
    h,w=line_mask.shape; uh=max(4,int(round(h*upper_frac)))
    top=(line_mask[:uh]>0).astype(np.uint8)
    # horizontal dilation only: joins separated pieces of one upper gallows without using lower word structure
    ker=np.ones((1,max(1,int(join_px))),np.uint8)
    dil=cv2.dilate(top,ker,iterations=1)
    n,lab,stats,_=cv2.connectedComponentsWithStats(dil,8)
    arr=[]
    for c in range(1,n):
        x,y,bw,bh,area=stats[c]
        if bw<2 or bh<2: continue
        # original-ink evidence in the dilated component's x/y box
        sub=top[y:y+bh,x:x+bw]
        orig=int(sub.sum())
        if orig<8: continue
        ys,xs=np.nonzero(sub)
        if not len(xs): continue
        # remove tiny flecks even if dilation inflated them
        obw=int(xs.max()-xs.min()+1); obh=int(ys.max()-ys.min()+1)
        if obw<2 or obh<3: continue
        arr.append({'x':int(x),'y':int(y),'w':int(bw),'h':int(bh),'orig_area':orig,
                    'cx':float(x+bw/2),'upper_h':uh})
    arr.sort(key=lambda z:z['cx'])
    return arr,top


def calibrate(line_masks, seqs):
    grid=[]
    for uf in [0.30,0.35,0.40,0.45,0.50,0.55,0.60]:
        for jp in [1,3,5,7,9,12,16,20,24]:
            exact=0; abs_err=0; eligible=0; zero_ok=0; zero_n=0
            for ln in DEV_LINES:
                g=len(seqs[ln]); s,_=structures(line_masks[ln],uf,jp); n=len(s)
                if g>0:
                    eligible+=1; exact+=int(n==g); abs_err+=abs(n-g)
                else:
                    zero_n+=1; zero_ok+=int(n==0)
            rate=exact/max(1,eligible); mae=abs_err/max(1,eligible); zrate=zero_ok/max(1,zero_n)
            grid.append({'upper_frac':uf,'join_px':jp,'exact':exact,'eligible':eligible,
                         'exact_rate':rate,'mae':mae,'zero_rate':zrate})
    # frozen objective: exact rate first, then MAE, then zero-line correctness, then smaller join
    grid.sort(key=lambda r:(-r['exact_rate'],r['mae'],-r['zero_rate'],r['join_px'],r['upper_frac']))
    return grid[0],grid


def shape_features(top,st,line_h):
    x=max(0,st['x']); x2=min(top.shape[1],st['x']+st['w'])
    g=(top[:,x:x2]>0).astype(np.uint8)
    ys,xs=np.nonzero(g)
    if len(xs)<8: return None,None
    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max(); g=g[y0:y1+1,x0:x1+1]
    h,w=g.shape
    rows=[]; spans=[]
    for yy in range(h):
        xx=np.flatnonzero(g[yy])
        if len(xx): rows.append((yy,float(xx.mean()))); spans.append((yy,float(xx.max()-xx.min()+1)))
    yy=np.array([q[0] for q in rows],float); xc=np.array([q[1] for q in rows],float)
    slant=float(np.polyfit(yy/max(1,h-1),xc/max(1,w-1),1)[0]) if len(yy)>=3 else 0.0
    def sb(a,b):
        v=[s for y,s in spans if a*h<=y<b*h]
        return float(np.mean(v)/w) if v else 0.0
    maxrun=max((v1.longest_run(row) for row in g),default=0)/max(1,w)
    ry,rx=np.nonzero(g); rowsd=float(np.std(ry)/max(1,h))
    contours,hier=cv2.findContours((g*255).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    hole=0.0
    if hier is not None:
        for i,c in enumerate(contours):
            if hier[0][i][3]>=0: hole+=abs(cv2.contourArea(c))
    f={'aspect':float(w/max(1,h)),'occupancy':float(g.mean()),'slant':slant,
       'top_span':sb(0,.333),'mid_span':sb(.333,.667),'bottom_span':sb(.667,1.01),
       'max_horiz_run_top':float(maxrun),'row_sd':rowsd,'hole_ratio':float(hole/max(1,w*h)),
       'width_lineh':float(w/max(1,line_h))}
    return f,g


def ascii_art(g,w=26,h=24):
    if g is None or g.size==0:return ['(empty)']
    gh,gw=g.shape; s=min((w-2)/max(1,gw),(h-2)/max(1,gh)); nw=max(1,int(round(gw*s))); nh=max(1,int(round(gh*s)))
    r=cv2.resize((g*255).astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)>0
    can=np.zeros((h,w),bool); oy=(h-nh)//2; ox=(w-nw)//2; can[oy:oy+nh,ox:ox+nw]=r
    return [''.join('██' if q else '  ' for q in row) for row in can]


def aggregate(rows,features):
    by=defaultdict(list)
    for r in rows: by[r['line']].append(np.array([r['features'][f] for f in features],float))
    return {ln:np.mean(v,axis=0) for ln,v in by.items()}


def contrast(vec,features,A_lines,B_lines):
    la=[x for x in A_lines if x in vec]; lb=[x for x in B_lines if x in vec]
    if len(la)<2 or len(lb)<2:return {'ok':False,'nA':len(la),'nB':len(lb),'linesA':la,'linesB':lb}
    A=np.vstack([vec[x] for x in la]);B=np.vstack([vec[x] for x in lb]);X=np.vstack([A,B])
    sd=X.std(0,ddof=1);sd=np.where(sd>1e-9,sd,1.0);eff=(A.mean(0)-B.mean(0))/sd
    obs=float(np.sqrt(np.mean(eff**2)));nA=len(A);n=len(X);vals=[];comb=math.comb(n,nA)
    if comb<=50000:
        its=itertools.combinations(range(n),nA); exact=True
    else:
        rng=np.random.default_rng(20260916);its=(rng.choice(n,nA,replace=False) for _ in range(20000));exact=False
    for ix in its:
        q=np.zeros(n,bool);q[list(ix)]=True;e=(X[q].mean(0)-X[~q].mean(0))/sd;vals.append(float(np.sqrt(np.mean(e**2))))
    vals=np.asarray(vals);p=float((np.sum(vals>=obs)+(0 if exact else 1))/(len(vals)+(0 if exact else 1)))
    return {'ok':True,'nA':len(la),'nB':len(lb),'linesA':la,'linesB':lb,'stat':obs,'p':p,'exact':exact,'null_n':len(vals),
            'feature_effects':{f:float(e) for f,e in zip(features,eff)}}


def main():
    imgp=OUT/'1006274_f115r.jpg'
    if not imgp.exists():v1.download(IMAGE_URL,imgp)
    img=cv2.imread(str(imgp));H,W=img.shape[:2];gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);mask=v1.adaptive_ink(gray)
    trans=v1.parse_f_transcription(v1.fetch_text(TR_URL)); seqs={r['line']:gallows_seq(r) for r in trans}
    x0,x1=int(.105*W),int(.925*W);y0,y1=int(.038*H),int(.865*H)
    centers,edges,_=v1.detect_lines(mask,x0,x1,y0,y1)
    line_masks={ln:mask[int(edges[ln-1]):int(edges[ln]),x0:x1] for ln in range(1,46)}
    best,grid=calibrate(line_masks,seqs);uf=best['upper_frac'];jp=best['join_px']

    assigned=[];count_rows=[];ascii_rows=[]
    for rec in trans:
        ln=rec['line']; expected=seqs[ln]; sts,top=structures(line_masks[ln],uf,jp)
        exact=len(sts)==len(expected)
        count_rows.append({'line':ln,'expected':len(expected),'detected':len(sts),'exact':exact,
                           'expected_types':''.join(z['type'] for z in expected)})
        if not exact: continue
        for st,lab in zip(sts,expected):
            f,g=shape_features(top,st,line_masks[ln].shape[0])
            if f is None:continue
            row={'line':ln,'type':lab['type'],'token':lab['token'],'word_index':lab['word_index'],'features':f,'structure':st}
            assigned.append(row)
            if 7<=ln<=24 and lab['type']=='k':
                ascii_rows.append(f"LINE {ln:02d} token={lab['token']} x={st['cx']:.1f} upper_frac={uf} join={jp}")
                ascii_rows.extend('  '+s for s in ascii_art(g));ascii_rows.append('')

    kval=[r for r in assigned if r['type']=='k'];kv=aggregate(kval,KFEATS)
    inkrows=[]
    for ln in range(1,46):
        ya,yb=int(edges[ln-1]),int(edges[ln]);f=v1.line_ink_features(img,mask,ya,yb,x0,x1)
        if f:inkrows.append({'line':ln,'features':f})
    iv=aggregate(inkrows,INKFEATS)
    w=lambda a,b:list(range(a,b+1))
    contrasts={
      'k_davis_7_12_vs_13_18':contrast(kv,KFEATS,w(7,12),w(13,18)),
      'k_ink_13_18_vs_19_24':contrast(kv,KFEATS,w(13,18),w(19,24)),
      'ink_davis_7_12_vs_13_18':contrast(iv,INKFEATS,w(7,12),w(13,18)),
      'ink_session_13_18_vs_19_24':contrast(iv,INKFEATS,w(13,18),w(19,24)),
    }
    scan=[]
    for b in PARA_BOUNDARIES:
        if b-6<1 or b+5>45:continue
        z=contrast(kv,KFEATS,w(b-6,b-1),w(b,b+5));z['boundary']=b;scan.append(z)
    dev_pos=[r for r in count_rows if r['line'] in DEV_LINES and r['expected']>0]
    dev_rate=sum(r['exact'] for r in dev_pos)/max(1,len(dev_pos))
    kd=contrasts['k_davis_7_12_vs_13_18'];ki=contrasts['k_ink_13_18_vs_19_24'];cd=contrasts['ink_davis_7_12_vs_13_18'];ci=contrasts['ink_session_13_18_vs_19_24']
    admiss=dev_rate>=.70 and kd.get('ok') and kd.get('nA',0)>=4 and kd.get('nB',0)>=4
    if not admiss:decision='UNDERPOWERED_OR_QC_FAIL'
    elif kd['p']<=.05 and (not ki.get('ok') or kd['stat']>ki['stat']) and cd.get('p',1)>.10:decision='DAVIS_LOCAL_SIGNAL'
    elif kd['p']>=.20:decision='NO_LOCAL_12_13_SIGNAL'
    else:decision='AMBIGUOUS'
    result={'schema':'f115r-gallows-boundary-v0.2','image_shape':[H,W],'calibration':best,'calibration_grid':grid,
            'development_exact_rate':dev_rate,'count_rows':count_rows,'assigned_total':len(assigned),'assigned_k':len(kval),
            'k_lines':sorted(kv),'contrasts':contrasts,'paragraph_scan':scan,'decision':decision,
            'labels_used_in_extraction':False,'target_used_in_calibration':False}
    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    (OUT/'k_ascii_qc.txt').write_text('\n'.join(ascii_rows),encoding='utf-8')
    # concise report
    def ff(o):
        return 'insufficient' if not o.get('ok') else f"stat={o['stat']:.3f}, p={o['p']:.4f}, lines={o['nA']}/{o['nB']}"
    txt=["# f115r gallows-only boundary assay v0.2 — result","",
         f"Development calibration: upper_frac={uf}, join_px={jp}; exact gallows-count rate={dev_rate:.3f} ({sum(r['exact'] for r in dev_pos)}/{len(dev_pos)} positive-gallows lines).",
         f"Assigned k structures: {len(kval)} across {len(kv)} physical lines.","",
         "## Frozen contrasts",f"- Davis 12/13 k morphology: {ff(kd)}",f"- Ink 18/19 k morphology: {ff(ki)}",
         f"- Davis 12/13 ink colour: {ff(cd)}",f"- Ink 18/19 ink colour: {ff(ci)}","",
         f"## Decision\n\n**{decision}**","","This result is local to f115r and does not by itself adjudicate the full five-scribe model."]
    (OUT/'RESULT.md').write_text('\n'.join(txt))
    print('V02='+json.dumps({'calibration':best,'development_exact_rate':dev_rate,'assigned_k':len(kval),'k_lines':len(kv),'contrasts':contrasts,'decision':decision},separators=(',',':')))

if __name__=='__main__':main()
