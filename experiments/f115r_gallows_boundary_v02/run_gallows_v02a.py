#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path
import cv2, numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_gallows_v02 as v2
v1=v2.v1

OUT=HERE/'results_v02a'; OUT.mkdir(parents=True,exist_ok=True)
CAL_LINES=list(range(25,35))
VAL_LINES=list(range(35,46))
TARGET_LINES=list(range(7,25))


def structures_filtered(line_mask, upper_frac, join_px, min_height_frac, min_area):
    h,w=line_mask.shape; uh=max(4,int(round(h*upper_frac)))
    top=(line_mask[:uh]>0).astype(np.uint8)
    ker=np.ones((1,max(1,int(join_px))),np.uint8)
    dil=cv2.dilate(top,ker,iterations=1)
    n,lab,stats,_=cv2.connectedComponentsWithStats(dil,8)
    arr=[]
    for c in range(1,n):
        x,y,bw,bh,area=stats[c]
        if bw<2 or bh<max(3,int(round(min_height_frac*uh))): continue
        sub=top[y:y+bh,x:x+bw]
        orig=int(sub.sum())
        if orig<min_area: continue
        ys,xs=np.nonzero(sub)
        if not len(xs): continue
        obw=int(xs.max()-xs.min()+1); obh=int(ys.max()-ys.min()+1)
        if obw<2 or obh<3: continue
        arr.append({'x':int(x),'y':int(y),'w':int(bw),'h':int(bh),'orig_area':orig,
                    'cx':float(x+bw/2),'upper_h':uh})
    arr.sort(key=lambda z:z['cx'])
    return arr,top


def eval_params(line_masks,seqs,lines,p):
    exact=0; pos=0; abs_err=0; zero_n=0; zero_ok=0; rows=[]
    for ln in lines:
        exp=len(seqs[ln]); sts,_=structures_filtered(line_masks[ln],**p); det=len(sts)
        rows.append({'line':ln,'expected':exp,'detected':det,'exact':det==exp,'types':''.join(x['type'] for x in seqs[ln])})
        if exp>0:
            pos+=1; exact+=int(det==exp); abs_err+=abs(det-exp)
        else:
            zero_n+=1; zero_ok+=int(det==0)
    return {'exact':exact,'positive':pos,'exact_rate':exact/max(1,pos),'mae':abs_err/max(1,pos),
            'zero_rate':zero_ok/max(1,zero_n),'rows':rows}


def calibrate(line_masks,seqs):
    grid=[]
    for uf in [0.25,0.30,0.35,0.40,0.45,0.50]:
      for jp in [12,16,20,24,30,36]:
       for mh in [0.25,0.35,0.45,0.55,0.65,0.75]:
        for ma in [10,20,30,45,60,80,120]:
            p={'upper_frac':uf,'join_px':jp,'min_height_frac':mh,'min_area':ma}
            e=eval_params(line_masks,seqs,CAL_LINES,p)
            grid.append({**p,'cal_exact':e['exact'],'cal_positive':e['positive'],'cal_exact_rate':e['exact_rate'],
                         'cal_mae':e['mae'],'cal_zero_rate':e['zero_rate']})
    grid.sort(key=lambda r:(-r['cal_exact_rate'],r['cal_mae'],-r['cal_zero_rate'],-r['min_height_frac'],-r['min_area'],r['join_px'],r['upper_frac']))
    return grid[0],grid


def aggregate(rows,features): return v2.aggregate(rows,features)
def contrast(vec,features,A,B): return v2.contrast(vec,features,A,B)


def main():
    imgp=OUT/'1006274_f115r.jpg'
    if not imgp.exists(): v1.download(v1.IMAGE_URL,imgp)
    img=cv2.imread(str(imgp)); H,W=img.shape[:2]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); mask=v1.adaptive_ink(gray)
    trans=v1.parse_f_transcription(v1.fetch_text(v1.TR_URL)); seqs={r['line']:v2.gallows_seq(r) for r in trans}
    x0,x1=int(.105*W),int(.925*W); y0,y1=int(.038*H),int(.865*H)
    centers,edges,_=v1.detect_lines(mask,x0,x1,y0,y1)
    line_masks={ln:mask[int(edges[ln-1]):int(edges[ln]),x0:x1] for ln in range(1,46)}

    best,grid=calibrate(line_masks,seqs)
    p={k:best[k] for k in ['upper_frac','join_px','min_height_frac','min_area']}
    cal=eval_params(line_masks,seqs,CAL_LINES,p)
    val=eval_params(line_masks,seqs,VAL_LINES,p)
    qualified=val['exact_rate']>=0.70

    result={'schema':'f115r-gallows-boundary-v0.2a','parameters':p,'calibration':{k:v for k,v in cal.items() if k!='rows'},
            'validation':{k:v for k,v in val.items() if k!='rows'},'calibration_rows':cal['rows'],'validation_rows':val['rows'],
            'top10_calibration_grid':grid[:10],'validation_qualified':qualified,'target_opened':False}

    ascii_rows=[]
    if qualified:
        result['target_opened']=True
        assigned=[]; target_rows=[]
        for rec in trans:
            ln=rec['line']
            if ln not in TARGET_LINES: continue
            sts,top=structures_filtered(line_masks[ln],**p); exp=seqs[ln]; exact=len(sts)==len(exp)
            target_rows.append({'line':ln,'expected':len(exp),'detected':len(sts),'exact':exact,'types':''.join(x['type'] for x in exp)})
            if not exact: continue
            for st,lab in zip(sts,exp):
                f,g=v2.shape_features(top,st,line_masks[ln].shape[0])
                if f is None: continue
                r={'line':ln,'type':lab['type'],'token':lab['token'],'word_index':lab['word_index'],'features':f,'structure':st}
                assigned.append(r)
                if lab['type']=='k':
                    ascii_rows.append(f"LINE {ln:02d} token={lab['token']} x={st['cx']:.1f} params={p}")
                    ascii_rows.extend('  '+s for s in v2.ascii_art(g)); ascii_rows.append('')
        kval=[r for r in assigned if r['type']=='k']; kv=aggregate(kval,v2.KFEATS)
        inkrows=[]
        for ln in TARGET_LINES:
            ya,yb=int(edges[ln-1]),int(edges[ln]);f=v1.line_ink_features(img,mask,ya,yb,x0,x1)
            if f: inkrows.append({'line':ln,'features':f})
        iv=aggregate(inkrows,v2.INKFEATS); w=lambda a,b:list(range(a,b+1))
        con={
          'k_davis_7_12_vs_13_18':contrast(kv,v2.KFEATS,w(7,12),w(13,18)),
          'k_ink_13_18_vs_19_24':contrast(kv,v2.KFEATS,w(13,18),w(19,24)),
          'ink_davis_7_12_vs_13_18':contrast(iv,v2.INKFEATS,w(7,12),w(13,18)),
          'ink_session_13_18_vs_19_24':contrast(iv,v2.INKFEATS,w(13,18),w(19,24))}
        kd=con['k_davis_7_12_vs_13_18'];ki=con['k_ink_13_18_vs_19_24'];cd=con['ink_davis_7_12_vs_13_18']
        admiss=kd.get('ok') and kd.get('nA',0)>=4 and kd.get('nB',0)>=4
        if not admiss:decision='UNDERPOWERED_OR_QC_FAIL'
        elif kd['p']<=.05 and (not ki.get('ok') or kd['stat']>ki['stat']) and cd.get('p',1)>.10: decision='DAVIS_LOCAL_SIGNAL'
        elif kd['p']>=.20: decision='NO_LOCAL_12_13_SIGNAL'
        else: decision='AMBIGUOUS'
        result.update({'target_rows':target_rows,'assigned_k':len(kval),'k_lines':sorted(kv),'contrasts':con,'decision':decision})
    else:
        result['decision']='VALIDATION_GATE_FAIL_TARGET_UNOPENED'

    (OUT/'result.json').write_text(json.dumps(result,indent=2))
    (OUT/'k_ascii_qc.txt').write_text('\n'.join(ascii_rows),encoding='utf-8')
    lines=['# f115r gallows-only boundary assay v0.2a — result','',
           f"Selected parameters on lines 25–34: `{p}`.",
           f"Calibration exact count: {cal['exact']}/{cal['positive']} = {cal['exact_rate']:.3f}.",
           f"Held-out validation exact count (lines 35–45): {val['exact']}/{val['positive']} = **{val['exact_rate']:.3f}**.",
           f"Validation gate (>=0.70): **{'PASS' if qualified else 'FAIL'}**.",
           f"Target opened: **{result['target_opened']}**.",'']
    if qualified:
        for k,o in result['contrasts'].items():
            if o.get('ok'): lines.append(f"- {k}: stat={o['stat']:.3f}, p={o['p']:.4f}, lines={o['nA']}/{o['nB']}")
            else: lines.append(f"- {k}: insufficient, lines={o.get('nA',0)}/{o.get('nB',0)}")
        lines += ['',f"## Decision\n\n**{result['decision']}**"]
    else:
        lines += ['Target morphology was not computed because the external-to-target validation gate failed.','',f"## Decision\n\n**{result['decision']}**"]
    (OUT/'RESULT.md').write_text('\n'.join(lines))
    try: imgp.unlink()
    except FileNotFoundError: pass
    print('V02A='+json.dumps({'parameters':p,'cal_rate':cal['exact_rate'],'val_rate':val['exact_rate'],'qualified':qualified,'target_opened':result['target_opened'],'decision':result['decision']},separators=(',',':')))

if __name__=='__main__': main()
