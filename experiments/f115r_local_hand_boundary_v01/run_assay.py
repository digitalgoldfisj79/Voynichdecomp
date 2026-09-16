#!/usr/bin/env python3
"""Same-page f115r hand-boundary assay v0.1.

No learned embeddings.  Extraction is blind to Davis hand labels; labels are
applied only after line/word/glyph morphology has been computed.
"""
from __future__ import annotations

import itertools, json, math, os, re, sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import requests
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
IMAGE_URL = "https://zodiackillerciphers.com/images/voynich-yale/1006274.jpg"
TR_URL = "https://www.voynich.nu/q20/f115r_tr.txt"
N_LINES = 45
DAVIS_BOUNDARY = 13   # new physical line; S2=1..12, S3=13..
INK_BOUNDARY = 19     # reported visible ink change after paragraph 5 / line 18
PARA_BOUNDARIES = [4,8,10,13,19,22,26,29,35,37,40,42]
K_FEATURES = [
    "aspect","occupancy","slant","top_span","mid_span","bottom_span",
    "hole_ratio","max_horiz_run_top","upper_row_sd"
]
MN_FEATURES = ["tail_height","tail_top","upper_right_occupancy","right_edge_span"]
INK_FEATURES = ["contrast_r","contrast_g","contrast_b","rg","bg"]


def download(url: str, path: Path) -> None:
    h={"User-Agent":"Mozilla/5.0 (f115r-local-assay/0.1)"}
    r=requests.get(url,headers=h,timeout=90)
    r.raise_for_status(); path.write_bytes(r.content)


def fetch_text(url: str) -> str:
    h={"User-Agent":"Mozilla/5.0 (f115r-local-assay/0.1)"}
    r=requests.get(url,headers=h,timeout=60); r.raise_for_status(); return r.text


def clean_token(s: str) -> str:
    s=re.sub(r"\{[^}]*\}","",s)
    s=s.replace("!","").replace("-","").replace("=","")
    s=re.sub(r"[^a-z]","",s.lower())
    return s


def parse_f_transcription(txt: str):
    lines=txt.splitlines(); out=[]; i=0
    tag_re=re.compile(r"^<f115r\.P\.([^;>]+);F>\s*(.*)$")
    while i<len(lines):
        m=tag_re.match(lines[i])
        if not m: i+=1; continue
        label=m.group(1); buf=m.group(2).strip(); i+=1
        while i<len(lines) and not lines[i].startswith("<") and not lines[i].startswith("#"):
            buf += lines[i].strip(); i+=1
        raw=buf.strip()
        # dots are EVA word separators; annotation ! is not treated as a boundary
        toks=[clean_token(x) for x in raw.replace("=","").replace("-","").split(".")]
        toks=[x for x in toks if x]
        out.append({"record":label,"raw":raw,"tokens":toks})
    if len(out)!=N_LINES:
        raise RuntimeError(f"Expected {N_LINES} F-record physical lines, got {len(out)}")
    for j,r in enumerate(out,1): r["line"]=j
    return out


def longest_run(v: np.ndarray) -> int:
    best=cur=0
    for b in v.astype(bool):
        if b: cur+=1; best=max(best,cur)
        else: cur=0
    return best


def adaptive_ink(gray: np.ndarray) -> np.ndarray:
    # line art is dark; local threshold handles paper shading
    b=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY_INV,51,13)
    # reject isolated speckle without joining neighbouring glyphs
    n,lab,stats,_=cv2.connectedComponentsWithStats((b>0).astype(np.uint8),8)
    out=np.zeros_like(b)
    min_area=max(3,int(gray.size*1e-7))
    for k in range(1,n):
        if stats[k,cv2.CC_STAT_AREA]>=min_area: out[lab==k]=255
    return out


def detect_lines(mask: np.ndarray, x0:int,x1:int,y0:int,y1:int):
    m=(mask[y0:y1,x0:x1]>0)
    ys,xs=np.nonzero(m)
    if len(ys)>300000:
        rng=np.random.default_rng(20260916)
        ys=rng.choice(ys,300000,replace=False)
    km=KMeans(n_clusters=N_LINES,random_state=20260916,n_init=20).fit(ys.reshape(-1,1))
    centers=np.sort(km.cluster_centers_.ravel())+y0
    # refine each center to local maximum in smoothed horizontal projection
    proj=m.sum(axis=1).astype(float)
    proj=gaussian_filter1d(proj,sigma=max(2,(y1-y0)/2500))
    spacing=np.median(np.diff(centers))
    rr=[]
    for c in centers:
        q=int(round(c-y0)); rad=max(4,int(spacing*.28)); a=max(0,q-rad); z=min(len(proj),q+rad+1)
        rr.append(y0+a+int(np.argmax(proj[a:z])))
    centers=np.array(rr,dtype=int)
    # force monotone uniqueness by falling back to cluster center if refinement collided
    if np.any(np.diff(centers)<=0): centers=np.round(np.sort(km.cluster_centers_.ravel())+y0).astype(int)
    mids=((centers[:-1]+centers[1:])//2).tolist()
    first=max(y0,int(centers[0]-spacing*.48)); last=min(y1,int(centers[-1]+spacing*.48))
    edges=np.array([first]+mids+[last],dtype=int)
    return centers,edges,proj


def runs_1d(v):
    idx=np.flatnonzero(v)
    if not len(idx): return []
    out=[]; s=p=int(idx[0])
    for x in idx[1:]:
        x=int(x)
        if x>p+1: out.append((s,p)); s=x
        p=x
    out.append((s,p)); return out


def segment_words(line_mask: np.ndarray, expected:int):
    col=(line_mask>0).sum(axis=0)
    ink=col>=max(1,int(line_mask.shape[0]*0.025))
    runs=runs_1d(ink)
    if not runs: return [],{"ok":False,"reason":"no_ink"}
    gaps=[]
    for i in range(len(runs)-1):
        a=runs[i][1]+1; b=runs[i+1][0]-1
        if b>=a: gaps.append((b-a+1,(a+b)//2))
    need=max(0,expected-1)
    if len(gaps)<need:
        return [],{"ok":False,"reason":f"only_{len(gaps)}_gaps_for_{expected}_words"}
    chosen=sorted(sorted(gaps,reverse=True)[:need],key=lambda t:t[1]) if need else []
    breaks=[x for _,x in chosen]
    left=runs[0][0]; right=runs[-1][1]
    cuts=[left]+breaks+[right+1]
    boxes=[]
    for i in range(expected):
        a=cuts[i] if i==0 else cuts[i]+1
        b=cuts[i+1]
        if b<=a: b=a+1
        boxes.append((int(a),int(b)))
    sel=[g for g,_ in chosen]
    unsel=[g for g,_ in gaps if (g,_) not in chosen]
    gap_sep=(float(np.median(sel))/max(1.0,float(np.median(unsel)))) if sel and unsel else float("nan")
    return boxes,{"ok":True,"gap_sep":gap_sep,"all_gaps":[g for g,_ in gaps],"chosen_gaps":sel}


def choose_k_component(word_mask: np.ndarray, token:str):
    ys,xs=np.nonzero(word_mask>0)
    if not len(xs): return None,None
    xlo,xhi=xs.min(),xs.max(); ylo,yhi=ys.min(),ys.max()
    if xhi<=xlo or yhi<=ylo: return None,None
    # slight close within strokes; do not join across large inter-glyph gaps
    mm=cv2.morphologyEx((word_mask>0).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab,stats,_=cv2.connectedComponentsWithStats(mm,8)
    pred=(token.index("k")+.5)/max(1,len(token))
    xp=xlo+pred*(xhi-xlo+1)
    cand=[]
    H=max(1,yhi-ylo+1); W=max(1,xhi-xlo+1)
    for c in range(1,n):
        x,y,w,h,area=stats[c]
        if area<5 or h < .45*H: continue
        cx=x+w/2
        score=(h/H)*2.0 - abs(cx-xp)/W
        cand.append((score,c,(x,y,w,h,area),abs(cx-xp)/W))
    if not cand: return None,None
    cand.sort(reverse=True); _,c,st,pred_err=cand[0]
    cm=(lab==c).astype(np.uint8)
    return cm,{"component":int(c),"pred_err":float(pred_err),"bbox":[int(v) for v in st]}


def k_features(cm: np.ndarray):
    ys,xs=np.nonzero(cm)
    if len(xs)<8: return None
    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max(); w=x1-x0+1; h=y1-y0+1
    g=cm[y0:y1+1,x0:x1+1]
    # row spans and centroid drift
    rows=[]; spans=[]
    for yy in range(h):
        xx=np.flatnonzero(g[yy])
        if len(xx): rows.append((yy,float(xx.mean()))); spans.append((yy,float(xx.max()-xx.min()+1)))
    yy=np.array([r[0] for r in rows],float); xc=np.array([r[1] for r in rows],float)
    slope=float(np.polyfit(yy/max(1,h-1),xc/max(1,w-1),1)[0]) if len(yy)>=3 else 0.0
    def span_band(a,b):
        vals=[s for y,s in spans if a*h<=y<b*h]
        return float(np.mean(vals)/w) if vals else 0.0
    # holes in selected connected component
    contours,hier=cv2.findContours((g*255).astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    hole=0.0
    if hier is not None:
        for i,c in enumerate(contours):
            if hier[0][i][3]>=0: hole+=abs(cv2.contourArea(c))
    top=g[:max(1,int(.55*h))]
    maxhr=max((longest_run(row) for row in top),default=0)/w
    ty,tx=np.nonzero(top)
    ursd=float(np.std(ty)/h) if len(ty) else 0.0
    return {
        "aspect":float(w/h),"occupancy":float(g.mean()),"slant":slope,
        "top_span":span_band(0,.333),"mid_span":span_band(.333,.667),"bottom_span":span_band(.667,1.01),
        "hole_ratio":float(hole/max(1,w*h)),"max_horiz_run_top":float(maxhr),"upper_row_sd":ursd
    }


def mn_features(word_mask: np.ndarray):
    ys,xs=np.nonzero(word_mask>0)
    if len(xs)<8: return None
    x0,x1=xs.min(),xs.max(); y0,y1=ys.min(),ys.max(); w=x1-x0+1; h=y1-y0+1
    g=(word_mask[y0:y1+1,x0:x1+1]>0)
    xx0=max(0,int(.82*w)); ry,rx=np.nonzero(g[:,xx0:])
    if not len(ry): return None
    tail_height=1.0-float(np.median(ry))/max(1,h-1)
    tail_top=1.0-float(np.min(ry))/max(1,h-1)
    ur=float(g[:max(1,h//2),max(0,int(.7*w)):].mean())
    span=float((ry.max()-ry.min()+1)/h)
    return {"tail_height":tail_height,"tail_top":tail_top,"upper_right_occupancy":ur,"right_edge_span":span}


def line_ink_features(img, mask, y0,y1,x0,x1):
    rgb=cv2.cvtColor(img[y0:y1,x0:x1],cv2.COLOR_BGR2RGB).astype(float)
    m=mask[y0:y1,x0:x1]>0
    if m.sum()<20: return None
    # avoid darkest 1% artifacts and use central dark-ink distribution
    ink=rgb[m]; paper=rgb[~m]
    if len(paper)<20: return None
    ip=np.median(ink,axis=0); pp=np.median(paper,axis=0)
    c=np.log((pp+1)/(ip+1))
    return {"contrast_r":float(c[0]),"contrast_g":float(c[1]),"contrast_b":float(c[2]),
            "rg":float(np.log((ip[0]+1)/(ip[1]+1))),"bg":float(np.log((ip[2]+1)/(ip[1]+1)))}


def aggregate(instances, features):
    by=defaultdict(list)
    for r in instances:
        by[r["line"]].append(np.array([r["features"][f] for f in features],float))
    return {ln:np.mean(v,axis=0) for ln,v in by.items()}


def contrast(linevec, feature_names, a_lines, b_lines):
    A=[linevec[x] for x in a_lines if x in linevec]; B=[linevec[x] for x in b_lines if x in linevec]
    la=[x for x in a_lines if x in linevec]; lb=[x for x in b_lines if x in linevec]
    if len(A)<2 or len(B)<2:
        return {"ok":False,"nA":len(A),"nB":len(B),"linesA":la,"linesB":lb}
    A=np.vstack(A); B=np.vstack(B); X=np.vstack([A,B])
    sd=X.std(axis=0,ddof=1); sd=np.where(sd>1e-9,sd,1.0)
    eff=(A.mean(0)-B.mean(0))/sd
    obs=float(np.sqrt(np.mean(eff**2)))
    nA=len(A); n=len(X)
    combs=math.comb(n,nA)
    vals=[]
    if combs<=50000:
        for idx in itertools.combinations(range(n),nA):
            q=np.zeros(n,bool); q[list(idx)]=True
            e=(X[q].mean(0)-X[~q].mean(0))/sd
            vals.append(float(np.sqrt(np.mean(e**2))))
        exact=True
    else:
        rng=np.random.default_rng(20260916); exact=False
        for _ in range(20000):
            idx=rng.choice(n,nA,replace=False); q=np.zeros(n,bool); q[idx]=True
            e=(X[q].mean(0)-X[~q].mean(0))/sd; vals.append(float(np.sqrt(np.mean(e**2))))
    vals=np.asarray(vals); p=float((np.sum(vals>=obs)+(0 if exact else 1))/(len(vals)+(0 if exact else 1)))
    return {"ok":True,"nA":len(A),"nB":len(B),"linesA":la,"linesB":lb,"stat":obs,"p":p,
            "exact":exact,"null_n":int(len(vals)),"feature_effects":{f:float(e) for f,e in zip(feature_names,eff)}}


def make_qc_image(img, centers, edges, x0,x1, word_rows, k_instances, path):
    q=img.copy(); H,W=q.shape[:2]
    for i,c in enumerate(centers,1):
        col=(0,0,255) if i in (12,13,18,19) else (80,180,80)
        cv2.line(q,(x0,int(c)),(x1,int(c)),col,max(1,W//2500))
        cv2.putText(q,str(i),(max(2,x0-70),int(c)+8),cv2.FONT_HERSHEY_SIMPLEX,.5,col,1,cv2.LINE_AA)
    # word boxes: light blue; candidate k word boxes magenta
    kset={(r["line"],r["word_index"]) for r in k_instances}
    for r in word_rows:
        ln=r["line"]; ya=int(edges[ln-1]); yb=int(edges[ln]);
        for wi,(xa,xb) in enumerate(r["boxes"]):
            col=(255,0,255) if (ln,wi) in kset else (180,120,40)
            cv2.rectangle(q,(x0+xa,ya),(x0+xb,yb),col,1)
    # downsize for repository QC while preserving source scan separately only on runner
    if W>1800:
        s=1800/W; q=cv2.resize(q,(1800,int(H*s)),interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path),q,[cv2.IMWRITE_JPEG_QUALITY,88])


def write_report(result):
    p=OUT/"RESULT.md"
    lc=result["line_qc"]
    k1=result["contrasts"]["k_davis_7_12_vs_13_18"]; k2=result["contrasts"]["k_ink_13_18_vs_19_24"]
    i1=result["contrasts"]["ink_davis_7_12_vs_13_18"]; i2=result["contrasts"]["ink_session_13_18_vs_19_24"]
    lines=[]
    lines += ["# f115r local hand-boundary assay v0.1 — result","",f"Line segmentation: **{lc['status']}**; median spacing {lc['median_spacing']:.1f}px; spacing CV {lc['spacing_cv']:.3f}.",f"Median line-level token-width Spearman rho: {result['word_qc']['median_width_length_rho']:.3f}.",f"Eligible EVA-k instances: **{result['counts']['k_instances']}** on {result['counts']['k_lines']} physical lines.",f"Eligible final-m/n instances: **{result['counts']['mn_instances']}** on {result['counts']['mn_lines']} physical lines.",""]
    def fmt(name,o):
        if not o.get('ok'): return f"- {name}: insufficient ({o.get('nA',0)} vs {o.get('nB',0)} eligible lines)."
        return f"- {name}: RMS standardized shape distance **{o['stat']:.3f}**, permutation p={o['p']:.4f}, eligible lines {o['nA']} vs {o['nB']}."
    lines += ["## Frozen primary contrasts",fmt("EVA-k morphology, Davis 12/13 boundary",k1),fmt("EVA-k morphology, ink 18/19 boundary",k2),fmt("Ink/colour, Davis 12/13 boundary",i1),fmt("Ink/colour, ink 18/19 boundary",i2),""]
    lines += ["## Paragraph-boundary scan", ""]
    for r in result['paragraph_scan']:
        lines.append(f"- new line {r['boundary']}: k-stat={r.get('stat','NA')} p={r.get('p','NA')} n={r.get('nA',0)}/{r.get('nB',0)}")
    lines += ["", "## Automated classification", "", f"**{result['decision']}**", "", "This classification is page-local only and does not validate or refute the full five-hand model.", ""]
    p.write_text("\n".join(lines))


def main():
    img_path=OUT/"1006274_f115r.jpg"
    if not img_path.exists(): download(IMAGE_URL,img_path)
    txt=fetch_text(TR_URL); trans=parse_f_transcription(txt)
    (OUT/"transcription_F_records.json").write_text(json.dumps(trans,indent=2))
    img=cv2.imread(str(img_path));
    if img is None: raise RuntimeError("image decode failed")
    H,W=img.shape[:2]; gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); mask=adaptive_ink(gray)
    x0,x1=int(.105*W),int(.925*W); y0,y1=int(.038*H),int(.865*H)
    centers,edges,proj=detect_lines(mask,x0,x1,y0,y1)
    d=np.diff(centers); spacing=float(np.median(d)); spacing_cv=float(np.std(d)/np.mean(d))
    line_status="PASS" if len(centers)==N_LINES and spacing_cv<.25 and np.min(d)>.45*np.median(d) else "FAIL"

    word_rows=[]; width_rhos=[]; kinst=[]; mninst=[]; ink_rows=[]
    for rec in trans:
        ln=rec['line']; ya,yb=int(edges[ln-1]),int(edges[ln]); lm=mask[ya:yb,x0:x1]
        boxes,qc=segment_words(lm,len(rec['tokens']))
        row={"line":ln,"boxes":boxes,"qc":qc,"tokens":rec['tokens']}; word_rows.append(row)
        if not boxes: continue
        widths=np.array([b-a for a,b in boxes],float); lens=np.array([len(t) for t in rec['tokens']],float)
        if len(widths)>=3 and len(set(lens))>1:
            rho=float(spearmanr(widths,lens).statistic); width_rhos.append(rho)
        inf=line_ink_features(img,mask,ya,yb,x0,x1)
        if inf: ink_rows.append({"line":ln,"features":inf})
        for wi,(tok,(xa,xb)) in enumerate(zip(rec['tokens'],boxes)):
            wm=lm[:,xa:xb]
            if tok.count('k')==1 and not any(c in tok for c in 'tpf'):
                cm,meta=choose_k_component(wm,tok)
                if cm is not None:
                    feat=k_features(cm)
                    if feat is not None and meta['pred_err']<=.38:
                        kinst.append({"line":ln,"word_index":wi,"token":tok,"features":feat,"meta":meta})
            if tok.endswith(('m','n')):
                feat=mn_features(wm)
                if feat is not None: mninst.append({"line":ln,"word_index":wi,"token":tok,"features":feat})

    kv=aggregate(kinst,K_FEATURES); mnv=aggregate(mninst,MN_FEATURES); iv=aggregate(ink_rows,INK_FEATURES)
    def win(a,b): return list(range(a,b+1))
    contrasts={
      "k_davis_7_12_vs_13_18":contrast(kv,K_FEATURES,win(7,12),win(13,18)),
      "k_ink_13_18_vs_19_24":contrast(kv,K_FEATURES,win(13,18),win(19,24)),
      "mn_davis_7_12_vs_13_18":contrast(mnv,MN_FEATURES,win(7,12),win(13,18)),
      "mn_ink_13_18_vs_19_24":contrast(mnv,MN_FEATURES,win(13,18),win(19,24)),
      "ink_davis_7_12_vs_13_18":contrast(iv,INK_FEATURES,win(7,12),win(13,18)),
      "ink_session_13_18_vs_19_24":contrast(iv,INK_FEATURES,win(13,18),win(19,24)),
    }
    scan=[]
    for b in PARA_BOUNDARIES:
        if b-6<1 or b+5>N_LINES: continue
        z=contrast(kv,K_FEATURES,win(b-6,b-1),win(b,b+5)); z['boundary']=b; scan.append(z)

    # Conservative automated label; final interpretation remains manual/QC-aware.
    kd=contrasts['k_davis_7_12_vs_13_18']; ki=contrasts['k_ink_13_18_vs_19_24']
    cd=contrasts['ink_davis_7_12_vs_13_18']; ci=contrasts['ink_session_13_18_vs_19_24']
    adequate=kd.get('ok') and kd.get('nA',0)>=4 and kd.get('nB',0)>=4
    if adequate and kd['p']<=.05 and (not ki.get('ok') or kd['stat']>ki['stat']) and (not cd.get('ok') or cd['p']>.10):
        decision='DAVIS_BOUNDARY_SUPPORTED'
    elif adequate and kd['p']>=.20 and ci.get('ok') and ci['p']<=.05 and (not ki.get('ok') or ki['stat']>=kd['stat']):
        decision='INK_SESSION_ONLY'
    elif adequate and kd['p']>=.20:
        decision='F115R_BOUNDARY_CONTRADICTED'
    else:
        decision='AMBIGUOUS_OR_UNDERPOWERED'

    result={
      "schema":"f115r-local-hand-boundary-v0.1","image":{"url":IMAGE_URL,"shape":[H,W]},
      "line_qc":{"status":line_status,"centers":[int(x) for x in centers],"median_spacing":spacing,"spacing_cv":spacing_cv,"diffs":[int(x) for x in d]},
      "word_qc":{"median_width_length_rho":float(np.nanmedian(width_rhos)) if width_rhos else float('nan'),"n_rhos":len(width_rhos),
                  "median_gap_sep":float(np.nanmedian([r['qc'].get('gap_sep',np.nan) for r in word_rows]))},
      "counts":{"k_instances":len(kinst),"k_lines":len(kv),"mn_instances":len(mninst),"mn_lines":len(mnv)},
      "contrasts":contrasts,"paragraph_scan":scan,"decision":decision,
      "k_instances":kinst,"mn_instances":mninst,
      "blind_extraction":True,"labels_applied_after_feature_extraction":True
    }
    (OUT/"result.json").write_text(json.dumps(result,indent=2,allow_nan=True))
    make_qc_image(img,centers,edges,x0,x1,word_rows,kinst,OUT/"qc_overlay.jpg")
    write_report(result)
    # keep original scan out of git; workflow deletes it before committing outputs
    print("RESULT_JSON="+json.dumps({k:result[k] for k in ['counts','contrasts','decision']},separators=(',',':')))
    print(f"IMAGE_SHAPE={W}x{H} LINE_QC={line_status} spacing_cv={spacing_cv:.4f} width_rho={result['word_qc']['median_width_length_rho']:.3f}")

if __name__=='__main__': main()
