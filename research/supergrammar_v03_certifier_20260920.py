#!/usr/bin/env python3
import argparse, collections, hashlib, json, math, os, pickle, re, urllib.request
from pathlib import Path
import numpy as np

VERSION="supergrammar-v03-certifier-20260920-v1"
CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/92ec41cb26d233a388b6f65fa1a4b7c45d7ad8c5/voynich_transcriptions_slim.json"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
ROWS_CANON_SHA="74f7310ea35922dc5ed71012f0825ef9480d051a1fa516c79fc5ca53f88a925f"
FOLDS_CANON_SHA="e774001ca046d88f24f56bf70f29213007d9cac3500d50f414e1782688d74888"
LAYERS=("ZLZI","ZLZB","TTLI","JSLI","VDRB","TTIA")
ALPH=tuple("abcdefghijklmnopqrstuvwxyz")
ALPH_N=len(ALPH)
ALPH_SET=set(ALPH)
ALPHAS=(4.0,16.0,64.0,256.0)
HARD_ZEROS=("ci","dh","dn","kk","km","kn","kp","lh","ln","pl","pn","pp","pt","tm","tn","tp","tr","tt","yn")
CHECKPOINT=Path("/tmp/supergrammar_v03_certifier.pkl")
PAIRS=[(1,8),(2,7),(3,6),(4,5),(9,16),(10,15),(11,14),(17,24),(18,23),(19,22),(20,21),
       (25,32),(26,31),(27,30),(28,29),(33,40),(34,39),(35,38),(36,37),(41,48),(42,47),
       (43,46),(44,45),(49,56),(50,55),(51,54),(52,53),(57,66),(58,65),(67,68),(69,70),
       (71,72),(75,84),(76,83),(77,82),(78,81),(79,80),(85,86),(87,90),(88,89),(93,96),
       (94,95),(99,102),(100,101),(103,116),(104,115),(105,114),(106,113),(107,112),(108,111)]
BIF_BY_NUM={n:f"B{a:03d}_{b:03d}" for a,b in PAIRS for n in (a,b)}

def atomic_pickle(obj,path=CHECKPOINT):
    tmp=str(path)+".tmp"
    with open(tmp,"wb") as f: pickle.dump(obj,f,pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,path)

def canon_sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def parse_num(f):
    m=re.match(r"f(\d+)",str(f)); return int(m.group(1)) if m else None

def section(f):
    n=parse_num(f)
    if n is None:return "UNK"
    if n<=66:return "HERBAL"
    if n<=73:return "ASTRO"
    if 75<=n<=84:return "BIO"
    if 85<=n<=102:return "PHARMA"
    if 103<=n<=116:return "RECIPES"
    return "UNK"

def davis_hand(folio,line_no):
    s=str(folio);n=parse_num(s);side="r" if "r" in s else ("v" if "v" in s else "");ln=int(line_no or 0)
    if n is None:return "UNKNOWN"
    if n==115 and side=="r":return "S2" if ln<=12 else "S3"
    if 1<=n<=24:return "S1"
    maps=[{25:"S1",26:"S2",27:"S1",28:"S1",29:"S1",30:"S1",31:"S2",32:"S1"},
          {33:"S2",34:"S2",35:"S1",36:"S1",37:"S1",38:"S1",39:"S2",40:"S2"},
          {41:"S5",42:"S1",43:"S2",44:"S1",45:"S1",46:"S2",47:"S1",48:"S5"},
          {49:"S1",50:"S2",51:"S1",52:"S1",53:"S1",54:"S1",55:"S2",56:"S1"}]
    for mp in maps:
        if n in mp:return mp[n]
    if n==57:return "S1" if side=="v" else "S5"
    if n in (58,65):return "S3"
    if n==66:return "S5"
    if 67<=n<=73:return "S4"
    if 75<=n<=84:return "S2"
    if 85<=n<=86:return "MIXED_ROSE"
    if 87<=n<=90:return "S1"
    if n==93:return "S1"
    if n in (94,95):return "S3"
    if n==96:return "S1"
    if 99<=n<=102:return "S1"
    if 103<=n<=116:return "S3"
    return "UNKNOWN"

def lenbin_token(t):
    n=len(t); return "12" if n<=2 else ("34" if n<=4 else ("56" if n<=6 else "7p"))

def family(t): return f"{t[0]}|{t[-1]}|{lenbin_token(t)}"

def posclass(pos,n):
    if pos==0:return "FIRST"
    if pos==n-1:return "LAST"
    return "MID"

def build_events(obj):
    rows=[];eid=0
    for fol,ld in obj["pages"].items():
        n=parse_num(fol)
        if n not in BIF_BY_NUM:continue
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get("ZLZI","")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            for pos,t in enumerate(toks):
                rows.append(dict(eid=eid,folio=fol,line=int(ls),pos=pos,line_len=len(toks),page=fol,
                                 bifolium=BIF_BY_NUM[n],section=section(fol),hand=davis_hand(fol,int(ls)),
                                 token=t,family=family(t)));eid+=1
    return rows

def fold_assignment(rows):
    by=collections.defaultdict(list)
    for r in rows:by[r["bifolium"]].append(r)
    secs=sorted({r["section"] for r in rows});hands=sorted({r["hand"] for r in rows});feats={}
    for bif,rs in by.items():
        cs=collections.Counter(r["section"] for r in rs);ch=collections.Counter(r["hand"] for r in rs)
        feats[bif]=np.array([len(rs)]+[cs[x] for x in secs]+[ch[x] for x in hands],float)
    total=sum(feats.values(),np.zeros(1+len(secs)+len(hands)));target=total/5;scale=np.maximum(target,1)
    loads=[np.zeros_like(target) for _ in range(5)];counts=[0]*5;assign={}
    for bif in sorted(feats,key=lambda z:(-feats[z][0],z)):
        cand=[]
        for f in range(5):
            trial=[x.copy() for x in loads];trial[f]+=feats[bif];cc=counts.copy();cc[f]+=1
            sc=float(sum(np.sum(((x-target)/scale)**2) for x in trial))+0.002*sum(x*x for x in cc)
            cand.append((sc,loads[f][0],counts[f],f))
        f=min(cand)[-1];assign[bif]=f;loads[f]+=feats[bif];counts[f]+=1
    return assign

def load():
    p=Path("/tmp/voynich_transcriptions_slim.json")
    urllib.request.urlretrieve(CORPUS_URL,p)
    got=hashlib.sha256(p.read_bytes()).hexdigest()
    if got!=CORPUS_SHA: raise RuntimeError(f"corpus sha mismatch {got}")
    obj=json.load(open(p))
    rows=build_events(obj); folds=fold_assignment(rows)
    if len(rows)!=34087: raise RuntimeError(f"event count {len(rows)}")
    if canon_sha(rows)!=ROWS_CANON_SHA: raise RuntimeError(f"row sha {canon_sha(rows)}")
    if canon_sha(folds)!=FOLDS_CANON_SHA: raise RuntimeError(f"fold sha {canon_sha(folds)}")
    return obj,rows,folds

def build_lines(obj,layer):
    out=[]
    for fol,ld in obj["pages"].items():
        n=parse_num(fol)
        if n not in BIF_BY_NUM:continue
        for ls,rec in ld.items():
            if "P" not in str(rec.get("u","")):continue
            txt=rec.get("t",{}).get(layer,"")
            toks=[t.lower() for t in txt.split() if re.fullmatch(r"[a-z]+",t.lower())]
            if not toks:continue
            line_label=str(ls)
            mm=re.match(r"(\\d+)",line_label)
            line_no=int(mm.group(1)) if mm else 0
            out.append(dict(folio=fol,line=line_no,line_label=line_label,line_order=(line_no,line_label),
                            tokens=toks,bifolium=BIF_BY_NUM[n],section=section(fol),
                            hand=davis_hand(fol,line_no)))
    return out

def signflip_stats(per_bif):
    vals=[]
    for bif,(s,n) in sorted(per_bif.items()):
        if n: vals.append(s/n)
    a=np.asarray(vals,float)
    effect=float(a.mean()) if len(a) else float("nan")
    null_sd=float(np.sqrt(np.sum(a*a))/len(a)) if len(a) else float("nan")
    ratio=float(abs(effect)/null_sd) if null_sd>0 else None
    return dict(n_bif=len(a),effect=effect,null_sd=null_sd,effect_over_null_sd=ratio,
                min_bif=float(a.min()) if len(a) else None,max_bif=float(a.max()) if len(a) else None,
                positive_bif=int((a>0).sum()) if len(a) else 0,negative_bif=int((a<0).sum()) if len(a) else 0)

class CharCounts:
    def __init__(self,lines,reverse,max_order):
        self.c0=collections.defaultdict(collections.Counter)
        self.cs=[None]+[collections.defaultdict(collections.Counter) for _ in range(max_order)]
        self.max_order=max_order
        for r in lines:
            sec=r["section"]
            for tok0 in r["tokens"]:
                tok=tok0[::-1] if reverse else tok0
                for j in range(max_order,len(tok)):
                    y=tok[j]
                    if y not in ALPH_SET:continue
                    self.c0[sec][y]+=1
                    for k in range(1,max_order+1):
                        ctx=tok[j-k:j]
                        self.cs[k][(sec,ctx)][y]+=1
    def p0(self,sec,y):
        c=self.c0.get(sec,collections.Counter()); n=sum(c.values())
        return (c.get(y,0)+0.5)/(n+0.5*ALPH_N)
    def prob(self,order,sec,hist,y,alpha):
        if order==0:return self.p0(sec,y)
        parent=self.prob(order-1,sec,hist[-(order-1):] if order>1 else "",y,alpha)
        ctx=hist[-order:]
        c=self.cs[order].get((sec,ctx),collections.Counter());n=sum(c.values())
        return (c.get(y,0)+alpha*parent)/(n+alpha)

def char_context_stage(layer_lines,folds):
    results=[]
    for reverse in (False,True):
        for child_order in (2,3):
            per_alpha={a:collections.defaultdict(lambda:[0.0,0]) for a in ALPHAS}
            for f in range(5):
                train=[r for r in layer_lines if folds[r["bifolium"]]!=f]
                test=[r for r in layer_lines if folds[r["bifolium"]]==f]
                model=CharCounts(train,reverse,child_order)
                for r in test:
                    sec=r["section"];bif=r["bifolium"]
                    for tok0 in r["tokens"]:
                        tok=tok0[::-1] if reverse else tok0
                        for j in range(child_order,len(tok)):
                            y=tok[j]
                            if y not in ALPH_SET:continue
                            hist=tok[:j]
                            for a in ALPHAS:
                                pp=max(model.prob(child_order-1,sec,hist,y,a),1e-300)
                                pc=max(model.prob(child_order,sec,hist,y,a),1e-300)
                                gain=math.log2(pc/pp)
                                z=per_alpha[a][bif]; z[0]+=gain; z[1]+=1
            for a in ALPHAS:
                st=signflip_stats(per_alpha[a])
                st.update(alpha=a,direction="RTL_REVERSED" if reverse else "LTR",contrast=f"ORDER{child_order-1}_TO_ORDER{child_order}")
                results.append(st)
    return results

def build_junction_pairs(lines):
    out=[]
    bypage=collections.defaultdict(list)
    for r in lines:bypage[r["folio"]].append(r)
    for fol,ls in bypage.items():
        ls=sorted(ls,key=lambda x:x.get("line_order",(x["line"],str(x["line"]))))
        for r in ls:
            toks=r["tokens"]
            for i in range(len(toks)-1):
                lft,rgt=toks[i],toks[i+1]
                out.append(dict(bifolium=r["bifolium"],section=r["section"],hand=r["hand"],boundary="SPACE",
                                right_pos=posclass(i+1,len(toks)),right_len=lenbin_token(rgt),
                                left_last=lft[-1],target=rgt[0]))
        for a,b in zip(ls,ls[1:]):
            if not a["tokens"] or not b["tokens"]:continue
            lft,rgt=a["tokens"][-1],b["tokens"][0]
            out.append(dict(bifolium=a["bifolium"],section=b["section"],hand=b["hand"],boundary="LINE_BREAK",
                            right_pos="FIRST",right_len=lenbin_token(rgt),left_last=lft[-1],target=rgt[0]))
    return out

class JunctionCounts:
    def __init__(self,pairs):
        self.c0=collections.defaultdict(collections.Counter)
        self.cb=collections.defaultdict(collections.Counter)
        self.ce=collections.defaultdict(collections.Counter)
        for r in pairs:
            k0=(r["boundary"],r["section"])
            kb=(r["boundary"],r["section"],r["hand"],r["right_pos"],r["right_len"])
            ke=kb+(r["left_last"],)
            y=r["target"]
            self.c0[k0][y]+=1;self.cb[kb][y]+=1;self.ce[ke][y]+=1
    def p0(self,r,y):
        c=self.c0[(r["boundary"],r["section"])];n=sum(c.values())
        return (c.get(y,0)+0.5)/(n+0.5*ALPH_N)
    def pbase(self,r,y,a):
        kb=(r["boundary"],r["section"],r["hand"],r["right_pos"],r["right_len"])
        c=self.cb[kb];n=sum(c.values());p=self.p0(r,y)
        return (c.get(y,0)+a*p)/(n+a)
    def penh(self,r,y,a):
        kb=(r["boundary"],r["section"],r["hand"],r["right_pos"],r["right_len"])
        ke=kb+(r["left_last"],);c=self.ce[ke];n=sum(c.values());p=self.pbase(r,y,a)
        return (c.get(y,0)+a*p)/(n+a)

def junction_stage(lines,folds):
    pairs=build_junction_pairs(lines)
    per={a:{"SPACE":collections.defaultdict(lambda:[0.0,0]),"LINE_BREAK":collections.defaultdict(lambda:[0.0,0])} for a in ALPHAS}
    counts=collections.Counter(r["boundary"] for r in pairs)
    for f in range(5):
        train=[r for r in pairs if folds[r["bifolium"]]!=f]
        test=[r for r in pairs if folds[r["bifolium"]]==f]
        m=JunctionCounts(train)
        for r in test:
            y=r["target"];bif=r["bifolium"];bt=r["boundary"]
            for a in ALPHAS:
                pb=max(m.pbase(r,y,a),1e-300);pe=max(m.penh(r,y,a),1e-300)
                g=math.log2(pe/pb);z=per[a][bt][bif];z[0]+=g;z[1]+=1
    out=[]
    for a in ALPHAS:
        ss=signflip_stats(per[a]["SPACE"]); ss.update(alpha=a,contrast="SPACE_EDGE_GAIN",pair_count=counts["SPACE"]);out.append(ss)
        sl=signflip_stats(per[a]["LINE_BREAK"]); sl.update(alpha=a,contrast="LINEBREAK_EDGE_GAIN",pair_count=counts["LINE_BREAK"]);out.append(sl)
        cc={}
        for bif in sorted(set(per[a]["SPACE"])&set(per[a]["LINE_BREAK"])):
            s1,n1=per[a]["SPACE"][bif];s2,n2=per[a]["LINE_BREAK"][bif]
            if n1 and n2:cc[bif]=[s1/n1-s2/n2,1]
        sc=signflip_stats(cc);sc.update(alpha=a,contrast="SPACE_MINUS_LINEBREAK_GAIN",pair_count=None);out.append(sc)
    return out

def hard_zero_stage(lines_by_layer):
    lines=lines_by_layer["ZLZI"]
    groups={}
    direct=collections.Counter()
    for r in lines:
        fol=r["folio"]
        for tok in r["tokens"]:
            for j in range(len(tok)-1):
                a,b=tok[j],tok[j+1]; direct[a+b]+=1
                key=(fol,j,lenbin_token(tok))
                g=groups.setdefault(key,[0,collections.Counter(),collections.Counter()])
                g[0]+=1;g[1][a]+=1;g[2][b]+=1
    rows=[]
    for pat in HARD_ZEROS:
        a,b=pat;obs=direct[pat];exp=0.0;var=0.0;opp=0
        for N,L,R in groups.values():
            nl=L.get(a,0);kr=R.get(b,0)
            if not nl or not kr:continue
            opp+=nl
            p=kr/N;exp+=nl*p
            if N>1:var+=nl*p*(1-p)*(N-nl)/(N-1)
        sd=math.sqrt(max(var,0.0));eff=exp-obs;ratio=eff/sd if sd>0 else None
        alt={}
        for layer,ls in lines_by_layer.items():
            c=0
            for r in ls:
                for tok in r["tokens"]:c+=sum(1 for j in range(len(tok)-1) if tok[j:j+2]==pat)
            alt[layer]=c
        rows.append(dict(pattern=pat,observed=obs,expected=exp,null_sd=sd,effect_expected_minus_observed=eff,
                         effect_over_null_sd=ratio,left_opportunities=opp,alt_observed=alt))
    return rows

def is_ed1(a,b):
    la,lb=len(a),len(b)
    if abs(la-lb)>1:return False
    if la==lb:return sum(x!=y for x,y in zip(a,b))==1
    if la>lb:a,b=b,a;la,lb=lb,la
    i=j=0;used=0
    while i<la and j<lb:
        if a[i]==b[j]:i+=1;j+=1
        else:
            used+=1
            if used>1:return False
            j+=1
    return True

def assignment_stats(lefts,rights):
    n=len(lefts)
    if n==0:return (0.0,0.0)
    M=np.zeros((n,n),dtype=np.float64)
    for i,a in enumerate(lefts):
        for j,b in enumerate(rights):M[i,j]=1.0 if is_ed1(a,b) else 0.0
    T=float(M.sum()); E=T/n
    if n==1:return E,0.0
    rs=M.sum(axis=1);cs=M.sum(axis=0)
    cross=T*T-float(np.sum(rs*rs))-float(np.sum(cs*cs))+T
    es2=E+cross/(n*(n-1))
    return E,max(0.0,es2-E*E)

def boundary_ed1_stage(lines):
    bypage=collections.defaultdict(list)
    for r in lines:bypage[r["folio"]].append(r)
    boundaries=[]
    pagegroups={}
    qgroups={}
    for fol,ls in bypage.items():
        ls=sorted(ls,key=lambda x:x["line"])
        pp=[]
        for i,(a,b) in enumerate(zip(ls,ls[1:])):
            if not a["tokens"] or not b["tokens"]:continue
            pp.append((a["tokens"][-1],b["tokens"][0],i))
        if not pp:continue
        pagegroups[fol]=pp
        m=len(pp)
        for idx,(lft,rgt,_) in enumerate(pp):
            q=min(3,int(4*idx/m))
            qgroups.setdefault((fol,q),[]).append((lft,rgt))
            boundaries.append((lft,rgt))
    N=len(boundaries); actual=sum(is_ed1(a,b) for a,b in boundaries); actual_rate=actual/N
    def null_for(groups):
        E=0.0;V=0.0;mov=0
        for g in groups.values():
            left=[x[0] for x in g];right=[x[1] for x in g]
            e,v=assignment_stats(left,right);E+=e;V+=v
            if len(g)>=2:mov+=len(g)
        mean=E/N;sd=math.sqrt(V)/N;eff=mean-actual_rate;z=eff/sd if sd>0 else None
        return dict(actual_count=actual,total=N,actual_rate=actual_rate,null_mean=mean,null_sd=sd,
                    effect_expected_minus_actual=eff,effect_over_null_sd=z,movable_fraction=mov/N)
    pgs={k:[(a,b) for a,b,_ in v] for k,v in pagegroups.items()}
    return dict(page_shuffle=null_for(pgs),page_vertical_quartile_shuffle=null_for(qgroups))

def summarize_decisions(res):
    out={}
    cc=[r for r in res["char_context"] if r["layer"]=="ZLZI"]
    for contrast in ("ORDER1_TO_ORDER2","ORDER2_TO_ORDER3"):
        rr=[r for r in cc if r["contrast"]==contrast]
        out[contrast]=dict(pass_all=all(r["effect"]>0 and (r["effect_over_null_sd"] or 0)>=2 for r in rr),
                           min_ratio=min(r["effect_over_null_sd"] for r in rr))
    jj=[r for r in res["junction"] if r["layer"]=="ZLZI"]
    for contrast in ("SPACE_EDGE_GAIN","SPACE_MINUS_LINEBREAK_GAIN"):
        rr=[r for r in jj if r["contrast"]==contrast]
        out[contrast]=dict(pass_all=all(r["effect"]>0 and (r["effect_over_null_sd"] or 0)>=2 for r in rr),
                           min_ratio=min(r["effect_over_null_sd"] for r in rr))
    hz=res["hard_zero"]
    out["HARD_ZERO_19"]=dict(all_zero=all(r["observed"]==0 for r in hz),
                             all_alt_zero=all(all(v==0 for v in r["alt_observed"].values()) for r in hz),
                             min_ratio=min(r["effect_over_null_sd"] for r in hz if r["effect_over_null_sd"] is not None))
    b=res["boundary_ed1"]
    out["BOUNDARY_ED1_INDEPENDENT_AVOIDANCE"]=dict(
        pass_independent=bool((b["page_shuffle"]["effect_over_null_sd"] or -999)>=2 and
                              (b["page_vertical_quartile_shuffle"]["effect_over_null_sd"] or -999)>=2 and
                              b["page_vertical_quartile_shuffle"]["movable_fraction"]>=0.60),
        hostile_ratio=b["page_vertical_quartile_shuffle"]["effect_over_null_sd"])
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--stage",default="all",choices=["all","char","junction","hardzero","boundary"])
    args=ap.parse_args()
    obj,rows,folds=load()
    result=dict(version=VERSION,corpus_sha256=CORPUS_SHA,rows_sha256=ROWS_CANON_SHA,folds_sha256=FOLDS_CANON_SHA,
                n_events=len(rows),n_bifolia=len(folds),layers={},char_context=[],junction=[],hard_zero=None,boundary_ed1=None)
    lines_by_layer={}
    for layer in LAYERS:
        ls=build_lines(obj,layer);lines_by_layer[layer]=ls
        result["layers"][layer]=dict(lines=len(ls),tokens=sum(len(r["tokens"]) for r in ls))
    atomic_pickle(result)
    if args.stage in ("all","char"):
        for layer in LAYERS:
            rr=char_context_stage(lines_by_layer[layer],folds)
            for x in rr:x["layer"]=layer
            result["char_context"].extend(rr);atomic_pickle(result)
    if args.stage in ("all","junction"):
        for layer in LAYERS:
            rr=junction_stage(lines_by_layer[layer],folds)
            for x in rr:x["layer"]=layer
            result["junction"].extend(rr);atomic_pickle(result)
    if args.stage in ("all","hardzero"):
        result["hard_zero"]=hard_zero_stage(lines_by_layer);atomic_pickle(result)
    if args.stage in ("all","boundary"):
        result["boundary_ed1"]=boundary_ed1_stage(lines_by_layer["ZLZI"]);atomic_pickle(result)
    if args.stage=="all":
        result["decisions"]=summarize_decisions(result)
    payload=json.dumps(result,sort_keys=True,separators=(",",":"))
    result["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    atomic_pickle(result)
    print("SGV03_RESULT="+json.dumps(result,sort_keys=True))

if __name__=="__main__":main()
