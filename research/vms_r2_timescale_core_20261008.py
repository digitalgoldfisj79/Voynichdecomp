#!/usr/bin/env python3
# VMS R2 timescale core — exact Recipes star-entry mapping + causal observed slow summaries.
import collections,hashlib,json,re,urllib.request
import numpy as np

R1_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/b02193969cb42745ee9cf3cea4985fa7664602b2/research/vms_r1_channel_core_20261008.py"
ZL_URL="https://raw.githubusercontent.com/noah-chelednik/voynich-data/472ef7366606a799fc8f1044c037e06b413f6ddd/data_sources/cache/ZL3b-n.txt"
ZL_SHA="bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"

src=urllib.request.urlopen(R1_URL,timeout=120).read().decode()
r1={"__name__":"r1core"}
exec(compile(src,R1_URL,"exec"),r1)
K=r1["K"]

zl=urllib.request.urlopen(ZL_URL,timeout=120).read()
if hashlib.sha256(zl).hexdigest()!=ZL_SHA:
    raise RuntimeError("ZL_SHA_MISMATCH")
zltxt=zl.decode("utf-8")

TARGET=set(f"f{n}{s}" for n in list(range(103,109))+list(range(111,117)) for s in ("r","v"))
TARGET.discard("f116v")
entries=[]; current={}
pat=re.compile(r"^<(f\d+[rv]\d*)\.(\d+),[^>]+>\s+(.*)$")
for ln in zltxt.splitlines():
    m=pat.match(ln)
    if not m: continue
    fol,lno,txt=m.group(1),int(m.group(2)),m.group(3)
    if fol not in TARGET: continue
    if "<%>" in txt:
        if fol in current: raise RuntimeError(("nested_entry",fol,lno))
        current[fol]={"folio":fol,"eno":1+sum(e["folio"]==fol for e in entries),"lines":[]}
    if fol in current: current[fol]["lines"].append(lno)
    if "<$>" in txt and fol in current: entries.append(current.pop(fol))
if current or len(entries)!=285:
    raise RuntimeError(("ENTRY_PARSE_FAIL",len(entries),sorted(current)))

LINE_TO_ENTRY={}
ENTRY_META={}
for eid,e in enumerate(entries):
    for li,lno in enumerate(e["lines"]):
        key=(e["folio"],int(lno))
        if key in LINE_TO_ENTRY: raise RuntimeError(("duplicate_line_entry",key))
        LINE_TO_ENTRY[key]=(eid,li,len(e["lines"]))
    ENTRY_META[eid]={"folio":e["folio"],"eno":e["eno"],"lines":[int(x) for x in e["lines"]]}

def annotate(lines):
    out=[];mapped=0
    for s in lines:
        if not s: continue
        fol=str(s[0]["folio"]); lno=int(s[0]["line_no"])
        mp=LINE_TO_ENTRY.get((fol,lno))
        if mp is None: raise RuntimeError(("UNMAPPED_R2_LINE",fol,lno))
        eid,eli,en=mp; zz=[]
        for e in s:
            x=dict(e);x["eid"]=eid;x["entry_line_index"]=eli;x["entry_line_count"]=en
            zz.append(x);mapped+=1
        out.append(zz)
    return out,mapped

def build_parent():
    lines,meta,n=r1["build"]()
    a,m=annotate(lines)
    if m!=n or n!=9616:raise RuntimeError(("R2_MAP_COUNT",m,n))
    return a,meta,n

def summarize_line(seq):
    y=np.array([int(e["y"]) for e in seq],int)
    kc=np.bincount(y,minlength=K).astype(float)
    kc/=max(len(y),1)
    fc=np.zeros(K,float); cb=np.zeros(4,float); pb=np.zeros(4,float)
    bits=[]; pear=[]
    for e in seq:
        fc[int(e["prev_final_class"])]+=1.
        cb[int(e["prev_char_bin"])]+=1.
        pb[int(e["prev_piece_bin"])]+=1.
        p=np.asarray(e["p"],float); yy=int(e["y"])
        bits.append(-np.log2(max(float(p[yy]),1e-15)))
        pear.append((1.0-float(p[yy]))/np.sqrt(max(float(p[yy])*(1-float(p[yy])),1e-12)))
    fc/=max(len(seq),1);cb/=max(len(seq),1);pb/=max(len(seq),1)
    return {
      "k12":kc,"final":fc,"char":cb,"piece":pb,
      "length":float(len(seq)),
      "mean_surprise":float(np.mean(bits)) if bits else 0.,
      "mean_pearson":float(np.mean(pear)) if pear else 0.,
      "first":int(y[0]) if len(y) else -1,
      "last":int(y[-1]) if len(y) else -1
    }

def attach_timescale_features(lines):
    # canonical physical order within entries; only prior lines can contribute.
    by=collections.defaultdict(list)
    for s in lines:
        if s: by[int(s[0]["eid"])].append(s)
    out=[]
    for eid,ss in by.items():
        ss=sorted(ss,key=lambda s:int(s[0]["entry_line_index"]))
        hist=[]
        cum_k=np.zeros(K,float);cum_n=0.
        for s in ss:
            curidx=int(s[0]["entry_line_index"])
            # Previous-line summary only if physically preceding line is actually present.
            prevsum=hist[-1] if hist and int(ss[len(hist)-1][0]["entry_line_index"])==curidx-1 else None
            t1=(cum_k/cum_n) if cum_n>0 else np.zeros(K,float)
            for e in s:
                x=dict(e)
                x["t1_cum_k12"]=t1.copy()
                x["t1_prior_lines"]=float(len(hist))
                if prevsum is None:
                    x["t2_has_prev"]=0.
                    x["t2_prev_k12"]=np.zeros(K,float)
                    x["t2_prev_final"]=np.zeros(K,float)
                    x["t2_prev_char"]=np.zeros(4,float)
                    x["t2_prev_piece"]=np.zeros(4,float)
                    x["t2_prev_length"]=0.
                    x["t2_prev_surprise"]=0.
                    x["t2_prev_pearson"]=0.
                else:
                    x["t2_has_prev"]=1.
                    x["t2_prev_k12"]=prevsum["k12"].copy()
                    x["t2_prev_final"]=prevsum["final"].copy()
                    x["t2_prev_char"]=prevsum["char"].copy()
                    x["t2_prev_piece"]=prevsum["piece"].copy()
                    x["t2_prev_length"]=prevsum["length"]
                    x["t2_prev_surprise"]=prevsum["mean_surprise"]
                    x["t2_prev_pearson"]=prevsum["mean_pearson"]
                zz=x
                # replace in place later
                e.clear();e.update(zz)
            sm=summarize_line(s);hist.append(sm);cum_k+=np.bincount([int(e["y"]) for e in s],minlength=K);cum_n+=len(s)
            out.append(s)
    # return canonical fold/folio/line order
    out.sort(key=lambda s:(int(s[0]["fold"]),str(s[0]["folio"]),int(s[0]["line_no"])))
    return out

def feature_block(e,level):
    if level==1:
        return np.r_[np.asarray(e["t1_cum_k12"],float),float(e["t1_prior_lines"]>0)]
    if level==2:
        return np.r_[
          np.asarray(e["t1_cum_k12"],float),float(e["t1_prior_lines"]>0),
          float(e["t2_has_prev"]),
          np.asarray(e["t2_prev_k12"],float),
          np.asarray(e["t2_prev_final"],float),
          np.asarray(e["t2_prev_char"],float),
          np.asarray(e["t2_prev_piece"],float),
          np.log1p(float(e["t2_prev_length"])),
          float(e["t2_prev_surprise"]),
          float(e["t2_prev_pearson"])
        ]
    raise ValueError(level)

if __name__=="__main__":
    lines,meta,n=build_parent();lines=attach_timescale_features(lines)
    eids={int(s[0]["eid"]) for s in lines}
    multi=sum(1 for eid in eids if len(ENTRY_META[eid]["lines"])>1)
    out={"programme":"VMS-R2-ALIGN","status":"complete","n_events":n,"n_lines":len(lines),
         "n_entries":len(eids),"multi_line_entries":multi,
         "mapped_fraction":1.0,"zl_sha":ZL_SHA}
    print("R2_ALIGNMENT="+json.dumps(out,separators=(",",":")),flush=True)
