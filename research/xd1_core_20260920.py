#!/usr/bin/env python3
"""
XD1 generic cross-domain metric engine.
Frozen scientific definitions are in PROTOCOL_supergrammar_v03_crossdomain_XD1_20260920.md.
Input schema: JSON object {"label":..., "lines":[{"block":str,"unit":str,"line_order":int|str,
"tokens":[str,...],"writer":str|null}, ...]}.
"""
import argparse, collections, hashlib, json, math, unicodedata
import numpy as np

VERSION="XD1-core-20260920-v2-orderfix"
ALPHAS=(4.0,16.0,64.0,256.0)
LAMBDAS=(16.0,64.0,256.0,1024.0,4096.0,16384.0,1e9)
PARENT_ALPHA=64.0
P5_BANDS=((1,1),(2,5),(6,16),(17,64))
P5_NPERM=200

def canon_sha(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()

def nfc(s): return unicodedata.normalize("NFC",str(s))

def lenbin(t):
    n=len(t)
    return "12" if n<=2 else ("34" if n<=4 else ("56" if n<=6 else "7p"))

def posclass(i,n):
    if i==0:return "FIRST"
    if i==n-1:return "LAST"
    return "MID"

def order_key(v):
    if isinstance(v,(int,float)): return (0,float(v),"")
    s=str(v)
    try: return (0,float(s),"")
    except: return (1,0.0,s)

def normalize_input(obj):
    out=[]
    for r in obj["lines"]:
        toks=[nfc(x) for x in r.get("tokens",[]) if nfc(x)]
        if not toks: continue
        out.append(dict(block=str(r["block"]),unit=str(r.get("unit",r["block"])),
                        segment=str(r.get("segment",r.get("unit",r["block"]))),
                        line_order=r.get("line_order",0),tokens=toks,
                        writer=None if r.get("writer") is None else str(r.get("writer"))))
    out.sort(key=lambda r:(r["block"],order_key(r["line_order"]),r["segment"]))
    return out

def block_weights(lines):
    c=collections.Counter()
    for r in lines:c[r["block"]]+=len(r["tokens"])
    return c

def assign_folds(lines,nfold=5,salt="XD1-outer"):
    w=block_weights(lines); loads=[0]*nfold; assign={}
    for b,n in sorted(w.items(),key=lambda x:(-x[1],x[0])):
        f=min(range(nfold),key=lambda j:(loads[j],j))
        assign[b]=f;loads[f]+=n
    return assign

def inner_assign(lines,outer,salt="XD1-inner",nfold=4):
    # balance only by token count; deterministic tie break is hash then block id
    w=block_weights(lines);loads=[0]*nfold;assign={}
    def h(b): return hashlib.sha256((salt+"|"+str(outer)+"|"+b).encode()).hexdigest()
    for b,n in sorted(w.items(),key=lambda x:(-x[1],h(x[0]),x[0])):
        f=min(range(nfold),key=lambda j:(loads[j],j));assign[b]=f;loads[f]+=n
    return assign

def signflip(values):
    a=np.asarray([x for x in values if np.isfinite(x)],float)
    if len(a)==0:return dict(n_blocks=0,effect=None,null_sd=None,effect_over_null_sd=None,positive=0,negative=0)
    effect=float(a.mean()); null_sd=float(np.sqrt(np.sum(a*a))/len(a))
    return dict(n_blocks=len(a),effect=effect,null_sd=null_sd,
                effect_over_null_sd=(abs(effect)/null_sd if null_sd>0 else None),
                positive=int((a>0).sum()),negative=int((a<0).sum()))

def alphabet_from_lines(lines):
    chars=sorted({ch for r in lines for t in r["tokens"] for ch in t})
    return tuple(chars)+("<UNK>",)

def ckey(ch,aset): return ch if ch in aset else "<UNK>"

class CharModel:
    def __init__(self,lines,max_order):
        self.alphabet=alphabet_from_lines(lines);self.aset=set(self.alphabet);self.V=len(self.alphabet)
        self.c0=collections.Counter();self.cs=[None]+[collections.defaultdict(collections.Counter) for _ in range(max_order)]
        for r in lines:
            for tok in r["tokens"]:
                for j,ch in enumerate(tok):
                    y=ckey(ch,self.aset);self.c0[y]+=1
                    for k in range(1,min(max_order,j)+1):
                        ctx=tuple(ckey(x,self.aset) for x in tok[j-k:j])
                        self.cs[k][ctx][y]+=1
    def p0(self,y):
        y=ckey(y,self.aset);n=sum(self.c0.values())
        return (self.c0.get(y,0)+0.5)/(n+0.5*self.V)
    def prob(self,order,hist,y,alpha):
        y=ckey(y,self.aset)
        if order<=0:return self.p0(y)
        ph=hist[-(order-1):] if order>1 else ()
        parent=self.prob(order-1,ph,y,alpha)
        ctx=tuple(ckey(x,self.aset) for x in hist[-order:])
        c=self.cs[order].get(ctx,collections.Counter());n=sum(c.values())
        return (c.get(y,0)+alpha*parent)/(n+alpha)

def reverse_lines(lines):
    return [dict(r,tokens=[t[::-1] for t in r["tokens"]]) for r in lines]

def p1_memory(lines,folds):
    out=[]
    for rev in (False,True):
        work=reverse_lines(lines) if rev else lines
        for child in (2,3):
            per={a:collections.defaultdict(lambda:[0.0,0]) for a in ALPHAS}
            for f in range(5):
                tr=[r for r in work if folds[r["block"]]!=f];te=[r for r in work if folds[r["block"]]==f]
                m=CharModel(tr,child)
                for r in te:
                    b=r["block"]
                    for tok in r["tokens"]:
                        for j in range(child,len(tok)):
                            y=tok[j];hist=tuple(tok[:j])
                            for a in ALPHAS:
                                pp=max(m.prob(child-1,hist,y,a),1e-300);pc=max(m.prob(child,hist,y,a),1e-300)
                                z=per[a][b];z[0]+=math.log2(pc/pp);z[1]+=1
            for a in ALPHAS:
                vals=[s/n for s,n in per[a].values() if n]
                st=signflip(vals);st.update(metric_id="P1",contrast=f"ORDER{child-1}_TO_ORDER{child}",
                                              direction="RTL_REVERSED" if rev else "LTR",alpha=a)
                out.append(st)
    return out

def junction_pairs(lines):
    by=collections.defaultdict(list)
    for r in lines:by[r["block"]].append(r)
    out=[]
    for b,ls in by.items():
        ls=sorted(ls,key=lambda r:order_key(r["line_order"]))
        for r in ls:
            toks=r["tokens"]
            for i in range(len(toks)-1):
                out.append(dict(block=b,boundary="SPACE",right_pos=posclass(i+1,len(toks)),
                                right_len=lenbin(toks[i+1]),left_last=toks[i][-1],target=toks[i+1][0]))
        for a,c in zip(ls,ls[1:]):
            if a["segment"]!=c["segment"]: continue
            if a["tokens"] and c["tokens"]:
                out.append(dict(block=b,boundary="LINE_BREAK",right_pos="FIRST",right_len=lenbin(c["tokens"][0]),
                                left_last=a["tokens"][-1][-1],target=c["tokens"][0][0]))
    return out

class JunctionModel:
    def __init__(self,pairs):
        chars=sorted({r["target"] for r in pairs}|{r["left_last"] for r in pairs});self.aset=set(chars);self.V=len(chars)+1
        self.c0=collections.Counter();self.cb=collections.defaultdict(collections.Counter);self.ce=collections.defaultdict(collections.Counter)
        for r in pairs:
            y=r["target"];self.c0[y]+=1
            kb=(r["boundary"],r["right_pos"],r["right_len"]);self.cb[kb][y]+=1
            self.ce[kb+(r["left_last"],)][y]+=1
    def p0(self,y):
        n=sum(self.c0.values());return (self.c0.get(y,0)+0.5)/(n+0.5*self.V)
    def pbase(self,r,y,a):
        c=self.cb[(r["boundary"],r["right_pos"],r["right_len"])];n=sum(c.values())
        return (c.get(y,0)+a*self.p0(y))/(n+a)
    def penh(self,r,y,a):
        kb=(r["boundary"],r["right_pos"],r["right_len"]);c=self.ce[kb+(r["left_last"],)];n=sum(c.values())
        return (c.get(y,0)+a*self.pbase(r,y,a))/(n+a)

def p2_boundary(lines,folds):
    pairs=junction_pairs(lines);out=[];counts=collections.Counter(r["boundary"] for r in pairs)
    per={a:{bt:collections.defaultdict(lambda:[0.0,0]) for bt in ("SPACE","LINE_BREAK")} for a in ALPHAS}
    for f in range(5):
        tr=[r for r in pairs if folds[r["block"]]!=f];te=[r for r in pairs if folds[r["block"]]==f]
        m=JunctionModel(tr)
        for r in te:
            for a in ALPHAS:
                pb=max(m.pbase(r,r["target"],a),1e-300);pe=max(m.penh(r,r["target"],a),1e-300)
                z=per[a][r["boundary"]][r["block"]];z[0]+=math.log2(pe/pb);z[1]+=1
    for a in ALPHAS:
        vals={}
        for bt in ("SPACE","LINE_BREAK"):
            d={b:s/n for b,(s,n) in per[a][bt].items() if n};vals[bt]=d
            st=signflip(list(d.values()));st.update(metric_id="P2",contrast=bt+"_EDGE_GAIN",alpha=a,pair_count=counts[bt]);out.append(st)
        common=sorted(set(vals["SPACE"])&set(vals["LINE_BREAK"]))
        st=signflip([vals["SPACE"][b]-vals["LINE_BREAK"][b] for b in common])
        st.update(metric_id="P2",contrast="SPACE_MINUS_LINEBREAK_GAIN",alpha=a,pair_count=None);out.append(st)
    return out

def transitions(lines):
    by=collections.defaultdict(list)
    for r in lines:by[r["block"]].append(r)
    out=[]
    for b,ls in by.items():
        ls=sorted(ls,key=lambda r:order_key(r["line_order"]));prev=None;prev_segment=None
        for r in ls:
            if prev_segment is not None and r["segment"]!=prev_segment: prev=None
            for j,t in enumerate(r["tokens"]):
                boundary="PAGE_START" if prev is None else ("LINE_BREAK" if j==0 else "SPACE")
                out.append(dict(block=b,boundary=boundary,prev=prev,target=t))
                prev=t
            prev_segment=r["segment"]
    return out

def morph(prev):
    if prev is None:return ("<START>","<START>",0)
    return (prev[:1],prev[-2:],len(prev))
def exact(prev):return "<START>" if prev is None else prev

class TokenParent:
    def __init__(self,rows):
        self.vocab=set(r["target"] for r in rows);self.V=len(self.vocab)+1
        self.globalc=collections.Counter(r["target"] for r in rows);self.N=sum(self.globalc.values())
        self.ctx=collections.defaultdict(collections.Counter)
        for r in rows:self.ctx[r["boundary"]][r["target"]]+=1
    def y(self,y):return y if y in self.vocab else "<UNK>"
    def gp(self,y):
        y=self.y(y);return (self.globalc.get(y,0)+0.5)/(self.N+0.5*self.V)
    def prob(self,r,y):
        y=self.y(y);c=self.ctx[r["boundary"]];n=sum(c.values())
        return (c.get(y,0)+PARENT_ALPHA*self.gp(y))/(n+PARENT_ALPHA)

class TokenChild:
    def y(self,y):
        return self.parent.y(y)
    def __init__(self,rows,keyfun,parent,lam):
        self.keyfun=keyfun;self.parent=parent;self.lam=lam
        self.c=collections.defaultdict(collections.Counter)
        for r in rows:self.c[(r["boundary"],keyfun(r["prev"]))][self.y(r["target"])]+=1
    def prob(self,r,y):
        yy=self.y(y);c=self.c[(r["boundary"],self.keyfun(r["prev"]))];n=sum(c.values())
        return (c.get(yy,0)+self.lam*self.parent.prob(r,y))/(n+self.lam)

def loss(model,rows):
    if not rows:return float("inf")
    return -sum(math.log2(max(model.prob(r,r["target"]),1e-300)) for r in rows)/len(rows)

def tune_morph(train,outer):
    ia=inner_assign([dict(block=r["block"],tokens=[r["target"]],unit=r["block"],line_order=0,writer=None) for r in train],outer)
    best=None
    for lam in LAMBDAS:
        ls=[]
        for f in range(4):
            tr=[r for r in train if ia[r["block"]]!=f];va=[r for r in train if ia[r["block"]]==f]
            if not va:continue
            p=TokenParent(tr);m=TokenChild(tr,morph,p,lam);ls.append(loss(m,va))
        x=float(np.mean(ls))
        if best is None or (x,lam)<best:best=(x,lam)
    return best[1]

def tune_exact(train,outer,lm):
    ia=inner_assign([dict(block=r["block"],tokens=[r["target"]],unit=r["block"],line_order=0,writer=None) for r in train],outer)
    best=None
    for lam in LAMBDAS:
        ls=[]
        for f in range(4):
            tr=[r for r in train if ia[r["block"]]!=f];va=[r for r in train if ia[r["block"]]==f]
            if not va:continue
            p=TokenParent(tr);m=TokenChild(tr,morph,p,lm);e=TokenChild(tr,exact,m,lam);ls.append(loss(e,va))
        x=float(np.mean(ls))
        if best is None or (x,lam)<best:best=(x,lam)
    return best[1]

def p34_morph_exact(lines,folds):
    rows=transitions(lines);mvals=[];evals=[];fold_rows=[]
    for outer in range(5):
        tr=[r for r in rows if folds[r["block"]]!=outer];te=[r for r in rows if folds[r["block"]]==outer]
        lm=tune_morph(tr,outer);le=tune_exact(tr,outer,lm)
        p=TokenParent(tr);m=TokenChild(tr,morph,p,lm);e=TokenChild(tr,exact,m,le)
        by=collections.defaultdict(lambda:[0.0,0.0,0])
        for r in te:
            pp=max(p.prob(r,r["target"]),1e-300);pm=max(m.prob(r,r["target"]),1e-300);pe=max(e.prob(r,r["target"]),1e-300)
            z=by[r["block"]];z[0]+=math.log2(pm/pp);z[1]+=math.log2(pe/pm);z[2]+=1
        mb=[v[0]/v[2] for v in by.values() if v[2]];eb=[v[1]/v[2] for v in by.values() if v[2]]
        mvals.extend(mb);evals.extend(eb)
        fold_rows.append(dict(fold=outer,n_test=len(te),lambda_morph=lm,lambda_exact=le,
                              morph_effect=float(np.mean(mb)) if mb else None,
                              exact_effect=float(np.mean(eb)) if eb else None))
    ms=signflip(mvals);ms.update(metric_id="P3",contrast="PREV_MORPH_GAIN")
    es=signflip(evals);es.update(metric_id="P4",contrast="EXACT_AFTER_MORPH_GAIN",
                                exact_off_folds=sum(r["lambda_exact"]>=16384 for r in fold_rows))
    return ms,es,fold_rows

def page_sequences(lines):
    by=collections.defaultdict(list)
    for r in lines:by[r["block"]].append(r)
    out={}
    for b,ls in by.items():
        seq=[]
        for r in sorted(ls,key=lambda x:order_key(x["line_order"])):seq.extend(r["tokens"])
        if seq:out[b]=seq
    return out

def band_rate(ids,lo,hi):
    n=len(ids);eligible=max(0,n-lo)
    if eligible==0:return None
    # Exact vectorized equivalent of: any(ids[max(0,i-hi):i-lo+1] == ids[i])
    # for target positions i=lo..n-1. Sentinel is outside encoded token IDs.
    w=hi-lo+1
    padded=np.concatenate((np.full(hi,-1,dtype=ids.dtype),ids))
    windows=np.lib.stride_tricks.sliding_window_view(padded,w)
    prev=windows[lo:n]
    target=ids[lo:n]
    return float(np.any(prev==target[:,None],axis=1).mean())

def p5_recurrence(lines,nperm=P5_NPERM,seed=20260920):
    seqs=page_sequences(lines);rng=np.random.default_rng(seed);out=[]
    encoded={}
    for b,s in seqs.items():
        mp={t:i for i,t in enumerate(sorted(set(s)))};encoded[b]=np.asarray([mp[t] for t in s],dtype=np.int32)
    for lo,hi in P5_BANDS:
        eligible={b:a for b,a in encoded.items() if len(a)>lo}
        actual={b:band_rate(a,lo,hi) for b,a in eligible.items()}
        reps=np.zeros(nperm,float);page_null={b:[] for b in eligible}
        for k in range(nperm):
            vals=[]
            for b,a in eligible.items():
                q=rng.permutation(a);r=band_rate(q,lo,hi);vals.append(r);page_null[b].append(r)
            reps[k]=float(np.mean(vals)) if vals else np.nan
        actual_mean=float(np.mean(list(actual.values()))) if actual else None
        null_mean=float(np.nanmean(reps)) if len(reps) else None
        null_sd=float(np.nanstd(reps,ddof=1)) if len(reps)>1 else None
        effect=(actual_mean-null_mean) if actual_mean is not None else None
        out.append(dict(metric_id="P5",contrast=f"LAG_{lo}_{hi}",n_blocks=len(actual),nperm=nperm,
                        observed=actual_mean,null_mean=null_mean,null_sd=null_sd,effect=effect,
                        effect_over_null_sd=(abs(effect)/null_sd if null_sd and null_sd>0 else None),
                        positive_blocks=sum(actual[b]>float(np.mean(page_null[b])) for b in actual),
                        negative_blocks=sum(actual[b]<float(np.mean(page_null[b])) for b in actual)))
    return out

def run(obj):
    lines=normalize_input(obj);folds=assign_folds(lines)
    res=dict(version=VERSION,label=obj.get("label"),input_sha256=canon_sha(obj),n_lines=len(lines),
             n_tokens=sum(len(r["tokens"]) for r in lines),n_blocks=len(set(r["block"] for r in lines)),
             fold_counts=collections.Counter(folds.values()),P1=None,P2=None,P3=None,P4=None,P5=None)
    res["P1"]=p1_memory(lines,folds)
    res["P2"]=p2_boundary(lines,folds)
    p3,p4,fr=p34_morph_exact(lines,folds);res["P3"]=p3;res["P4"]=p4;res["P34_folds"]=fr
    res["P5"]=p5_recurrence(lines)
    payload=json.dumps(res,sort_keys=True,separators=(",",":"),ensure_ascii=False)
    res["result_sha256"]=hashlib.sha256(payload.encode()).hexdigest()
    return res

def main():
    ap=argparse.ArgumentParser();ap.add_argument("input");ap.add_argument("--output")
    a=ap.parse_args();obj=json.load(open(a.input,encoding="utf-8"));res=run(obj)
    s=json.dumps(res,ensure_ascii=False,sort_keys=True)
    if a.output:open(a.output,"w",encoding="utf-8").write(s)
    print("XD1_RESULT="+s)

if __name__=="__main__":main()
