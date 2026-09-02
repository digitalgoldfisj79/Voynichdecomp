#!/usr/bin/env python3
"""Fractionation-composition falsification gate v0.1.

Frozen decision rule BEFORE Voynich exposure:
  candidate = (internally calibrated max coherence Z >= 5.0) AND (best block b >= 2)
Synthetic gate must pass before any Voynich text is scored:
  planted TPR >= .90
  exact planted-b recovery >= .80
  positional-matched-null FPR <= .05
  f57v Scribal Manual FPR <= .05
  P70-C Scribal generator FPR <= .05

The internal null preserves exact token lengths and the empirical symbol
histogram at every (token-length, absolute-position) cell. It therefore
preserves positional/slot effects while destroying within-token coupling.

Important scope: word-reset row/column fractionation with block regrouping
b=2..8, optional bounded homophony/role overlap, and optional suffix nulls.
It is NOT a test of every conceivable fractionating cipher.
"""
from __future__ import annotations
import json, math, random, statistics, importlib.util, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
B_VALUES = tuple(range(1, 9))
Z_GATE = 5.0


def H(xs):
    n=len(xs)
    if not n: return 0.0
    c=Counter(xs)
    return -sum((v/n)*math.log2(v/n) for v in c.values())


def MI(xs, ys):
    n=len(xs)
    if not n: return 0.0
    cx,cy,cxy=Counter(xs),Counter(ys),Counter(zip(xs,ys))
    return sum((v/n)*math.log2((v*n)/(cx[a]*cy[b])) for (a,b),v in cxy.items())


def NMI(xs, ys):
    d=min(H(xs),H(ys))
    return MI(xs,ys)/d if d>1e-12 else 0.0


def features(tokens, b):
    syms=[]; phases=[]; rs=[]; cs=[]; u0=[]; u1=[]
    for tok0 in tokens:
        tok=list(tok0)
        # Frozen bounded-null model: one optional suffix null on odd tokens.
        if len(tok)%2: tok=tok[:-1]
        latent=[]; p=0
        while p < len(tok):
            rem=len(tok)-p
            clen=min(2*b, rem)
            if clen%2: clen-=1
            if clen<2: break
            half=clen//2
            rr=tok[p:p+half]; cc=tok[p+half:p+2*half]
            syms.extend(rr); syms.extend(cc)
            phases.extend([0]*half); phases.extend([1]*half)
            rs.extend(rr); cs.extend(cc)
            latent.extend(zip(rr,cc))
            p += 2*half
        for a,z in zip(latent,latent[1:]):
            u0.append(a); u1.append(z)
    phase=NMI(syms,phases)
    coord=NMI(rs,cs)
    bigram=NMI(u0,u1)
    return {"b":b,"phase":phase,"coord":coord,"bigram":bigram,
            "score":phase+coord+bigram}


def scan(tokens):
    rows=[features(tokens,b) for b in B_VALUES]
    mx=max(r["score"] for r in rows)
    # Decision-rule audit: smallest b wins numerical ties.
    tied=[r for r in rows if r["score"] >= mx-1e-10]
    best=min(tied,key=lambda r:r["b"])
    return best, rows


def positional_shuffle(tokens, rng):
    # Preserve exact token length and each absolute-position symbol histogram.
    out=[list(t) for t in tokens]
    groups=defaultdict(list)
    for ti,t in enumerate(tokens):
        for p,s in enumerate(t): groups[(len(t),p)].append((ti,s))
    for (L,p), vals in groups.items():
        ss=[s for _,s in vals]; rng.shuffle(ss)
        for (ti,_),s in zip(vals,ss): out[ti][p]=s
    return out


def calibrated(tokens, seed, nsh=8):
    obs,_=scan(tokens)
    rng=random.Random(seed)
    ns=[]
    for _ in range(nsh):
        sh=positional_shuffle(tokens,rng)
        b,_=scan(sh); ns.append(b["score"])
    mu=statistics.mean(ns)
    sd=statistics.stdev(ns) if len(ns)>1 else 0.0
    z=(obs["score"]-mu)/sd if sd>1e-12 else float("inf")
    return {**obs,"null_mean":mu,"null_sd":sd,"z":z,
            "decision": bool(z>=Z_GATE and obs["b"]>=2)}


# ---------- synthetic positive family ----------
def weighted_choice(rng, weights):
    x=rng.random()*sum(weights); s=0.0
    for i,w in enumerate(weights):
        s+=w
        if x<=s:return i
    return len(weights)-1


def markov_words(seed, n=500, K=25, maxL=14):
    rng=random.Random(seed)
    # Uneven unigram plus symbol-specific successor boosts.
    base=[rng.gammavariate(.55,1.0) for _ in range(K)]
    sb=sum(base); base=[x/sb for x in base]
    trans=[]
    for i in range(K):
        row=[.45*x for x in base]
        fav=rng.sample(range(K),5)
        for j in fav: row[j]+=rng.uniform(.04,.16)
        sr=sum(row); trans.append([x/sr for x in row])
    lens=list(range(2,maxL+1)); lw=[math.exp(-.22*(L-2)) for L in lens]
    words=[]
    for _ in range(n):
        L=lens[weighted_choice(rng,lw)]
        w=[weighted_choice(rng,base)]
        for _ in range(L-1): w.append(weighted_choice(rng,trans[w[-1]]))
        words.append(w)
    return words


def make_maps(rng, overlap=.0, hom=1):
    rpool=list(range(12)); cpool=list(range(12,24))
    ov=int(round(overlap*6))
    if ov: cpool=c_pool=list(range(12,24-ov))+list(range(6,6+ov))
    def alloc(pool):
        first=rng.sample(pool,5); out=[]
        for f in first:
            vals=[f]
            while len(vals)<hom:
                q=rng.choice(pool)
                if q not in vals: vals.append(q)
            out.append(vals)
        return out
    return alloc(rpool),alloc(cpool)


def frac_encode(words, seed, b, overlap=0.0, hom=1, nullp=0.0):
    rng=random.Random(seed)
    perm=list(range(25)); rng.shuffle(perm)
    rc=[(perm[u]//5,perm[u]%5) for u in range(25)]
    rm,cm=make_maps(rng,overlap,hom)
    out=[]
    for w in words:
        t=[]
        for st in range(0,len(w),b):
            co=[rc[u] for u in w[st:st+b]]
            t.extend(rng.choice(rm[r]) for r,c in co)
            t.extend(rng.choice(cm[c]) for r,c in co)
        if rng.random()<nullp: t.append(rng.randrange(24))
        out.append(t)
    return out


def run_synthetics():
    variants={"clean":(0.0,1,0.0),"hom":(.35,2,0.0),"null":(.35,2,.08)}
    positives=[]; matched=[]
    case=0
    for vname,(ov,hom,np_) in variants.items():
        for j in range(10):
            seed=10000+case*101; true_b=2+((j+3*case)%7)
            words=markov_words(seed,n=500)
            tok=frac_encode(words,seed+1,true_b,ov,hom,np_)
            r=calibrated(tok,seed+2,nsh=8)
            positives.append({"variant":vname,"true_b":true_b,**r})
            # Explicit matched negative on first 4 cases per variant.
            if j<4:
                q=positional_shuffle(tok,random.Random(seed+3))
                qr=calibrated(q,seed+4,nsh=8)
                matched.append({"source_variant":vname,**qr})
            case+=1
    return positives,matched


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
    spec.loader.exec_module(mod); return mod


def fast_control(mod, spec, producer, ncases, n_tokens, seed0, label):
    rows=[]
    for i in range(ncases):
        toks=producer(spec,n_tokens=n_tokens,seed=seed0+i)
        toks=[list(t) for t in toks if t]
        best,_=scan(toks)
        if best["b"]>=2:
            c=calibrated(toks,seed0+5000+i,nsh=8)
        else:
            c={**best,"z":None,"decision":False,"null_mean":None,"null_sd":None}
        rows.append({"control":label,"seed":seed0+i,**c})
    return rows


def get_tokens(data,tid):
    out=[]
    for fid,lines in data.get("pages",{}).items():
        for lnum,line in lines.items():
            txt=line.get("t",{}).get(tid,"")
            if txt: out.extend(txt.split())
    return out


def collapse_benches(token):
    # Alternative glyph representation; longer EVA bench forms first.
    reps=[("ckh","Ḱ"),("cth","Ṱ"),("cph","Ṕ"),("cfh","Ḟ"),("ch","Č"),("sh","Š")]
    for a,b in reps: token=token.replace(a,b)
    return token


def clean_tokens(tokens):
    # Slim normalized text is expected to be clean. Remove only empty/control tokens;
    # do not tune a glyph whitelist from Voynich.
    return [list(t) for t in tokens if t and not t.startswith("<")]


def main():
    results={"protocol":{"z_gate":Z_GATE,"b_gate":2,"B_values":list(B_VALUES),
                         "tie_break":"smallest b within 1e-10 of max",
                         "voynich_sealed_until_gate":True}}
    positives,matched=run_synthetics()
    tpr=sum(r["decision"] for r in positives)/len(positives)
    brec=sum(r["b"]==r["true_b"] for r in positives)/len(positives)
    mfpr=sum(r["decision"] for r in matched)/len(matched)
    results["synthetic"]={"positives":positives,"matched_nulls":matched,
                          "tpr":tpr,"exact_b_recovery":brec,"matched_null_fpr":mfpr}

    # Exact existing f57v Scribal Manual control.
    g7=load_module("g7",ROOT/"Paper/Generators/gen_scribal_manual.py")
    slim_path=ROOT/"voynich_transcriptions_slim.json"
    g7spec=g7.load_f57v_spec(str(slim_path))
    g7rows=fast_control(g7,g7spec,g7.produce_manuscript,50,600,30000,"G7_f57v_scribal")
    g7fpr=sum(r["decision"] for r in g7rows)/len(g7rows)
    results["g7"]={"rows":g7rows,"fpr":g7fpr}

    # Rich P70-C slot/copy-mutate generator control.
    gp=load_module("gp70c",ROOT/"Paper/Generators/gen_scribal_p70c.py")
    p70spec=gp.build_p70c_spec(str(ROOT/"Paper/p70c_full_spec_v1.json"),
                               str(ROOT/"Paper/enriched_records.pkl"))
    p70rows=fast_control(gp,p70spec,gp.produce_manuscript,30,600,40000,"P70C_scribal")
    p70fpr=sum(r["decision"] for r in p70rows)/len(p70rows)
    results["p70c"]={"rows":p70rows,"fpr":p70fpr}

    gate=(tpr>=.90 and brec>=.80 and mfpr<=.05 and g7fpr<=.05 and p70fpr<=.05)
    results["gate_passed"]=gate

    if gate:
        data=json.loads(slim_path.read_text())
        vms={}
        for tid in ("ZLZI","TTLI"):
            raw=get_tokens(data,tid)
            if not raw: continue
            for rep,vals in [("raw",raw),("bench_collapsed",[collapse_benches(t) for t in raw])]:
                toks=clean_tokens(vals)
                c=calibrated(toks,seed=50000+len(vms),nsh=16)
                c["n_tokens"]=len(toks)
                c["odd_rate"]=sum(len(t)%2 for t in toks)/len(toks)
                vms[f"{tid}_{rep}"]=c
        results["voynich_exposed"]=True
        results["voynich"]=vms
    else:
        results["voynich_exposed"]=False
        results["voynich"]={}

    (OUT/"fractionation_gate_v01.json").write_text(json.dumps(results,indent=2,allow_nan=False))

    lines=["# Fractionation-composition gate v0.1",""]
    lines += [f"- Synthetic TPR: {tpr:.3f}",f"- Exact planted-b recovery: {brec:.3f}",
              f"- Position-matched-null FPR: {mfpr:.3f}",f"- f57v Scribal Manual FPR: {g7fpr:.3f}",
              f"- P70-C Scribal FPR: {p70fpr:.3f}",f"- **Gate passed: {gate}**",""]
    if gate:
        lines.append("## Frozen Voynich exposure")
        for k,r in results["voynich"].items():
            lines.append(f"- {k}: b={r['b']}, Z={r['z']:.3f}, score={r['score']:.6f}, null SD={r['null_sd']:.6f}, decision={r['decision']}, odd-rate={r['odd_rate']:.3f}, n={r['n_tokens']}")
    else:
        lines.append("Voynich remained sealed because the synthetic/control gate failed.")
    (OUT/"FRACTIONATION_GATE_V01_RESULT.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))

if __name__=="__main__": main()
