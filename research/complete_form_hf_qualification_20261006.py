#!/usr/bin/env python3
import argparse, base64, collections, gzip, hashlib, importlib.util, json, math, os, pathlib, re, sys, urllib.request
import numpy as np

CORPUS_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/19fc6f2262dc2d184b370fb7c6960f11278f4778/voynich_transcriptions_slim.json"
CORPUS_SHA="26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f"
EXPECTED={"n_lines":4117,"n_tokens":34229,"n_within":30112,"fold_tokens":{"0":6794,"1":6374,"2":8253,"3":6434,"4":6374}}

def load_harness(path):
    spec=importlib.util.spec_from_file_location("cfh",path)
    h=importlib.util.module_from_spec(spec); spec.loader.exec_module(h); return h

def load_json_b64(path):
    return json.loads(gzip.decompress(base64.b64decode(pathlib.Path(path).read_text().strip())))

def folio_number(f):
    m=re.match(r"f(\d+)",f)
    return int(m.group(1)) if m else None

def build_folio_fold(meta):
    by={}
    for k,v in meta["folds"].items():
        if k.startswith("UNPAIRED_"):
            by[k[len("UNPAIRED_"):]]=(k,int(v)); continue
        if k.startswith("B"):
            a,b=map(int,k[1:].split("_"))
            by[a]=(k,int(v)); by[b]=(k,int(v))
    return by

def section(f):
    n=folio_number(f)
    if n is None:return "UNK"
    if n<=66:return "HERBAL"
    if n<=73:return "ASTRO"
    if n<=84:return "BIO"
    if n<=102:return "PHARMA"
    if n<=116:return "RECIPES"
    return "UNK"

def recover_from_public(corpus_path,meta):
    obj=json.load(open(corpus_path))
    exact_unpaired={x for x in ("f13r","f13v","f73r","f73v")}
    fold_by=build_folio_fold(meta); starts={(f,int(n)) for f,n in meta["paragraph_starts"]}
    para_ctr=collections.Counter(); prevline={}; rows=[]; dropped=0; order=0
    for f,ld in obj["pages"].items():
        if f in exact_unpaired:
            ff=fold_by.get(f)
        else:
            n=folio_number(f); ff=fold_by.get(n)
        if ff is None: continue
        bif,fold=ff
        for ls,rec in ld.items():
            u=str(rec.get("u",""))
            if "P" not in u: continue
            try: ln=int(ls)
            except Exception: continue
            raw=str(rec.get("t",{}).get("ZLZI",""))
            toks=[]
            for t in raw.split():
                t=t.lower()
                if re.fullmatch(r"[a-z]+",t): toks.append(t)
                elif t: dropped+=1
            if not toks: continue
            if (f,ln) in starts or para_ctr[f]==0: para_ctr[f]+=1
            p=int(para_ctr[f])
            prev=prevline.get(f); prevop=None
            if prev is not None and prev["para"]==p and ln==prev["line"]+1 and prev["tokens"]:
                prevop=prev["tokens"][0][0]
            rr=dict(order=order,folio=f,line=ln,tokens=toks,section=section(f),
                    currier=meta["currier"].get(f,"UNK"),unit=u,para=p,
                    bifolium=bif,fold=int(fold),prev_line_opener=prevop)
            rows.append(rr); prevline[f]=rr; order+=1
    aud=dict(layer="ZLZI",n_lines=len(rows),n_tokens=sum(len(r["tokens"]) for r in rows),
             n_within=sum(max(0,len(r["tokens"])-1) for r in rows),dropped=dropped,
             fold_tokens={str(i):sum(len(r["tokens"]) for r in rows if r["fold"]==i) for i in range(5)})
    if aud["n_lines"]!=EXPECTED["n_lines"] or aud["n_tokens"]!=EXPECTED["n_tokens"] or aud["n_within"]!=EXPECTED["n_within"] or aud["fold_tokens"]!=EXPECTED["fold_tokens"]:
        raise RuntimeError("public reconstruction mismatch "+json.dumps(aud,sort_keys=True))
    return rows,aud

def compact(out):
    d={"seed":out["seed"],"audit":out["audit"],"results":{}}
    for r in out["results"]:
        d["results"][r["source"]]={
            "CB_all":r["test"]["C_minus_B"]["all"]["mean"],
            "CB_first":r["test"]["C_minus_B"]["first_unseen_distinct"]["mean"],
            "CS_all":r["test"]["C_minus_S"]["all"]["mean"],
            "CS_first":r["test"]["C_minus_S"]["first_unseen_distinct"]["mean"],
            "n_first":r["test"]["C_minus_B"]["first_unseen_distinct"]["n"],
            "selected":r["selected"],
            "shuffle_changed":r["shuffle_audit"]["changed_fraction"],
        }
    return d

def run_seed(args):
    cp=pathlib.Path(args.corpus)
    if cp.exists() and hashlib.sha256(cp.read_bytes()).hexdigest()==CORPUS_SHA:
        raw=cp.read_bytes()
    else:
        raw=urllib.request.urlopen(CORPUS_URL,timeout=120).read()
        if hashlib.sha256(raw).hexdigest()!=CORPUS_SHA: raise RuntimeError("corpus SHA mismatch")
        tmp=cp.with_suffix(".tmp"); tmp.write_bytes(raw); os.replace(tmp,cp)
    meta=load_json_b64(args.meta)
    h=load_harness(args.harness)
    rows,aud=recover_from_public(args.corpus,meta)
    def rec(layer="ZLZI"):
        if layer!="ZLZI": raise RuntimeError("qualification worker is ZLZI only")
        return rows,dict(aud)
    h.recover_population=rec
    out=h.pilot(args.seed,fast=False)
    c=compact(out)
    pathlib.Path(args.outdir).mkdir(parents=True,exist_ok=True)
    p=pathlib.Path(args.outdir)/f"qual_{args.seed}.json"
    tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(c,separators=(",",":"))); os.replace(tmp,p)
    print("QUAL_SEED="+json.dumps(c,separators=(",",":")),flush=True)

def summarize(args):
    rows=[]
    for p in sorted(pathlib.Path(args.outdir).glob("qual_*.json")):
        rows.append(json.loads(p.read_text()))
    if len(rows)<args.expect: raise RuntimeError(f"expected {args.expect}, found {len(rows)}")
    metrics=["CB_all","CB_first","CS_all","CS_first"]; sources=["A","B","BROKEN","C"]
    res={"n_seeds":len(rows),"seed_min":min(x["seed"] for x in rows),"seed_max":max(x["seed"] for x in rows),"metrics":{}}
    for m in metrics:
        null=np.array([x["results"][s][m] for x in rows for s in ["A","B","BROKEN"]],float)
        alt=np.array([x["results"]["C"][m] for x in rows],float)
        mu=float(null.mean()); sd=float(null.std(ddof=1)); q95=float(np.quantile(null,.95)); q99=float(np.quantile(null,.99))
        # empirical one-sided 5% threshold from pooled null; familywise calibration is reported separately below.
        res["metrics"][m]=dict(null_n=len(null),alt_n=len(alt),null_mean=mu,null_sd=sd,
            centered_effect=float(alt.mean()-mu),effect_over_null_sd=float((alt.mean()-mu)/(sd or 1)),
            null_q95=q95,null_q99=q99,false_alarm_q95=float(np.mean(null>q95)),
            power_q95=float(np.mean(alt>q95)),power_q99=float(np.mean(alt>q99)),
            alt_positive=float(np.mean(alt>0)),alt_mean=float(alt.mean()),alt_min=float(alt.min()),alt_max=float(alt.max()))
    # Familywise max-stat across four standardized metrics, using each null regime/seed as one replicate.
    mus={m:np.mean([x["results"][s][m] for x in rows for s in ["A","B","BROKEN"]]) for m in metrics}
    sds={m:np.std([x["results"][s][m] for x in rows for s in ["A","B","BROKEN"]],ddof=1) for m in metrics}
    nullmax=[]; altmax=[]
    for x in rows:
        for s in ["A","B","BROKEN"]:
            nullmax.append(max((x["results"][s][m]-mus[m])/(sds[m] or 1) for m in metrics))
        altmax.append(max((x["results"]["C"][m]-mus[m])/(sds[m] or 1) for m in metrics))
    thr=float(np.quantile(nullmax,.95))
    res["familywise"]=dict(null_replicates=len(nullmax),alt_replicates=len(altmax),maxstat_q95=thr,
        empirical_false_alarm=float(np.mean(np.array(nullmax)>thr)),alt_detection=float(np.mean(np.array(altmax)>thr)))
    print("QUAL_SUMMARY="+json.dumps(res,separators=(",",":")),flush=True)

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("seed"); p.add_argument("--seed",type=int,required=True); p.add_argument("--harness",required=True); p.add_argument("--meta",required=True); p.add_argument("--corpus",default="/tmp/voynich_slim.json"); p.add_argument("--outdir",default="/tmp/qual")
    p=sp.add_parser("summarize"); p.add_argument("--outdir",default="/tmp/qual"); p.add_argument("--expect",type=int,default=100)
    a=ap.parse_args(); run_seed(a) if a.cmd=="seed" else summarize(a)
if __name__=="__main__":main()
