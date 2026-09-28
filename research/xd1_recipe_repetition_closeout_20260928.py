#!/usr/bin/env python3
"""
XD1-CLOSEOUT-20260928 historical recipe-register lag-1/lag-2 control.
Protocol: research/PROTOCOL_XD1_closeout_20260928.md

Two-stage by design:
  --preflight fetches/parses sources, performs QC only, and writes checkpoint_A_sources.pkl.
  --run refuses to fetch or tune sources; it consumes the frozen checkpoint and computes metrics.

No repetition statistic is computed during --preflight.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, pickle, re, sys, unicodedata, urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=ROOT/"research"/"xd1_closeout_results_20260928"
OUT.mkdir(exist_ok=True)
CHECKPOINT_A=OUT/"checkpoint_A_sources.pkl"
CHECKPOINT_B=OUT/"checkpoint_B_metrics.pkl"
CHECKPOINT_C=OUT/"checkpoint_C_adjudication.pkl"

PROTOCOL_ID="XD1-CLOSEOUT-20260928"
PRIMARY=("A1","BS1","SO1","W1")
SECONDARY=("A1B_LATIN","BS2","KA1","KO1","W2")
DESCRIPTIVE=("KA3",)

# Counts and folio scopes are pre-existing SG89 source-manifest metadata,
# inspected before this protocol's repetition outcomes.
WIT={
 "A1": dict(pid="o:corema.a1", lines=799,tokens=9606,first="59v",last="70r",role="PRIMARY"),
 "A1B_LATIN": dict(pid="o:corema.a1", lines=232,tokens=2814,first="64r",last="67r",role="SECONDARY",
                    first_line=9,last_line=5),
 "BS1":dict(pid="o:corema.bs1",lines=3563,tokens=24534,first="17r",last="108v",role="PRIMARY"),
 "BS2":dict(pid="o:corema.bs2",lines=1316,tokens=5782,first="300r",last="310v",role="SECONDARY"),
 "KA1":dict(pid="o:corema.ka1",lines=752,tokens=6665,first="108r",last="120r",role="SECONDARY"),
 "KA3":dict(pid="o:corema.ka3",lines=156,tokens=1682,first="11r",last="14v",role="DESCRIPTIVE"),
 "KO1":dict(pid="o:corema.ko1",lines=264,tokens=3397,first="24v",last="97v",role="SECONDARY"),
 "SO1":dict(pid="o:corema.so1",lines=1767,tokens=15491,first="1r",last="30v",role="PRIMARY"),
 "W1": dict(pid="o:corema.w1",lines=1914,tokens=20836,first="1r",last="29v",role="PRIMARY"),
 "W2": dict(pid="o:corema.w2",lines=483,tokens=2238,first="77v",last="80r",role="SECONDARY"),
}
BINS=(("ALL",2,10**9),("2-5",2,5),("6-10",6,10),("11-20",11,20),("21+",21,10**9))
LAGS=(1,2)
NPERM_PRIMARY=200
NPERM_SENS=2000
NBOOT=2000
BASE_SEED=20260928

def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha256_obj(x)->str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def local(tag): return tag.split("}")[-1]
def nfc(s): return unicodedata.normalize("NFC",str(s))

def tokenize(text):
    text=nfc(text).replace("\n"," ")
    out=[];buf=[]
    for ch in text:
        cat=unicodedata.category(ch)
        if cat[0] in ("L","M") or cat=="Nd":
            buf.append(ch.lower())
        else:
            if buf: out.append("".join(buf));buf=[]
    if buf:out.append("".join(buf))
    return [t for t in out if any(unicodedata.category(ch).startswith("L") for ch in t)]

def req(url):
    r=urllib.request.Request(url,headers={"User-Agent":"XD1-CLOSEOUT-20260928/1.0"})
    with urllib.request.urlopen(r,timeout=90) as f:
        return f.read()

def fetch_tei(pid):
    urls=[
      f"https://gams.uni-graz.at/archive/objects/{pid}/datastreams/TEI_SOURCE/content",
      f"https://gams.uni-graz.at/{pid}/TEI_SOURCE",
    ]
    errs=[]
    for u in urls:
        try:
            b=req(u)
            # hard source gate: XML, not an HTML error or metadata page
            ET.fromstring(b)
            if b"<TEI" not in b and b":TEI" not in b[:1000]:
                # namespace TEI roots may not literally contain <TEI in first bytes;
                # parsed root name is the final check.
                root=ET.fromstring(b)
                if local(root.tag)!="TEI": raise ValueError("root is not TEI")
            return u,b
        except Exception as e:
            errs.append(f"{u}: {type(e).__name__}: {e}")
    raise RuntimeError("TEI fetch failed; "+" | ".join(errs))

def folio_key(s):
    if s is None:return None
    m=re.search(r"(\d+)\s*([rv])",str(s).lower())
    if not m:return None
    return int(m.group(1))*2+(1 if m.group(2)=="v" else 0)

def line_num(s, fallback):
    if s is None:return fallback
    m=re.search(r"(\d+)",str(s))
    return int(m.group(1)) if m else fallback

SKIP={"teiHeader","note","fw","head","del","supplied","figDesc","desc"}
CHOICE_PREF=("abbr","orig","sic")

def parse_physical_lines(xml_bytes):
    root=ET.fromstring(xml_bytes)
    text=next((x for x in root.iter() if local(x.tag)=="text"),None)
    if text is None: raise RuntimeError("TEI has no <text>")
    state={"page":None,"line":None,"seq":0,"buf":[]}
    rows=[]

    def flush():
        if state["page"] is None or state["line"] is None:return
        s="".join(state["buf"]).strip()
        toks=tokenize(s)
        if toks:
            rows.append({"folio":state["page"],"line_order":state["line"],"tokens":toks})
        state["buf"]=[]

    def append(txt):
        if txt and state["page"] is not None and state["line"] is not None:
            state["buf"].append(txt)

    def walk(el):
        tag=local(el.tag)
        if tag in SKIP:
            return
        if tag=="pb":
            flush(); state["page"]=el.attrib.get("n") or el.attrib.get("facs") or el.attrib.get("{http://www.w3.org/XML/1998/namespace}id")
            state["line"]=None; state["seq"]=0
            return
        if tag=="lb":
            flush(); state["seq"]+=1; state["line"]=line_num(el.attrib.get("n"),state["seq"])
            return
        if tag=="choice":
            chosen=None
            for want in CHOICE_PREF:
                chosen=next((c for c in el if local(c.tag)==want),None)
                if chosen is not None:break
            if chosen is None and len(el): chosen=el[0]
            if chosen is not None:
                append("".join(chosen.itertext()))
            else:
                append(el.text)
            return
        if tag in ("reg","corr","expan"):
            # Only use these when they are not an alternative inside <choice>.
            append("".join(el.itertext()))
            return
        append(el.text)
        for ch in el:
            walk(ch)
            append(ch.tail)

    walk(text); flush()
    # stable de-duplication in case the TEI carries duplicated presentational branches
    seen=set();out=[]
    for r in rows:
        k=(str(r["folio"]),int(r["line_order"])," ".join(r["tokens"]))
        if k not in seen: seen.add(k);out.append(r)
    return out

def scoped(rows,spec,wid):
    a,b=folio_key(spec["first"]),folio_key(spec["last"])
    out=[]
    for r in rows:
        fk=folio_key(r["folio"])
        if fk is None or fk<a or fk>b:continue
        # A1B is a source-only embedded Latin block with line-qualified endpoints.
        if wid=="A1B_LATIN":
            if fk==a and r["line_order"]<spec["first_line"]:continue
            if fk==b and r["line_order"]>spec["last_line"]:continue
        out.append(r)
    return out

def source_preflight():
    bundle={"protocol_id":PROTOCOL_ID,"stage":"SOURCE_PREFLIGHT","witnesses":{}}
    failures=[]
    for wid,spec in WIT.items():
        url,b=fetch_tei(spec["pid"])
        allrows=parse_physical_lines(b)
        rows=scoped(allrows,spec,wid)
        nlines=len(rows); ntok=sum(len(r["tokens"]) for r in rows)
        line_dev=abs(nlines-spec["lines"])/spec["lines"]
        tok_dev=abs(ntok-spec["tokens"])/spec["tokens"]
        gate=(line_dev<=0.03 and tok_dev<=0.10) if spec["role"]=="PRIMARY" else True
        rec=dict(witness_id=wid,pid=spec["pid"],role=spec["role"],url=url,
                 source_sha256=sha256_bytes(b),n_lines=nlines,n_tokens=ntok,
                 prior_manifest_lines=spec["lines"],prior_manifest_tokens=spec["tokens"],
                 line_rel_deviation=line_dev,token_rel_deviation=tok_dev,qc_pass=gate,
                 lines=rows)
        bundle["witnesses"][wid]=rec
        if spec["role"]=="PRIMARY" and not gate: failures.append(wid)
    # Store source extraction before any repetition result exists.
    with open(CHECKPOINT_A,"wb") as f: pickle.dump(bundle,f,protocol=5)
    public={k:{kk:vv for kk,vv in v.items() if kk!="lines"} for k,v in bundle["witnesses"].items()}
    result={"protocol_id":PROTOCOL_ID,"stage":"SOURCE_PREFLIGHT","primary_failures":failures,"witnesses":public}
    result["checkpoint_sha256"]=sha256_bytes(CHECKPOINT_A.read_bytes())
    (OUT/"source_qc.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    print("SOURCE_QC="+json.dumps(result,sort_keys=True))
    if failures: raise SystemExit(3)

def encode(toks):
    mp={t:i for i,t in enumerate(sorted(set(toks)))}
    return np.asarray([mp[t] for t in toks],dtype=np.int32)

def analytic_line(toks,lag):
    n=len(toks)
    if n<=lag:return None
    opp=n-lag
    hits=sum(toks[i]==toks[i-lag] for i in range(lag,n))
    c=collections.Counter(toks)
    p=sum(v*(v-1) for v in c.values())/(n*(n-1))
    return hits,opp,opp*p

def metric(lines,lag,nperm,seed,lo=2,hi=10**9):
    use=[r["tokens"] if isinstance(r,dict) else r for r in lines if lo<=len(r["tokens"] if isinstance(r,dict) else r)<=hi]
    use=[x for x in use if len(x)>lag]
    if not use:return None
    rng=np.random.default_rng(seed)
    obs_hits=0;opp=0;analytic_exp=0.0
    null_hits=np.zeros(nperm,dtype=np.int64)
    for toks in use:
        h,o,e=analytic_line(toks,lag);obs_hits+=h;opp+=o;analytic_exp+=e
        a=encode(toks)
        # vectorized batches keep peak memory bounded for long lines.
        for k in range(nperm):
            q=rng.permutation(a)
            null_hits[k]+=int(np.sum(q[lag:]==q[:-lag]))
    obs=obs_hits/opp
    nr=null_hits/opp
    nm=float(nr.mean()); sd=float(nr.std(ddof=1)); eff=obs-nm
    an=analytic_exp/opp
    return dict(lag=lag,n_lines=len(use),opportunities=opp,observed_hits=obs_hits,
                observed=obs,null_mean=nm,null_sd=sd,effect=eff,
                effect_over_null_sd=(abs(eff)/sd if sd>0 else None),
                observed_over_null=(obs/nm if nm>0 else None),
                analytic_null_mean=an,observed_over_analytic_null=(obs/an if an>0 else None),
                mc_minus_analytic=nm-an,nperm=nperm)

def all_bins(lines,lag,nperm,seed):
    out={}
    for j,(name,lo,hi) in enumerate(BINS):
        out[name]=metric(lines,lag,nperm,seed+10000*j,lo,hi)
    return out

def bootstrap(lines,lag,seed,nboot=NBOOT):
    # Stratified physical-line bootstrap using frozen bins excluding the ALL pseudo-bin.
    bins=[]
    for name,lo,hi in BINS[1:]:
        arr=[]
        for r in lines:
            toks=r["tokens"] if isinstance(r,dict) else r
            if lo<=len(toks)<=hi and len(toks)>lag:
                h,o,e=analytic_line(toks,lag);arr.append((h,o,e))
        if arr: bins.append(np.asarray(arr,float))
    rng=np.random.default_rng(seed)
    vals=np.empty((nboot,2),float) # effect, ratio
    for b in range(nboot):
        H=O=E=0.0
        for arr in bins:
            idx=rng.integers(0,len(arr),size=len(arr))
            z=arr[idx].sum(axis=0);H+=z[0];O+=z[1];E+=z[2]
        obs=H/O; null=E/O
        vals[b]=[obs-null, obs/null if null>0 else np.nan]
    eff=vals[:,0];rat=vals[:,1]
    return dict(nboot=nboot,effect_mean=float(np.nanmean(eff)),effect_sd=float(np.nanstd(eff,ddof=1)),
                effect_ci95=[float(x) for x in np.nanpercentile(eff,[2.5,97.5])],
                ratio_mean=float(np.nanmean(rat)),ratio_sd=float(np.nanstd(rat,ddof=1)),
                ratio_ci95=[float(x) for x in np.nanpercentile(rat,[2.5,97.5])])

def vms_lines():
    sys.path.insert(0,str(HERE))
    import xd1_within_line_primary_fast_20260920 as old
    sha,lines=old.local_vms(ROOT)
    return sha,[{"folio":"VMS","line_order":i,"tokens":x} for i,x in enumerate(lines)]

def standardize_to_vms(vms_lines_,control_lines,lag):
    # Analytic-null effect standardized to VMS opportunity mass in frozen bins.
    def per(lines):
        d={}
        for name,lo,hi in BINS[1:]:
            z=metric(lines,lag,1,BASE_SEED+911,lo,hi)
            if z:d[name]=z
        return d
    v=per(vms_lines_);c=per(control_lines)
    common=[n for n,_,__ in BINS[1:] if n in v and n in c and v[n]["n_lines"]>=50 and c[n]["n_lines"]>=50]
    if not common:return None
    denom=sum(v[n]["opportunities"] for n in common)
    weights={n:v[n]["opportunities"]/denom for n in common}
    ce=0.0;ve=0.0
    for n in common:
        # use analytic expectation for standardization to avoid MC noise.
        ve+=weights[n]*(v[n]["observed"]-v[n]["analytic_null_mean"])
        ce+=weights[n]*(c[n]["observed"]-c[n]["analytic_null_mean"])
    return dict(common_bins=common,weights=weights,vms_effect=ve,control_effect=ce,difference=ve-ce)

def run_metrics():
    if not CHECKPOINT_A.exists(): raise SystemExit("missing checkpoint_A_sources.pkl; run --preflight first")
    bundle=pickle.loads(CHECKPOINT_A.read_bytes())
    failures=[w for w in PRIMARY if not bundle["witnesses"][w]["qc_pass"]]
    if failures: raise SystemExit("primary source QC failed: "+",".join(failures))
    vsha,vlines=vms_lines()
    corp={"VMS":dict(role="TARGET",lines=vlines,source_sha256=vsha)}
    for wid,r in bundle["witnesses"].items():
        corp[wid]=dict(role=r["role"],lines=r["lines"],source_sha256=r["source_sha256"])
    res={"protocol_id":PROTOCOL_ID,"stage":"METRICS","vms_source_sha256":vsha,"corpora":{}}
    order=["VMS"]+list(WIT)
    for idx,wid in enumerate(order):
        r=corp[wid]; lines=r["lines"]; rr={"role":r["role"],"source_sha256":r["source_sha256"],
             "n_lines":len(lines),"n_tokens":sum(len(x["tokens"]) for x in lines),"lags":{}}
        for lag in LAGS:
            base=all_bins(lines,lag,NPERM_PRIMARY,BASE_SEED+idx*100000+lag*1000)
            boot=bootstrap(lines,lag,BASE_SEED+idx*100000+lag*1000+77)
            entry={"primary_200":base,"bootstrap":boot}
            if wid=="VMS" or wid in PRIMARY:
                entry["sensitivity_2000_ALL"]=metric(lines,lag,NPERM_SENS,BASE_SEED+idx*100000+lag*1000+333)
            rr["lags"][str(lag)]=entry
        if wid!="VMS":
            rr["vms_standardized"]={str(lag):standardize_to_vms(vlines,lines,lag) for lag in LAGS}
        res["corpora"][wid]=rr
    with open(CHECKPOINT_B,"wb") as f:pickle.dump(res,f,protocol=5)
    res["checkpoint_sha256"]=sha256_bytes(CHECKPOINT_B.read_bytes())
    (OUT/"metrics.json").write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding="utf-8")
    print("METRICS_SHA="+res["checkpoint_sha256"])
    adjudicate(res)

def adjudicate(res):
    v=res["corpora"]["VMS"]
    rows=[]; falsifiers=[]
    for wid in PRIMARY:
        c=res["corpora"][wid]
        l1=c["lags"]["1"]["sensitivity_2000_ALL"];l2=c["lags"]["2"]["sensitivity_2000_ALL"]
        fires=bool(l1["observed_over_null"]>=0.90 and l2["observed_over_null"]>=1.10)
        if fires:falsifiers.append(wid)
        vb=v["lags"]["2"]["bootstrap"];cb=c["lags"]["2"]["bootstrap"]
        diff=v["lags"]["2"]["primary_200"]["ALL"]["effect"]-c["lags"]["2"]["primary_200"]["ALL"]["effect"]
        pooled=math.sqrt(vb["effect_sd"]**2+cb["effect_sd"]**2)
        z=abs(diff)/pooled if pooled>0 else None
        vpoint=v["lags"]["2"]["primary_200"]["ALL"]["observed_over_null"]
        outside=not (cb["ratio_ci95"][0]<=vpoint<=cb["ratio_ci95"][1])
        # formal length-strata opposition
        strata={}
        for bn in ("6-10","11-20"):
            vv=v["lags"]["2"]["primary_200"][bn];cc=c["lags"]["2"]["primary_200"][bn]
            formal=bool(vv and cc and vv["n_lines"]>=50 and cc["n_lines"]>=50)
            strata[bn]=dict(formal=formal,
                vms_ratio=None if not vv else vv["observed_over_null"],
                control_ratio=None if not cc else cc["observed_over_null"],
                opposite=(formal and vv["effect"]>0 and cc["effect"]<=0))
        rows.append(dict(control=wid,falsifier_fires=fires,lag1_ratio=l1["observed_over_null"],
                         lag2_ratio=l2["observed_over_null"],lag2_effect_difference=diff,
                         pooled_bootstrap_sd=pooled,difference_over_pooled_bootstrap_sd=z,
                         vms_lag2_ratio=vpoint,control_lag2_ratio_ci95=cb["ratio_ci95"],
                         vms_point_outside_control_ci95=outside,strata=strata))
    v2=v["lags"]["2"]["sensitivity_2000_ALL"]
    criterion1=not falsifiers
    criterion2=(v2["effect_over_null_sd"] is not None and v2["effect_over_null_sd"]>=2)
    criterion3=all(r["difference_over_pooled_bootstrap_sd"] is not None and r["difference_over_pooled_bootstrap_sd"]>=2
                   and r["vms_point_outside_control_ci95"] for r in rows)
    formal_strata=[s for r in rows for s in r["strata"].values() if s["formal"]]
    criterion4=bool(formal_strata) and all(s["opposite"] for s in formal_strata)
    # criterion 5 (independent representation) is deliberately unresolved here.
    decision=dict(protocol_id=PROTOCOL_ID,stage="ADJUDICATION",
      frozen_falsifier_primary_controls=falsifiers,
      criterion1_no_recipe_falsifier=criterion1,
      criterion2_vms_lag2_resolved=criterion2,
      criterion3_all_primary_cross_corpus_bootstrap_resolved=criterion3,
      criterion4_formal_length_strata_opposition=criterion4,
      criterion5_independent_representation="PENDING_OR_BLOCKED",
      promote_to_kernel=False,
      promotion_reason="Promotion prohibited until independent non-EVA representation criterion 5 is passed.",
      comparisons=rows)
    with open(CHECKPOINT_C,"wb") as f:pickle.dump(decision,f,protocol=5)
    decision["checkpoint_sha256"]=sha256_bytes(CHECKPOINT_C.read_bytes())
    (OUT/"adjudication.json").write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding="utf-8")
    print("ADJUDICATION="+json.dumps(decision,sort_keys=True))

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preflight",action="store_true");g.add_argument("--run",action="store_true")
    a=ap.parse_args()
    if a.preflight:source_preflight()
    else:run_metrics()
if __name__=="__main__":main()
