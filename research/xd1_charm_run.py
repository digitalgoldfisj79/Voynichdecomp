#!/usr/bin/env python3
"""Run frozen XD1-CHARM recurrence test on CUL Add.9308 diplomatic physical lines."""
import json,re,html,urllib.request,hashlib,unicodedata,random,math,statistics
from collections import Counter
from pathlib import Path

BASE="https://services.cudl.lib.cam.ac.uk"; ITEM="MS-ADD-09308"
UA={"User-Agent":"Voynichdecomp-XD1-CHARM-outcome/1.0"}
NPERM=2000

def get(u):
    req=urllib.request.Request(u,headers=UA)
    with urllib.request.urlopen(req,timeout=60) as r:return r.read()

def clean_html(s):
    s=re.sub(r"<script\b.*?</script>","",s,flags=re.I|re.S)
    s=re.sub(r"<style\b.*?</style>","",s,flags=re.I|re.S)
    return s

def textify(s):
    s=re.sub(r"<[^>]+>"," ",s)
    return re.sub(r"\s+"," ",html.unescape(s)).strip()

def parse_lines(s):
    s=clean_html(s)
    ms=list(re.finditer(r"<br\b([^>]*)>",s,re.I));out=[]
    for i,m in enumerate(ms):
        end=ms[i+1].start() if i+1<len(ms) else len(s)
        attrs=m.group(1); frag=s[m.end():end]
        xid=re.search(r"(?:xml:id|id)=['\"]([^'\"]+)['\"]",attrs,re.I)
        pts=re.search(r"data-points=['\"]([^'\"]+)['\"]",attrs,re.I)
        if not pts: continue
        t=textify(frag)
        assert "querySelectorAll" not in t and "const paragraphs" not in t, t
        out.append({"line_index":len(out)+1,"line_id":xid.group(1) if xid else None,
                    "geometry":pts.group(1),"text":t})
    return out

def tokenize(s):
    # Frozen XD1 semantics: Unicode NFC/lower, L/M/N characters inside tokens, require >=1 letter.
    s=unicodedata.normalize("NFC",s).lower()
    toks=[];cur=[]
    def flush():
        nonlocal cur
        if cur:
            z="".join(cur)
            if any(unicodedata.category(c).startswith("L") for c in z): toks.append(z)
            cur=[]
    for c in s:
        cat=unicodedata.category(c)
        if cat[0] in ("L","M","N"): cur.append(c)
        else: flush()
    flush()
    return toks

def analytic_null(lines,d):
    num=den=0.0
    for toks in lines:
        n=len(toks); opp=max(n-d,0)
        if opp<=0: continue
        den+=opp
        c=Counter(toks)
        p=sum(v*(v-1) for v in c.values())/(n*(n-1)) if n>1 else 0
        num+=opp*p
    return num/den if den else float("nan")

def observed(lines,d):
    hits=opp=0
    for toks in lines:
        for i in range(d,len(toks)):
            opp+=1; hits += (toks[i]==toks[i-d])
    return hits/opp if opp else float("nan"),hits,opp

def seed_for(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],"big")

def metric(lines,d,label,nperm=NPERM):
    obs,hits,opp=observed(lines,d)
    ana=analytic_null(lines,d)
    rng=random.Random(seed_for(label+f"|d{d}|n{nperm}"))
    vals=[]
    for _ in range(nperm):
        h=o=0
        for toks in lines:
            n=len(toks)
            if n<=d: continue
            x=toks.copy(); rng.shuffle(x)
            o += n-d
            h += sum(x[i]==x[i-d] for i in range(d,n))
        vals.append(h/o if o else 0.0)
    mean=statistics.fmean(vals); sd=statistics.pstdev(vals)
    eff=obs-mean
    z=eff/sd if sd else (math.inf if eff>0 else -math.inf if eff<0 else 0.0)
    ratio=obs/mean if mean>0 else (math.inf if obs>0 else 1.0)
    mc_se=sd/math.sqrt(nperm) if nperm else float("nan")
    return {"lag":d,"lines":len(lines),"tokens":sum(map(len,lines)),"observed_hits":hits,
            "opportunities":opp,"observed_rate":obs,"analytic_null_mean":ana,
            "null_mean":mean,"null_sd":sd,"effect":eff,"signed_effect_null_sd":z,
            "observed_null_ratio":ratio,"mc_mean_minus_analytic":mean-ana,
            "mc_mean_se":mc_se,"permutations":nperm}

def score_cohort(lineobjs,label,nperm=NPERM):
    lines=[x["tokens"] for x in lineobjs if len(x["tokens"])>=2]
    return {"label":label,"physical_lines_total":len(lineobjs),
            "physical_lines_ge2tokens":len(lines),
            "lag1":metric(lines,1,label,nperm),"lag2":metric(lines,2,label,nperm)}

def ranges_to_lines(pages,ranges):
    out=[]
    for fol,a,b in ranges:
        if fol not in pages: raise RuntimeError(f"missing folio {fol}")
        by={x["line_index"]:x for x in pages[fol]}
        for i in range(a,b+1):
            if i not in by: raise RuntimeError(f"missing {fol}:{i}")
            z=dict(by[i]);z["folio"]=fol;out.append(z)
    return out

def gate(m):
    a=m["lag1"];b=m["lag2"]
    return (a["observed_null_ratio"]>=0.90 and a["signed_effect_null_sd"]>-2.0 and
            b["observed_null_ratio"]>=1.10 and b["signed_effect_null_sd"]>=2.0)

def bootstrap_units(units,n=2000,label="primary"):
    rng=random.Random(seed_for("bootstrap|"+label))
    names=list(units); out={1:[],2:[]}
    for _ in range(n):
        samp=[rng.choice(names) for __ in names]
        lines=[]
        for k in samp: lines.extend(units[k])
        for d in (1,2):
            toks=[x["tokens"] for x in lines if len(x["tokens"])>=2]
            obs,_,_=observed(toks,d); nul=analytic_null(toks,d)
            out[d].append({"effect":obs-nul,"ratio":obs/nul if nul>0 else math.inf})
    def q(v,p):
        v=sorted(v); x=(len(v)-1)*p; lo=int(x);hi=min(lo+1,len(v)-1);f=x-lo
        return v[lo]*(1-f)+v[hi]*f
    return {f"lag{d}":{
      "effect_ci95":[q([x["effect"] for x in out[d]],.025),q([x["effect"] for x in out[d]],.975)],
      "ratio_ci95":[q([x["ratio"] for x in out[d]],.025),q([x["ratio"] for x in out[d]],.975)]
    } for d in (1,2)}

def main():
    manifest=json.load(open("research/XD1_CHARM_MANIFEST_20260928.json"))
    meta=json.loads(get(f"{BASE}/v1/metadata/json/{ITEM}"))
    # Fetch main compilation pages through 89r plus any selected charm page.
    wanted=set()
    for u in manifest["primary_units"]:
        wanted|={r[0] for r in u["ranges"]}
    pages={}
    for p in meta["pages"]:
        label=p.get("label");u=p.get("transcriptionDiplomaticURL")
        if not u: continue
        # all Arabic folios 1v..89r for descriptive background; every manifest folio regardless
        mm=re.fullmatch(r"(\d+)([rv])",str(label or ""))
        in_main=bool(mm and 1<=int(mm.group(1))<=89)
        if not (in_main or label in wanted): continue
        s=get(BASE+u).decode("utf-8","replace")
        ls=parse_lines(s)
        for x in ls:x["tokens"]=tokenize(x["text"])
        pages[label]=ls
    # Parser regression: known page-final lines must not contain service JS.
    assert pages["14v"][15]["text"].endswith("tempere")
    assert pages["22v"][15]["text"].endswith("ech")
    # Build units.
    unit_lines={}
    unit_results={}
    for u in manifest["primary_units"]:
        ls=ranges_to_lines(pages,u["ranges"])
        unit_lines[u["id"]]=ls
        unit_results[u["id"]]={"status":u["status"],"ranges":u["ranges"],
                               "metrics":score_cohort(ls,"UNIT|"+u["id"])}
    primary=[x for k in unit_lines for x in unit_lines[k]]
    strict=[x for k in manifest["strict_explicit_subset"] for x in unit_lines[k]]
    primary_metrics=score_cohort(primary,"PRIMARY_14")
    strict_metrics=score_cohort(strict,"STRICT_EXPLICIT_10")
    # Leave-one-unit-out for primary population; full 2k null each, frozen requirement.
    loo={}
    for omit in unit_lines:
        ls=[x for k in unit_lines if k!=omit for x in unit_lines[k]]
        m=score_cohort(ls,"LOO|"+omit)
        m["downgrade_gate_fires"]=gate(m)
        loo[omit]=m
    # Descriptive medical background (not pure recipe).
    selected={(r[0],i) for u in manifest["primary_units"] for r in u["ranges"] for i in range(r[1],r[2]+1)}
    mixed={(f,i) for f,i in manifest["excluded_mixed_boundary_lines"]}
    bg=[]
    for fol,ls in pages.items():
        mm=re.fullmatch(r"(\d+)([rv])",str(fol))
        if not (mm and 1<=int(mm.group(1))<=89): continue
        # 1r is poem/front intro; main compilation is catalogued 1v-89r.
        if fol=="1r": continue
        for x in ls:
            key=(fol,x["line_index"])
            if key in selected or key in mixed: continue
            z=dict(x);z["folio"]=fol;bg.append(z)
    background_metrics=score_cohort(bg,"ADD9308_MEDICAL_BACKGROUND")
    # length strata for primary and strict
    def strata(ls,prefix):
        bins={"2-5":lambda n:2<=n<=5,"6-10":lambda n:6<=n<=10,
              "11-20":lambda n:11<=n<=20,"21+":lambda n:n>=21}
        o={}
        for name,fn in bins.items():
            sub=[x for x in ls if fn(len(x["tokens"]))]
            o[name]=score_cohort(sub,prefix+"|"+name) if sub else None
        return o
    result={
      "protocol":"XD1-CHARM-20260928-v1.1",
      "source_item":manifest["item"],"source_item_id":ITEM,
      "manifest_sha256":hashlib.sha256(open("research/XD1_CHARM_MANIFEST_20260928.json","rb").read()).hexdigest(),
      "nperm":NPERM,
      "primary":primary_metrics,
      "strict_explicit":strict_metrics,
      "primary_downgrade_gate_fires":gate(primary_metrics),
      "strict_downgrade_gate_fires":gate(strict_metrics),
      "leave_one_unit_out":loo,
      "leave_one_out_all_fire":all(v["downgrade_gate_fires"] for v in loo.values()),
      "unit_results":unit_results,
      "bootstrap_primary_units":bootstrap_units(unit_lines,2000,"primary14"),
      "length_strata_primary":strata(primary,"PRIMARY14"),
      "length_strata_strict":strata(strict,"STRICT10"),
      "secondary_medical_background":background_metrics,
      "frozen_gate":manifest["downgrade_gate"],
      "power_counts":{"independent_units":len(unit_lines),
                      "lag2_opportunities":primary_metrics["lag2"]["opportunities"],
                      "minimum_units":manifest["downgrade_gate"]["minimum_independent_units"],
                      "minimum_lag2_opportunities":manifest["downgrade_gate"]["minimum_lag2_opportunities"]}
    }
    result["power_gate_passes"]=(result["power_counts"]["independent_units"]>=result["power_counts"]["minimum_units"] and
                                 result["power_counts"]["lag2_opportunities"]>=result["power_counts"]["minimum_lag2_opportunities"])
    Path("xd1_charm_outcome").mkdir(exist_ok=True)
    Path("xd1_charm_outcome/xd1_charm_result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    # compact TSV
    with open("xd1_charm_outcome/summary.tsv","w") as f:
        f.write("cohort\tlag\tlines\ttokens\topportunities\tobserved\tnull_mean\teffect\tnull_sd\teffect_sd\tobs_null\n")
        for name,m in [("PRIMARY_14",primary_metrics),("STRICT_10",strict_metrics),("MEDICAL_BACKGROUND",background_metrics)]:
            for lag in ("lag1","lag2"):
                z=m[lag];f.write(f"{name}\t{lag}\t{z['lines']}\t{z['tokens']}\t{z['opportunities']}\t{z['observed_rate']:.12g}\t{z['null_mean']:.12g}\t{z['effect']:.12g}\t{z['null_sd']:.12g}\t{z['signed_effect_null_sd']:.6f}\t{z['observed_null_ratio']:.6f}\n")
    print(json.dumps({
      "power_gate_passes":result["power_gate_passes"],
      "primary_gate_fires":result["primary_downgrade_gate_fires"],
      "strict_gate_fires":result["strict_downgrade_gate_fires"],
      "loo_all_fire":result["leave_one_out_all_fire"],
      "primary_lag1":{"z":primary_metrics["lag1"]["signed_effect_null_sd"],"ratio":primary_metrics["lag1"]["observed_null_ratio"]},
      "primary_lag2":{"z":primary_metrics["lag2"]["signed_effect_null_sd"],"ratio":primary_metrics["lag2"]["observed_null_ratio"]},
      "strict_lag1":{"z":strict_metrics["lag1"]["signed_effect_null_sd"],"ratio":strict_metrics["lag1"]["observed_null_ratio"]},
      "strict_lag2":{"z":strict_metrics["lag2"]["signed_effect_null_sd"],"ratio":strict_metrics["lag2"]["observed_null_ratio"]},
    },indent=2))
if __name__=="__main__":main()
