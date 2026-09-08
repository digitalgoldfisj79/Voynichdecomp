#!/usr/bin/env python3
"""La Sfera Stage-2: test frozen visual-neighbour sets against independent textual/toponym traditions.

Frozen before textual inspection (from Voynich f68r visual test):
  left/middle pair: Laur2, Yale4
  right set: Fn12, He1, He2, NYPL2, Spe, Par4, Barb4, Cap1, Urb2, Vat3

Uses the public La Sfera Django fixture. Primary text metric uses curated StanzaVariant
records when available; a full-stanza exact-reading metric is reported as robustness.
Toponym metric uses LocationAlias manuscript readings. Lower distance = closer.
Nulls are coverage-matched random pairs/groups. No post-hoc witness substitution.
"""
from __future__ import annotations
import json, math, os, random, re, statistics, sys, urllib.request
from collections import defaultdict, Counter
from html import unescape
from pathlib import Path

URL = "https://raw.githubusercontent.com/chnm/lasfera/main/manuscript/fixtures/lasfera_all_data.json"
SEED = 20260908
PAIR = ("Laur2", "Yale4")
RIGHT_FROZEN = ("Fn12", "He1", "He2", "NYPL2", "Spe", "Par4", "Barb4", "Cap1", "Urb2", "Vat3")
N_GROUP_NULL = 100000


def norm(s):
    if s is None: return ""
    s = unescape(str(s))
    s = re.sub(r"<[^>]+>", " ", s)
    s = s.lower().replace("’", "'").replace("‘", "'")
    s = re.sub(r"[^\wà-ÿ']+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def mean(xs): return sum(xs)/len(xs) if xs else float("nan")
def sd(xs): return statistics.stdev(xs) if len(xs)>1 else float("nan")
def z_closer(obs, null):
    m, s = mean(null), sd(null)
    return (m-obs)/s if s and not math.isnan(s) else float("nan")
def pct_le(obs, null):
    return (sum(x <= obs for x in null)+1)/(len(null)+1) if null else float("nan")


def get_model(by, suffix):
    for k in by:
        if k.endswith(suffix): return by[k]
    return []


def fkey(v):
    # dumpdata FK is normally int; tolerate natural-key singleton/list
    if isinstance(v, list) and len(v)==1: return v[0]
    return v


def pair_distance(readings, coverage, a, b, min_common=1):
    loci = coverage.get(a,set()) & coverage.get(b,set())
    if len(loci) < min_common: return None
    mism=0
    for locus in loci:
        ra = readings.get(a,{}).get(locus, ("__BASE__",))
        rb = readings.get(b,{}).get(locus, ("__BASE__",))
        if ra != rb: mism += 1
    return mism/len(loci), len(loci)


def build_pair_null(readings, coverage, target_pair, min_common=1, tol=.15):
    a,b=target_pair
    target=pair_distance(readings,coverage,a,b,min_common)
    if not target: return None
    obs, cov=target
    mss=sorted(coverage)
    allp=[]
    for i,x in enumerate(mss):
        for y in mss[i+1:]:
            d=pair_distance(readings,coverage,x,y,min_common)
            if d: allp.append((d[0],d[1],x,y))
    if not allp: return None
    lo,hi=cov*(1-tol),cov*(1+tol)
    null=[d for d,c,_,_ in allp if lo<=c<=hi and {_,} is not None]
    # previous comprehension intentionally only selects d/c; if narrow, broaden
    if len(null)<30:
        lo,hi=cov*.70,cov*1.30
        null=[d for d,c,x,y in allp if lo<=c<=hi]
    return {"observed":obs,"common_loci":cov,"null":null,"all_pairs":allp,
            "null_mean":mean(null),"null_sd":sd(null),"z_closer":z_closer(obs,null),
            "empirical_p_lower":pct_le(obs,null),"n_null":len(null)}


def group_stat(readings, coverage, group, min_common=1):
    vals=[]; covs=[]; missing=[]
    for i,a in enumerate(group):
        for b in group[i+1:]:
            r=pair_distance(readings,coverage,a,b,min_common)
            if r: vals.append(r[0]); covs.append(r[1])
            else: missing.append((a,b))
    if not vals: return None
    return mean(vals), mean(covs), len(vals), missing


def group_null(readings, coverage, group, min_common=1, rng=None):
    rng=rng or random.Random(SEED)
    obs=group_stat(readings,coverage,group,min_common)
    if not obs: return None
    obs_d,obs_cov,n_pairs,missing=obs
    mss=sorted(coverage)
    k=len(group)
    if len(mss)<k: return None
    candidates=[]
    # sample broadly then coverage-match to target mean overlap and valid-pair count
    for _ in range(N_GROUP_NULL):
        g=rng.sample(mss,k)
        s=group_stat(readings,coverage,g,min_common)
        if not s: continue
        d,c,npairs,_=s
        if npairs < max(1, int(.90*n_pairs)): continue
        candidates.append((d,c,npairs))
    if not candidates: return None
    for tol in (.10,.20,.35,.60):
        lo,hi=obs_cov*(1-tol),obs_cov*(1+tol)
        null=[d for d,c,n in candidates if lo<=c<=hi]
        if len(null)>=100: break
    return {"observed":obs_d,"mean_pair_common_loci":obs_cov,"valid_pairs":n_pairs,
            "missing_pairs":[list(x) for x in missing],"null":null,
            "null_mean":mean(null),"null_sd":sd(null),"z_closer":z_closer(obs_d,null),
            "empirical_p_lower":pct_le(obs_d,null),"n_null":len(null)}


def summarize_test(x):
    if not x: return None
    return {k:(round(v,6) if isinstance(v,float) and math.isfinite(v) else v)
            for k,v in x.items() if k not in ("null","all_pairs")}


def main():
    outdir=Path(sys.argv[1] if len(sys.argv)>1 else "research_out")
    outdir.mkdir(parents=True,exist_ok=True)
    datafile=outdir/"lasfera_all_data.json"
    if not datafile.exists():
        print("Downloading fixture...", flush=True)
        urllib.request.urlretrieve(URL,datafile)
    records=json.loads(datafile.read_text(encoding="utf-8"))
    by=defaultdict(list)
    for r in records: by[r["model"]].append(r)
    counts={k:len(v) for k,v in sorted(by.items())}
    print("MODEL COUNTS",json.dumps(counts,indent=2),flush=True)

    msrecs=get_model(by,"singlemanuscript")
    folrecs=get_model(by,"folio")
    strecs=get_model(by,"stanza")
    varrecs=get_model(by,"stanzavariant")
    locrecs=get_model(by,"location")
    aliasrecs=get_model(by,"locationalias")
    print("Selected counts",len(msrecs),len(folrecs),len(strecs),len(varrecs),len(locrecs),len(aliasrecs),flush=True)

    pk_to_sig={r["pk"]:r["fields"].get("siglum") for r in msrecs if r["fields"].get("siglum")}
    sig_to_pk={v:k for k,v in pk_to_sig.items()}
    folio_to_ms={r["pk"]:pk_to_sig.get(fkey(r["fields"].get("manuscript"))) for r in folrecs}
    # stanza -> manuscripts via folio.stanzas and stanza.folios, tolerating either schema direction
    stanza_to_mss=defaultdict(set)
    for fr in folrecs:
        sig=folio_to_ms.get(fr["pk"])
        if not sig: continue
        for spk in fr["fields"].get("stanzas",[]) or []: stanza_to_mss[fkey(spk)].add(sig)
    folio_ids={r["pk"] for r in folrecs}
    for sr in strecs:
        fs=sr["fields"].get("folios",[]) or []
        for fpk in fs:
            sig=folio_to_ms.get(fkey(fpk))
            if sig: stanza_to_mss[sr["pk"]].add(sig)
        # older schema possible direct related_manuscript
        rm=sr["fields"].get("related_manuscript")
        if rm:
            sig=pk_to_sig.get(fkey(rm))
            if sig: stanza_to_mss[sr["pk"]].add(sig)

    stanza_by_pk={r["pk"]:r for r in strecs}
    # Coverage by manuscript at curated variant loci: a locus is covered if corresponding stanza exists in that MS.
    all_var_loci=set()
    var_locus_stanza_prefix={}
    for vr in varrecs:
        code=vr["fields"].get("stanza_variation_line_code_starts")
        if code:
            all_var_loci.add(code)
            # strip terminal variant letter, then book+stanza identify containing stanza
            base=re.sub(r"[a-z]$","",code)
            parts=base.split(".")
            var_locus_stanza_prefix[code]=".".join(parts[:2])

    ms_stanza_prefixes=defaultdict(set)
    full_text=defaultdict(dict)
    for sr in strecs:
        code=sr["fields"].get("stanza_line_code_starts")
        if not code: continue
        prefix=".".join(code.split(".")[:2])
        text=norm(sr["fields"].get("stanza_text"))
        for sig in stanza_to_mss.get(sr["pk"],[]):
            ms_stanza_prefixes[sig].add(prefix)
            # exact stanza code as robustness reading; if duplicates, aggregate deterministically
            full_text[sig].setdefault(code,[]).append(text)
    full_text={s:{c:tuple(sorted(set(v))) for c,v in d.items()} for s,d in full_text.items()}
    full_cov={s:set(d) for s,d in full_text.items()}

    var_read_temp=defaultdict(lambda:defaultdict(list))
    mapped_var=0
    for vr in varrecs:
        f=vr["fields"]; code=f.get("stanza_variation_line_code_starts"); spk=fkey(f.get("stanza"))
        if not code or not spk: continue
        reading=norm(f.get("stanza_variation")) or "__EMPTYVAR__"
        for sig in stanza_to_mss.get(spk,[]):
            var_read_temp[sig][code].append(reading); mapped_var+=1
    var_read={s:{c:tuple(sorted(set(v))) for c,v in d.items()} for s,d in var_read_temp.items()}
    var_cov={}
    for sig,prefs in ms_stanza_prefixes.items():
        loci={loc for loc,pref in var_locus_stanza_prefix.items() if pref in prefs}
        if loci: var_cov[sig]=loci

    # Toponym aliases by location; infer manuscript from M2M manuscripts or folios.
    topo_temp=defaultdict(lambda:defaultdict(list))
    mapped_alias=0
    for ar in aliasrecs:
        f=ar["fields"]; loc=fkey(f.get("location")); label=norm(f.get("placename_alias") or f.get("placename_from_mss"))
        if not loc or not label: continue
        sigs=set()
        for mpk in f.get("manuscripts",[]) or []:
            sig=pk_to_sig.get(fkey(mpk))
            if sig: sigs.add(sig)
        for fpk in f.get("folios",[]) or []:
            sig=folio_to_ms.get(fkey(fpk))
            if sig: sigs.add(sig)
        # tolerate singular legacy fields
        if f.get("manuscript"):
            sig=pk_to_sig.get(fkey(f.get("manuscript")))
            if sig: sigs.add(sig)
        if f.get("folio"):
            sig=folio_to_ms.get(fkey(f.get("folio")))
            if sig: sigs.add(sig)
        for sig in sigs:
            topo_temp[sig][str(loc)].append(label); mapped_alias+=1
    topo_read={s:{l:tuple(sorted(set(v))) for l,v in d.items()} for s,d in topo_temp.items()}
    topo_cov={s:set(d) for s,d in topo_read.items() if d}

    # Restrict matrices to manuscripts with meaningful coverage. Keep frozen witnesses if present.
    def eligible(cov, min_n): return {s:c for s,c in cov.items() if len(c)>=min_n}
    # Curated variants: choose threshold adaptively, never based on target distance.
    vc_sizes=sorted(len(x) for x in var_cov.values())
    vmin=max(10, int(statistics.median(vc_sizes)*.25)) if vc_sizes else 10
    v_elig=eligible(var_cov,vmin)
    # full text threshold
    fc_sizes=sorted(len(x) for x in full_cov.values())
    fmin=max(10, int(statistics.median(fc_sizes)*.25)) if fc_sizes else 10
    f_elig=eligible(full_cov,fmin)
    # topo threshold low because common-place overlap is sparser
    tc_sizes=sorted(len(x) for x in topo_cov.values())
    tmin=max(3, int(statistics.median(tc_sizes)*.20)) if tc_sizes else 3
    t_elig=eligible(topo_cov,tmin)

    # Frozen sets intersect available/eligible data only; do not substitute missing witnesses.
    right_var=[s for s in RIGHT_FROZEN if s in v_elig]
    right_full=[s for s in RIGHT_FROZEN if s in f_elig]
    right_topo=[s for s in RIGHT_FROZEN if s in t_elig]
    missing={
      "variant":[s for s in RIGHT_FROZEN if s not in v_elig],
      "full_text":[s for s in RIGHT_FROZEN if s not in f_elig],
      "toponym":[s for s in RIGHT_FROZEN if s not in t_elig],
    }

    rng=random.Random(SEED)
    # Pair minimum common = 20% of median eligible locus count, bounded.
    v_common_min=max(5,int(statistics.median([len(c) for c in v_elig.values()])*.15)) if v_elig else 5
    f_common_min=max(5,int(statistics.median([len(c) for c in f_elig.values()])*.15)) if f_elig else 5
    t_common_min=max(2,int(statistics.median([len(c) for c in t_elig.values()])*.10)) if t_elig else 2

    var_pair=build_pair_null(var_read,v_elig,PAIR,v_common_min) if all(s in v_elig for s in PAIR) else None
    var_group=group_null(var_read,v_elig,right_var,v_common_min,rng) if len(right_var)>=3 else None
    full_pair=build_pair_null(full_text,f_elig,PAIR,f_common_min) if all(s in f_elig for s in PAIR) else None
    full_group=group_null(full_text,f_elig,right_full,f_common_min,rng) if len(right_full)>=3 else None
    topo_pair=build_pair_null(topo_read,t_elig,PAIR,t_common_min) if all(s in t_elig for s in PAIR) else None
    topo_group=group_null(topo_read,t_elig,right_topo,t_common_min,rng) if len(right_topo)>=3 else None

    # Nearest neighbours of each pair target in each independent matrix (descriptive, not selection).
    def nearest(readings,cov,target,min_common,n=10):
        arr=[]
        if target not in cov: return arr
        for s in cov:
            if s==target: continue
            d=pair_distance(readings,cov,target,s,min_common)
            if d: arr.append((d[0],d[1],s))
        return sorted(arr)[:n]

    result={
      "fixture_url":URL,
      "model_counts":counts,
      "mapping_diagnostics":{
        "manuscripts":len(pk_to_sig),"folios":len(folrecs),"stanzas":len(strecs),
        "stanza_variants":len(varrecs),"mapped_variant_assignments":mapped_var,
        "location_aliases":len(aliasrecs),"mapped_alias_assignments":mapped_alias,
        "variant_loci":len(all_var_loci),"variant_matrix_mss":len(var_cov),
        "full_text_matrix_mss":len(full_cov),"toponym_matrix_mss":len(topo_cov),
      },
      "frozen":{
        "left_middle_pair":list(PAIR),"right_set":list(RIGHT_FROZEN),
        "right_available_variant":right_var,"right_available_full_text":right_full,
        "right_available_toponym":right_topo,"missing_or_below_coverage":missing,
      },
      "thresholds":{
        "eligible_variant_loci":vmin,"eligible_full_stanzas":fmin,"eligible_toponyms":tmin,
        "pair_min_common_variant":v_common_min,"pair_min_common_full":f_common_min,
        "pair_min_common_toponym":t_common_min,
      },
      "primary_curated_text_variants":{
        "left_middle_pair":summarize_test(var_pair),
        "right_group":summarize_test(var_group),
        "Laur2_nearest":nearest(var_read,v_elig,"Laur2",v_common_min),
        "Yale4_nearest":nearest(var_read,v_elig,"Yale4",v_common_min),
      },
      "robustness_full_stanza_text":{
        "left_middle_pair":summarize_test(full_pair),
        "right_group":summarize_test(full_group),
      },
      "independent_toponym_variants":{
        "left_middle_pair":summarize_test(topo_pair),
        "right_group":summarize_test(topo_group),
      },
    }

    (outdir/"result.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    # Markdown summary, with automatic decision language.
    def decision(x):
        if not x: return "NOT TESTABLE from current data."
        z=x.get("z_closer")
        if z is None or not isinstance(z,(int,float)) or not math.isfinite(z): return "NOT RESOLVED."
        if z>=2: return f"RESOLVES as unusually textually tight ({z:.2f} null SD)."
        if z<=-2: return f"RESOLVES in the opposite direction: unusually dispersed ({z:.2f} null SD)."
        return f"THE METRIC DOES NOT RESOLVE ({z:.2f} null SD)."
    vp=result["primary_curated_text_variants"]["left_middle_pair"]
    vg=result["primary_curated_text_variants"]["right_group"]
    tp=result["independent_toponym_variants"]["left_middle_pair"]
    tg=result["independent_toponym_variants"]["right_group"]
    md=f"""# La Sfera Stage-2 textual/toponym tradition test\n\n## Frozen visual selections\n- Left/middle: Laur2 + Yale4\n- Right: {', '.join(RIGHT_FROZEN)}\n- No missing witness was replaced post hoc.\n\n## Primary curated textual-variant test\n- Laur2↔Yale4: **{decision(vp)}**\n- Right frozen set (available intersection): **{decision(vg)}**\n\n## Independent toponym-variant test\n- Laur2↔Yale4: **{decision(tp)}**\n- Right frozen set (available intersection): **{decision(tg)}**\n\n## Availability\n```json\n{json.dumps(result['frozen'],indent=2,ensure_ascii=False)}\n```\n\n## Statistics\n```json\n{json.dumps({'text_pair':vp,'text_group':vg,'toponym_pair':tp,'toponym_group':tg,'full_text_robustness':result['robustness_full_stanza_text'],'diagnostics':result['mapping_diagnostics'],'thresholds':result['thresholds']},indent=2,ensure_ascii=False)}\n```\n\nInterpretation rule was frozen before inspection: ≥2 null SD tighter is required for a visual-selected set to count as tracking an independent textual/toponym lineage.\n"""
    (outdir/"summary.md").write_text(md,encoding="utf-8")
    print(md,flush=True)

if __name__=="__main__": main()
