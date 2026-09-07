#!/usr/bin/env python3
"""Voynich topology marginalisation v0.1.

FROZEN BEFORE TARGET OUTCOMES.

Purpose: audit canonical text measurements for dependence on (a) the folio as
unit and (b) the current codex order, after codicological work showed that the
bifolium is often a production-functional unit and current nested-quire order
is not safe as universal original production topology.

Hard stop rules:
- Never optimise an order to improve a Voynich statistic.
- Missing bifolia receive no imputed text.
- |effect|/null SD < 2 => "the metric does not resolve this".
- A topology-dependence headline requires >=3/4 ZL/RF/IT/GC transcriptions at
  |z|>=2 in the same direction.
- R64 metrics here are topology diagnostics (exact recurrence within 64), NOT
  a reproduction of canonical WorkingSetV2+R64 predictive codelength.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import requests

PROTOCOL = "vms_topology_marginal_20260907_v01"
RUN_ID = os.environ.get("TOPO_RUN_ID", "topomarg_v01_20260907")
OUT = Path(os.environ.get("TOPO_OUT", "artifacts/vms_topology_marginal_v01"))
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20260907
N_ORDER = int(os.environ.get("TOPO_N_ORDER", "512"))
N_PAIR = int(os.environ.get("TOPO_N_PAIR", "5000"))
N_ADJ = int(os.environ.get("TOPO_N_ADJ", "5000"))

CORPORA = {
    "ZL": ("https://www.voynich.nu/data/ZL3b-n.txt", "bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc"),
    "RF": ("https://www.voynich.nu/data/RF1b-er.txt", "eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2"),
    "IT": ("https://www.voynich.nu/data/IT2a-n.txt", "7f27a8b0feed8f6de0a99900df6bf912dd1d295c38e5f830bac8b41c3f536fb5"),
    "GC": ("https://www.voynich.nu/data/GC2a-n.txt", "b09570cb6c993bc2d87134d115e60a978650a8a6495483ddbb1f6005a586096f"),
}

# Explicit conjoint map frozen independently from text scoring.
UNITS = [
("q01_b1_8","q01",1,8),("q01_b2_7","q01",2,7),("q01_b3_6","q01",3,6),("q01_b4_5","q01",4,5),
("q02_b9_16","q02",9,16),("q02_b10_15","q02",10,15),("q02_b11_14","q02",11,14),("q02_b12_13","q02",12,13),
("q03_b17_24","q03",17,24),("q03_b18_23","q03",18,23),("q03_b19_22","q03",19,22),("q03_b20_21","q03",20,21),
("q04_b25_32","q04",25,32),("q04_b26_31","q04",26,31),("q04_b27_30","q04",27,30),("q04_b28_29","q04",28,29),
("q05_b33_40","q05",33,40),("q05_b34_39","q05",34,39),("q05_b35_38","q05",35,38),("q05_b36_37","q05",36,37),
("q06_b41_48","q06",41,48),("q06_b42_47","q06",42,47),("q06_b43_46","q06",43,46),("q06_b44_45","q06",44,45),
("q07_b49_56","q07",49,56),("q07_b50_55","q07",50,55),("q07_b51_54","q07",51,54),("q07_b52_53","q07",52,53),
("q08_b57_66","q08",57,66),("q08_b58_65","q08",58,65),("q08_b59_64_missing","q08",59,64),("q08_b60_63_missing","q08",60,63),("q08_b61_62_missing","q08",61,62),
("q09_b67_68","q09",67,68),("q10_b69_70","q10",69,70),("q11_b71_72","q11",71,72),("q12_b73_74","q12",73,74),
("q13_b75_84","q13",75,84),("q13_b76_83","q13",76,83),("q13_b77_82","q13",77,82),("q13_b78_81","q13",78,81),("q13_b79_80","q13",79,80),
("q14_b85_86","q14",85,86),("q15_b87_90","q15",87,90),("q15_b88_89","q15",88,89),("q16_b91_92_missing","q16",91,92),
("q17_b93_96","q17",93,96),("q17_b94_95","q17",94,95),("q18_b97_98_missing","q18",97,98),("q19_b99_102","q19",99,102),("q19_b100_101","q19",100,101),
("q20_b103_116","q20",103,116),("q20_b104_115","q20",104,115),("q20_b105_114","q20",105,114),("q20_b106_113","q20",106,113),("q20_b107_112","q20",107,112),("q20_b108_111","q20",108,111),("q20_b109_110_missing","q20",109,110),
]

PAGE_RE = re.compile(r"^<(f\d+[rv]\d*)>\s*(?:<!\s*(.*?)>)?")
LINE_RE = re.compile(r"^<(f\d+[rv]\d*)\.\d+,[^>]*>\s*(.*)$")
META_RE = re.compile(r"\$(\w)=([^\s>]+)")
ALT_RE = re.compile(r"\[([^:\]]+):[^\]]+\]")
BRACE_RE = re.compile(r"\{([^}]*)\}")
TAG_RE = re.compile(r"<[^>]*>")
AT_RE = re.compile(r"@\d+;")


def fetch_text(url: str, expected_sha: str) -> str:
    r = requests.get(url, timeout=90, headers={"User-Agent":"VoynichTopologyMarginal/0.1"})
    r.raise_for_status()
    raw = r.content
    got = hashlib.sha256(raw).hexdigest()
    if got != expected_sha:
        raise RuntimeError(f"hash mismatch {url}: {got} != {expected_sha}")
    return raw.decode("utf-8", errors="replace")


def base_folio(page: str) -> int:
    m = re.match(r"f(\d+)", page)
    if not m:
        raise ValueError(page)
    return int(m.group(1))


def clean_tokens(s: str) -> List[str]:
    # Preserve text content, remove editorial apparatus. Alternatives choose the
    # left reading deterministically; this rule is frozen for all transcriptions.
    s = s.replace("<->", ".")
    s = ALT_RE.sub(lambda m: m.group(1), s)
    s = BRACE_RE.sub(lambda m: m.group(1), s)
    s = AT_RE.sub("", s)
    s = TAG_RE.sub("", s)
    out = []
    for piece in re.split(r"[\s.,;:/|]+", s.lower()):
        t = re.sub(r"[^a-z]", "", piece)
        if t:
            out.append(t)
    return out


def parse_ivtff(body: str):
    folio_tokens: Dict[int,List[str]] = defaultdict(list)
    page_meta: Dict[str,dict] = {}
    page_tokens: Dict[str,List[str]] = defaultdict(list)
    for raw in body.splitlines():
        if not raw or raw.startswith("#"):
            continue
        pm = PAGE_RE.match(raw)
        if pm and "." not in pm.group(1):
            page = pm.group(1)
            meta = {k:v for k,v in META_RE.findall(pm.group(2) or "")}
            page_meta[page] = meta
            continue
        lm = LINE_RE.match(raw)
        if not lm:
            continue
        page, txt = lm.groups()
        toks = clean_tokens(txt)
        if not toks:
            continue
        page_tokens[page].extend(toks)
        folio_tokens[base_folio(page)].extend(toks)
    # Folio metadata by token-weighted majority over its page-sides/subpages.
    folio_meta = {}
    for f, toks in folio_tokens.items():
        votes = {"L":Counter(), "H":Counter(), "I":Counter(), "Q":Counter()}
        for p, pt in page_tokens.items():
            if base_folio(p) != f: continue
            w=max(1,len(pt)); m=page_meta.get(p,{})
            for k in votes:
                if k in m: votes[k][m[k]] += w
        folio_meta[f] = {k:(v.most_common(1)[0][0] if v else None) for k,v in votes.items()}
    return dict(folio_tokens), folio_meta, dict(page_tokens), page_meta


def entropy(tokens: List[str]) -> float:
    if not tokens: return float("nan")
    c=Counter(tokens); n=len(tokens)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def lex(tokens: List[str]):
    c=Counter(tokens); n=len(tokens); v=len(c); h=sum(x==1 for x in c.values())
    return {
        "H1": entropy(tokens),
        "TTR": v/n if n else float("nan"),
        "hapax_token_ratio": h/n if n else float("nan"),
        "hapax_type_ratio": h/v if v else float("nan"),
    }

def cosine(a: List[str], b: List[str]) -> float:
    ca,cb=Counter(a),Counter(b)
    if not ca or not cb: return 0.0
    dot=sum(v*cb.get(k,0) for k,v in ca.items())
    na=math.sqrt(sum(v*v for v in ca.values())); nb=math.sqrt(sum(v*v for v in cb.values()))
    return dot/(na*nb) if na and nb else 0.0


def summarize_null(obs: float, vals: Iterable[float], tail="two"):
    a=np.asarray(list(vals),float); m=float(a.mean()); sd=float(a.std(ddof=1)) if len(a)>1 else 0.0
    z=(obs-m)/sd if sd>0 else float("nan")
    if tail=="lower": p=float((1+np.sum(a<=obs))/(len(a)+1))
    elif tail=="upper": p=float((1+np.sum(a>=obs))/(len(a)+1))
    else: p=float((1+np.sum(np.abs(a-m)>=abs(obs-m)))/(len(a)+1))
    return {"observed":float(obs),"n_null":int(len(a)),"null_mean":m,"null_sd":sd,"effect":float(obs-m),"z":float(z),"p_empirical":p}


def unit_maps(folio_tokens: Dict[int,List[str]]):
    unit_by_folio={}; units=[]; byq=defaultdict(list)
    for uid,q,a,b in UNITS:
        present=[f for f in (a,b) if f in folio_tokens and folio_tokens[f]]
        if not present: continue
        d={"id":uid,"q":q,"folios":present,"complete":len(present)==2}
        units.append(d); byq[q].append(d)
        for f in present: unit_by_folio[f]=uid
    return units,byq,unit_by_folio


def random_matching(xs: List[int], rng: random.Random):
    y=xs[:]; rng.shuffle(y); return [(y[i],y[i+1]) for i in range(0,len(y),2)]


def unit_assays(folio_tokens, byq, folio_meta, rng):
    # Eligible quire strata require >=2 complete observed bifolia.
    actual=[]; q_folios={}
    for q,us in byq.items():
        comp=[u for u in us if u["complete"]]
        if len(comp)<2: continue
        pairs=[tuple(u["folios"]) for u in comp]
        actual.extend([(q,a,b) for a,b in pairs])
        q_folios[q]=[x for _,a,b in [(q,*p) for p in pairs] for x in (a,b)]
    if not actual: return {}
    def stats(pairs):
        rows=[]
        for q,a,b in pairs:
            comb=folio_tokens[a]+folio_tokens[b]; l=lex(comb)
            l["pair_cosine"]=cosine(folio_tokens[a],folio_tokens[b])
            la,lb=folio_meta.get(a,{}).get("L"),folio_meta.get(b,{}).get("L")
            l["same_currier"]=1.0 if la and lb and la==lb else (0.0 if la and lb else float("nan"))
            rows.append(l)
        out={}
        for k in rows[0]:
            vals=[r[k] for r in rows if math.isfinite(r[k])]
            out[k]=float(np.mean(vals)) if vals else float("nan")
        return out
    obs=stats(actual); nulls={k:[] for k in obs}
    for _ in range(N_PAIR):
        pairs=[]
        for q,xs in q_folios.items():
            pairs.extend((q,a,b) for a,b in random_matching(xs,rng))
        s=stats(pairs)
        for k,v in s.items(): nulls[k].append(v)
    return {k:summarize_null(obs[k],nulls[k]) for k in obs if math.isfinite(obs[k])}


def mattr(tokens: List[str], w=100) -> float:
    n=len(tokens)
    if not n: return float("nan")
    if n<=w: return len(set(tokens))/n
    c=Counter(tokens[:w]); total=len(c)/w; k=1
    for i in range(w,n):
        old=tokens[i-w]; c[old]-=1
        if c[old]==0: del c[old]
        c[tokens[i]]+=1; total+=len(c)/w; k+=1
    return total/k


def vocab_auc(tokens: List[str]) -> float:
    if not tokens: return float("nan")
    V=len(set(tokens)); seen=set(); area=0.0
    for t in tokens:
        seen.add(t); area += len(seen)/V
    return area/len(tokens)


def sequence_metrics(tokens: List[str], unit_ids: List[str]):
    assert len(tokens)==len(unit_ids)
    last={}; n=len(tokens); bins=Counter(); hit2_64=0; same_unit=0; cross_unit=0
    for i,t in enumerate(tokens):
        if t in last:
            j=last[t]; lag=i-j
            if lag==1: bins["lag1"]+=1
            elif 2<=lag<=5: bins["lag2_5"]+=1
            elif 6<=lag<=16: bins["lag6_16"]+=1
            elif 17<=lag<=64: bins["lag17_64"]+=1
            if 2<=lag<=64:
                hit2_64+=1
                if unit_ids[i]==unit_ids[j]: same_unit+=1
                else: cross_unit+=1
        last[t]=i
    return {
        "vocab_growth_auc":vocab_auc(tokens),
        "mattr100":mattr(tokens,100),
        "exact_lag1_rate":bins["lag1"]/n,
        "exact_lag2_5_rate":bins["lag2_5"]/n,
        "exact_lag6_16_rate":bins["lag6_16"]/n,
        "exact_lag17_64_rate":bins["lag17_64"]/n,
        "r64_exact_hit_rate":hit2_64/n,
        "r64_same_unit_rate":same_unit/n,
        "r64_cross_unit_rate":cross_unit/n,
        "r64_cross_share":cross_unit/hit2_64 if hit2_64 else 0.0,
    }


def current_stream(folio_tokens, unit_by_folio):
    toks=[]; ids=[]
    for f in sorted(folio_tokens):
        ft=folio_tokens[f]
        uid=unit_by_folio.get(f,f"unmapped_f{f}")
        toks.extend(ft); ids.extend([uid]*len(ft))
    return toks,ids


def block_stream(order_units, folio_tokens, rng=None, random_orientation=False):
    toks=[]; ids=[]
    for u in order_units:
        fs=u["folios"][:]
        if random_orientation and len(fs)==2 and rng.random()<.5: fs.reverse()
        else: fs.sort()
        for f in fs:
            ft=folio_tokens[f]; toks.extend(ft); ids.extend([u["id"]]*len(ft))
    return toks,ids


def order_ensemble(units, byq, folio_tokens, n, seed, mode):
    rng=random.Random(seed); out=[]
    # order current quires numerically; only observed units with text enter.
    qkeys=sorted(byq, key=lambda x:int(re.sub(r"\D","",x) or 999))
    for _ in range(n):
        if mode=="global":
            us=units[:]; rng.shuffle(us)
        elif mode=="within_quire":
            us=[]
            for q in qkeys:
                qq=byq[q][:]; rng.shuffle(qq); us.extend(qq)
        else: raise ValueError(mode)
        t,i=block_stream(us,folio_tokens,rng,True)
        out.append(sequence_metrics(t,i))
    return out


def adjacent_assay(folio_tokens, folio_meta, unit_by_folio, rng):
    fs=sorted(folio_tokens)
    obs_pairs=[]
    pools=defaultdict(list)
    for i,a in enumerate(fs):
        ia=folio_meta.get(a,{}).get("I")
        for b in fs[i+1:]:
            if ia and folio_meta.get(b,{}).get("I")==ia and unit_by_folio.get(a)!=unit_by_folio.get(b):
                pools[ia].append((a,b))
    for a,b in zip(fs[:-1],fs[1:]):
        ia=folio_meta.get(a,{}).get("I")
        if ia and folio_meta.get(b,{}).get("I")==ia and unit_by_folio.get(a)!=unit_by_folio.get(b):
            obs_pairs.append((ia,a,b))
    if not obs_pairs: return None
    obs=float(np.mean([cosine(folio_tokens[a],folio_tokens[b]) for _,a,b in obs_pairs]))
    vals=[]
    for _ in range(N_ADJ):
        chosen=[]
        for sec,_,__ in obs_pairs:
            if pools[sec]: chosen.append(rng.choice(pools[sec]))
        vals.append(float(np.mean([cosine(folio_tokens[a],folio_tokens[b]) for a,b in chosen])))
    r=summarize_null(obs,vals)
    r["n_observed_pairs"]=len(obs_pairs)
    return r


def currier_assays(folio_tokens, folio_meta, byq, rng):
    fs=[f for f in sorted(folio_tokens) if folio_meta.get(f,{}).get("L") in ("A","B")]
    if len(fs)<4: return {}
    def same_rate(labels):
        return float(np.mean([labels[fs[i]]==labels[fs[i+1]] for i in range(len(fs)-1)]))
    labels={f:folio_meta[f]["L"] for f in fs}; obs=same_rate(labels)
    # Section-conditioned label permutation: strongest confound-preserving null available from IVTFF headers.
    bysec=defaultdict(list)
    for f in fs: bysec[folio_meta.get(f,{}).get("I")].append(f)
    vals=[]
    for _ in range(N_PAIR):
        lab=labels.copy()
        for sec,xs in bysec.items():
            vv=[labels[x] for x in xs]; rng.shuffle(vv)
            for x,v in zip(xs,vv): lab[x]=v
        vals.append(same_rate(lab))
    return {"current_adjacent_same_currier_section_null":summarize_null(obs,vals)}


def classification(results_by_code):
    # Group exact assay/metric/null labels and enforce >=3/4 same-direction |z|>=2.
    groups=defaultdict(list)
    for code,rows in results_by_code.items():
        for r in rows:
            z=r.get("effect_over_null_sd")
            if z is None or not math.isfinite(z): continue
            groups[(r["assay_family"],r["metric_name"],r["null_type"])].append((code,z))
    out=[]
    for k,xs in groups.items():
        pos=sum(z>=2 for _,z in xs); neg=sum(z<=-2 for _,z in xs)
        if pos>=3: status="TOPOLOGY_DEPENDENT_POSITIVE"
        elif neg>=3: status="TOPOLOGY_DEPENDENT_NEGATIVE"
        elif any(abs(z)>=2 for _,z in xs): status="REPRESENTATION_DEPENDENT"
        else: status="UNRESOLVED"
        out.append({"assay_family":k[0],"metric_name":k[1],"null_type":k[2],"status":status,"z_by_code":dict(xs),"n_pos_ge2":pos,"n_neg_le_minus2":neg})
    return out


def row(code,family,metric,null_type,res,interpretation=""):
    z=res.get("z")
    status="RESOLVED_SINGLE_REP" if z is not None and math.isfinite(z) and abs(z)>=2 else "UNRESOLVED_SINGLE_REP"
    return {"transcription":code,"assay_family":family,"metric_name":metric,"null_type":null_type,
            "observed":res.get("observed"),"n_null":res.get("n_null"),"null_mean":res.get("null_mean"),"null_sd":res.get("null_sd"),
            "effect":res.get("effect"),"effect_over_null_sd":z,"p_empirical":res.get("p_empirical"),"status":status,"interpretation":interpretation}


def main():
    all_rows={}; qa={}; hashes={}
    for ci,(code,(url,sha)) in enumerate(CORPORA.items()):
        body=fetch_text(url,sha); hashes[code]=sha
        ft,fm,pt,pm=parse_ivtff(body)
        units,byq,ubf=unit_maps(ft)
        nt=sum(len(x) for x in ft.values())
        qa[code]={"parsed_tokens":nt,"folios":len(ft),"pages_with_tokens":len(pt),"units_with_text":len(units),"complete_units":sum(u["complete"] for u in units)}
        rng=random.Random(SEED+1000*ci)
        rows=[]
        # Global invariants: descriptive only; exact order permutation sanity checked later.
        glob=lex([t for f in sorted(ft) for t in ft[f]])
        for k,v in glob.items():
            rows.append({"transcription":code,"assay_family":"global_invariant","metric_name":k,"null_type":"none_order_invariant","observed":v,"n_null":0,"null_mean":v,"null_sd":0.0,"effect":0.0,"effect_over_null_sd":None,"p_empirical":None,"status":"INVARIANT","interpretation":"Order-invariant by definition."})
        # Unit-definition assays.
        ua=unit_assays(ft,byq,fm,rng)
        for k,res in ua.items(): rows.append(row(code,"unit_definition",k,"within_quire_folio_repairing",res,"True bifolia versus same-quire size-matched alternative folio pairings."))
        # Current sequence vs global and within-quire bifolium-order ensembles.
        ct,ciids=current_stream(ft,ubf); obs=sequence_metrics(ct,ciids)
        e2=order_ensemble(units,byq,ft,N_ORDER,SEED+2000+ci,"global")
        e3=order_ensemble(units,byq,ft,N_ORDER,SEED+3000+ci,"within_quire")
        for k,v in obs.items():
            rows.append(row(code,"sequence_topology",k,"global_bifolium_order_ensemble",summarize_null(v,[x[k] for x in e2]),"Current numeric folio order versus globally permuted observed bifolium blocks."))
            rows.append(row(code,"sequence_topology",k,"within_quire_bifolium_order_ensemble",summarize_null(v,[x[k] for x in e3]),"Current numeric folio order versus bifolium-block permutations restricted within current quire."))
        # Deterministic unit reset / block representation, reported as sensitivity not a null.
        bt,bids=block_stream(units,ft,random.Random(SEED),False); bmet=sequence_metrics(bt,bids)
        for k in obs:
            rows.append({"transcription":code,"assay_family":"bifolium_block_sensitivity","metric_name":k,"null_type":"deterministic_bifolium_blocks","observed":obs[k],"n_null":0,"null_mean":bmet[k],"null_sd":None,"effect":obs[k]-bmet[k],"effect_over_null_sd":None,"p_empirical":None,"status":"SENSITIVITY_ONLY","interpretation":"Current numeric folio stream minus deterministic bifolium-block representation; no null SD, so no decision claim."})
        adj=adjacent_assay(ft,fm,ubf,rng)
        if adj: rows.append(row(code,"adjacent_folio","same_section_word_cosine","same_section_nonconjoint_pair_null",adj,"Current adjacent folios after matching broad IVTFF section and excluding conjoint pairs."))
        ca=currier_assays(ft,fm,byq,rng)
        for k,res in ca.items(): rows.append(row(code,"currier_topology",k,"section_conditioned_label_permutation",res,"Current Currier adjacency after preserving broad IVTFF section composition."))
        all_rows[code]=rows
        print(code,qa[code],flush=True)
    classes=classification(all_rows)
    flat=[r for code in CORPORA for r in all_rows[code]]
    with (OUT/"topology_marginal_results.csv").open("w",newline="") as f:
        fields=["transcription","assay_family","metric_name","null_type","observed","n_null","null_mean","null_sd","effect","effect_over_null_sd","p_empirical","status","interpretation"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(flat)
    (OUT/"topology_marginal_classification.json").write_text(json.dumps(classes,indent=2,sort_keys=True))
    summary={"protocol_id":PROTOCOL,"run_id":RUN_ID,"corpus_hashes":hashes,"qa":qa,"n_order":N_ORDER,"n_pair":N_PAIR,"n_adj":N_ADJ,
             "decision_rule":">=3/4 transcriptions with |z|>=2 in same direction; otherwise unresolved/representation-dependent",
             "classes":classes,
             "r64_warning":"Exact recurrence/R64 outputs are topology diagnostics only, not canonical WorkingSetV2+R64 predictive codelength.",
             "prohibited":"No output may be used to select/reconstruct an original order."}
    (OUT/"topology_marginal_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True))
    # Compact human-readable headline list.
    lines=["# VMS topology marginalisation v0.1 — machine closeout","","## STOP RULES","- No downstream metric selects an original order.","- <2 null SD => the metric does not resolve this.","- Topology dependence requires >=3/4 transcriptions at |z|>=2 in the same direction.","- R64 here is an exact-recurrence topology proxy, not the canonical predictive model.","","## Cross-representation classifications"]
    for x in classes:
        lines.append(f"- {x['assay_family']} / {x['metric_name']} / {x['null_type']}: **{x['status']}**; z={x['z_by_code']}")
    (OUT/"TOPOLOGY_MARGINAL_CLOSEOUT.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
