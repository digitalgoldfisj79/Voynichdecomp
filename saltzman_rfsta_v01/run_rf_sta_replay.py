#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,math,re,sys
import numpy as np
import scorer

ROOT=Path(__file__).resolve().parent
PROTOCOL_SHA="8952d3d7990a349e3f8b6cbce4786a69067ace95bed6e07ef935efb072741592"
REPS=("RF_MEMBER","STA_FAMILY","AAA_CONNECTED")
SYNREPS=("ATOMIC","LITERAL")

def verify_protocol():
    b=(ROOT/"protocol.json").read_bytes();got=hashlib.sha256(b).hexdigest();exp=(ROOT/"PROTOCOL_SHA256").read_text().strip()
    assert got==exp==PROTOCOL_SHA,(got,exp,PROTOCOL_SHA);return json.loads(b)

def parse_rules(path):
    rules={}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#") or ln.startswith("<") or ln.startswith("-"):continue
        f=ln.split()
        if len(f)>=2 and re.fullmatch(r"[A-Z][0-9a-z]",f[0]):rules[f[0]]=f[1]
    assert len(rules)>=250,len(rules);return rules

STA_WORD=re.compile(r"(?:[A-Z][0-9a-z])+")
def flush(cur,out):
    if cur:out.append(cur[:]);cur.clear()

def parse_clean_rf(path,rules):
    lines=[];raw_line_records=excluded_words=kept_words=interruptions=0
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        m=re.match(r"^<([^>]+)>\s+(.*)$",ln)
        if not m:continue
        tag,body=m.group(1),m.group(2)
        if "." not in tag:continue
        raw_line_records+=1;segments=body.split("<->");interruptions+=max(0,len(segments)-1)
        for seg in segments:
            cur=[]
            for raw in seg.split("."):
                w=raw.strip()
                if not w:flush(cur,lines);continue
                if STA_WORD.fullmatch(w):
                    members=re.findall(r"[A-Z][0-9a-z]",w)
                    if "".join(members)==w and all(x in rules for x in members):cur.append(members);kept_words+=1;continue
                excluded_words+=1;flush(cur,lines)
            flush(cur,lines)
    lines=[z for z in lines if z];assert kept_words>10000 and len(lines)>1000,(kept_words,len(lines))
    return lines,{"raw_text_line_records":raw_line_records,"clean_segments":len(lines),"clean_words":kept_words,"excluded_or_uncertain_words":excluded_words,"drawing_interruptions":interruptions}

def unit_lines(rf_lines,rules,rep):
    if rep=="RF_MEMBER":words_by_line=[[members[:] for members in line] for line in rf_lines]
    elif rep=="STA_FAMILY":words_by_line=[[[m[0] for m in members] for members in line] for line in rf_lines]
    elif rep=="AAA_CONNECTED":
        words_by_line=[]
        for line in rf_lines:
            out=[]
            for members in line:
                units=[]
                for m in members:
                    parts=rules[m].split("~");assert all(re.fullmatch(r"[a-z][0-9](?::[a-z][0-9])*",x) for x in parts),(m,rules[m]);units.extend(parts)
                out.append(units)
            words_by_line.append(out)
    else:raise ValueError(rep)
    vocab=sorted({u for line in words_by_line for word in line for u in word});assert 0<len(vocab)<6000,len(vocab)
    mp={u:chr(0xE000+i) for i,u in enumerate(vocab)}
    out=[["".join(mp[u] for u in word) for word in line] for line in words_by_line];assert all(w for line in out for w in line)
    return out,{"unit_vocabulary":len(vocab),"words":sum(map(len,out)),"segments":len(out)}

def target_gate(q):
    if len(q)<4 or not all(x is not None and math.isfinite(x) and x>0 for x in q):return False
    return bool(q[0]>=1.10 and 0.90<=q[3]<=1.10 and q[0]>q[1]>q[2] and 0.95<=q[2]<=1.10)

def qualitative(q):
    if len(q)<4 or not all(x is not None and math.isfinite(x) and x>0 for x in q):return False
    return bool(q[0]>1 and q[0]>q[1]>q[2] and abs(math.log(q[2]))<abs(math.log(q[0])) and 0.90<=q[3]<=1.10)

def target_mode(P,rf_path,rules_path,out_path):
    rules=parse_rules(rules_path);rf_lines,parse_stats=parse_clean_rf(rf_path,rules);targets={}
    nseed=int(P["target_estimation"]["null_seeds"]);nperm=int(P["target_estimation"]["per_seed_permutations"])
    for rep in REPS:
        lines,stats=unit_lines(rf_lines,rules,rep);e1avail=scorer.adj_repeats(lines)>0;assert e1avail,rep;rows=[]
        for i in range(nseed):
            seed=scorer.seed_of(P["target_estimation"]["seed_namespace"],rep,i);r=scorer.one_eval(scorer.prep(lines),seed,e1avail,nperm)
            q=[None if x is None else float(x) for x in r["Q4"]];rows.append(q);print("TARGET",rep,i+1,nseed,q,flush=True)
        arr=np.asarray(rows,float);med=[float(x) for x in np.median(arr,axis=0)];pass_rows=sum(target_gate(q) for q in rows)
        targets[rep]={"median_Q4":med,"rows_Q4":rows,"median_structural_gate":target_gate(med),"qualitative_median":qualitative(med),"seed_gate_passes":pass_rows,"seed_gate_fraction":pass_rows/len(rows),"e1_available":e1avail,**stats}
    minfrac=float(P["target_robustness"]["seed_stability_min_fraction"]);binding=all(targets[r]["median_structural_gate"] and targets[r]["seed_gate_fraction"]>=minfrac for r in REPS)
    result={"experiment":P["experiment"],"protocol_sha256":PROTOCOL_SHA,"scorer_sha256":P["target_estimation"]["scorer_sha256"],"parse_stats":parse_stats,"targets":targets,"binding_target_representation_robust":binding}
    Path(out_path).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

def ldist(q,t,n):
    if q is None or t is None or len(q)<n or len(t)<n:return math.inf
    vals=[]
    for a,b in zip(q[:n],t[:n]):
        if a is None or b is None or a<=0 or b<=0 or not math.isfinite(a) or not math.isfinite(b):return math.inf
        vals.append(math.log(a/b))
    return math.sqrt(sum(x*x for x in vals))

def robust_maps(cells,target,arm,n):
    by={}
    for c in cells:
        if c["arm"]!=arm or c["representation"] not in SYNREPS:continue
        q=c["median_Q3"] if n==3 else c["median_Q4"];by.setdefault(c["corpus"],{})[c["representation"]]=ldist(q,target,n)
    return {doc:max(z[r] for r in SYNREPS) for doc,z in by.items() if all(r in z for r in SYNREPS)}

def robust_e1(cells,target_e1,arm):
    by={}
    for c in cells:
        if c["arm"]!=arm or c["representation"] not in SYNREPS:continue
        q=c.get("median_Q4");e=q[3] if q and len(q)>3 else None;v=math.inf if e is None or e<=0 or not math.isfinite(e) else abs(math.log(e/target_e1));by.setdefault(c["corpus"],{})[c["representation"]]=v
    return {doc:max(z[r] for r in SYNREPS) for doc,z in by.items() if all(r in z for r in SYNREPS)}

def seedint(*parts):return int.from_bytes(hashlib.sha256("|".join(map(str,parts)).encode()).digest()[:8],"big")%(2**63-1)

def boot_gain(a,b,P,label):
    docs=sorted(set(a)&set(b));vals=np.asarray([a[d]-b[d] for d in docs if math.isfinite(a[d]) and math.isfinite(b[d])],float)
    if not len(vals):return {"n":0,"median":None,"ci95":[None,None]}
    rng=np.random.default_rng(seedint(P["primary_test"]["bootstrap_seed_namespace"],label));B=int(P["primary_test"]["bootstrap_resamples"]);meds=np.empty(B);n=len(vals)
    for i in range(0,B,500):m=min(500,B-i);idx=rng.integers(0,n,size=(m,n));meds[i:i+m]=np.median(vals[idx],axis=1)
    return {"n":n,"median":float(np.median(vals)),"ci95":[float(np.quantile(meds,.025)),float(np.quantile(meds,.975))]}

def medfinite(d):
    v=[x for x in d.values() if math.isfinite(x)];return {"n_finite":len(v),"n_total":len(d),"median":float(np.median(v)) if v else None,"infinite":sum(not math.isfinite(x) for x in d.values())}

def replay_mode(P,target_path,combined_path,out_path):
    T=json.load(open(target_path));assert T["protocol_sha256"]==PROTOCOL_SHA;C=json.load(open(combined_path));cells=C["cells"];assert len({c["corpus"] for c in cells})==190
    arms=P["synthetic_replay"]["arms"];out={};all_gain=all_close=full_close=True;any_positive=False
    for rep in REPS:
        target=T["targets"][rep]["median_Q4"];row={"target_Q4":target};maps3={k:robust_maps(cells,target,arm,3) for k,arm in arms.items()};maps4={k:robust_maps(cells,target,arm,4) for k,arm in arms.items()};e1={k:robust_e1(cells,target[3],arm) for k,arm in arms.items()}
        g=boot_gain(maps3["baseline"],maps3["canonical"],P,rep+"|D3|CANON");gs=boot_gain(maps3["baseline"],maps3["sensitivity"],P,rep+"|D3|SENS");g4=boot_gain(maps4["baseline"],maps4["canonical"],P,rep+"|D4|CANON");ge=boot_gain(e1["baseline"],e1["canonical"],P,rep+"|E1|CANON")
        row["d3"]={k:medfinite(v) for k,v in maps3.items()};row["d3_gain_canonical"]=g;row["d3_gain_continuous_sensitivity"]=gs;row["d4"]={k:medfinite(v) for k,v in maps4.items()};row["d4_gain_canonical_finite_pairs"]=g4;row["e1_log_error"]={k:medfinite(v) for k,v in e1.items()};row["e1_gain_canonical_finite_pairs"]=ge;out[rep]=row
        gainok=g["n"]>=180 and g["ci95"][0] is not None and g["ci95"][0]>0;close=row["d3"]["canonical"]["median"] is not None and row["d3"]["canonical"]["median"]<=0.15 and row["d3"]["canonical"]["n_finite"]>=180;fclose=row["d4"]["canonical"]["n_finite"]>=180 and row["d4"]["canonical"]["median"] is not None and row["d4"]["canonical"]["median"]<=0.15
        all_gain&=gainok;all_close&=close;full_close&=fclose;any_positive|=(g["median"] is not None and g["median"]>0)
    target_robust=bool(T["binding_target_representation_robust"])
    if not target_robust:verdict="VMS_Q_NOT_REPRESENTATION_ROBUST"
    elif all_gain and all_close and full_close:verdict="REPRESENTATION_ROBUST_FULL_Q_MATCH"
    elif all_gain and all_close:verdict="REPRESENTATION_ROBUST_ED1_MATCH_ONLY"
    elif all_gain:verdict="REPRESENTATION_ROBUST_ED1_GAIN_NOT_MATCH"
    elif any_positive:verdict="REPRESENTATION_SENSITIVE_OR_UNRESOLVED"
    else:verdict="NO_REPRESENTATION_ROBUST_GAIN"
    result={"experiment":P["experiment"],"protocol_sha256":PROTOCOL_SHA,"phase8_protocol_sha256":C.get("protocol_sha256"),"target_representation_robust":target_robust,"replay":out,"adjudication":verdict,"notes":{"synthetic_scores_reused":"Exact Phase-8 median Q vectors; no ReM simulation or parameter fitting was rerun.","primary":"ED1 d3 transfer across RF_MEMBER, STA_FAMILY, AAA_CONNECTED.","full_Q4":"Secondary; zero/nonpositive E1 produces non-finite d4 and is never imputed."}}
    Path(out_path).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

def main():
    P=verify_protocol();scorer.self_test_ed1()
    if sys.argv[1]=="target":target_mode(P,sys.argv[2],sys.argv[3],sys.argv[4])
    elif sys.argv[1]=="replay":replay_mode(P,sys.argv[2],sys.argv[3],sys.argv[4])
    else:raise SystemExit("target RF1b STA-aaa.bit OUT | replay TARGETS COMBINED OUT")
if __name__=="__main__":main()
