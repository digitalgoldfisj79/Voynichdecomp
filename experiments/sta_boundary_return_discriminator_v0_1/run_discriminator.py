#!/usr/bin/env python3
"""
STA boundary-template vs exact-return discriminator v0.1.

Primary models:
B0 = inherited exact-edge independent baseline
B1 = B0 + exact segment length
B2 = B1 + one shared latent boundary template (identity-blind k-modes)
B3 = B2 + one global boundary-local lag-2 identity-return probability q
CG = B2 + global lag-2 return placebo

All target fitting is folio-held-out. See PREREG_20260815.md and
IMPLEMENTATION_FREEZE_20260815.md.
"""
import argparse, collections, hashlib, importlib.util, json, math, random
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "sta_boundary_robustness_v0_1" / "run_sta_robustness.py"
spec = importlib.util.spec_from_file_location("sta_parent", PARENT)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)

SEED = 20260815
NFOLD = 5
NREP = 50
K_PRIMARY = 8
Q_GRID = np.linspace(0.0, 0.25, 501)
MIN_COUNT = 30
MIN_TYPES = 5
MIN_Z_LINES = 10
EDGE_LABELS = ("L0","L1","L2","L3","R3","R2","R1","R0")
LEN_FEATURES = ((1,2,"1-2"),(3,4,"3-4"),(5,6,"5-6"),(7,10**9,"7+"))

def med(xs): return float(np.median(np.asarray(xs, dtype=float)))
def qtile(xs,q): return float(np.quantile(np.asarray(xs, dtype=float),q))

def token_len_bucket(tok):
    n=len(tok)
    for lo,hi,s in LEN_FEATURES:
        if lo<=n<=hi: return s
    return "7+"

def boundary_eligible_target(j,n):
    return j>=2 and ((j-2)<2 or j>=n-2)

def boundary_feature(line):
    t=line["tokens"]; n=len(t)
    pos={}
    for i,tok in enumerate(t):
        ep=p.edge_pos(i,n)
        if ep!="M": pos[ep]=(token_len_bucket(tok), tok[0], tok[-1])
    out=[]
    for lab in EDGE_LABELS: out.extend(pos.get(lab,("NA","NA","NA")))
    return tuple(out)

def hamming(a,b): return sum(x!=y for x,y in zip(a,b))

def kmodes(lines,K):
    feats=[boundary_feature(x) for x in lines]
    counts=collections.Counter(feats); uniq=sorted(counts)
    if not uniq: return [], []
    K=min(K,len(uniq)); mx=max(counts.values())
    first=min(v for v in uniq if counts[v]==mx); prot=[first]
    while len(prot)<K:
        scored=[]
        for v in uniq:
            if v in prot: continue
            scored.append((min(hamming(v,z) for z in prot),v))
        maxd=max(d for d,_ in scored)
        prot.append(min(v for d,v in scored if d==maxd))
    assign=None
    for _ in range(30):
        new=[]
        for v in feats:
            ds=[hamming(v,z) for z in prot]
            new.append(min(range(K), key=lambda z:(ds[z],z)))
        if assign==new: break
        assign=new
        for z in range(K):
            members=[feats[i] for i,a in enumerate(assign) if a==z]
            if not members: continue
            nv=[]
            for col in zip(*members):
                c=collections.Counter(col); m=max(c.values())
                nv.append(min(k for k,v in c.items() if v==m))
            prot[z]=tuple(nv)
    return prot,assign

def usable(c,min_count=MIN_COUNT,min_types=MIN_TYPES):
    return bool(c) and sum(c.values())>=min_count and len(c)>=min_types

class FoldModel:
    def __init__(self, train, K=K_PRIMARY):
        self.train=train
        self.vocab=collections.Counter(tok for x in train for tok in x["tokens"])
        self.V=max(len(self.vocab),1)
        self.tables=collections.defaultdict(collections.Counter)
        self.zpriors=collections.defaultdict(collections.Counter)
        self.prototypes,self.assignments=kmodes(train,K); self.K=len(self.prototypes)
        self._build()

    def _build(self):
        for ix,x in enumerate(self.train):
            t=x["tokens"]; n=len(t); sec=x["section"]; lb=p.lbucket(n); z=self.assignments[ix]
            self.zpriors[("ZLB",sec,lb)][z]+=1
            self.zpriors[("ZSEC",sec)][z]+=1
            self.zpriors[("ZG",)][z]+=1
            for i,tok in enumerate(t):
                ep=p.edge_pos(i,n)
                for k in (("G",),("SEC",sec),("LB",sec,lb),("P",sec,ep),
                          ("B0",sec,lb,ep),("B1",sec,n,ep),("ZP",z,ep),
                          ("ZLB",z,sec,lb,ep),("ZFULL",z,sec,n,ep)):
                    self.tables[k][tok]+=1

    def z_counter(self,sec,lb):
        for k in (("ZLB",sec,lb),("ZSEC",sec),("ZG",)):
            c=self.zpriors.get(k)
            if c and sum(c.values())>=MIN_Z_LINES: return c
        return self.zpriors[("ZG",)]

    def z_probs(self,sec,lb):
        c=self.z_counter(sec,lb); s=sum(c.values())
        return {z:v/s for z,v in c.items()} if s else {0:1.0}

    def counter(self,model,sec,n,ep,z=None):
        lb=p.lbucket(n)
        if model in ("B2","B3","CG") and z is not None:
            for k in (("ZFULL",z,sec,n,ep),("ZLB",z,sec,lb,ep),("ZP",z,ep)):
                c=self.tables.get(k)
                if usable(c): return c
        if model in ("B1","B2","B3","CG"):
            c=self.tables.get(("B1",sec,n,ep))
            if usable(c): return c
        for k in (("B0",sec,lb,ep),("P",sec,ep),("LB",sec,lb),("SEC",sec),("G",)):
            c=self.tables.get(k)
            if usable(c): return c
        return self.tables[("G",)]

    def sample_z(self,sec,lb,rng):
        c=self.z_counter(sec,lb)
        return rng.choices(list(c),weights=list(c.values()),k=1)[0]

    def sample_token(self,model,sec,n,ep,rng,z=None):
        c=self.counter(model,sec,n,ep,z)
        return rng.choices(list(c),weights=list(c.values()),k=1)[0]

    def kt_prob(self,c,tok):
        N=sum(c.values())
        return (c.get(tok,0)+0.5)/(N+0.5*self.V) if N else 1.0/self.V

    def b2_marginal_prob(self,sec,n,ep,tok):
        lb=p.lbucket(n); s=0.0
        for z,w in self.z_probs(sec,lb).items():
            s += w*self.kt_prob(self.counter("B2",sec,n,ep,z),tok)
        return max(s,1e-300)

def fit_q(model, lines, eligible_only=True):
    p0=[]; eq=[]
    for x in lines:
        t=x["tokens"]; n=len(t); sec=x["section"]
        for j in range(2,n):
            if eligible_only and not boundary_eligible_target(j,n): continue
            p0.append(model.b2_marginal_prob(sec,n,p.edge_pos(j,n),t[j]))
            eq.append(1.0 if t[j]==t[j-2] else 0.0)
    if not p0: return 0.0, float("-inf"), 0
    p0=np.asarray(p0); eq=np.asarray(eq); best=(float("-inf"),0.0)
    for q in Q_GRID:
        prob=q*eq+(1.0-q)*p0
        ll=float(np.log(np.maximum(prob,1e-300)).sum())
        if ll>best[0]+1e-12: best=(ll,float(q))
    return best[1],best[0],len(p0)

def generate_line(x, fm, model, rng, q=0.0):
    n=len(x["tokens"]); sec=x["section"]; lb=p.lbucket(n)
    z=fm.sample_z(sec,lb,rng) if model in ("B2","B3","CG") else None
    g=[]
    for j in range(n):
        ep=p.edge_pos(j,n)
        if model=="B3" and boundary_eligible_target(j,n) and rng.random()<q:
            g.append(g[j-2]); continue
        if model=="CG" and j>=2 and rng.random()<q:
            g.append(g[j-2]); continue
        base_model="B2" if model in ("B3","CG") else model
        g.append(fm.sample_token(base_model,sec,n,ep,rng,z))
    return {"folio":x["folio"],"section":sec,"tokens":tuple(g)}

def split_folds(lines):
    by={f:[] for f in range(NFOLD)}
    for x in lines: by[p.fold_of(x["folio"])].append(x)
    return by

def fit_fold_models(lines,K=K_PRIMARY):
    by=split_folds(lines); out={}
    for f in range(NFOLD):
        train=[x for g in range(NFOLD) if g!=f for x in by[g]]
        fm=FoldModel(train,K); qb,_,_=fit_q(fm,train,True); qg,_,_=fit_q(fm,train,False)
        out[f]={"train":train,"test":by[f],"fm":fm,"q_boundary":qb,"q_global":qg}
    return out

def generate_oof(fits,model,rep):
    out=[]; offset={"B0":0,"B1":100,"B2":200,"B3":300,"CG":400}[model]
    for f,d in fits.items():
        rng=random.Random(SEED+100000*rep+1000*f+offset)
        q=d["q_boundary"] if model=="B3" else (d["q_global"] if model=="CG" else 0.0)
        for x in d["test"]: out.append(generate_line(x,d["fm"],model,rng,q))
    return out

def group_counters(tokens,groups):
    return [collections.Counter(tokens[int(i)] for i in g) for g in groups]

def eq_pair_prob(c1,c2,same):
    n1=sum(c1.values()); n2=sum(c2.values())
    if same:
        return sum(v*(v-1) for v in c1.values())/(n1*(n1-1)) if n1>=2 else 0.0
    return sum(v*c2.get(k,0) for k,v in c1.items())/(n1*n2) if n1 and n2 else 0.0

def rel_pair_prob(c1,c2,same,rel):
    n1=sum(c1.values()); n2=sum(c2.values())
    if same:
        den=n1*(n1-1)
        if den<=0:return 0.0
        num=0
        for a,fa in c1.items():
            for b,fb in c1.items():
                pairs=fa*(fb-(1 if a==b else 0))
                if pairs>0 and rel(a,b): num+=pairs
        return num/den
    den=n1*n2
    if den<=0:return 0.0
    return sum(fa*fb for a,fa in c1.items() for b,fb in c2.items() if rel(a,b))/den

def expected_relation(lines,lag,null,rel,starts_fn=None):
    total=0.0
    for x in lines:
        t=x["tokens"]; n=len(t)
        if n<=lag: continue
        groups=p.groups_for(n,null); gmap={int(pos):gi for gi,g in enumerate(groups) for pos in g}; cs=group_counters(t,groups)
        starts=range(n-lag) if starts_fn is None else starts_fn(n,lag)
        for i in starts:
            j=i+lag; ga,gb=gmap[i],gmap[j]
            total += eq_pair_prob(cs[ga],cs[gb],ga==gb) if rel=="eq" else rel_pair_prob(cs[ga],cs[gb],ga==gb,rel)
    return total

def observed_relation(lines,lag,rel,starts_fn=None):
    tot=0
    for x in lines:
        t=x["tokens"]; n=len(t)
        if n<=lag: continue
        starts=range(n-lag) if starts_fn is None else starts_fn(n,lag)
        for i in starts:
            a,b=t[i],t[i+lag]; tot += (a==b) if rel=="eq" else bool(rel(a,b))
    return tot

def n3_starts(n,lag): return range(2,max(2,n-4)) if lag==2 else range(0)
def edn3_starts(n,lag): return range(2,max(2,n-3))

def score(lines):
    out={}
    for lag in (1,2,3,4):
        obs=observed_relation(lines,lag,"eq"); exp=expected_relation(lines,lag,"N0","eq")
        out[f"E{lag}_N0"]=obs/exp if exp>0 else None
    obs=observed_relation(lines,2,"eq"); exp=expected_relation(lines,2,"N1","eq")
    out["E2_N1"]=obs/exp if exp>0 else None
    ob3=observed_relation(lines,2,"eq",n3_starts); ex3=expected_relation(lines,2,"N1","eq",n3_starts)
    out["E2_N3"]=ob3/ex3 if ex3>0 else None
    oed=observed_relation(lines,1,p.is_ed1); eed=expected_relation(lines,1,"N0",p.is_ed1)
    out["ED1_N0"]=oed/eed if eed>0 else None
    oed3=observed_relation(lines,1,p.is_ed1,edn3_starts); eed3=expected_relation(lines,1,"N1",p.is_ed1,edn3_starts)
    out["ED1_N3"]=oed3/eed3 if eed3>0 else None
    return out

def gates(median,real):
    p1=1.13<=median["E2_N0"]<=1.23
    p2=median["E2_N1"]<=1.10 and median["E2_N3"]<=1.08
    p3=(median["E2_N0"]-median["E2_N3"])>=0.08
    p4=all(median[f"E{k}_N0"]<=max(1.10,real[f"E{k}_N0"]+0.05) for k in (1,3,4))
    p5=(median["ED1_N3"]<=max(1.08,real["ED1_N3"]+0.04) and median["ED1_N0"]<=real["ED1_N0"]+0.06)
    return {"P1_magnitude":bool(p1),"P2_attenuation":bool(p2),"P3_gap":bool(p3),"P4_lag_specificity":bool(p4),"P5_ED1_collateral":bool(p5)}

def predictive_identity(fits,bootstrap=2000):
    folio_delta=collections.defaultdict(lambda:[0.0,0]); fold_means={}; qvals=[]; oov=0; total=0
    for f,d in fits.items():
        fm=d["fm"]; q=d["q_boundary"]; qvals.append(q); ds=[]; nn=0
        for x in d["test"]:
            t=x["tokens"]; n=len(t); sec=x["section"]
            for j in range(2,n):
                if not boundary_eligible_target(j,n): continue
                total+=1
                if t[j] not in fm.vocab: oov+=1; continue
                p0=fm.b2_marginal_prob(sec,n,p.edge_pos(j,n),t[j])
                pp=q*(1.0 if t[j]==t[j-2] else 0.0)+(1.0-q)*p0
                delta=math.log(max(pp,1e-300))-math.log(max(p0,1e-300))
                folio_delta[x["folio"]][0]+=delta; folio_delta[x["folio"]][1]+=1; ds.append(delta); nn+=1
        fold_means[str(f)]=float(sum(ds)/nn) if nn else None
    items=[(a,b) for a,b in folio_delta.values() if b]
    point=sum(a for a,b in items)/sum(b for a,b in items) if items else float("nan")
    rng=random.Random(SEED+700000); boots=[]
    for _ in range(bootstrap if items else 0):
        samp=[items[rng.randrange(len(items))] for _ in range(len(items))]; den=sum(b for a,b in samp)
        boots.append(sum(a for a,b in samp)/den if den else 0.0)
    ci=[qtile(boots,.025),qtile(boots,.975)] if boots else [None,None]
    positives=sum(v is not None and v>0 for v in fold_means.values()); ceiling=sum(abs(q-.25)<1e-12 for q in qvals)
    passed=bool(ci[0] is not None and ci[0]>0 and positives>=4 and ceiling<2)
    return {"mean_delta_loglik_per_event":point,"bootstrap95":ci,"fold_means":fold_means,"positive_folds":positives,
            "q_by_fold":qvals,"ceiling_folds":ceiling,"eligible_total":total,"excluded_oov":oov,"pass":passed}

def n1_shuffle(lines,rng):
    out=[]
    for x in lines:
        t=list(x["tokens"]); n=len(t)
        for g in p.groups_for(n,"N1"):
            vals=[t[int(i)] for i in g]; rng.shuffle(vals)
            for pos,val in zip(g,vals): t[int(pos)]=val
        out.append({"folio":x["folio"],"section":x["section"],"tokens":tuple(t)})
    return out

def calibration_cn1(lines,n=100,smoke=False):
    qs=[]; positive=0; reps=3 if smoke else n
    for r in range(reps):
        sh=n1_shuffle(lines,random.Random(SEED+900000+r)); fits=fit_fold_models(sh,K_PRIMARY)
        qs.append(med([d["q_boundary"] for d in fits.values()]))
        positive += bool(predictive_identity(fits,bootstrap=(100 if smoke else 2000))["pass"])
    return {"n":reps,"median_q":med(qs),"p95_q":qtile(qs,.95),"positive_gate_fraction":positive/reps,
            "pass":bool(med(qs)<=.02 and positive/reps<=.05)}

def synthetic_controls(lines,smoke=False):
    fits=fit_fold_models(lines,K_PRIMARY); reps=2 if smoke else 8; q0=[]; q8=[]; pred0=[]; pred8=[]
    for r in range(reps):
        syn=generate_oof(fits,"B2",1000+r); f0=fit_fold_models(syn,K_PRIMARY)
        q0.append(med([d["q_boundary"] for d in f0.values()])); pred0.append(bool(predictive_identity(f0,bootstrap=(100 if smoke else 2000))["pass"]))
        planted=[]; rng=random.Random(SEED+950000+r)
        for x in generate_oof(fits,"B2",2000+r):
            t=list(x["tokens"]); n=len(t)
            for j in range(2,n):
                if boundary_eligible_target(j,n) and rng.random()<.08: t[j]=t[j-2]
            planted.append({"folio":x["folio"],"section":x["section"],"tokens":tuple(t)})
        f8=fit_fold_models(planted,K_PRIMARY)
        q8.append(med([d["q_boundary"] for d in f8.values()])); pred8.append(bool(predictive_identity(f8,bootstrap=(100 if smoke else 2000))["pass"]))
    ct=med(q0)<=.02 and sum(pred0)==0; cr=abs(med(q8)-.08)<=.03 and (sum(pred8)/len(pred8))>=.75
    return {"template_control":{"median_q":med(q0),"predictive_pass_fraction":sum(pred0)/len(pred0),"pass":bool(ct)},
            "return_control":{"planted_q":.08,"median_q":med(q8),"predictive_pass_fraction":sum(pred8)/len(pred8),"pass":bool(cr)},
            "pass":bool(ct and cr)}

def model_aggregate(fits,real,models=("B0","B1","B2","B3","CG"),nrep=NREP):
    out={}; keys=("E1_N0","E2_N0","E2_N1","E2_N3","E3_N0","E4_N0","ED1_N0","ED1_N3")
    for model in models:
        reps=[score(generate_oof(fits,model,r)) for r in range(nrep)]
        median={k:med([x[k] for x in reps]) for k in keys}; gs=gates(median,real)
        out[model]={"median":median,"p10_p90":{k:[qtile([x[k] for x in reps],.1),qtile([x[k] for x in reps],.9)] for k in keys},
                    "gates":gs,"all_primary":all(gs.values())}
    return out

def adjudicate(cal,models,pred):
    if not cal["pass"]: return "INSTRUMENT_FAIL"
    if models["B0"]["all_primary"]: return "BASELINE_REPRODUCES_RESIDUAL"
    if models["B1"]["all_primary"]: return "EXACT_LENGTH_POSITION_SUFFICIENT"
    if models["B2"]["all_primary"]: return "LATENT_BOUNDARY_TEMPLATE_SUFFICIENT"
    if models["B3"]["all_primary"] and pred["pass"]:
        cg=models["CG"]["gates"]
        if cg["P2_attenuation"] and cg["P3_gap"]: return "RETURN_LOCALIZATION_UNRESOLVED"
        return "MINIMAL_BOUNDARY_RETURN_SUPPORTED"
    if models["B3"]["all_primary"] and not pred["pass"]: return "DESCRIPTIVE_RETURN_FIT_ONLY"
    return "TESTED_TEMPLATE_AND_MINIMAL_RETURN_INSUFFICIENT"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("rf1b",nargs="?",default="/tmp/RF1b.txt"); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--skip-cn1",action="store_true"); args=ap.parse_args()
    sections=p.load_sections(); raw,lines,parse_audit=p.parse_rf(args.rf1b,sections); sha=hashlib.sha256(raw).hexdigest(); real=score(lines)
    validation={"source_sha256":sha,"header_ok":raw.startswith(b"#=IVTFF STA1 2.0"),"folios":len({x["folio"] for x in lines}),
                "segments":len(lines),"tokens":sum(len(x["tokens"]) for x in lines),"parser":parse_audit,
                "anchor_windows":{"E2_N0":1.16<=real["E2_N0"]<=1.20,"E2_N1":1.05<=real["E2_N1"]<=1.09,"E2_N3":1.02<=real["E2_N3"]<=1.07}}
    validation["pass"]=(validation["header_ok"] and sha=="81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17" and all(validation["anchor_windows"].values()))
    result={"metadata":{"seed":SEED,"nfold":NFOLD,"nrep":(3 if args.smoke else NREP),"K":K_PRIMARY,"q_grid_step":.0005},"validation":validation,"real":real}
    if not validation["pass"]: result["verdict"]="IMPLEMENTATION_VALIDATION_FAIL"
    else:
        syn=synthetic_controls(lines,args.smoke); cn1={"pass":True,"skipped":True} if args.skip_cn1 else calibration_cn1(lines,100,args.smoke)
        cal={"synthetic":syn,"cn1":cn1,"pass":bool(syn["pass"] and cn1["pass"])}; result["calibration"]=cal
        fits=fit_fold_models(lines,K_PRIMARY)
        result["fit_audit"]={"q_boundary_by_fold":[fits[f]["q_boundary"] for f in range(NFOLD)],"q_global_by_fold":[fits[f]["q_global"] for f in range(NFOLD)],"K_realized_by_fold":[fits[f]["fm"].K for f in range(NFOLD)]}
        pred=predictive_identity(fits,bootstrap=(200 if args.smoke else 2000)); result["predictive_identity"]=pred
        models=model_aggregate(fits,real,nrep=(3 if args.smoke else NREP)); result["models"]=models; result["verdict"]=adjudicate(cal,models,pred)
        if not args.smoke:
            result["K_sensitivity"]={str(K):model_aggregate(fit_fold_models(lines,K),real,models=("B2","B3"),nrep=20) for K in (4,12)}
    outdir=Path("results/sta_boundary_return_discriminator_v0_1"); outdir.mkdir(parents=True,exist_ok=True); suffix="SMOKE" if args.smoke else "RESULTS"
    (outdir/f"{suffix}_20260815.json").write_text(json.dumps(result,indent=2)+"\n")
    md=["# STA boundary-template vs exact-return discriminator v0.1","",f"Mode: **{'SMOKE' if args.smoke else 'PRIMARY'}**",f"Validation: **{'PASS' if validation['pass'] else 'FAIL'}**",f"Verdict: **{result.get('verdict','NOT_ADJUDICATED')}**","","## Real anchors",f"- E2 N0: {real['E2_N0']:.4f}",f"- E2 N1: {real['E2_N1']:.4f}",f"- E2 N3: {real['E2_N3']:.4f}",f"- ED1 N0/N3: {real['ED1_N0']:.4f} / {real['ED1_N3']:.4f}"]
    if "models" in result:
        md += ["","## Model medians","|model|E2 N0|E2 N1|E2 N3|gap|all gates|","|---|---:|---:|---:|---:|---|"]
        for model in ("B0","B1","B2","B3","CG"):
            a=result["models"][model]; m=a["median"]; md.append(f"|{model}|{m['E2_N0']:.4f}|{m['E2_N1']:.4f}|{m['E2_N3']:.4f}|{m['E2_N0']-m['E2_N3']:.4f}|{a['all_primary']}|")
        md += ["",f"Predictive return gate: `{result['predictive_identity']}`","",f"Calibration: `{result['calibration']}`"]
    (outdir/f"{suffix}_20260815.md").write_text("\n".join(md)+"\n"); print("\n".join(md))

if __name__=="__main__": main()
