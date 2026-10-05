#!/usr/bin/env python3
"""
NK4 diagnostic / falsification upper bound.
NK3 strict gate failed, so this is NOT a confirmatory promotion test.

Question:
If we give frozen FORM the *oracle* NK3 K=12 family of each held-out novel token,
can post-generation family conditioning close the real-vs-generated Stolfi/Mauro
whole-token residual?

Family coordinate:
- K=12 fixed from NK3, learned from folds 2/3 only.
- internal frozen FORM features only; no Stolfi/Mauro labels, no external context.
- exact token identity excluded from features.

Generator:
- true frozen exact-piece+depth STOP
- K8 continuation response grouping
- 99% hard continuation crib
- gallows-used rho=.25
- first FORM piece fixed to observed target
- exact raw character length fixed to observed target
- training types and exact target excluded
- family model NEVER changes local transition probabilities: it is an upstream
  rejection/conditioning layer over complete generated forms.

Final material: physical folds 0/1 only. FORM fit uses folds 2/3/4.
"""
import collections, hashlib, json, math, random, re, urllib.request
import numpy as np

SEED=20261005
NK3_URL="https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/eb094abf9590499732fdd47b666df17289d0609e/research/naibbe_kernel_NK3_20261005.py"
src=urllib.request.urlopen(NK3_URL,timeout=120).read().decode()
prefix=src.split("# Select K only on ZLZI validation external context.")[0]
ns={"__name__":"nk4_nk3base"}
exec(compile(prefix,NK3_URL,"exec"),ns)

build_rows=ns["build_rows"]; type_stats=ns["type_stats"]; fit_cluster=ns["fit_cluster"]
internal_feature=ns["internal_feature"]; safe_form=ns["safe_form"]; ST=ns["ST"]; G8=ns["G8"]
KSEL=12
PIECES=sorted(ST)
GAL=set("fkpt");RHO=.25

def hasgal(p): return any(c in GAL for c in p)

# --- classical whole-token diagnostics: exact definitions reused from prior closure work ---
MSLOTS=[['','ch','sh','y'],['','eee','ee','e','q','a'],['','o'],['','iiii','iii','ii','i','d'],
        ['','l','k','r','s','t','p','f','cth','ckh','cph','cfh','n','m','y']]
MCH=set()
def _mr(si,s):
    if si==len(MSLOTS):
        if s:MCH.add(s)
        return
    for x in MSLOTS[si]:_mr(si+1,s+x)
_mr(0,'')
MSET=set(MCH);MMAX=max(map(len,MCH));_MU={}
def mauro_ok(tok):
    if tok in _MU:return _MU[tok]
    n=len(tok);cur={0};ans=False
    for _ in range(5):
        nxt=set()
        for pos in cur:
            for j in range(pos+1,min(n,pos+MMAX)+1):
                if tok[pos:j] in MSET:nxt.add(j)
        if n in nxt:ans=True;break
        cur=nxt
        if not cur:break
    _MU[tok]=ans;return ans

R='(?:d|l|r|s|n|x)';O='(?:o|a|y)';Y='(?:y|o)';A='(?:a|o)';N='(?:n|r|l|m|s)'
IN=f'(?:i|ii|iii){N}';Final=f'(?:{Y}|{A}m|{A}{IN})';OptOFinal=f'(?:|{Final}|{O}{Final})'
OR=f'(?:{R}|{O}{R}|{O}{O}{R})';CrP=f'(?:|{OR}|{OR}{OR})';Q=f'(?:q|{Y}q)'
CrustPrefix=f'(?:{CrP}|{Q}{CrP})';CrS=f'(?:|{OR}|{OR}{OR}|{OR}{OR}{OR})'
CrustSuffix=f'(?:{CrS}{OptOFinal})';CrW='(?:'+'|'.join(['']+[OR*i for i in range(1,6)])+')'
WholeCrust=f'(?:{CrW}{OptOFinal}|{Q}{CrW}{OptOFinal})';OE='(?:e|oee?)'.replace('oeee','oee')
# preserve historical operational regex exactly
OE='(?:e|oe)';OEE='(?:ee|oee)';CH='(?:ch|sh)';OCH=f'(?:{CH}|o{CH}|y{CH})'
MtP=f'(?:|{OE}|{OEE}|{OEE}{OE})';MantlePrefix=f'(?:{MtP}|{OCH}{MtP})'
MtS=f'(?:{OEE}|{OEE}{OCH}|{OCH}|{OCH}{OE}|{OCH}{OEE}|{OCH}{OCH}|{OCH}{OE}{OCH}|{OCH}{OE}{OEE}|{OCH}{OCH}{OE})'
WholeMantle=f'(?:{MtS}|{OE}|{OE}{MtS})';G='(?:t|p|k|f)';Gallows=f'(?:{G}|c{G}h)'
OGallows=f'(?:{Gallows}|o{Gallows}|y{Gallows})';Core=f'(?:{OGallows}|{OGallows}{OE})'
MantleSuffix=f'(?:|{MtS})';MantleCore=f'(?:{MantlePrefix}{Core}{MantleSuffix}|{WholeMantle})'
Normal=f'(?:{CrustPrefix}{MantleCore}{CrustSuffix}|{WholeCrust})'
STRE=re.compile('^'+Normal+'$');_ST={}
def stolfi_ok(tok):
    if tok not in _ST:_ST[tok]=bool(STRE.fullmatch(tok))
    return _ST[tok]

ACC={"stolfi":stolfi_ok,"mauro":mauro_ok,"joint":lambda t:stolfi_ok(t) and mauro_ok(t)}

# --- frozen FORM generator ---
def fit_form(rows):
    h=collections.defaultdict(collections.Counter);hg=collections.Counter()
    cont={g:collections.Counter() for g in range(8)}
    dest=collections.defaultdict(collections.Counter)
    for r in rows:
        ps=r["ps"];cs=r["cs"]
        for i,p in enumerate(ps):
            end=int(i==len(ps)-1);d=min(i,3);h[(p,d)][end]+=1;hg[end]+=1
            if not end:
                g=cs[i];cont[g][ps[i+1]]+=1;dest[g][cs[i+1]]+=1
    allowed=set()
    for g,co in dest.items():
        tot=sum(co.values());cum=0
        for dc,n in sorted(co.items(),key=lambda z:(-z[1],z[0])):
            allowed.add((g,dc));cum+=n
            if cum/tot>=.99:break
    return h,hg,cont,allowed

def stopprob(M,p,d):
    h,hg,_,_=M;c=h.get((p,min(d,3)),hg);a=.25
    return (c[1]+a)/(sum(c.values())+2*a)

def contweights(M,cur_piece,hist):
    _,_,cont,allow=M;g=G8[ST[cur_piece]];co=cont[g]
    seen=any(hasgal(p) for p in hist);w=[]
    for p in PIECES:
        x=co[p]+.5
        if (g,G8[ST[p]]) not in allow:x=0.
        if seen and hasgal(p):x*=RHO
        w.append(x)
    s=sum(w)
    return [x/s for x in w] if s else [1/len(PIECES)]*len(PIECES)

def draw_exact(first,M,target_len,forbidden,rng,maxatt=250):
    for _ in range(maxatt):
        ps=[first];rawlen=len(first)
        if rawlen>target_len:return None
        for __ in range(16):
            p=ps[-1]
            if rng.random()<stopprob(M,p,len(ps)-1):
                t=''.join(ps)
                if len(t)==target_len and t not in forbidden:return t
                break
            q=rng.choices(PIECES,weights=contweights(M,p,ps),k=1)[0]
            if rawlen+len(q)>target_len:break
            ps.append(q);rawlen+=len(q)
    return None

def fam_of(tok,model):
    f=internal_feature(tok)
    if f is None:return None
    sc,km=model
    return int(km.predict(sc.transform(f[None,:]))[0])

def block_stat(records,key):
    # key value is per-target improvement: +1 means family candidate matches
    # real grammar status better than baseline; -1 means worse.
    vals=[r[key] for r in records]
    by=collections.Counter()
    for r in records:by[r["bif"]]+=r[key]
    N=len(vals);eff=float(np.mean(vals)) if vals else None
    sd=(math.sqrt(sum(v*v for v in by.values()))/N) if N else None
    z=(eff/sd if sd and sd>0 else None)
    return dict(n=N,effect=eff,block_sd=sd,z=z,positive_blocks=sum(v>0 for v in by.values()),blocks=len(by))

def rates(records,name):
    fn=ACC[name]
    rr=np.array([fn(r["real"]) for r in records],float)
    bb=np.array([fn(r["base"]) for r in records],float)
    ff=np.array([fn(r["family"]) for r in records],float)
    rg=float(rr.mean());bg=float(bb.mean());fg=float(ff.mean())
    base_gap=rg-bg;fam_gap=rg-fg
    # absolute-error improvement is target-wise and block-testable.
    for r,a,b,c in zip(records,rr,bb,ff):
        r["_imp_"+name]=abs(a-b)-abs(a-c)
    st=block_stat(records,"_imp_"+name)
    closure=(1-abs(fam_gap)/abs(base_gap)) if abs(base_gap)>1e-12 else None
    return dict(real_rate=rg,baseline_generated_rate=bg,family_generated_rate=fg,
                baseline_gap=base_gap,family_gap=fam_gap,residual_closure_fraction=closure,
                improvement_stat=st)

def run_tid(tid):
    rows=build_rows(tid)
    # NK3 family coordinate frozen exactly: learn only on discovery folds 2/3.
    fam_train_stats=type_stats([r for r in rows if r["fold"] in (2,3)])
    fam_model=fit_cluster(fam_train_stats,KSEL)

    # True frozen FORM final fit on all non-test material.
    fit=[r for r in rows if r["fold"] in (2,3,4)]
    test=[r for r in rows if r["fold"] in (0,1)]
    forbidden={r["token"] for r in fit}
    M=fit_form(fit)

    # Completely unseen test occurrences; retain physical block identity.
    real=[r for r in test if r["token"] not in forbidden]
    rng=random.Random(SEED+1000+sum(map(ord,tid)))
    rec=[];total_family_candidate_draws=0;family_hits=0;base_fail=0;fam_fail=0

    for idx,r in enumerate(real):
        target=r["token"];first=r["ps"][0];L=len(target);tf=fam_of(target,fam_model)
        if tf is None:continue
        # exact target itself excluded in addition to all fitted types
        ban=set(forbidden);ban.add(target)
        b=draw_exact(first,M,L,ban,rng,350)
        if b is None:
            base_fail+=1;continue
        # Family oracle: sample ordinary frozen-FORM candidates and reject only
        # after a complete form is produced. Local FORM is never changed.
        f=None;draws=0
        for _ in range(250):
            g=draw_exact(first,M,L,ban,rng,250)
            if g is None:continue
            draws+=1;total_family_candidate_draws+=1
            if fam_of(g,fam_model)==tf:
                f=g;family_hits+=1;break
        if f is None:
            fam_fail+=1;continue
        rec.append(dict(bif=r["bif"],fold=r["fold"],real=target,base=b,family=f,
                        family_id=tf,family_draws=draws,raw_len=L,first=first))

    out=dict(tid=tid,K=KSEL,n_rows=len(rows),n_fit=len(fit),n_test=len(test),
             n_novel_test_occurrences=len(real),n_paired=len(rec),
             pair_rate=(len(rec)/len(real) if real else 0),
             base_fail=base_fail,family_fail=fam_fail,
             family_candidate_draws=total_family_candidate_draws,family_hits=family_hits,
             family_hit_rate=(family_hits/total_family_candidate_draws if total_family_candidate_draws else None),
             family_condition_cost_bits=(-math.log2(family_hits/total_family_candidate_draws)
                                         if family_hits and total_family_candidate_draws else None))

    out["closure"]={k:rates(rec,k) for k in ("stolfi","mauro","joint")}

    # diversity / support diagnostics
    if rec:
        bt=[r["base"] for r in rec];ft=[r["family"] for r in rec]
        out["diversity"]={
          "baseline_unique_types":len(set(bt)),"family_unique_types":len(set(ft)),
          "baseline_unique_share":len(set(bt))/len(bt),"family_unique_share":len(set(ft))/len(ft),
          "baseline_unseen_share":float(np.mean([t not in forbidden for t in bt])),
          "family_unseen_share":float(np.mean([t not in forbidden for t in ft])),
          "exact_length_match_share":float(np.mean([len(r["real"])==len(r["base"])==len(r["family"]) for r in rec])),
          "same_first_piece_share":float(np.mean([safe_form(r["base"])[0][0]==r["first"] and safe_form(r["family"])[0][0]==r["first"] for r in rec]))
        }
        # family post-conditioning support yield by target physical block
        by=collections.defaultdict(list)
        for r in rec:by[r["bif"]].append(r["family_draws"])
        bits=[math.log2(max(1,r["family_draws"])) for r in rec]
        out["support_cost"]={
          "mean_log2_draws":float(np.mean(bits)),
          "median_draws":float(np.median([r["family_draws"] for r in rec])),
          "p90_draws":float(np.quantile([r["family_draws"] for r in rec],.9))
        }
    return out

results=[run_tid(t) for t in ("ZLZI","ZLZB","TTLI")]

# Diagnostic adjudication: because NK3 failed, even a positive result cannot promote.
# A strong falsification is: family oracle fails to reduce BOTH Stolfi and Mauro residual
# by >50% in the same direction in all three transcriptions.
closures={r["tid"]:{k:r["closure"][k]["residual_closure_fraction"] for k in ("stolfi","mauro")} for r in results}
upper_bound_closes=all(
    closures[t]["stolfi"] is not None and closures[t]["mauro"] is not None and
    closures[t]["stolfi"]>0.5 and closures[t]["mauro"]>0.5
    for t in closures
)

out={
 "phase":"NK4","status":"complete","mode":"diagnostic_oracle_upper_bound_after_NK3_failure",
 "family_coordinate":"NK3 K=12, learned folds2/3 only, no grammar labels",
 "test_folds":[0,1],
 "form_fit_folds":[2,3,4],
 "conditioning":"post-generation rejection on oracle family; local FORM unchanged",
 "results":results,
 "oracle_closes_both_residuals_all_transcriptions":upper_bound_closes,
 "interpretation_rule":"Positive cannot promote because NK3 failed and family is oracle. Negative is strong evidence against this K12 family coordinate as the missing complete-token support object."
}
print("NK4_JSON="+json.dumps(out,separators=(",",":")),flush=True)
