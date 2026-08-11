# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import concurrent.futures, collections, json, statistics
import numpy as np
import amadi_driver_v1c as d
m=d.m

# Q2 correction frozen before qualification: PWA exact-rule accuracy is N/A, not 0,
# if PWA itself failed Q1 and is absent from the admitted blind universe.
def q2_fixed(std,vc,r12,gates,smoke=False,workers=1):
    rows=[]
    if gates.get("R12H"):
        for rep in range(2 if smoke else 8):
            ctl=m.make_control(std,vc,r12,"R12H",1,"italian",rep,"Q2R",400 if smoke else 1500,400 if smoke else 1500)
            r=m.cand_fit(ctl,std,vc,r12,"R12H",1,"italian",f"Q2R:{rep}",smoke)
            rows.append({"truth_family":"R12H","selected_family":"R12H","truth_rule":1,"selected_rule":1,"truth_language":"italian","selected_language":"italian","recovery":r["recovery"],"converged":r["converged"]})
    fams=[f for f in ["VC","PWA","GH"] if gates.get(f)]; controls=[]
    for fam in fams:
        langs=(["latin","italian"] if smoke else list(std))
        for i,lang in enumerate(langs):
            rule=(m.PWA_RULES[i%4] if fam=="PWA" else (5 if fam=="GH" else 1)); controls.append((fam,rule,lang,i))
    def one(j):
        tf,tr,tl,rep=j; ctl=m.make_control(std,vc,r12,tf,tr,tl,rep,"Q2",400 if smoke else 1500,400 if smoke else 1500); cand=[]
        for fam in fams:
            rules=m.PWA_RULES if fam=="PWA" else ([5] if fam=="GH" else [1])
            for rule in rules:
                for lang in list(std):
                    r=m.cand_fit(ctl,std,vc,r12,fam,rule,lang,f"Q2:{tf}:{tr}:{tl}:{rep}:{fam}:{rule}:{lang}",smoke); cand.append((r["hold_score"],fam,rule,lang,r))
        cand.sort(key=lambda x:(-x[0],x[1],x[2],x[3])); x=cand[0]
        return {"truth_family":tf,"selected_family":x[1],"truth_rule":tr,"selected_rule":x[2],"truth_language":tl,"selected_language":x[3],"recovery":x[4]["recovery"],"converged":x[4]["converged"],"top_score":x[0]}
    if controls:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for r in ex.map(one,controls): rows.append(r); print("Q2",json.dumps(r,sort_keys=True),flush=True)
    z=[r for r in rows if r["truth_family"]!="R12H"]; rz=[r for r in rows if r["truth_family"]=="R12H"]
    famacc=sum(r["truth_family"]==r["selected_family"] for r in z)/max(1,len(z)); langacc=sum(r["truth_language"]==r["selected_language"] for r in z)/max(1,len(z)); pw=[r for r in z if r["truth_family"]=="PWA"]
    ruleacc=(sum(r["truth_rule"]==r["selected_rule"] for r in pw)/len(pw)) if pw else 1.0
    med=statistics.median([r["recovery"] for r in z]) if z else 0.0
    perlang={l:(sum(r["selected_language"]==l for r in z if r["truth_language"]==l),sum(1 for r in z if r["truth_language"]==l)) for l in list(std)}
    ok=(bool(z) and famacc>=.90 and ruleacc>=.85 and langacc>=.90 and med>=.90 and all(n<4 or c/n>=.75 for c,n in perlang.values()))
    rok=bool(rz) and all(r["converged"] and r["recovery"]>=.85 for r in rz) and statistics.median(r["recovery"] for r in rz)>=.95
    return rows,{"multifamily_pass":bool(ok),"family_accuracy":famacc,"pwa_rule_accuracy":ruleacc,"pwa_rule_accuracy_applicable":bool(pw),"language_accuracy":langacc,"median_recovery":med,"per_language":perlang,"R12H_pass":bool(rok),"admitted_multifamily_universe":fams}
m.q2=q2_fixed

# Frozen GHOUSE target-only positive requirements.
GH_MIN_FIT_PAYLOAD_CHARS_PER_STATE=500
GH_MIN_STATE_AB_AGREEMENT=0.90

def gh_target_fixed(pages,fitfs,h2fs,std,q3res,smoke=False):
    fitraw=m.combine(pages,fitfs); fitx,fm,fcen=m.target_extract_gh(fitraw)
    h2x=[]; hm=[]; blocks=[]; census_by_folio={}; pos=0
    for fol in h2fs:
        q,s,c=m.target_extract_gh(pages[fol]); h2x.extend(q); hm.extend(s); blocks.append((fol,pos,pos+len(q))); pos+=len(q); census_by_folio[fol]=c
    support=[0]*5
    for w,s in zip(fitx,fm): support[s]+=len(w)
    best=None
    for lang in list(std):
        fs=m.make_stats(fitx,m.state_words(fitx,"GH",5,fm),5); hs=m.make_stats(h2x,m.state_words(h2x,"GH",5,hm),5)
        sol=m.solve_bij(fs,std[lang],f"TARGET:GH:{lang}",smoke); hscore=m.fixed_score(hs,std[lang],sol["dec"])
        if best is None or sol["fit_score"]>best[0]: best=(sol["fit_score"],lang,sol,hscore,hs)
    fit_score,lang,sol,hscore,hs=best
    ctl={"fit":fitx,"hold":h2x,"mfit":fm,"mhold":hm,"fit_plain":[],"hold_plain":[]}; base=m.baseline(ctl,std,lang,f"TARGETBASE:GH:{lang}",smoke)
    cell=q3res["cells"][f"GH:{lang}"]; delta=hscore-base["hold_score"]
    perm_scores=[]
    for rep in range(256):
        pm=list(hm)
        for fol,lo,hi in blocks:
            rg=np.random.default_rng(m.seed(m.NS,"ghperm",rep,fol)); arr=np.array(pm[lo:hi],dtype=np.int32); rg.shuffle(arr); pm[lo:hi]=[int(x) for x in arr]
        ps=m.make_stats(h2x,m.state_words(h2x,"GH",5,pm),5); perm_scores.append(m.fixed_score(ps,std[lang],sol["dec"]))
    p99=float(np.quantile(np.array(perm_scores),.99,method="linear")); sag=sol.get("state_agreement") or []
    stab_support=all(x>=GH_MIN_FIT_PAYLOAD_CHARS_PER_STATE for x in support); stab_ag=len(sag)==5 and all(x>=GH_MIN_STATE_AB_AGREEMENT for x in sag)
    r={"family":"GH","language":lang,"rule":5,"fit_score":fit_score,"H2_score":hscore,"baseline_H2":base["hold_score"],"delta":delta,"ABS_FLOOR":cell["ABS_FLOOR"],"DELTA_FLOOR":cell["DELTA_FLOOR"],"abs_pass":hscore>=cell["ABS_FLOOR"],"delta_pass":delta>=cell["DELTA_FLOOR"],"agreement":sol["agreement"],"state_agreement":sag,"converged":sol["converged"],"fit_selector_census":fcen,"H2_selector_census_by_folio":census_by_folio,"fit_payload_chars_by_state":support,"state_support_pass":stab_support,"state_stability_pass":stab_ag,"selector_permutation_gate":{"n":256,"real_score":hscore,"permuted_p99":p99,"permuted_max":max(perm_scores),"pass":hscore>p99}}
    return r

def run_target_fixed(std,vc,r12,qual,smoke=False):
    man=m.manifest(); pages,_=m.parse_rf(); fitfs=man["FIT_A"]["folios"]; h2fs=man["H2"]["folios"]; fitw=m.combine(pages,fitfs); h2w=m.combine(pages,h2fs); out={"manifest":man,"families":{},"C2_opened":False}
    q3r=qual["q3_summary"]; active=q3r["active"] if qual["q4_summary"]["pass"] else []
    for fam in active:
        if fam=="GH":
            r=gh_target_fixed(pages,fitfs,h2fs,std,q3r,smoke)
        else:
            r,_=m.target_fit_family(fitw,h2w,std,vc,r12,fam,q3r,smoke)
        r["q4_pass"]=qual["q4_summary"]["pass"]
        positive=r["abs_pass"] and r["delta_pass"] and r["converged"] and qual["q4_summary"]["pass"]
        if fam=="PWA": positive &= r.get("reset_pass",False)
        if fam=="GH": positive &= r["selector_permutation_gate"]["pass"] and r["state_support_pass"] and r["state_stability_pass"]
        if fam=="R12H": positive &= r.get("agreement",0)>=.95
        r["verdict"]="H2_CANDIDATE" if positive else ("CLOSED_NEGATIVE_INCOMPATIBLE_V1" if not r["abs_pass"] and r["converged"] else "COMPATIBLE_NONSPECIFIC")
        out["families"][fam]=r
    return out
m.run_target=run_target_fixed

if __name__=="__main__": m.main()
