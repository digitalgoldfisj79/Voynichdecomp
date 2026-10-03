"""Implementation-only fast scorer for MG3.
Scientific metric definitions and decision rules are unchanged from MG3_PREREG_20261003.md.
The only change is to precompute each outer fold's capped raw-character Levenshtein matrix once
with RapidFuzz, after an exact deterministic cross-check against joint_model.distance_raw.
"""
from __future__ import annotations
import collections as C, json, os, pickle, sys
import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
import joint_model as J

OUTM=J.OUT/'MG3'

def build_fold_mats(rs):
    out=[]
    for f in range(5):
        vocab=sorted({t for r in rs if r['fold']!=f for t in r['tokens']})
        vi={t:i for i,t in enumerate(vocab)}
        D=process.cdist(vocab,vocab,scorer=Levenshtein.distance,score_cutoff=3,workers=-1,dtype=np.uint8)
        # RapidFuzz distance with cutoff=3 returns 4 for >3; verify exact equivalence to frozen native metric.
        assert D.shape==(len(vocab),len(vocab))
        assert np.all(np.diag(D)==0)
        assert np.array_equal(D,D.T)
        n=len(vocab)
        rng=np.random.default_rng(90361003+f)
        for _ in range(2500):
            i=int(rng.integers(n)); j=int(rng.integers(n))
            a=int(D[i,j]); b=min(4,J.ed(vocab[i],vocab[j]))
            if a!=b: raise AssertionError(('ED mismatch',f,vocab[i],vocab[j],a,b))
        out.append((vocab,vi,D))
        print(json.dumps(dict(stage='ed_matrix',fold=f,V=len(vocab),bytes=int(D.nbytes),verified_pairs=2500)),flush=True)
    return out

def folio_seen_metrics(lines, mats):
    bypage=C.defaultdict(list)
    for r in lines:bypage[(r['fold'],r['folio'])].extend(r['tokens'])
    sM=0.;nM=0;uniq_sum=0;seen_n=0;per_fold={}
    for f in range(5):
        vocab,vi,D=mats[f];fs=0.;fn=0;fu=0;fseen=0
        for (ff,fol),toks in bypage.items():
            if ff!=f:continue
            prior=[]; prior_set=set(); fol_seen=[]
            for t in toks:
                j=vi.get(t)
                if j is None:continue
                fol_seen.append(t)
                if prior:
                    if j in prior_set:
                        cand=[k for k in prior if k!=j]
                    else:cand=prior
                    m=int(D[j,cand].min()) if cand else 4
                else:m=4
                fs+=m;fn+=1
                if j not in prior_set:prior_set.add(j);prior.append(j)
            fu+=len(set(fol_seen));fseen+=len(fol_seen)
        per_fold[str(f)]=dict(Mplus=fs/max(fn,1),TTR=fu/max(fseen,1),n=fn)
        sM+=fs;nM+=fn;uniq_sum+=fu;seen_n+=fseen
    return dict(folio_seen_Mplus=sM/max(nM,1),folio_seen_TTR=uniq_sum/max(seen_n,1),per_fold=per_fold,n_seen=nM)

def main(nrep):
    from run_joint import diagnostics
    rs=J.load_lines(os.environ.get('MG_LAYER','ZLZI'));mats=build_fold_mats(rs)
    obs=diagnostics(rs);obs_extra=folio_seen_metrics(rs,mats);order={(r['folio'],r['line']):i for i,r in enumerate(rs)}
    sims=[];extras=[];runaway=[]
    for rep in range(nrep):
        lines=[]
        for f in range(5):
            gl=pickle.load(open(OUTM/f'gen_fold{f}_rep{rep:03d}.pkl','rb'));lines+=gl
            tk=[t for r in gl for t in r['tokens']]
            if C.Counter(tk).most_common(1)[0][1]>0.06*len(tk):runaway.append((rep,f))
        lines.sort(key=lambda r:order[r['folio'],r['line']]);sims.append(diagnostics(lines));extras.append(folio_seen_metrics(lines,mats))
    keys=[k for k in obs if k!='hard_zero_violation_rate']
    X=np.array([[s[k] for k in keys] for s in sims]);mu=X.mean(0);sd=X.std(0,ddof=1);y=np.array([obs[k] for k in keys])
    z=np.divide(y-mu,sd,out=np.zeros_like(y),where=sd>0)
    simz=np.divide(X-mu,sd,out=np.zeros_like(X),where=sd>0);null95=float(np.quantile(np.abs(simz).max(1),.95))
    ek=['folio_seen_Mplus','folio_seen_TTR'];EX=np.array([[e[k] for k in ek] for e in extras]);emu=EX.mean(0);esd=EX.std(0,ddof=1);ey=np.array([obs_extra[k] for k in ek]);ez=np.divide(ey-emu,esd,out=np.zeros_like(ey),where=esd>0)
    zd=dict(zip(keys,z.tolist()));ezd=dict(zip(ek,ez.tolist()))
    untargeted=['n_types','hapax_type_fraction','hapax_token_fraction','token_length_mean','atom_length_mean','q_kt_tail_CMI','q_kt_opportunities_per_token','irun_terminator_CMI','irun_opportunities_per_token']
    reuse=[k for k in keys if k.startswith('page_reuse_')]
    core=(float(np.abs(z).max())<=null95 and all(abs(zd[k])<=3 for k in untargeted) and len(runaway)==0 and all(abs(zd[k])<=3 for k in reuse) and all(abs(ezd[k])<=3 for k in ek))
    pre='FALSIFIED' if(float(np.abs(z).max())>10 or len(runaway)>=3) else('CORE_CLOSE_PENDING_C2ST' if core else'NOT_CLOSED_PARTIAL')
    res=dict(n_rep=nrep,observed=obs,sim_mean=dict(zip(keys,mu.tolist())),sim_sd=dict(zip(keys,sd.tolist())),z=zd,
      max_abs_z=float(np.abs(z).max()),argmax=keys[int(np.abs(z).argmax())],null95=null95,
      hard_zero_sim=[s['hard_zero_violation_rate'] for s in sims],runaway_reps=runaway,
      observed_extra={k:obs_extra[k] for k in ek},sim_extra_mean=dict(zip(ek,emu.tolist())),
      sim_extra_sd=dict(zip(ek,esd.tolist())),z_extra=ezd,real_extra_per_fold=obs_extra['per_fold'],
      core_close=core,pre_c2st_verdict=pre,score_impl='rapidfuzz_matrix_verified_against_native_12500_pairs')
    J.atomic(res,OUTM/'score.pkl');json.dump(res,open(J.ROOT.parent/'mg3_results.json','w'),indent=1)
    for k in sorted(keys,key=lambda k:-abs(zd[k])):
        print(f"{k:28s} obs {obs[k]:10.4f} sim {res['sim_mean'][k]:10.4f} sd {res['sim_sd'][k]:9.5f} z {zd[k]:7.2f}")
    for k in ek:print(f"{k:28s} obs {obs_extra[k]:10.5f} sim {res['sim_extra_mean'][k]:10.5f} sd {res['sim_extra_sd'][k]:9.6f} z {ezd[k]:7.2f}")
    print('MG3_PRE_C2ST='+json.dumps(dict(verdict=pre,max_abs_z=res['max_abs_z'],argmax=res['argmax'],null95=null95,runaway=len(runaway),extra_z=ezd),sort_keys=True),flush=True)
if __name__=='__main__':main(int(sys.argv[1]))
