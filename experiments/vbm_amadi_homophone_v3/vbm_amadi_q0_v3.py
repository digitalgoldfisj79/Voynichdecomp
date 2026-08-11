# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0,<5", "numpy>=1.26,<2.2", "numba>=0.60,<0.62", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import concurrent.futures,json,statistics,sys
import numpy as np
sys.path.insert(0,'experiments/amadi_residuals_v1');sys.path.insert(0,'experiments/amadi_expanded_vbm_v1');sys.path.insert(0,'experiments/vbm_hmm_v2')
import vbm_hmm_moment_v2 as m
b=m.b
NS='VBMAMADIV3'; b.NS=NS
CORE_REGIMES=['ANTI_SQRT','UNIFORM','SQRT_FREQ','FREQ_PROP','SUPER_FREQ','DIRICHLET_SKEW']
BRIDGE_SCHEDULES=['FLAT','CYCLE']
# source ratio a:e:i:o:u = 3:2:3:4:2 scaled to 123 by frozen largest remainder
BR_COUNTS={'a':26,'e':18,'i':26,'o':35,'u':18}

def core_counts(lm,regime,rng):
    states=b.CIDX;m=len(states);rem=b.KCORE-m;f=np.array([lm.freq[int(x)] for x in states],float);f=np.maximum(f,1e-8)
    if regime=='ANTI_SQRT':w=f**-0.5
    elif regime=='UNIFORM':w=np.ones(m)
    elif regime=='SQRT_FREQ':w=f**0.5
    elif regime=='FREQ_PROP':w=f
    elif regime=='SUPER_FREQ':w=f**1.5
    elif regime=='DIRICHLET_SKEW':w=rng.dirichlet(np.full(m,.20))*f**.5
    else:raise ValueError(regime)
    return np.ones(m,dtype=int)+b.largest_remainder(rem,w)

def make_key(lm,core_regime,bridge_schedule,tag):
    rng=np.random.default_rng(b.seed(NS,'key',tag,core_regime,bridge_schedule));pools={};probs={};cycle_phase={}
    cc=core_counts(lm,core_regime,rng);sur=np.arange(0,b.KCORE,dtype=int);rng.shuffle(sur);k=0
    for st,c in zip(b.CIDX,cc):
        q=sur[k:k+int(c)].copy();k+=int(c);pools[int(st)]=q;probs[int(st)]=np.full(len(q),1/len(q));cycle_phase[int(st)]=0
    assert k==b.KCORE
    sur=np.arange(b.KCORE,b.NOBS,dtype=int);rng.shuffle(sur);k=0
    for ch in 'aeiou':
        st=b.P2I[ch];c=BR_COUNTS[ch];q=sur[k:k+c].copy();k+=c;pools[st]=q;probs[st]=np.full(len(q),1/len(q));cycle_phase[st]=int(rng.integers(0,len(q)))
    assert k==b.KBR
    census={b.PLAIN[s]:len(pools[s]) for s in range(b.A)}
    return pools,probs,cycle_phase,census

def encrypt_v3(seqs,pools,probs,phase,bridge_schedule,tag):
    out=[];truth=[];global_counts={int(s):0 for s in b.VIDX};vset=set(global_counts)
    for si,s in enumerate(seqs):
        rng=np.random.default_rng(b.seed(NS,'emit',tag,si));q=[];z=[]
        for ch in s:
            st=b.P2I[ch];pool=pools[st]
            if st in vset and bridge_schedule=='CYCLE':
                j=(phase[st]+global_counts[st])%len(pool);global_counts[st]+=1;obs=int(pool[j])
            else:obs=int(pool[int(rng.choice(len(pool),p=probs[st]))])
            q.append(obs);z.append(st)
        if q:out.append(np.array(q,np.int32));truth.append(np.array(z,np.int32))
    return out,truth

def one_control(lms,truth_name,core_regime,bridge_schedule):
    truth=lms[truth_name];tag=f'Q0:{truth_name}:{core_regime}:{bridge_schedule}';fw,hw=b.plain_span(truth.control,tag,18000,7000);p,u,ph,census=make_key(truth,core_regime,bridge_schedule,tag);fc,ft=encrypt_v3(fw,p,u,ph,bridge_schedule,tag+':F');hc,ht=encrypt_v3(hw,p,u,ph,bridge_schedule,tag+':H');cand=[]
    for la,lm in lms.items():
        r=m.paired_fit_moment(fc,hc,lm,f'{tag}:{la}',ht if la==truth_name else None,40);cand.append({'language':la,'score_A':r['A_eval']['score'],'score_B':r['B_eval']['score'],'score_mean':r['score'],'score_gap':r['score_gap'],'truth_recovery_diag':r['recovery']})
    meanrank=sorted(cand,key=lambda x:(-x['score_mean'],x['language']));Arank=sorted(cand,key=lambda x:(-x['score_A'],x['language']));Brank=sorted(cand,key=lambda x:(-x['score_B'],x['language']));tm=next(x for x in cand if x['language']==truth_name);margin=meanrank[0]['score_mean']-meanrank[1]['score_mean'];qual=bool(meanrank[0]['language']==truth_name and Arank[0]['language']==truth_name and Brank[0]['language']==truth_name and margin>=.02 and tm['score_gap']<=.10)
    return {'truth':truth_name,'core_regime':core_regime,'bridge_schedule':bridge_schedule,'winner_A':Arank[0]['language'],'winner_B':Brank[0]['language'],'winner_mean':meanrank[0]['language'],'margin_mean':margin,'truth_score_gap':tm['score_gap'],'truth_recovery_diag':tm['truth_recovery_diag'],'qualified':qual,'homophone_counts':census,'fit_events':sum(map(len,fc)),'hold_events':sum(map(len,hc)),'candidates':cand}

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();lms=b.load_lms();jobs=[(la,r,s) for la in lms for r in CORE_REGIMES for s in BRIDGE_SCHEDULES];rows=[]
    def one(j):return one_control(lms,*j)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for z in ex.map(one,jobs):rows.append(z);print('Q0V3',json.dumps({k:z[k] for k in ['truth','core_regime','bridge_schedule','winner_A','winner_B','winner_mean','margin_mean','truth_score_gap','truth_recovery_diag','qualified']},sort_keys=True),flush=True)
    per={};floors={};cycle_ok=True
    for la in lms:
        z=[x for x in rows if x['truth']==la];cyc=[x for x in z if x['bridge_schedule']=='CYCLE'];per[la]={'qualified':sum(x['qualified'] for x in z),'total':12,'cycle_qualified':sum(x['qualified'] for x in cyc),'median_margin':statistics.median(x['margin_mean'] for x in z),'min_margin':min(x['margin_mean'] for x in z)};floors[la]=float(np.quantile(np.array([x['margin_mean'] for x in z]),.05,method='linear'));cycle_ok &= per[la]['cycle_qualified']>=5
    bavfp=all(x['qualified'] for x in rows if x['truth']=='bavarian' and x['core_regime']=='FREQ_PROP');overall=sum(x['qualified'] for x in rows);passed=bool(overall>=32 and all(per[la]['qualified']>=10 for la in per) and per['bavarian']['qualified']>=10 and cycle_ok and bavfp)
    out={'namespace':NS,'source_bridge_counts':BR_COUNTS,'overall_qualified':overall,'total':36,'per_language':per,'margin_floors':floors,'cycle_gate_all_languages':cycle_ok,'bavarian_freqprop_both_pass':bavfp,'pass':passed,'rows':rows}
    print('RESULT_JSON',json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
