#!/usr/bin/env python3
"""Corrected pre-outcome analysis for Alpine–Venetian Corridor v0.1.

Amendment 005: independent convergence families are representation arms,
not visual object classes. Visual classes remain diagnostic only.
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics
import numpy as np
import psycopg

SEED=20260808
N_PERM=100000
ALPHA=0.01
FDR_Q=0.05
MIN_POSITIVE_ARMS=3
MIN_FDR_ARMS=2


def db():
    u=os.environ.get('SUPABASE_DB_URL')
    if not u: raise SystemExit('SUPABASE_DB_URL required')
    return psycopg.connect(u)

def h(s:str)->int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)

def signflip_p(diffs:np.ndarray,seed:int)->float:
    if not len(diffs): return float('nan')
    obs=float(np.mean(diffs)); rng=np.random.default_rng(seed); exc=0
    for start in range(0,N_PERM,10000):
        m=min(10000,N_PERM-start)
        signs=rng.choice(np.array([-1.,1.]),size=(m,len(diffs)))
        vals=(signs*diffs).mean(axis=1)
        exc += int(np.count_nonzero(np.abs(vals)>=abs(obs)))
    return (exc+1)/(N_PERM+1)

def bh(items):
    vals=sorted([(k,p) for k,p in items if np.isfinite(p)],key=lambda x:x[1])
    out={}; prev=1.; m=len(vals)
    for i in range(m-1,-1,-1):
        k,p=vals[i]; q=min(prev,p*m/(i+1));out[k]=q;prev=q
    return out

def aggregate(rows):
    # candidate -> arm -> class -> scores
    d={}
    for cand,cls,arm,score in rows:
        d.setdefault(cand,{}).setdefault(arm,{}).setdefault(cls,[]).append(float(score))
    # each arm gets equal weight across its available classes per manuscript
    arm_scores={}
    for cand,arms in d.items():
        arm_scores[cand]={}
        for arm,classes in arms.items():
            class_means=[statistics.fmean(v) for v in classes.values() if v]
            if class_means: arm_scores[cand][arm]=statistics.fmean(class_means)
    # primary composite equal weight over available arms
    comp={c:statistics.fmean(a.values()) for c,a in arm_scores.items() if a}
    return d,arm_scores,comp

def diffs_for(scores,match):
    ds=[]; keys=[]
    for c,ctrls in match.items():
        vals=[scores[x] for x in ctrls if x in scores]
        if c in scores and vals:
            ds.append(scores[c]-statistics.fmean(vals));keys.append(c)
    return np.asarray(ds,float),keys

def run(label:str):
    with db() as con, con.cursor() as cur:
        cur.execute('select run_id from public.corridor_runs where run_label=%s',(label,));row=cur.fetchone()
        if not row: raise SystemExit('unknown run')
        run_id=row[0]
        cur.execute('select candidate_key,object_class,arm,calibrated_score from public.corridor_scores where run_id=%s and calibrated_score is not null',(run_id,))
        rows=cur.fetchall()
        if not rows: raise SystemExit('No calibrated corridor_scores for this run')
        raw,arm_scores,comp=aggregate(rows)
        cur.execute('select corridor_candidate_key,control_candidate_key from public.corridor_control_matches where run_id=%s order by corridor_candidate_key,match_rank',(run_id,))
        match={}
        for c,d in cur.fetchall(): match.setdefault(c,[]).append(d)

        primary_d,primary_keys=diffs_for(comp,match)
        primary_est=float(primary_d.mean()) if len(primary_d) else float('nan')
        primary_p=signflip_p(primary_d,SEED)

        all_arms=sorted({a for x in arm_scores.values() for a in x})
        arm_res=[]
        for arm in all_arms:
            scores={c:a[arm] for c,a in arm_scores.items() if arm in a}
            d,_=diffs_for(scores,match)
            est=float(d.mean()) if len(d) else float('nan')
            p=signflip_p(d,SEED+h('arm:'+arm)%100000)
            arm_res.append((arm,est,p,len(d)))
        qarm=bh([(a,p) for a,_,p,_ in arm_res])

        # visual class diagnostics: equal across arms available within class/candidate
        class_scores={}
        for cand,arms in raw.items():
            tmp={}
            for arm,classes in arms.items():
                for cls,vals in classes.items(): tmp.setdefault(cls,[]).append(statistics.fmean(vals))
            class_scores[cand]={cls:statistics.fmean(v) for cls,v in tmp.items()}
        all_cls=sorted({cls for x in class_scores.values() for cls in x})
        cls_res=[]
        for cls in all_cls:
            scores={c:x[cls] for c,x in class_scores.items() if cls in x}
            d,_=diffs_for(scores,match)
            est=float(d.mean()) if len(d) else float('nan')
            p=signflip_p(d,SEED+h('class:'+cls)%100000)
            cls_res.append((cls,est,p,len(d)))
        qcls=bh([(c,p) for c,_,p,_ in cls_res])

        contrib=np.abs(primary_d)
        max_frac=float(contrib.max()/contrib.sum()) if contrib.sum()>0 else 0.
        loo=[]
        if len(primary_d)>1:
            loo=[float(np.delete(primary_d,i).mean()) for i in range(len(primary_d))]
        loo_pos=float(np.mean(np.asarray(loo)>0)) if loo else float('nan')
        pos_arms=sum(1 for a,e,p,n in arm_res if np.isfinite(e) and e>0)
        fdr_arms=sum(1 for a,e,p,n in arm_res if e>0 and qarm.get(a,1)<=FDR_Q)
        convergence=(pos_arms>=MIN_POSITIVE_ARMS and fdr_arms>=MIN_FDR_ARMS)
        primary_sig=bool(np.isfinite(primary_est) and primary_est>0 and primary_p<ALPHA)
        verdict='positive' if primary_sig and convergence else 'not_established'

        cur.execute('delete from public.corridor_results where run_id=%s',(run_id,))
        cur.execute('''insert into public.corridor_results(run_id,result_key,analysis_family,estimate,p_value,n_corridor,n_control,verdict,detail)
                       values(%s,'primary_composite','composite',%s,%s,%s,%s,%s,%s::jsonb)''',
                    (run_id,primary_est,primary_p,len(primary_d),sum(len(v) for v in match.values()),verdict,
                     json.dumps({'max_single_manuscript_effect_fraction':max_frac,'loo_positive_fraction':loo_pos,
                                 'positive_arms':pos_arms,'fdr_positive_arms':fdr_arms,'convergence_pass':convergence,
                                 'amendment':'005'})))
        for arm,est,p,n in arm_res:
            cur.execute('''insert into public.corridor_results(run_id,result_key,analysis_family,estimate,p_value,q_value,n_corridor,verdict,detail)
                           values(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)''',
                        (run_id,'arm:'+arm,'arm:'+arm,est,p,qarm.get(arm),n,
                         'positive' if est>0 else 'nonpositive',json.dumps({'role':'convergence_family'})))
        for cls,est,p,n in cls_res:
            cur.execute('''insert into public.corridor_results(run_id,result_key,analysis_family,estimate,p_value,q_value,n_corridor,verdict,detail)
                           values(%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)''',
                        (run_id,'class:'+cls,'class:'+cls,est,p,qcls.get(cls),n,
                         'positive' if est>0 else 'nonpositive',json.dumps({'role':'diagnostic_only'})))
        con.commit()
    print(json.dumps({'primary_estimate':primary_est,'primary_p':primary_p,'verdict':verdict,
      'convergence_pass':convergence,'arms':[{'arm':a,'estimate':e,'p':p,'q':qarm.get(a),'n':n} for a,e,p,n in arm_res],
      'classes':[{'class':c,'estimate':e,'p':p,'q':qcls.get(c),'n':n} for c,e,p,n in cls_res]},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run-label',required=True);a=ap.parse_args();run(a.run_label)
