#!/usr/bin/env python3
"""Post-closeout diagnostics for BLOCKED P1 v1.
Uses only already-generated development/calibration/validation synthetic logs.
No target access and no qualification decision changes.
"""
import argparse, importlib.util, json, subprocess, tempfile, math
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf


def getlogs(ids, td):
    paths=[]
    for jid in ids:
        text=subprocess.check_output(['hf','jobs','logs',jid],text=True,stderr=subprocess.STDOUT)
        p=Path(td)/(jid+'.log'); p.write_text(text); paths.append(str(p))
    return paths


def concat_z(o,rep,fit,sc):
    zz=[]
    for f in range(5):
        fm=fit['foldmods'][f]; x=sc.fold_features(o,rep,f); zz.extend(((x-fm['mu'])/fm['sd']).tolist())
    return np.asarray(zz,float)


def class_dist(o,rep,fit,cls,sc):
    z=concat_z(o,rep,fit,sc); e=fit['env'][cls]; r=z-e['location']; d=float(np.sqrt(max(r@e['precision']@r,0)))
    return d, d/e['threshold']


def score_variants(o,rep,fit,sc):
    # trial_score gives posterior + predicted class + all frozen eligibility pieces
    s=sc.trial_score(o,rep,fit)
    raw=s['predicted']
    post_ok=(s['top_p']>=.70 and s['margin']>=.20)
    return {'raw':raw,'post':raw if post_ok else None,'full':s['prediction'],'ood':s['ood_score'],'top_p':s['top_p'],'margin':s['margin']}


def confusion(rows, classes):
    return {a:{b:sum(1 for y,p in rows if y==a and p==b) for b in classes+['ABSTAIN']} for a in classes}


def perm_feature_z(A,B,seed=0,nperm=999):
    # A/B: manuscript x feature values (already averaged over five physical folds)
    obs=A.mean(0)-B.mean(0); X=np.vstack([A,B]); labs=np.array([0]*len(A)+[1]*len(B)); rng=np.random.default_rng(seed)
    null=[]
    for _ in range(nperm):
        q=rng.permutation(labs); null.append(X[q==0].mean(0)-X[q==1].mean(0))
    null=np.asarray(null); nm=null.mean(0); ns=null.std(0,ddof=1); z=np.divide(np.abs(obs-nm),ns,out=np.full_like(obs,np.inf),where=ns>0)
    return obs,nm,ns,z


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dev',nargs='+',required=True); ap.add_argument('--cal',nargs='+',required=True); ap.add_argument('--val',nargs='+',required=True); a=ap.parse_args()
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('sc',here/'memory_recurrence_score_v1_1.py'); sc=importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    with tempfile.TemporaryDirectory() as td:
        objs=sc.load_jsonl(getlogs(a.dev+a.cal+a.val,td)); dev=sc.index_trials(objs,'development',sc.CLASSES,80); cal=sc.index_trials(objs,'calibration',sc.CLASSES,40); val=sc.index_trials(objs,'validation',sc.CLASSES,50); fit=sc.fit_models(dev)
        out={'status':'POST_CLOSEOUT_DIAGNOSTIC_ONLY','target_accessed':False,'n':{'dev':400,'cal':200,'val':250},'representations':{}}
        for ri,rep in enumerate(sc.REPS):
            R={'classification_variants':{},'true_class_ood':{},'calibrated_ood_exploration':{},'working_vs_line_reset_features':{}}
            # classification with raw argmax, posterior confidence only, frozen full OOD
            for split,data,n in [('calibration',cal,40),('validation',val,50)]:
                vars={k:[] for k in ('raw','post','full')}
                for cls in sc.CLASSES:
                    for t in range(n):
                        sv=score_variants(data[(cls,t,None)],rep,fit[rep],sc)
                        for k in vars: vars[k].append((cls,sv[k] if sv[k] is not None else 'ABSTAIN'))
                R['classification_variants'][split]={k:{'accuracy':sum(y==p for y,p in rows)/len(rows),'abstain_rate':sum(p=='ABSTAIN' for y,p in rows)/len(rows),'confusion':confusion(rows,sc.CLASSES)} for k,rows in vars.items()}
            # true-class OOD distributions. Development values are in-sample by construction.
            for cls in sc.CLASSES:
                vals={}
                dev_ratio=np.asarray(fit[rep]['env'][cls]['development_distances'])/fit[rep]['env'][cls]['threshold']; vals['development_in_sample']=dev_ratio
                for split,data,n in [('calibration',cal,40),('validation',val,50)]:
                    vals[split]=np.asarray([class_dist(data[(cls,t,None)],rep,fit[rep],cls,sc)[1] for t in range(n)])
                R['true_class_ood'][cls]={k:{'mean':float(v.mean()),'median':float(np.median(v)),'q95':float(np.quantile(v,.95)),'q975':float(np.quantile(v,.975)),'exceed_1_rate':float(np.mean(v>1))} for k,v in vals.items()}
            # Exploratory v2 idea: calibration-set conformal-ish class thresholds based on true-class distance.
            # Threshold ratio is higher 95% order statistic of 40 calibration true-class ratios, then applied untouched to v1 validation.
            cals={}; passrates={}
            for cls in sc.CLASSES:
                cr=np.sort([class_dist(cal[(cls,t,None)],rep,fit[rep],cls,sc)[1] for t in range(40)])
                q=float(cr[min(len(cr)-1, math.ceil((len(cr)+1)*.95)-1)])
                vr=np.asarray([class_dist(val[(cls,t,None)],rep,fit[rep],cls,sc)[1] for t in range(50)])
                cals[cls]=q; passrates[cls]=float(np.mean(vr<=q))
            R['calibrated_ood_exploration']={'calibration_95pct_ratio_threshold_by_true_class':cals,'validation_true_class_inlier_rate':passrates,'note':'exploratory only; v1 remains failed; v2 requires fresh confirmation'}
            # Existing feature resolution: working_set vs line_reset, manuscript-average over 5 physical folds.
            names=next(iter(dev.values()))['feature_names']; A=[];B=[]
            for cls,target in [('working_set_only',A),('line_reset_r64',B)]:
                for t in range(80): target.append(np.mean([sc.fold_features(dev[(cls,t,None)],rep,f) for f in range(5)],axis=0))
            A=np.asarray(A);B=np.asarray(B); obs,nm,ns,z=perm_feature_z(A,B,seed=88000+ri)
            order=np.argsort(-z)
            R['working_vs_line_reset_features']={'n_per_source':80,'top_features':[{'feature':names[i],'mean_difference_working_minus_line_reset':float(obs[i]),'null_sd':float(ns[i]),'effect_over_null_sd':float(z[i]),'resolved_ge_2sd':bool(z[i]>=2)} for i in order], 'n_resolved_ge_2sd':int(np.sum(z>=2))}
            out['representations'][rep]=R
        print('P1V1DIAG='+json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()
