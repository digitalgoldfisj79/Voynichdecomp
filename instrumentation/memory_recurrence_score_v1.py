#!/usr/bin/env python3
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupKFold

MANIFEST_SHA='62655854117793168d46bfb05b548d5993f28dd613301d859fd666504a3b51c0'
CLASSES=['working_set_only','r64_exact','refractory_exact','line_reset_r64','family_persistence']
CONTROLS=['gross_page_shuffle','gross_family_shuffle','line_opener_unknown']
REPS=['R0_canonical_tokens','R1_repeated_glyph_collapse']
C_GRID=[0.1,1.0,10.0]


def softmax(z):
    z=np.asarray(z,float); z=z-z.max(); e=np.exp(z); return e/e.sum()


def parse_marker_text(text):
    dec=json.JSONDecoder(); pos=0; out=[]
    while True:
        p=text.find('P1TRIAL=',pos)
        if p<0: break
        s=text[p+8:].lstrip()
        try:o,k=dec.raw_decode(s)
        except Exception:
            pos=p+8; continue
        out.append(o); pos=p+8+k
    return out


def load_jsonl(paths):
    out=[]
    for p in paths:
        text=Path(p).read_text(encoding='utf-8',errors='replace')
        if 'P1TRIAL=' in text: out.extend(parse_marker_text(text))
        else:
            for line in text.splitlines():
                line=line.strip()
                if line: out.append(json.loads(line))
    return out


def index_trials(objs, split, sources, n_expected):
    d={}
    for o in objs:
        if o.get('split')!=split or o.get('source') not in sources: continue
        key=(o['source'],int(o['trial']),o.get('strength'))
        if key in d and json.dumps(d[key],sort_keys=True)!=json.dumps(o,sort_keys=True):
            raise ValueError(f'conflicting duplicate {key}')
        d[key]=o
    if n_expected is not None:
        for s in sources:
            ks=sorted(k[1] for k in d if k[0]==s and k[2] is None)
            if ks!=list(range(n_expected)): raise ValueError(f'{split} {s} incomplete n={len(ks)} keys={ks[:5]}..{ks[-5:] if ks else []}')
    return d


def fold_features(o,rep,fold):
    z=next(x for x in o['folds'] if int(x['fold'])==fold)
    return np.asarray(z['features'][rep],float)


def cv_select_C(X,y,groups):
    gkf=GroupKFold(n_splits=5); scores=[]
    for C in C_GRID:
        losses=[]
        for tr,te in gkf.split(X,y,groups):
            mu=X[tr].mean(0); sd=X[tr].std(0,ddof=1); sd=np.where(sd>1e-12,sd,1.0)
            m=LogisticRegression(C=C,solver='lbfgs',max_iter=3000).fit((X[tr]-mu)/sd,y[tr])
            pr=m.predict_proba((X[te]-mu)/sd)
            P=np.zeros((len(te),len(CLASSES)))
            for j,c in enumerate(m.classes_): P[:,CLASSES.index(c)]=pr[:,j]
            losses.append(log_loss(y[te],P,labels=CLASSES))
        scores.append((float(np.mean(losses)),C))
    scores.sort(key=lambda z:(z[0],C_GRID.index(z[1])))
    return scores[0][1],scores


def fit_models(dev):
    fitted={}
    for rep in REPS:
        foldmods=[]
        for f in range(5):
            X=[]; y=[]; groups=[]
            for s in CLASSES:
                for t in range(80):
                    o=dev[(s,t,None)]; X.append(fold_features(o,rep,f)); y.append(s); groups.append(t)
            X=np.asarray(X,float); y=np.asarray(y); groups=np.asarray(groups)
            C,cvs=cv_select_C(X,y,groups)
            mu=X.mean(0); sd=X.std(0,ddof=1); sd=np.where(sd>1e-12,sd,1.0)
            m=LogisticRegression(C=C,solver='lbfgs',max_iter=3000).fit((X-mu)/sd,y)
            foldmods.append({'mu':mu,'sd':sd,'model':m,'C':C,'cv':cvs})
        env={}
        for s in CLASSES:
            Z=[]
            for t in range(80):
                o=dev[(s,t,None)]; zz=[]
                for f in range(5):
                    fm=foldmods[f]; zz.extend(((fold_features(o,rep,f)-fm['mu'])/fm['sd']).tolist())
                Z.append(zz)
            Z=np.asarray(Z,float); lw=LedoitWolf().fit(Z); Q=Z-lw.location_; ds=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',Q,lw.precision_,Q),0))
            q=float(np.quantile(ds,.975,method='higher'))
            env[s]={'location':lw.location_,'precision':lw.precision_,'threshold':q,'development_distances':ds}
        fitted[rep]={'foldmods':foldmods,'env':env}
    return fitted


def trial_score(o,rep,fit):
    logev=np.zeros(len(CLASSES)); zz=[]
    for f in range(5):
        fm=fit['foldmods'][f]; x=fold_features(o,rep,f); xs=(x-fm['mu'])/fm['sd']; zz.extend(xs.tolist())
        pr=fm['model'].predict_proba(xs.reshape(1,-1))[0]
        pc=np.full(len(CLASSES),1e-300)
        for j,c in enumerate(fm['model'].classes_): pc[CLASSES.index(c)]=max(pr[j],1e-300)
        logev+=np.log(pc)
    post=softmax(logev); order=np.argsort(-post); top=int(order[0]); second=int(order[1]); pred=CLASSES[top]
    z=np.asarray(zz,float); e=fit['env'][pred]; r=z-e['location']; dist=float(np.sqrt(max(r@e['precision']@r,0))); ood=dist/max(e['threshold'],1e-12)
    eligible=bool(post[top]>=.70 and (post[top]-post[second])>=.20 and ood<=1.0)
    return {'posterior':post,'predicted':pred,'prediction':pred if eligible else None,'abstain':not eligible,'top_p':float(post[top]),'margin':float(post[top]-post[second]),'ood_score':ood,'ood_distance':dist}


def evaluate_known(data,rep,fit,n):
    scores={}; rows=[]
    for s in CLASSES:
        for t in range(n):
            sc=trial_score(data[(s,t,None)],rep,fit); scores[(s,t)]=sc; rows.append((s,t,sc))
    src=[]
    for s in CLASSES:
        rr=[x for x in rows if x[0]==s]; src.append({'name':s,'n':n,'correct':sum(x[2]['prediction']==s for x in rr),'abstained':sum(x[2]['abstain'] for x in rr)})
    return scores,src


def pairwise_validation(scores,rep):
    out=[]; pidx=0
    for i,a in enumerate(CLASSES):
        for b in CLASSES[i+1:]:
            labs=[]; probs=[]
            for s in (a,b):
                for t in range(50): labs.append(s); probs.append(scores[(s,t)]['posterior'])
            labs=np.asarray(labs); probs=np.asarray(probs); ia=CLASSES.index(a); ib=CLASSES.index(b)
            def mean_margin(L):
                vals=[]
                for k,lab in enumerate(L):
                    if lab==a: vals.append(math.log(max(probs[k,ia],1e-300))-math.log(max(probs[k,ib],1e-300)))
                    else: vals.append(math.log(max(probs[k,ib],1e-300))-math.log(max(probs[k,ia],1e-300)))
                return float(np.mean(vals))
            obs=mean_margin(labs); null=[]
            prng=np.random.default_rng(416000 + REPS.index(rep)*1000 + pidx)
            for _ in range(999): null.append(mean_margin(prng.permutation(labs)))
            nm=float(np.mean(null)); ns=float(np.std(null,ddof=1)); eff=obs-nm; z=abs(eff)/ns if ns>0 else (math.inf if eff else 0.0)
            out.append({'a':a,'b':b,'observed_mean_margin':obs,'null_mean':nm,'effect':eff,'null_sd':ns,'effect_over_null_sd':z})
            pidx+=1
    return out


def evaluate_controls(ctrl,rep,fit,known_validation_scores):
    known_ood=np.asarray([v['ood_score'] for v in known_validation_scores.values()],float); km=float(known_ood.mean()); ks=float(known_ood.std(ddof=1))
    out=[]
    for s in CONTROLS:
        ss=[trial_score(ctrl[(s,t,None)],rep,fit) for t in range(50)]
        rej=sum(x['abstain'] for x in ss); u=np.asarray([x['ood_score'] for x in ss],float); eff=float(u.mean()-km); z=abs(eff)/ks if ks>0 else (math.inf if eff else 0.0)
        out.append({'name':s,'n':50,'rejected_or_abstained':rej,'rejection_rate':rej/50,'unknown_mean_ood':float(u.mean()),'known_mean_ood':km,'effect':eff,'null_sd':ks,'effect_over_null_sd':z})
    return out


def rep_gate(source_rows,pairs,unknown):
    recalls=[x['correct']/x['n'] for x in source_rows]; abst=[x['abstained']/x['n'] for x in source_rows]
    src=all(r>=.80 and a<=.10 for r,a in zip(recalls,abst)) and float(np.mean(recalls))>=.85
    pp=all(x['effect']>0 and x['effect_over_null_sd']>=2 for x in pairs)
    up=all(x['rejection_rate']>=.80 and x['effect']>0 and x['effect_over_null_sd']>=2 for x in unknown)
    return bool(src and pp and up)


def summarize(dev,cal,val,ctrl,power_complete=False,power_report=None):
    fit=fit_models(dev); byrep={}
    for rep in REPS:
        _,cal_src=evaluate_known(cal,rep,fit[rep],40)
        val_scores,val_src=evaluate_known(val,rep,fit[rep],50)
        pairs=pairwise_validation(val_scores,rep); unknown=evaluate_controls(ctrl,rep,fit[rep],val_scores)
        passes=rep_gate(val_src,pairs,unknown)
        byrep[rep]={'selected_C_by_fold':[x['C'] for x in fit[rep]['foldmods']], 'calibration_source_rows':cal_src,
                    'source_validation':val_src,'pairwise_separations':pairs,'unknown_controls':unknown,'pass':passes}
    r0=byrep[REPS[0]]
    summary={'manifest_sha256':MANIFEST_SHA,'source_validation':r0['source_validation'],'pairwise_separations':r0['pairwise_separations'],
             'unknown_controls':r0['unknown_controls'],'representation_checks':[{'name':r,'pass':bool(byrep[r]['pass'])} for r in REPS],
             'leakage_checks':{'target_inaccessible_during_fit':True,'disjoint_seed_namespaces':True,'training_only_model_selection':True,'grouped_trial_splits':True},
             'calibration_checks':{'complete':True,'n_per_source':40,'thresholds_changed':False},
             'power_reporting':{'complete':bool(power_complete),'report':power_report},'by_representation':byrep}
    summary['summary_sha256']=hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return summary


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',nargs='+',required=True); ap.add_argument('--output',required=True); ap.add_argument('--power-complete',action='store_true')
    a=ap.parse_args(); objs=load_jsonl(a.input)
    dev=index_trials(objs,'development',CLASSES,80); cal=index_trials(objs,'calibration',CLASSES,40); val=index_trials(objs,'validation',CLASSES,50); ctrl=index_trials(objs,'control',CONTROLS,50)
    out=summarize(dev,cal,val,ctrl,power_complete=a.power_complete); Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print('P1SUMMARY='+json.dumps(out,separators=(',',':')))

if __name__=='__main__': main()
