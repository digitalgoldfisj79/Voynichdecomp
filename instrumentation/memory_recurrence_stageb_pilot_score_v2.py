#!/usr/bin/env python3
import argparse,json,math,subprocess,tempfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

SOURCES=('working_set_only','line_reset_r64')
REPS=('R0_canonical_tokens','R1_repeated_glyph_collapse')
USE_IDX=(0,1,5,8,14)

def parse(text):
    dec=json.JSONDecoder(); out=[]; pos=0
    while True:
        p=text.find('P1V2TRIAL=',pos)
        if p<0: break
        s=text[p+10:].lstrip()
        try:o,k=dec.raw_decode(s)
        except Exception: pos=p+10; continue
        out.append(o);pos=p+10+k
    return out

def feat(o,rep): return np.mean([np.asarray(f['features'][rep],float) for f in o['folds']],axis=0)

def permz(A,B,seed):
    obs=A.mean(0)-B.mean(0);X=np.vstack([A,B]);lab=np.array([0]*len(A)+[1]*len(B));rng=np.random.default_rng(seed);nul=[]
    for _ in range(999):
        q=rng.permutation(lab);nul.append(X[q==0].mean(0)-X[q==1].mean(0))
    nul=np.asarray(nul);ns=nul.std(0,ddof=1);z=np.divide(np.abs(obs-nul.mean(0)),ns,out=np.full_like(obs,np.inf),where=ns>0);return obs,ns,z

def cvacc(X,y,groups):
    g=GroupKFold(5);acc=[];margins=[]
    for tr,te in g.split(X,y,groups):
        mu=X[tr].mean(0);sd=X[tr].std(0,ddof=1);sd=np.where(sd>1e-12,sd,1);m=LogisticRegression(C=1.0,solver='lbfgs',max_iter=3000).fit((X[tr]-mu)/sd,y[tr]);pr=m.predict_proba((X[te]-mu)/sd);pred=m.classes_[np.argmax(pr,axis=1)];acc.extend((pred==y[te]).tolist());ci={c:i for i,c in enumerate(m.classes_)}
        for row,true in zip(pr,y[te]):
            other=SOURCES[1] if true==SOURCES[0] else SOURCES[0];margins.append(math.log(max(row[ci[true]],1e-300))-math.log(max(row[ci[other]],1e-300)))
    return float(np.mean(acc)),float(np.mean(margins))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('job_ids',nargs='+');a=ap.parse_args();objs=[]
    with tempfile.TemporaryDirectory() as td:
        for jid in a.job_ids: objs+=parse(subprocess.check_output(['hf','jobs','logs',jid],text=True,stderr=subprocess.STDOUT))
    d={(o['source'],int(o['trial'])):o for o in objs if o.get('split')=='development'}
    for s in SOURCES:
        if sorted(t for x,t in d if x==s)!=list(range(20)):raise SystemExit(f'incomplete {s}')
    out={'n_per_source':20,'target_accessed':False,'representations':{}};names=next(iter(d.values()))['feature_names']
    for ri,rep in enumerate(REPS):
        A=np.asarray([feat(d[(SOURCES[0],t)],rep) for t in range(20)]);B=np.asarray([feat(d[(SOURCES[1],t)],rep) for t in range(20)]);obs,ns,z=permz(A,B,62000+ri);order=np.argsort(-z);X=np.vstack([A[:,USE_IDX],B[:,USE_IDX]]);y=np.array([SOURCES[0]]*20+[SOURCES[1]]*20);groups=np.array(list(range(20))*2);acc,margin=cvacc(X,y,groups);maxz=float(np.max(z));passed=bool(maxz>=3.0 or acc>=.80)
        out['representations'][rep]={'fixed_classifier_feature_names':[names[i] for i in USE_IDX],'grouped_cv_accuracy':acc,'mean_true_class_log_margin':margin,'max_feature_effect_over_null_sd':maxz,'pilot_go_pass':passed,'features_ranked':[{'feature':names[i],'working_minus_line_mean':float(obs[i]),'null_sd':float(ns[i]),'effect_over_null_sd':float(z[i])} for i in order]}
    out['overall_go']=bool(all(out['representations'][r]['pilot_go_pass'] for r in REPS));print('P1V2PILOT='+json.dumps(out,separators=(',',':')))
if __name__=='__main__':main()
