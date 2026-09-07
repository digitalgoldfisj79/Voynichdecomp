#!/usr/bin/env python3
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import memory_recurrence_score_v1 as score

TARGET_SPLIT='target'
TARGET_NS=415
TARGET_SOURCE='voynich_target'
TARGET_SOURCE_CODE=90

def load_source_runner(path='/workspace/source/instrumentation/memory_recurrence_runner_v1.py'):
    spec=importlib.util.spec_from_file_location('p1source',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    m.SPLIT_NS[TARGET_SPLIT]=TARGET_NS; m.SOURCE_CODE[TARGET_SOURCE]=TARGET_SOURCE_CODE
    return m

def load_t0b(path='/workspace/t0b/t0b_runner.py'):
    spec=importlib.util.spec_from_file_location('t0b_target',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def target_object(source_runner,t0b):
    folds=[]
    for f in range(5):
        tr,te=t0b.w.fold_data(f)
        lam,_=t0b.choose_lambda(tr,f); fit=t0b.RegionalModel(tr,f,lam=lam)
        gr=float(fit.score(te,'noregion')-fit.score(te,'full')); go=float(fit.score(te,'noorder')-fit.score(te,'full'))
        reps={}
        for rep in score.REPS:
            rr=source_runner.transform_rows(te,rep,t0b.s2)
            vv=source_runner.recurrence_features(rr,TARGET_SPLIT,TARGET_SOURCE,0,f,rep,t0b.s2)+[gr,go]
            if len(vv)!=20 or not all(np.isfinite(vv)): raise RuntimeError(f'bad target vector fold={f} rep={rep}')
            reps[rep]=[float(x) for x in vv]
        folds.append({'fold':f,'selected_lambda':float(lam),'features':reps})
    return {'assay_id':'memory_recurrence_source_id_v1','split':'target','source':TARGET_SOURCE,'trial':0,'strength':None,
            'feature_names':source_runner.FEATURE_NAMES,'folds':folds,'target_opened':True}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--development-jsonl',nargs='+',required=True); ap.add_argument('--output',required=True)
    a=ap.parse_args(); sr=load_source_runner(); t0b=load_t0b(); objs=score.load_jsonl(a.development_jsonl)
    dev=score.index_trials(objs,'development',score.CLASSES,80); models=score.fit_models(dev); o=target_object(sr,t0b)
    byrep={rep:score.trial_score(o,rep,models[rep]) for rep in score.REPS}
    preds=[byrep[r]['prediction'] for r in score.REPS]
    if any(byrep[r]['abstain'] for r in score.REPS): verdict='ABSTAIN'
    elif len(set(preds))!=1: verdict='ABSTAIN_REPRESENTATION_DEPENDENT'
    else: verdict=preds[0]
    out={'assay_id':'memory_recurrence_source_id_v1','target':'Voynich canonical 34,087-event surface','representations':byrep,
         'final_verdict':verdict,'representation_agreement':len(set(preds))==1,'target_opened':True,
         'licensed_scope':'Discrimination among the five frozen recurrence/active-repertoire source models only; no semantic, cipher, origin or authorship inference.'}
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    print('P1TARGET_RESULT='+json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()
