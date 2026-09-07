#!/usr/bin/env python3
import argparse, importlib.util, json, subprocess, tempfile
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('job_ids',nargs='+'); a=ap.parse_args()
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('sc',here/'memory_recurrence_score_v1.py')
    sc=importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    paths=[]
    with tempfile.TemporaryDirectory() as td:
        for jid in a.job_ids:
            text=subprocess.check_output(['hf','jobs','logs',jid],text=True,stderr=subprocess.STDOUT)
            p=Path(td)/(jid+'.log'); p.write_text(text); paths.append(str(p))
        objs=sc.load_jsonl(paths); dev=sc.index_trials(objs,'development',sc.CLASSES,80); fit=sc.fit_models(dev)
        out={'n_development_manuscripts':len(dev),'classes':sc.CLASSES,'representations':{}}
        for rep in sc.REPS:
            folds=[]
            for f,fm in enumerate(fit[rep]['foldmods']):
                folds.append({'fold':f,'selected_C':float(fm['C']),'cv_logloss_ranked':[{'mean_logloss':float(x[0]),'C':float(x[1])} for x in fm['cv']]})
            env={}
            for cls,e in fit[rep]['env'].items():
                ds=np.asarray(e['development_distances'],float)
                env[cls]={'ood_threshold':float(e['threshold']),'mean_dev_distance':float(ds.mean()),'sd_dev_distance':float(ds.std(ddof=1)),'max_dev_distance':float(ds.max())}
            out['representations'][rep]={'fold_models':folds,'ood_envelopes':env}
        print('P1DEVFIT='+json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()
