#!/usr/bin/env python3
import argparse, importlib.util, json, subprocess, tempfile
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dev-jobs',nargs='+',required=True); ap.add_argument('--cal-jobs',nargs='+',required=True); a=ap.parse_args()
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('sc',here/'memory_recurrence_score_v1_1.py')
    sc=importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    with tempfile.TemporaryDirectory() as td:
        paths=[]
        for jid in a.dev_jobs+a.cal_jobs:
            text=subprocess.check_output(['hf','jobs','logs',jid],text=True,stderr=subprocess.STDOUT)
            p=Path(td)/(jid+'.log'); p.write_text(text); paths.append(str(p))
        objs=sc.load_jsonl(paths)
        dev=sc.index_trials(objs,'development',sc.CLASSES,80); cal=sc.index_trials(objs,'calibration',sc.CLASSES,40)
        fit=sc.fit_models(dev); out={'n_development':len(dev),'n_calibration':len(cal),'thresholds_changed':False,'representations':{}}
        for rep in sc.REPS:
            scores,src=sc.evaluate_known(cal,rep,fit[rep],40)
            detail=[]
            for s in sc.CLASSES:
                ss=[scores[(s,t)] for t in range(40)]
                detail.append({'name':s,'n':40,'correct':sum(x['prediction']==s for x in ss),'abstained':sum(x['abstain'] for x in ss),'mean_top_p':float(np.mean([x['top_p'] for x in ss])),'mean_margin':float(np.mean([x['margin'] for x in ss])),'mean_ood':float(np.mean([x['ood_score'] for x in ss])),'max_ood':float(np.max([x['ood_score'] for x in ss]))})
            out['representations'][rep]={'selected_C_by_fold':[float(x['C']) for x in fit[rep]['foldmods']],'source_rows':src,'detail':detail}
        print('P1CAL='+json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()
