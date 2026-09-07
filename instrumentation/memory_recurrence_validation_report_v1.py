#!/usr/bin/env python3
import argparse, importlib.util, json, subprocess, tempfile
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dev-jobs',nargs='+',required=True); ap.add_argument('--val-jobs',nargs='+',required=True); a=ap.parse_args()
    here=Path(__file__).resolve().parent
    spec=importlib.util.spec_from_file_location('sc',here/'memory_recurrence_score_v1_1.py')
    sc=importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
    with tempfile.TemporaryDirectory() as td:
        paths=[]
        for jid in a.dev_jobs+a.val_jobs:
            text=subprocess.check_output(['hf','jobs','logs',jid],text=True,stderr=subprocess.STDOUT)
            p=Path(td)/(jid+'.log'); p.write_text(text); paths.append(str(p))
        objs=sc.load_jsonl(paths); dev=sc.index_trials(objs,'development',sc.CLASSES,80); val=sc.index_trials(objs,'validation',sc.CLASSES,50); fit=sc.fit_models(dev)
        out={'n_development':len(dev),'n_validation':len(val),'representations':{}}
        for rep in sc.REPS:
            scores,src=sc.evaluate_known(val,rep,fit[rep],50); pairs=sc.pairwise_validation(scores,rep)
            recalls=[x['correct']/x['n'] for x in src]; abst=[x['abstained']/x['n'] for x in src]
            src_gate=all(r>=.80 and a<=.10 for r,a in zip(recalls,abst)) and float(np.mean(recalls))>=.85
            pair_gate=all(x['effect']>0 and x['effect_over_null_sd']>=2 for x in pairs)
            out['representations'][rep]={'source_validation':src,'macro_recall':float(np.mean(recalls)),'source_gate':bool(src_gate),'pairwise_separations':pairs,'pairwise_gate':bool(pair_gate),'pass_without_unknown_controls':bool(src_gate and pair_gate)}
        print('P1VAL='+json.dumps(out,separators=(',',':')))
if __name__=='__main__': main()
