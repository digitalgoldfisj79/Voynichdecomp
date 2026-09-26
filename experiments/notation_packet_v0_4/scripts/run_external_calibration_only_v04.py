#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_historical_notation_classifier_v04 as h
import historical_classifier_corrections_v04 as corrections
h=corrections.install(h)


def balanced_frame(df,per_group=100,per_family=500):
    """Post-hoc robustness sample; never used for the frozen primary gate."""
    chunks=[]
    for _,g in df.groupby(['family','group'],sort=True):
        chunks.append(g.sample(n=min(per_group,len(g)),random_state=h.SEED))
    x=pd.concat(chunks,ignore_index=True); out=[]
    for _,g in x.groupby('family',sort=True):
        out.append(g.sample(n=min(per_family,len(g)),random_state=h.SEED+1))
    return pd.concat(out,ignore_index=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ammerbach-dir',type=Path,required=True)
    ap.add_argument('--gabc-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    reps,aa=h.load_ammerbach(args.ammerbach_dir); gg,ga=h.read_gabc_files(args.gabc_dir)
    results={}; frames={}
    for rep in ('paired','pitch','flattened'):
        df,features=h.dataframe_from_rows(h.build_external_rows(reps,gg,rep))
        broad=h.external_cv(df,features); fam=h.family_cv(df,features)
        robust_df=balanced_frame(df); robust_broad=h.external_cv(robust_df,features); robust_fam=h.family_cv(robust_df,features)
        results[rep]={'counts':df['family'].value_counts().to_dict(),'group_counts':df.groupby('family')['group'].nunique().to_dict(),'features':features,'broad':{k:v for k,v in broad.items() if k!='cv_predictions'},'family':fam,'balanced_sensitivity':{'counts':robust_df['family'].value_counts().to_dict(),'broad':{k:v for k,v in robust_broad.items() if k!='cv_predictions'},'family':robust_fam}}
        frames[rep]=df
        df.to_csv(args.out/f'external_features_{rep}_v0_4.csv',index=False)
        pd.DataFrame(broad['cv_predictions']).to_csv(args.out/f'external_cv_predictions_{rep}_v0_4.csv',index=False)
        pd.DataFrame(robust_broad['cv_predictions']).to_csv(args.out/f'external_cv_predictions_{rep}_balanced_sensitivity_v0_4.csv',index=False)
        print(rep,broad['metrics']['ensemble']['roc_auc'],broad['gate_pass'],fam['macro_f1'],'sensitivity',robust_broad['metrics']['ensemble']['roc_auc'],robust_fam['macro_f1'],flush=True)
    tie={'paired':2,'pitch':1,'flattened':0}
    selected=max(results,key=lambda r:(results[r]['broad']['metrics']['ensemble']['roc_auc'],results[r]['broad']['balanced_accuracy_threshold'],tie[r]))
    sr=results[selected]; gate=bool(sr['broad']['gate_pass']); family_gate=gate and bool(sr['family']['gate_pass'])
    frames[selected].to_csv(args.out/'selected_external_features_v0_4.csv',index=False)
    out={'schema':'historical-notation-external-calibration-v0.4-corrected','selected_representation':selected,'external_gate_pass':gate,'family_gate_pass':family_gate,'voynich_opened':False,'ammerbach_audit':aa,'gabc_audit':ga,'representations':results}
    (args.out/'external_notation_calibration_results_v0_4.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.out/'ammerbach_intake_audit_v0_4.json').write_text(json.dumps(aa,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# External historical-notation calibration v0.4 — corrected intake','',f'**Selected representation:** `{selected}`',f"**External gate:** {'PASS' if gate else 'FAIL / ABSTAIN'}",f"**Family gate:** {'PASS' if family_gate else 'FAIL / ABSTAIN'}",'','The Voynich target remained sealed during this run. The balanced sensitivity is post-hoc and does not alter the frozen gate.','']
    for rep,rr in results.items():
        b=rr['broad']; rb=rr['balanced_sensitivity']['broad']
        lines += [f'## {rep}','',f"- Counts: `{rr['counts']}`",f"- Groups: `{rr['group_counts']}`",f"- Ensemble ROC AUC: {b['metrics']['ensemble']['roc_auc']:.4f}",f"- Logistic ROC AUC: {b['metrics']['logistic']['roc_auc']:.4f}",f"- Forest ROC AUC: {b['metrics']['forest']['roc_auc']:.4f}",f"- Balanced accuracy: {b['balanced_accuracy_threshold']:.4f}",f"- Historical-family recalls: `{b['per_historical_family_recall']}`",f"- Family macro-F1: {rr['family']['macro_f1']:.4f}",f"- Frozen gate: {'PASS' if b['gate_pass'] else 'FAIL'}",f"- Balanced-sensitivity AUC: {rb['metrics']['ensemble']['roc_auc']:.4f}",f"- Balanced-sensitivity family macro-F1: {rr['balanced_sensitivity']['family']['macro_f1']:.4f}",'']
    (args.out/'EXTERNAL_HISTORICAL_NOTATION_CALIBRATION_v0_4.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'selected':selected,'gate':gate,'family_gate':family_gate},indent=2))
if __name__=='__main__': main()
