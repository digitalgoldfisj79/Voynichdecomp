#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_historical_notation_classifier_v04 as h


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ammerbach-dir',type=Path,required=True)
    ap.add_argument('--gabc-dir',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    reps,aa=h.load_ammerbach(args.ammerbach_dir)
    gg,ga=h.read_gabc_files(args.gabc_dir)
    results={}; frames={}
    for rep in ('paired','pitch','flattened'):
        rows=h.build_external_rows(reps,gg,rep)
        df,features=h.dataframe_from_rows(rows)
        broad=h.external_cv(df,features)
        fam=h.family_cv(df,features)
        results[rep]={'counts':df['family'].value_counts().to_dict(),'features':features,'broad':{k:v for k,v in broad.items() if k!='cv_predictions'},'family':fam}
        frames[rep]=df
        pd.DataFrame(broad['cv_predictions']).to_csv(args.out/f'external_cv_predictions_{rep}_v0_4.csv',index=False)
        print(rep,broad['metrics']['ensemble']['roc_auc'],broad['gate_pass'],fam['macro_f1'],flush=True)
    tie={'paired':2,'pitch':1,'flattened':0}
    selected=max(results,key=lambda r:(results[r]['broad']['metrics']['ensemble']['roc_auc'],results[r]['broad']['balanced_accuracy_threshold'],tie[r]))
    selected_result=results[selected]
    gate=bool(selected_result['broad']['gate_pass'])
    family_gate=gate and bool(selected_result['family']['gate_pass'])
    frames[selected].to_csv(args.out/'selected_external_features_v0_4.csv',index=False)
    out={'schema':'historical-notation-external-calibration-v0.4','selected_representation':selected,'external_gate_pass':gate,'family_gate_pass':family_gate,'voynich_opened':False,'ammerbach_audit':aa,'gabc_audit':ga,'representations':results}
    (args.out/'external_notation_calibration_results_v0_4.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    (args.out/'ammerbach_intake_audit_v0_4.json').write_text(json.dumps(aa,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# External historical-notation calibration v0.4','',f'**Selected representation:** `{selected}`',f"**External gate:** {'PASS' if gate else 'FAIL / ABSTAIN'}",f"**Family gate:** {'PASS' if family_gate else 'FAIL / ABSTAIN'}",'','The Voynich target remained sealed during this run.','']
    for rep,rr in results.items():
        b=rr['broad']; lines += [f'## {rep}','',f"- Ensemble ROC AUC: {b['metrics']['ensemble']['roc_auc']:.4f}",f"- Logistic ROC AUC: {b['metrics']['logistic']['roc_auc']:.4f}",f"- Forest ROC AUC: {b['metrics']['forest']['roc_auc']:.4f}",f"- Balanced accuracy: {b['balanced_accuracy_threshold']:.4f}",f"- Organ recall: {b['organ_recall_threshold']:.4f}",f"- Family macro-F1: {rr['family']['macro_f1']:.4f}",f"- Gate: {'PASS' if b['gate_pass'] else 'FAIL'}",'']
    (args.out/'EXTERNAL_HISTORICAL_NOTATION_CALIBRATION_v0_4.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'selected':selected,'gate':gate,'family_gate':family_gate},indent=2))
if __name__=='__main__': main()
