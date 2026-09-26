#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold


def make_external_cv(h):
    def external_cv(df,feature_cols):
        historical={"organ_tablature","neume_aquitanian","neume_square"}
        broad=df[~df["family"].isin(["procedural_synthetic","neume_unknown"])].copy()
        broad["y"]=broad["family"].isin(historical).astype(int)
        X=broad[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
        y=broad["y"].to_numpy(int); groups=broad["group"].astype(str).to_numpy()
        split=StratifiedGroupKFold(n_splits=min(5,len(set(groups))),shuffle=True,random_state=h.SEED)
        pred={name:np.zeros(len(y)) for name in h.models()}; folds=[]
        for fold,(tr,te) in enumerate(split.split(X,y,groups)):
            for name,model in h.models().items():
                pred[name][te]=clone(model).fit(X[tr],y[tr]).predict_proba(X[te])[:,1]
            folds.append({"fold":fold,"train":len(tr),"test":len(te),"test_groups":sorted(set(groups[te])),"test_positive":int(y[te].sum())})
        ensemble=np.mean(np.vstack(list(pred.values())),axis=0)
        metrics={name:{"roc_auc":float(roc_auc_score(y,p)),"average_precision":float(average_precision_score(y,p)),"balanced_accuracy_0_5":float(balanced_accuracy_score(y,p>=.5))} for name,p in {**pred,"ensemble":ensemble}.items()}
        candidates=[]
        for t in np.linspace(.01,.99,99):
            yp=ensemble>=t; tn=int(((y==0)&(~yp)).sum()); fp=int(((y==0)&yp).sum()); tp=int(((y==1)&yp).sum()); fn=int(((y==1)&(~yp)).sum())
            fpr=fp/max(1,fp+tn); tpr=tp/max(1,tp+fn); precision=tp/max(1,tp+fp)
            if fpr<=.10: candidates.append((tpr,precision,t,fpr))
        if candidates: tpr,precision,threshold,fpr=max(candidates,key=lambda x:(x[0],x[1],x[2]))
        else: threshold=.5; fpr=tpr=precision=float('nan')
        yp=ensemble>=threshold; fam_array=broad["family"].to_numpy()
        recalls={fam:(float(np.mean(yp[fam_array==fam])) if np.any(fam_array==fam) else None) for fam in sorted(historical)}
        organ=recalls.get("organ_tablature") or 0.0
        gate=metrics["logistic"]["roc_auc"]>=.80 and metrics["forest"]["roc_auc"]>=.80 and balanced_accuracy_score(y,yp)>=.70 and organ>=.60
        return {"rows":len(broad),"groups":len(set(groups)),"folds":folds,"metrics":metrics,"threshold":float(threshold),"threshold_cv":{"fpr":float(fpr),"tpr":float(tpr),"precision":float(precision)},"balanced_accuracy_threshold":float(balanced_accuracy_score(y,yp)),"organ_recall_threshold":float(organ),"per_historical_family_recall":recalls,"gate_pass":bool(gate),"cv_predictions":pd.DataFrame({"family":broad["family"],"group":broad["group"],"y":y,**pred,"ensemble":ensemble}).to_dict("records")}
    return external_cv


def make_family_cv(h):
    def family_cv(df,feature_cols):
        clean=df[df["family"]!="neume_unknown"].copy()
        X=clean[feature_cols].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
        labels=sorted(clean["family"].unique()); enc={x:i for i,x in enumerate(labels)}
        y=np.array([enc[x] for x in clean["family"].astype(str)]); groups=clean["group"].astype(str).to_numpy()
        split=StratifiedGroupKFold(n_splits=min(5,len(set(groups))),shuffle=True,random_state=h.SEED)
        probs={name:np.zeros((len(y),len(labels))) for name in h.models(False)}
        for tr,te in split.split(X,y,groups):
            for name,model in h.models(False).items():
                m=clone(model).fit(X[tr],y[tr]); raw=m.predict_proba(X[te]); classes=m.classes_ if hasattr(m,"classes_") else m.named_steps["clf"].classes_
                for j,c in enumerate(classes): probs[name][te,int(c)]=raw[:,j]
        ensemble=np.mean(np.stack(list(probs.values())),axis=0); pred=ensemble.argmax(axis=1)
        macro=float(f1_score(y,pred,average="macro"))
        return {"rows":len(clean),"labels":labels,"macro_f1":macro,"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"confusion_matrix":confusion_matrix(y,pred,labels=list(range(len(labels)))).tolist(),"classification_report":classification_report(y,pred,target_names=labels,output_dict=True,zero_division=0),"gate_pass":bool(macro>=.55)}
    return family_cv
