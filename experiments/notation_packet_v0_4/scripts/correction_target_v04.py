#!/usr/bin/env python3
import numpy as np
from sklearn.base import clone


def make_fit_and_predict(h):
    def fit_and_predict(external,target,features,broad_result,fam_result):
        historical={"organ_tablature","neume_aquitanian","neume_square"}
        broad=external[~external["family"].isin(["procedural_synthetic","neume_unknown"])].copy()
        y=broad["family"].isin(historical).astype(int).to_numpy()
        X=broad[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
        Xt=target[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
        bp=np.mean(np.vstack([clone(m).fit(X,y).predict_proba(Xt)[:,1] for m in h.models().values()]),axis=0)
        out=target[[c for c in ["group","section","window_id"] if c in target.columns]].copy()
        out["historical_notation_probability"]=bp
        out["above_external_threshold"]=bp>=broad_result["threshold"]
        fam_external=external[external["family"]!="neume_unknown"].copy()
        labels=fam_result["labels"]; enc={x:i for i,x in enumerate(labels)}
        yf=np.array([enc[x] for x in fam_external["family"].astype(str)])
        Xf=fam_external[features].replace([np.inf,-np.inf],np.nan).fillna(0).to_numpy(float)
        all_probs=[]
        for model in h.models(False).values():
            m=clone(model).fit(Xf,yf); raw=m.predict_proba(Xt); arr=np.zeros((len(Xt),len(labels)))
            classes=m.classes_ if hasattr(m,"classes_") else m.named_steps["clf"].classes_
            for j,c in enumerate(classes): arr[:,int(c)]=raw[:,j]
            all_probs.append(arr)
        fam_prob=np.mean(np.stack(all_probs),axis=0)
        for j,lab in enumerate(labels): out[f"p_{lab}"]=fam_prob[:,j]
        out["predicted_family"]=[labels[i] for i in fam_prob.argmax(axis=1)]
        summary={"windows":len(out),"mean_historical_notation_probability":float(bp.mean()),"median_historical_notation_probability":float(np.median(bp)),"fraction_above_threshold":float(np.mean(out["above_external_threshold"])),"mean_family_probabilities":{lab:float(fam_prob[:,j].mean()) for j,lab in enumerate(labels)},"predicted_family_counts":out["predicted_family"].value_counts().to_dict(),"by_section":{}}
        if "section" in out:
            for sec,d in out.groupby("section"):
                summary["by_section"][str(sec)]={"n":len(d),"mean_historical_probability":float(d["historical_notation_probability"].mean()),"fraction_above_threshold":float(d["above_external_threshold"].mean()),"family_counts":d["predicted_family"].value_counts().to_dict()}
        return {"summary":summary,"rows":out.to_dict("records")}
    return fit_and_predict
