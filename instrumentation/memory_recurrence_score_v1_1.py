#!/usr/bin/env python3
"""Pre-validation correction for memory_recurrence_score_v1.

The v1 scorer built probability columns in frozen CLASSES order but passed them to
sklearn.metrics.log_loss with string labels. sklearn orders those labels
lexicographically, so the development CV loss associated probabilities with the
wrong classes. This wrapper changes only cv_select_C: it computes the intended
multiclass NLL directly from the frozen class-column mapping. All other scoring,
OOD, abstention, validation, control and summary code remains the v1 code.
"""
import importlib.util
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

_here=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('memory_recurrence_score_v1_base',_here/'memory_recurrence_score_v1.py')
base=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)


def cv_select_C_fixed(X,y,groups):
    gkf=GroupKFold(n_splits=5); scores=[]
    for C in base.C_GRID:
        losses=[]
        for tr,te in gkf.split(X,y,groups):
            mu=X[tr].mean(0); sd=X[tr].std(0,ddof=1); sd=np.where(sd>1e-12,sd,1.0)
            m=LogisticRegression(C=C,solver='lbfgs',max_iter=3000).fit((X[tr]-mu)/sd,y[tr])
            pr=m.predict_proba((X[te]-mu)/sd)
            P=np.zeros((len(te),len(base.CLASSES)))
            for j,c in enumerate(m.classes_): P[:,base.CLASSES.index(c)]=pr[:,j]
            yi=np.fromiter((base.CLASSES.index(v) for v in y[te]),dtype=int,count=len(te))
            losses.append(float(-np.mean(np.log(np.maximum(P[np.arange(len(te)),yi],1e-300)))))
        scores.append((float(np.mean(losses)),C))
    scores.sort(key=lambda z:(z[0],base.C_GRID.index(z[1])))
    return scores[0][1],scores

base.cv_select_C=cv_select_C_fixed

# Re-export the corrected module API for reporting/final scoring.
for _name in dir(base):
    if not _name.startswith('_') and _name not in globals(): globals()[_name]=getattr(base,_name)
cv_select_C=cv_select_C_fixed

if __name__=='__main__':
    base.main()
