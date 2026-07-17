#!/usr/bin/env python3
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import io
import json
import re
import subprocess
import tarfile
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

SOURCE_JOB_ID = "6a5a1540d216bd6f3a1fb177"
EXPECTED_BUNDLE_SHA256 = "7cdeb84c9b533e1d14f89a5102f4c8f050bbb9fd260fce06a5f5be27a1e339fa"
SEED = 20260719
BOOTSTRAPS = 2000


def reconstruct() -> dict[str, np.ndarray]:
    p = subprocess.run(["hf", "jobs", "logs", SOURCE_JOB_ID], capture_output=True, text=True, check=True)
    chunks = {}
    count = None
    for line in p.stdout.splitlines():
        if line.startswith("V15_BUNDLE_BEGIN "):
            count = json.loads(line.split(" ", 1)[1])["chunks"]
        m = re.match(r"^V15_BUNDLE_CHUNK\s+(\d+)\s+(.+)$", line)
        if m:
            chunks[int(m.group(1))] = m.group(2).strip()
    if count is None or sorted(chunks) != list(range(count)):
        raise RuntimeError("incomplete source bundle")
    raw = base64.b64decode("".join(chunks[i] for i in range(count)), validate=True)
    if hashlib.sha256(raw).hexdigest() != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError("bundle SHA mismatch")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as t:
        b = t.extractfile("exact_features.npz").read()
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def pair_table(x: np.ndarray, writers: np.ndarray) -> dict[str, np.ndarray]:
    x = l2(x)
    i, j = np.triu_indices(len(x), 1)
    return {
        "score": np.sum(x[i] * x[j], axis=1),
        "label": (writers[i] == writers[j]).astype(np.int8),
        "writer_i": writers[i].astype(str),
        "writer_j": writers[j].astype(str),
    }


def eer(y: np.ndarray, score: np.ndarray, weight: np.ndarray | None = None) -> float:
    fpr, tpr, _ = roc_curve(y, score, sample_weight=weight)
    fnr = 1.0 - tpr
    k = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[k] + fnr[k]) / 2.0)


def metrics(table: dict[str, np.ndarray], weight: np.ndarray | None = None) -> dict[str, float]:
    y, s = table["label"], table["score"]
    prevalence = float(np.average(y, weights=weight)) if weight is not None else float(y.mean())
    ap = float(average_precision_score(y, s, sample_weight=weight))
    return {
        "roc_auc": float(roc_auc_score(y, s, sample_weight=weight)),
        "average_precision": ap,
        "prevalence": prevalence,
        "ap_lift": ap - prevalence,
        "equal_error_rate": eer(y, s, weight),
    }


def retrieval(x: np.ndarray, writers: np.ndarray) -> dict[str, float]:
    x = l2(x); sim = x @ x.T
    aps=[]; top1=[]; top5=[]
    for q in range(len(x)):
        rel = writers == writers[q]; rel[q] = False
        order = np.argsort(-sim[q]); order = order[order != q]
        rr = rel[order].astype(np.int8); ranks = np.flatnonzero(rr) + 1
        aps.append(float(np.mean(np.arange(1, len(ranks)+1) / ranks)))
        top1.append(float(rr[:1].any())); top5.append(float(rr[:5].any()))
    return {"map":float(np.mean(aps)),"top1":float(np.mean(top1)),"top5":float(np.mean(top5)),"eligible_queries":len(aps)}


def pair_weights(table: dict[str, np.ndarray], counts: dict[str, int]) -> np.ndarray:
    a = np.fromiter((counts.get(w,0) for w in table["writer_i"]), dtype=float, count=len(table["writer_i"]))
    b = np.fromiter((counts.get(w,0) for w in table["writer_j"]), dtype=float, count=len(table["writer_j"]))
    return np.where(table["label"] == 1, a, a*b)


def bootstrap(tables: dict[str, dict[str, np.ndarray]], writers: np.ndarray) -> dict[str, Any]:
    unique=np.array(sorted(set(writers.astype(str)))); rng=np.random.default_rng(SEED+20)
    values={name:defaultdict(list) for name in tables}; diffs=defaultdict(list)
    for _ in range(BOOTSTRAPS):
        sample=rng.choice(unique,size=len(unique),replace=True)
        keys,nums=np.unique(sample,return_counts=True); counts=dict(zip(keys,nums))
        rep={}
        for name,table in tables.items():
            w=pair_weights(table,counts); keep=w>0; t={k:v[keep] for k,v in table.items()}
            rep[name]=metrics(t,w[keep])
            for k,v in rep[name].items(): values[name][k].append(v)
        for nuisance in ["acquisition_scaled","ink_scaled","combined_scaled","combined_raw"]:
            diffs[nuisance].append(rep["selected"]["roc_auc"]-rep[nuisance]["roc_auc"])
    return {
        "replicates":BOOTSTRAPS,
        "metrics":{name:{k:{"lower_95":float(np.quantile(v,.025)),"upper_95":float(np.quantile(v,.975))} for k,v in d.items()} for name,d in values.items()},
        "selected_auc_minus_nuisance":{name:{"lower_95":float(np.quantile(v,.025)),"upper_95":float(np.quantile(v,.975))} for name,v in diffs.items()},
    }


def ece(y: np.ndarray, p: np.ndarray, bins: int=10) -> float:
    edges=np.linspace(0,1,bins+1); out=0.0
    for a,b in zip(edges[:-1],edges[1:]):
        mask=(p>=a)&((p<=b) if b==1 else (p<b))
        if mask.any(): out += mask.mean()*abs(float(p[mask].mean())-float(y[mask].mean()))
    return float(out)


def main() -> int:
    z=reconstruct(); writers=z["test_writers"].astype(str)
    unique=np.array(sorted(set(writers))); rng=np.random.default_rng(SEED+2); rng.shuffle(unique)
    calibration=set(unique[:30]); evaluation=set(unique[30:])
    fit_mask=np.array([w in calibration for w in writers]); eval_mask=np.array([w in evaluation for w in writers])
    acq=z["acquisition_test"].astype(float); ink=z["ink_test"].astype(float); combined=np.concatenate([acq,ink],axis=1)
    reps={
        "selected":z["test_selected"][eval_mask],
        "raw":z["test"][eval_mask],
        "acquisition_scaled":StandardScaler().fit(acq[fit_mask]).transform(acq[eval_mask]),
        "ink_scaled":StandardScaler().fit(ink[fit_mask]).transform(ink[eval_mask]),
        "combined_scaled":StandardScaler().fit(combined[fit_mask]).transform(combined[eval_mask]),
        "combined_raw":combined[eval_mask],
    }
    ew=writers[eval_mask]; tables={name:pair_table(x,ew) for name,x in reps.items()}
    point={name:metrics(t) for name,t in tables.items()}
    retr={name:retrieval(x,ew) for name,x in reps.items()}
    # Fit probability calibration solely on the disjoint calibration writers.
    cal_table=pair_table(z["test_selected"][fit_mask],writers[fit_mask])
    model=LogisticRegression(C=1.0,solver="lbfgs",max_iter=1000,random_state=SEED)
    model.fit(cal_table["score"][:,None],cal_table["label"])
    prob=model.predict_proba(tables["selected"]["score"][:,None])[:,1]
    y=tables["selected"]["label"]; prev=float(y.mean())
    result={
        "schema":"blind-pal-saghog-v1.6-stage0b-writer-disjoint-nuisance-audit",
        "source_job_id":SOURCE_JOB_ID,"bundle_sha256":EXPECTED_BUNDLE_SHA256,"seed":SEED,
        "calibration_writers":sorted(calibration),"evaluation_writers":sorted(evaluation),
        "calibration_pages":int(fit_mask.sum()),"evaluation_pages":int(eval_mask.sum()),
        "point_metrics":point,"retrieval":retr,"writer_cluster_bootstrap":bootstrap(tables,ew),
        "calibration":{"brier":float(brier_score_loss(y,prob)),"prevalence_predictor_brier":float(brier_score_loss(y,np.full(len(y),prev))),"ece_10_bins":ece(y,prob),"coefficient":float(model.coef_[0,0]),"intercept":float(model.intercept_[0])},
        "interpretation_constraint":"Post-stage0 audit prompted by terminal-fitted nuisance scaling. It is descriptive and cannot count as independent validation.",
        "seal":{"voynich_opened":False,"davis_labels_loaded":False,"f115r_loaded":False},
    }
    print("V16_STAGE0B_RESULT "+json.dumps(result,sort_keys=True),flush=True)
    return 0

if __name__=="__main__": raise SystemExit(main())
