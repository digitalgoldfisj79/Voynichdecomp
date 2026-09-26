#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

import run_corema_recoverability_v06 as base


def one_fold(df: pd.DataFrame, X_struct: np.ndarray, y: np.ndarray, groups: np.ndarray,
             eligible: list[str], fold: int, tr: np.ndarray, te: np.ndarray) -> dict:
    train = df.iloc[tr]
    test = df.iloc[te]
    ytr = y[tr]
    yte = y[te]
    preds: dict[str, list[str]] = {}

    majority = Counter(ytr).most_common(1)[0][0]
    preds["majority"] = [majority] * len(te)

    lexical = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
                                  max_features=80000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=2.0,
                                   n_jobs=1)),
    ])
    lexical.fit(train["lex_context"], ytr)
    preds["lexical"] = lexical.predict(test["lex_context"]).tolist()

    rank = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2,
                                  max_features=30000, token_pattern=r"[^ ]+")),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0,
                                   n_jobs=1)),
    ])
    rank.fit(train["rank_context"], ytr)
    preds["rank"] = rank.predict(test["rank_context"]).tolist()

    pattern = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 5), min_df=2,
                                  max_features=40000)),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=1.0,
                                   n_jobs=1)),
    ])
    pattern.fit(train["pattern_context"], ytr)
    preds["pattern"] = pattern.predict(test["pattern_context"]).tolist()

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=4,
        class_weight="balanced_subsample", random_state=base.SEED + fold,
        n_jobs=4, max_features="sqrt",
    )
    rf.fit(X_struct[tr], ytr)
    sp = rf.predict_proba(X_struct[te])
    expanded = np.full((len(te), len(base.ROLE_ORDER)), 1e-9)
    for j, cls in enumerate(rf.classes_):
        expanded[:, base.ROLE_TO_INT[cls]] = sp[:, j]
    expanded /= expanded.sum(axis=1, keepdims=True)
    preds["structural"] = [base.ROLE_ORDER[i] for i in expanded.argmax(axis=1)]
    init_log, trans_log = base.fit_transition(ytr, train["recipe_id"].to_numpy())
    preds["structural_hmm"] = base.hmm_decode_by_recipe(
        test.reset_index(drop=True), expanded, init_log, trans_log
    )

    fold_rows = []
    for model, p in preds.items():
        mm = base.metrics(yte, p, eligible)
        fold_rows.append({
            "fold": fold,
            "model": model,
            "test_manuscripts": sorted(set(groups[te])),
            **{k: v for k, v in mm.items() if k != "per_class"},
        })
    print(f"parallel fold {fold} complete: {len(te)} tokens", flush=True)
    return {"fold": fold, "te": te.tolist(), "preds": preds, "fold_rows": fold_rows}


def parallel_token_cv(df: pd.DataFrame, n_jobs: int) -> dict:
    eligible = base.eligible_roles(df)
    groups = df["manuscript"].to_numpy()
    y = df["role"].to_numpy()
    X_struct = base.structural_matrix(df)
    splits = list(GroupKFold(n_splits=min(5, len(np.unique(groups)))).split(df, y, groups))
    results = Parallel(n_jobs=min(n_jobs, len(splits)), backend="loky", verbose=10)(
        delayed(one_fold)(df, X_struct, y, groups, eligible, fold, tr, te)
        for fold, (tr, te) in enumerate(splits, 1)
    )
    models = ["majority", "lexical", "rank", "pattern", "structural", "structural_hmm"]
    pred_store = {m: np.empty(len(df), dtype=object) for m in models}
    fold_rows = []
    for result in results:
        te = np.asarray(result["te"], dtype=int)
        for model in models:
            pred_store[model][te] = result["preds"][model]
        fold_rows.extend(result["fold_rows"])
    summary = {m: base.metrics(y, pred_store[m].tolist(), eligible) for m in models}
    return {
        "eligible_roles": eligible,
        "folds": sorted(fold_rows, key=lambda x: (x["fold"], x["model"])),
        "summary": summary,
        "predictions": {m: p.tolist() for m, p in pred_store.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fold-jobs", type=int, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    acquisition = base.download_corema(args.data_dir)
    tok, rec, parse_audit = base.parse_corema(args.data_dir)
    tok = base.enrich_token_features(tok)
    if tok["manuscript"].nunique() < 8 or len(rec) < 100 or len(tok) < 5000:
        raise RuntimeError(
            f"Insufficient corpus: {tok.manuscript.nunique()} manuscripts, "
            f"{len(rec)} units, {len(tok)} tokens"
        )

    cv = parallel_token_cv(tok, args.fold_jobs)
    order = base.role_order_gain(tok)
    recipe_cv = base.run_recipe_type_cv(rec)
    eligible = cv["eligible_roles"]
    lex = cv["summary"]["lexical"]
    neutral = cv["summary"]["structural_hmm"]
    maj = cv["summary"]["majority"]
    strong_classes = sum(
        lex["per_class"].get(r, {}).get("f1-score", 0) >= 0.40 for r in eligible
    )
    gates = {
        "lexical_role": bool(
            lex["macro_f1_eligible"] >= 0.60
            and strong_classes >= min(3, len(eligible))
        ),
        "neutral_role": bool(
            neutral["macro_f1_eligible"] >= 0.35
            and neutral["macro_f1_eligible"] - maj["macro_f1_eligible"] >= 0.10
        ),
        "role_order": bool(
            order["mean_real_vs_shuffle_bpt"] >= 0.05
            and all(x["real_vs_shuffle_bpt"] > 0 for x in order["folds"])
        ),
    }
    gates["downstream_admissible"] = bool(
        gates["lexical_role"] and gates["neutral_role"] and gates["role_order"]
    )
    formal = "CALIBRATION_PASS" if gates["downstream_admissible"] else "CALIBRATION_FAILURE"
    result = {
        "schema": "corema-procedural-recoverability-v0.6",
        "execution": "parallel-fold implementation; frozen estimators and splits unchanged",
        "formal_verdict": formal,
        "corpus": {
            "manuscripts": int(tok["manuscript"].nunique()),
            "recipes": int(len(rec)),
            "tokens": int(len(tok)),
            "role_counts": tok["role"].value_counts().to_dict(),
            "type_counts": rec["recipe_type"].value_counts().to_dict(),
        },
        "acquisition": acquisition,
        "parse_audit": parse_audit,
        "token_role_cv": cv,
        "role_order": order,
        "recipe_type_cv": recipe_cv,
        "gates": gates,
    }
    out_json = args.out / "corema_recoverability_results_v0_6.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    base.write_report(result, args.out / "COREMA_RECOVERABILITY_REPORT_v0_6.md")
    pred = tok[["manuscript", "recipe_id", "position", "token", "role"]].copy()
    for model, vals in cv["predictions"].items():
        pred[f"pred_{model}"] = vals
    pred.to_csv(args.out / "corema_role_predictions_v0_6.csv.gz", index=False)
    tok.to_csv(args.out / "corema_token_rows_v0_6.csv.gz", index=False)
    rec.to_csv(args.out / "corema_recipe_rows_v0_6.csv.gz", index=False)
    print(json.dumps({"formal_verdict": formal, "corpus": result["corpus"], "gates": gates}, indent=2), flush=True)


if __name__ == "__main__":
    main()
