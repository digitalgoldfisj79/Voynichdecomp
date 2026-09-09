#!/usr/bin/env python3
"""Preregistered primary model for production_localisation_v1.

This script is intentionally conservative and should not be modified after primary
outputs are viewed without a version bump and decision-log entry.
"""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 4081425


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', type=Path, required=True)
    ap.add_argument('--sample', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    feat = pd.read_csv(args.features)
    samp = pd.read_csv(args.sample)
    df = samp.merge(feat, on='manuscript_id', how='inner', validate='one_to_one')
    df = df[df['production_region_strict'].isin(['IT', 'DE'])].copy()
    if len(df) == 0:
        raise SystemExit('No frozen IT/DE training rows found.')

    y = (df['production_region_strict'] == 'IT').astype(int).to_numpy()
    X = df.drop(columns=[c for c in ['production_region_strict', 'sample_order', 'included_primary',
                                     'date_bin', 'region_stratum', 'group_id', 'sampling_seed']
                         if c in df.columns])
    if 'manuscript_id' in X:
        X = X.drop(columns=['manuscript_id'])

    # Exclude features with >40% missingness before model fitting, as preregistered.
    missing = X.isna().mean()
    keep = missing[missing <= 0.40].index.tolist()
    X = X[keep]

    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    prep = ColumnTransformer([
        ('num', num_pipe, numeric),
        ('cat', cat_pipe, categorical),
    ])
    model = Pipeline([
        ('prep', prep),
        ('clf', LogisticRegression(penalty='l2', class_weight='balanced', max_iter=5000,
                                   random_state=SEED)),
    ])

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=SEED)
    p = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    pred = (p >= 0.5).astype(int)

    metrics = {
        'n_total': int(len(df)),
        'n_IT': int(y.sum()),
        'n_DE': int((1-y).sum()),
        'roc_auc': float(roc_auc_score(y, p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'brier_score': float(brier_score_loss(y, p)),
        'seed': SEED,
        'features_sha256': sha256(args.features),
        'sample_sha256': sha256(args.sample),
        'kept_features': keep,
        'excluded_over_40pct_missing': missing[missing > 0.40].index.tolist(),
        'status': 'training_validation_only',
        'note': 'VMS classification must not be run if balanced_accuracy < 0.65.',
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
