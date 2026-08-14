from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import adjusted_rand_score
from .common import GateFailure, atomic_json, load_config

META = {"bifolium", "fold", "quire", "currier", "hand", "section"}

def load_feature_matrix(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "bifolium" not in rows[0] or "fold" not in rows[0]:
        raise GateFailure("feature matrix requires bifolium and fold columns")
    feats = [c for c in rows[0] if c not in META and not c.endswith("__family")]
    X = np.array([[float(r[c]) if r[c] not in ("", "NA", "NaN", "nan") else np.nan for c in feats] for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])
    return rows, feats, X, folds

def equal_family_weights(feats: list[str], X: np.ndarray):
    fams = [f.split("::", 1)[0] if "::" in f else "UNREGISTERED" for f in feats]
    uniq = sorted(set(fams))
    W = np.ones(len(feats))
    for fam in uniq:
        idx = [i for i, x in enumerate(fams) if x == fam]
        W[idx] = 1.0 / np.sqrt(len(idx))
    return X * W, fams

def heldout_scores(X, folds, ks, seed):
    out = {k: [] for k in ks}
    for fold in sorted(set(folds)):
        tr, te = folds != fold, folds == fold
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sc = RobustScaler().fit(imp.transform(X[tr]))
        Xtr = sc.transform(imp.transform(X[tr]))
        Xte = sc.transform(imp.transform(X[te]))
        for k in ks:
            gm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4, n_init=20, random_state=seed + k * 100 + int(fold))
            gm.fit(Xtr)
            out[k].append(float(gm.score(Xte)))
    return out

def choose_k(scores):
    means = {k: float(np.mean(v)) for k, v in scores.items()}
    best = max(means, key=means.get)
    return best, means

def bootstrap_stability(X, k, seed, reps=200):
    rng = np.random.default_rng(seed)
    imp = SimpleImputer(strategy="median").fit(X)
    Xi = imp.transform(X)
    sc = RobustScaler().fit(Xi)
    Z = sc.transform(Xi)
    base = GaussianMixture(k, covariance_type="diag", reg_covar=1e-4, n_init=30, random_state=seed).fit(Z).predict(Z)
    aris = []
    n = len(Z)
    for b in range(reps):
        idx = rng.integers(0, n, n)
        gm = GaussianMixture(k, covariance_type="diag", reg_covar=1e-4, n_init=10, random_state=seed + b + 1).fit(Z[idx])
        pred = gm.predict(Z)
        aris.append(adjusted_rand_score(base, pred))
    return float(np.median(aris)), base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "PROGRAMME_CONFIG.json")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = load_config(a.config)
    uc = cfg["u3"]
    rows, feats, X, folds = load_feature_matrix(a.features)
    Xw, fams = equal_family_weights(feats, X)
    scores = heldout_scores(Xw, folds, uc["k_values"], cfg["global"]["default_seed"])
    k, means = choose_k(scores)
    med_ari, assign = bootstrap_stability(Xw, k, cfg["global"]["default_seed"], uc["bootstrap_reps"])
    fold_wins = sum(scores[k][i] > scores[1][i] for i in range(len(scores[1]))) if k != 1 else 0
    if k == 1:
        interpretation = "ONE_STATE_PRESELECTION_REQUIRES_SYNTHETIC_CALIBRATION"
    elif fold_wins >= uc["min_outer_fold_wins"] and med_ari >= uc["min_stability_ari"]:
        interpretation = f"DISCRETE_K{k}_CANDIDATE_REQUIRES_SYNTHETIC_CALIBRATION_AND_LOFO"
    else:
        interpretation = "UNSTABLE_DISCRETE_STRUCTURE"
    res = {
        "formal_verdict": "ABSTAIN_UNRESOLVED",
        "target_opened": False,
        "selected_k": k,
        "heldout_mean_loglik": means,
        "fold_scores": scores,
        "fold_wins_vs_k1": fold_wins,
        "median_bootstrap_ari": med_ari,
        "interpretation": interpretation,
        "warning": "This is preselection only. Synthetic calibration and leave-feature-family-out gates must pass before any regime claim or Currier/hand/section association is opened."
    }
    a.out.mkdir(parents=True, exist_ok=True)
    atomic_json(a.out / "U3_PRESELECTION.json", res)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
