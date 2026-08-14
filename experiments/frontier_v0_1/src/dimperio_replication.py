from __future__ import annotations
import argparse, csv, json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.covariance import LedoitWolf
from .common import GateFailure, atomic_json

HISTORICAL = {59, 60, 76, 79, 94}

def char_vector(text: str, alphabet: list[str]) -> np.ndarray:
    c = Counter(ch for ch in text if not ch.isspace())
    n = sum(c.values())
    return np.array([c[a] / n if n else 0.0 for a in alphabet], dtype=float)

def load_panel(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    req = {"currier_page", "folio", "label", "text"}
    if not rows or not req.issubset(rows[0]):
        raise GateFailure(f"U2 panel must contain columns {sorted(req)}")
    return rows

def consensus_flags(rows: list[dict]) -> dict:
    alphabet = sorted(set("".join(r["text"] for r in rows)) - set(" \t\r\n*?"))
    X = np.vstack([char_vector(r["text"], alphabet) for r in rows])
    labels = [r["label"] for r in rows]
    pages = [int(r["currier_page"]) for r in rows]
    uniq = sorted(set(labels))

    flags1 = set()
    for i, lab in enumerate(labels):
        centroids = {}
        for u in uniq:
            idx = [j for j, l in enumerate(labels) if l == u and j != i]
            if idx:
                centroids[u] = X[idx].mean(axis=0)
        pred = min(centroids, key=lambda u: cdist(X[i:i+1], centroids[u][None, :], metric="correlation")[0, 0])
        if pred != lab:
            flags1.add(pages[i])

    scores = []
    for i, lab in enumerate(labels):
        idx = [j for j, l in enumerate(labels) if l == lab and j != i]
        if len(idx) < 4:
            scores.append(float("nan"))
            continue
        fit = LedoitWolf().fit(X[idx])
        scores.append(fit.mahalanobis(X[i:i+1])[0])
    flags2 = set()
    for lab in uniq:
        idx = [i for i, l in enumerate(labels) if l == lab and np.isfinite(scores[i])]
        if not idx:
            continue
        vals = np.array([scores[i] for i in idx])
        thr = np.quantile(vals, .95, method="higher")
        for i in idx:
            if scores[i] >= thr:
                flags2.add(pages[i])

    Z = linkage(X, method="average", metric="correlation")
    cl = fcluster(Z, t=len(uniq), criterion="maxclust")
    majority = {}
    for k in set(cl):
        labs = [labels[i] for i, c in enumerate(cl) if c == k]
        majority[k] = Counter(labs).most_common(1)[0][0]
    flags3 = {pages[i] for i, c in enumerate(cl) if majority[c] != labels[i]}

    methods = {"loo_centroid": sorted(flags1), "robust_outlier": sorted(flags2), "agglomeration": sorted(flags3)}
    consensus = sorted(p for p in HISTORICAL if sum(p in set(v) for v in methods.values()) >= 2)
    return {"methods": methods, "historical_replicated": consensus, "n_replicated": len(consensus)}

def verdict(n: int) -> str:
    if n >= 4:
        return "CONFIRM_REPLICATION"
    if n <= 1:
        return "FALSIFY_REPLICATION"
    return "ABSTAIN_UNRESOLVED"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, required=True, help="Frozen 40-page CSV after page-to-folio mapping gate")
    ap.add_argument("--mapping-gate", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    gate = json.loads(a.mapping_gate.read_text(encoding="utf-8"))
    if gate.get("formal_verdict") != "PASS" or not gate.get("selected_mapping"):
        raise SystemExit("U2 target remains sealed: page mapping gate has not passed")
    rows = load_panel(a.panel)
    if len(rows) != 40:
        raise SystemExit("U2 requires exactly 40 frozen pages")
    res = consensus_flags(rows)
    res["formal_verdict"] = verdict(res["n_replicated"])
    res["target_opened"] = True
    res["selected_mapping"] = gate["selected_mapping"]
    a.out.mkdir(parents=True, exist_ok=True)
    atomic_json(a.out / "U2_RESULT.json", res)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
