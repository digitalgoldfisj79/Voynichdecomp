#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from surface_features_v04 import (
    canonicalize_events,
    extract_surface_features,
    parse_gabc_body,
    windows,
)

WINDOW = 48
STRIDE = 24
MINIMUM = 48
SEED = 20260724
META_FIELDS = {"corpus", "family", "group", "section", "window_id", "start", "end", "representation"}


def tokenise_line(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    if ":" in line and line.split(":", 1)[0].strip().lower() in {
        "duration", "durations", "special", "duration_special", "pitch", "pitches", "rest", "pitch_rest", "top", "bottom", "line1", "line2"
    }:
        line = line.split(":", 1)[1].strip()
    try:
        obj = ast.literal_eval(line)
        if isinstance(obj, (list, tuple)):
            return [str(x) for x in obj]
    except Exception:
        pass
    return [x for x in re.split(r"[\s,;]+", line) if x]


def parse_ammerbach_annotation(path: Path) -> dict[str, list[str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    seqs = [tokenise_line(x) for x in lines]
    seqs = [x for x in seqs if x]
    if not seqs:
        return {}
    if len(seqs) == 1:
        duration, pitch = [], seqs[0]
    else:
        duration, pitch = seqs[0], seqs[1]
    paired: list[str] = []
    if duration and pitch:
        m = max(len(duration), len(pitch))
        for i in range(m):
            a = duration[i] if i < len(duration) else "_"
            b = pitch[i] if i < len(pitch) else "_"
            paired.append(f"{a}|{b}")
    flattened = ([f"d{x}" for x in duration] + [f"p{x}" for x in pitch]) if duration else pitch[:]
    return {"pitch": pitch, "duration": duration, "paired": paired, "flattened": flattened, "raw_lines": seqs}


def infer_ammerbach_metadata(root: Path) -> tuple[dict[str, dict], dict]:
    mapping: dict[str, dict] = {}
    audit: dict = {"csv_files": [], "columns": {}, "source_counts": {}}
    for csv_path in sorted(root.rglob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            audit["csv_files"].append({"path": str(csv_path), "error": str(exc)})
            continue
        audit["csv_files"].append({"path": str(csv_path), "rows": int(len(df))})
        audit["columns"][str(csv_path)] = [str(x) for x in df.columns]
        for idx, row in df.iterrows():
            d = {str(k): (None if pd.isna(v) else str(v)) for k, v in row.to_dict().items()}
            keys = set()
            for col, val in d.items():
                if val is None:
                    continue
                lc = col.lower()
                if any(q in lc for q in ["file", "image", "name", "path", "index", "id"]):
                    keys.add(Path(val).stem)
                    keys.add(val)
            keys.add(str(idx))
            for k in keys:
                mapping[k] = d
    return mapping, audit


def source_from_meta(path: Path, row: dict | None) -> str:
    hay = " ".join([str(path)] + ([str(x) for x in row.values() if x is not None] if row else [])).lower()
    if "1583" in hay or "instrument" in hay:
        return "ammerbach_1583"
    if "1575" in hay or "künst" in hay or "kunst" in hay:
        return "ammerbach_1575"
    split = next((x for x in ("train", "val", "test") if x in {p.lower() for p in path.parts}), "unknown")
    m = re.search(r"(\d+)", path.stem)
    if m:
        idx = int(m.group(1))
        cut = {"train": 500, "val": 200, "test": 500}.get(split)
        if cut is not None:
            return f"ammerbach_inferred_{'bookA' if idx < cut else 'bookB'}"
    return f"ammerbach_{split}_unknown"


def load_ammerbach(root: Path) -> tuple[dict[str, list[list[str]]], dict]:
    mapping, audit = infer_ammerbach_metadata(root)
    reps: dict[str, list[tuple[str, list[str], str]]] = defaultdict(list)
    samples = []
    counts = Counter()
    lengths = defaultdict(list)
    txts = [p for p in root.rglob("*.txt") if p.is_file()]
    for path in sorted(txts):
        ann = parse_ammerbach_annotation(path)
        if not ann:
            continue
        row = mapping.get(path.stem) or mapping.get(path.name) or mapping.get(str(path))
        source = source_from_meta(path, row)
        counts[source] += 1
        for rep in ("pitch", "duration", "paired", "flattened"):
            seq = ann.get(rep, [])
            if seq:
                reps[rep].append((source, seq, str(path)))
                lengths[rep].append(len(seq))
        if len(samples) < 12:
            samples.append({"path": str(path), "source": source, "lines": ann["raw_lines"], "lengths": {k: len(ann.get(k, [])) for k in ("pitch", "duration", "paired", "flattened")}, "metadata": row})
    audit.update({
        "txt_files_seen": len(txts),
        "annotations_parsed": int(sum(counts.values())),
        "source_counts": dict(counts),
        "representation_lengths": {k: {"n": len(v), "mean": float(np.mean(v)) if v else 0.0, "median": float(np.median(v)) if v else 0.0, "min": min(v) if v else 0, "max": max(v) if v else 0} for k, v in lengths.items()},
        "samples": samples,
    })
    return reps, audit


def read_gabc_files(root: Path) -> tuple[dict[tuple[str, str], dict[str, list[str]]], dict]:
    grouped: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: {"notation": [], "lyrics": [], "files": []})
    audit = {"files": 0, "families": Counter(), "manuscripts": Counter(), "parse_failures": []}
    for path in sorted(root.rglob("*.gabc")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "%%" not in text:
            audit["parse_failures"].append(str(path))
            continue
        header, body = text.split("%%", 1)
        meta = {}
        for line in header.splitlines():
            if ":" in line:
                k, val = line.split(":", 1)
                meta[k.strip().lower()] = val.strip().strip(";")
        low = str(path).lower() + " " + header.lower()
        family = "aquitanian" if "aquit" in low else ("square" if "square" in low else "unknown")
        manuscript = meta.get("manuscript") or meta.get("manuscript-storage-place") or path.parent.name
        notation, lyrics = parse_gabc_body(body)
        if not notation:
            continue
        key = (family, manuscript)
        grouped[key]["notation"].extend(notation)
        grouped[key]["lyrics"].extend(lyrics)
        grouped[key]["files"].append(str(path))
        audit["files"] += 1
        audit["families"][family] += 1
        audit["manuscripts"][manuscript] += 1
    audit["families"] = dict(audit["families"])
    audit["manuscripts"] = dict(audit["manuscripts"])
    audit["groups"] = len(grouped)
    return grouped, audit


def monoalphabetic_cipher(events: Sequence[str], seed: int) -> list[str]:
    chars = sorted(set("".join(events)))
    targets = list("abcdefghijklmnopqrstuvwxyz")
    targets += [chr(0x3B1 + i) for i in range(25)]
    rng = random.Random(seed)
    rng.shuffle(targets)
    if len(chars) > len(targets):
        targets += [chr(0x430 + i) for i in range(len(chars) - len(targets))]
    mp = dict(zip(chars, targets))
    return ["".join(mp[ch] for ch in x) for x in events]


def shuffled_surface(events: Sequence[str], seed: int) -> list[str]:
    lens = [len(x) for x in events]
    chars = list("".join(events))
    random.Random(seed).shuffle(chars)
    out = []
    i = 0
    for n in lens:
        out.append("".join(chars[i : i + n]))
        i += n
    return out


def markov_surface(events: Sequence[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    starts = Counter(x[0] for x in events if x)
    trans = defaultdict(Counter)
    chars = Counter(ch for x in events for ch in x)
    for x in events:
        for a, b in zip(x[:-1], x[1:]):
            trans[a][b] += 1
    def draw(c: Counter):
        if not c:
            c = chars
        total = sum(c.values())
        z = rng.uniform(0, total)
        acc = 0.0
        for k, v in c.items():
            acc += v
            if acc >= z:
                return k
        return next(iter(c))
    out = []
    for x in events:
        if not x:
            out.append("")
            continue
        s = [draw(starts)]
        for _ in range(len(x) - 1):
            s.append(draw(trans[s[-1]]))
        out.append("".join(s))
    return out


def procedural_sequences(groups: int = 30, events_per_group: int = 480) -> list[tuple[str, list[str]]]:
    out = []
    for g in range(groups):
        rng = random.Random(SEED + 10000 + g)
        ops = list("oydschq")
        cores = ["".join(rng.choice("aeioux") for _ in range(rng.choice([1, 2, 2, 3]))) for _ in range(40)]
        suffixes = ["", "n", "r", "dy", "in", "ol"]
        state = rng.randrange(6)
        seq = []
        for _ in range(events_per_group):
            if rng.random() < 0.18:
                state = (state + rng.choice([-1, 1, 2])) % 6
            op = ops[(state + rng.randrange(3)) % len(ops)] if rng.random() < 0.75 else ""
            core = cores[(state * 7 + rng.randrange(12)) % len(cores)]
            suffix = suffixes[(state + rng.randrange(3)) % len(suffixes)] if rng.random() < 0.8 else ""
            seq.append(op + core + suffix)
        out.append((f"procedural_{g:02d}", seq))
    return out


def add_windows(rows: list[dict], events: Sequence[str], corpus: str, family: str, group: str, representation: str, section: str = ""):
    if len(events) < MINIMUM:
        return
    for wi, (a, b, w) in enumerate(windows(events, window=WINDOW, stride=STRIDE, minimum=MINIMUM)):
        f = extract_surface_features(canonicalize_events(w))
        rows.append({
            "corpus": corpus, "family": family, "group": group, "section": section,
            "representation": representation, "window_id": f"{group}:{wi}:{a}", "start": a, "end": b, **f,
        })


def build_external_rows(ammerbach_reps, gabc_groups, representation: str) -> list[dict]:
    rows: list[dict] = []
    by_source = defaultdict(list)
    for source, seq, path in ammerbach_reps.get(representation, []):
        by_source[source].extend(seq)
    for source, seq in sorted(by_source.items()):
        add_windows(rows, seq, "Ammerbach", "organ_tablature", source, representation)

    for (family, manuscript), d in sorted(gabc_groups.items()):
        notation = d["notation"]
        lyrics = d["lyrics"]
        fam = "neume_aquitanian" if family == "aquitanian" else ("neume_square" if family == "square" else "neume_unknown")
        base_group = f"gabc:{manuscript}"
        add_windows(rows, notation, "GABC", fam, base_group, "gabc_notation", family)
        if lyrics:
            add_windows(rows, lyrics, "GABC", "language_or_substitution", base_group, "latin_lyrics", family)
            add_windows(rows, monoalphabetic_cipher(lyrics, SEED + len(base_group)), "GABC", "language_or_substitution", base_group, "monoalphabetic_cipher", family)
        add_windows(rows, shuffled_surface(notation, SEED + 2 * len(base_group)), "GABC", "surface_null", base_group, "char_shuffle", family)
        add_windows(rows, markov_surface(notation, SEED + 3 * len(base_group)), "GABC", "surface_null", base_group, "char_markov", family)

    for group, seq in procedural_sequences():
        add_windows(rows, seq, "Synthetic", "procedural_synthetic", group, "procedural")
    return rows


def read_feature_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in META_FIELDS]
    return df, features


def models(binary: bool = True):
    lr = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced", random_state=SEED)),
    ])
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=3,
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1,
    )
    return {"logistic": lr, "forest": rf}


def external_cv(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    historical = {"organ_tablature", "neume_aquitanian", "neume_square"}
    broad = df[df["family"] != "procedural_synthetic"].copy()
    broad["y"] = broad["family"].isin(historical).astype(int)
    X = broad[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    y = broad["y"].to_numpy(int)
    groups = broad["group"].astype(str).to_numpy()
    n_groups = len(set(groups))
    folds = min(5, n_groups)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    pred = {name: np.zeros(len(y), dtype=float) for name in models()}
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter.split(X, y, groups)):
        for name, model in models().items():
            m = clone(model).fit(X[tr], y[tr])
            pred[name][te] = m.predict_proba(X[te])[:, 1]
        fold_rows.append({"fold": fold, "train": len(tr), "test": len(te), "test_groups": sorted(set(groups[te])), "test_positive": int(y[te].sum())})
    ensemble = np.mean(np.vstack(list(pred.values())), axis=0)
    metric = {}
    for name, p in {**pred, "ensemble": ensemble}.items():
        metric[name] = {
            "roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "balanced_accuracy_0_5": float(balanced_accuracy_score(y, p >= 0.5)),
        }
    candidates = []
    for t in np.linspace(0.01, 0.99, 99):
        yp = ensemble >= t
        tn = int(((y == 0) & (~yp)).sum()); fp = int(((y == 0) & yp).sum())
        tp = int(((y == 1) & yp).sum()); fn = int(((y == 1) & (~yp)).sum())
        fpr = fp / max(1, fp + tn); tpr = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        if fpr <= 0.10:
            candidates.append((tpr, precision, t, fpr))
    if candidates:
        tpr, precision, threshold, fpr = max(candidates, key=lambda x: (x[0], x[1], x[2]))
    else:
        threshold = 0.5; fpr = float("nan"); tpr = float("nan"); precision = float("nan")
    yp = ensemble >= threshold
    organ_mask = broad["family"].to_numpy() == "organ_tablature"
    gate = (
        metric["logistic"]["roc_auc"] >= 0.80 and
        metric["forest"]["roc_auc"] >= 0.80 and
        balanced_accuracy_score(y, yp) >= 0.70 and
        (float(np.mean(yp[organ_mask])) if organ_mask.any() else 0.0) >= 0.60
    )
    return {
        "rows": len(broad), "groups": n_groups, "folds": fold_rows,
        "metrics": metric, "threshold": float(threshold),
        "threshold_cv": {"fpr": float(fpr), "tpr": float(tpr), "precision": float(precision)},
        "balanced_accuracy_threshold": float(balanced_accuracy_score(y, yp)),
        "organ_recall_threshold": float(np.mean(yp[organ_mask])) if organ_mask.any() else None,
        "gate_pass": bool(gate),
        "cv_predictions": pd.DataFrame({"family": broad["family"], "group": broad["group"], "y": y, **pred, "ensemble": ensemble}).to_dict("records"),
    }


def family_cv(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    labels = sorted(df["family"].unique())
    y_text = df["family"].astype(str).to_numpy()
    enc = {x: i for i, x in enumerate(labels)}
    y = np.array([enc[x] for x in y_text], dtype=int)
    groups = df["group"].astype(str).to_numpy()
    folds = min(5, len(set(groups)))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=SEED)
    probs = {name: np.zeros((len(y), len(labels))) for name in models(False)}
    for tr, te in splitter.split(X, y, groups):
        for name, model in models(False).items():
            m = clone(model).fit(X[tr], y[tr])
            raw = m.predict_proba(X[te])
            classes = m.classes_ if hasattr(m, "classes_") else m.named_steps["clf"].classes_
            for j, c in enumerate(classes):
                probs[name][te, int(c)] = raw[:, j]
    ensemble = np.mean(np.stack(list(probs.values())), axis=0)
    pred = ensemble.argmax(axis=1)
    macro = float(f1_score(y, pred, average="macro"))
    gate = macro >= 0.55
    return {
        "labels": labels, "macro_f1": macro, "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(labels)))).tolist(),
        "classification_report": classification_report(y, pred, target_names=labels, output_dict=True, zero_division=0),
        "gate_pass": bool(gate),
    }


def fit_and_predict(external: pd.DataFrame, target: pd.DataFrame, features: list[str], broad_result: dict, fam_result: dict) -> dict:
    historical = {"organ_tablature", "neume_aquitanian", "neume_square"}
    broad = external[external["family"] != "procedural_synthetic"].copy()
    y = broad["family"].isin(historical).astype(int).to_numpy()
    X = broad[features].fillna(0.0).to_numpy(float)
    Xt = target[features].fillna(0.0).to_numpy(float)
    broad_probs = []
    for model in models().values():
        m = clone(model).fit(X, y)
        broad_probs.append(m.predict_proba(Xt)[:, 1])
    bp = np.mean(np.vstack(broad_probs), axis=0)
    out = target[[c for c in ["group", "section", "window_id"] if c in target.columns]].copy()
    out["historical_notation_probability"] = bp
    out["above_external_threshold"] = bp >= broad_result["threshold"]

    labels = fam_result["labels"]
    enc = {x: i for i, x in enumerate(labels)}
    yf = np.array([enc[x] for x in external["family"].astype(str)], dtype=int)
    Xf = external[features].fillna(0.0).to_numpy(float)
    fp = []
    for model in models(False).values():
        m = clone(model).fit(Xf, yf)
        raw = m.predict_proba(Xt)
        arr = np.zeros((len(Xt), len(labels)))
        classes = m.classes_ if hasattr(m, "classes_") else m.named_steps["clf"].classes_
        for j, c in enumerate(classes):
            arr[:, int(c)] = raw[:, j]
        fp.append(arr)
    fam_prob = np.mean(np.stack(fp), axis=0)
    for j, lab in enumerate(labels):
        out[f"p_{lab}"] = fam_prob[:, j]
    out["predicted_family"] = [labels[i] for i in fam_prob.argmax(axis=1)]

    summary = {
        "windows": len(out),
        "mean_historical_notation_probability": float(bp.mean()),
        "median_historical_notation_probability": float(np.median(bp)),
        "fraction_above_threshold": float(np.mean(out["above_external_threshold"])),
        "mean_family_probabilities": {lab: float(fam_prob[:, j].mean()) for j, lab in enumerate(labels)},
        "predicted_family_counts": out["predicted_family"].value_counts().to_dict(),
        "by_section": {},
    }
    if "section" in out:
        for sec, d in out.groupby("section"):
            summary["by_section"][str(sec)] = {
                "n": len(d),
                "mean_historical_probability": float(d["historical_notation_probability"].mean()),
                "fraction_above_threshold": float(d["above_external_threshold"].mean()),
                "family_counts": d["predicted_family"].value_counts().to_dict(),
            }
    return {"summary": summary, "rows": out.to_dict("records")}


def dataframe_from_rows(rows: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.DataFrame(rows)
    features = [c for c in df.columns if c not in META_FIELDS]
    return df, features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ammerbach-dir", type=Path, required=True)
    ap.add_argument("--gabc-dir", type=Path, required=True)
    ap.add_argument("--voynich-features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ammerbach_reps, ammer_audit = load_ammerbach(args.ammerbach_dir)
    gabc_groups, gabc_audit = read_gabc_files(args.gabc_dir)
    target, target_features = read_feature_csv(args.voynich_features)

    representation_results = {}
    external_frames = {}
    for rep in ("paired", "pitch", "flattened"):
        rows = build_external_rows(ammerbach_reps, gabc_groups, rep)
        df, features = dataframe_from_rows(rows)
        if set(features) != set(target_features):
            raise RuntimeError(f"feature mismatch for {rep}: external-only={set(features)-set(target_features)}, target-only={set(target_features)-set(features)}")
        features = target_features
        broad = external_cv(df, features)
        fam = family_cv(df, features)
        representation_results[rep] = {"counts": df["family"].value_counts().to_dict(), "broad": broad, "family": fam}
        external_frames[rep] = df
        print(rep, broad["metrics"]["ensemble"], broad["gate_pass"], fam["macro_f1"], flush=True)

    tie_order = {"paired": 2, "pitch": 1, "flattened": 0}
    selected = max(representation_results, key=lambda r: (
        representation_results[r]["broad"]["metrics"]["ensemble"]["roc_auc"],
        representation_results[r]["broad"]["balanced_accuracy_threshold"], tie_order[r]
    ))
    selected_result = representation_results[selected]
    unlocked = bool(selected_result["broad"]["gate_pass"])
    family_unlocked = unlocked and bool(selected_result["family"]["gate_pass"])
    target_result = None
    if unlocked:
        target_result = fit_and_predict(external_frames[selected], target, target_features, selected_result["broad"], selected_result["family"])
        pd.DataFrame(target_result["rows"]).to_csv(args.out / "voynich_predictions_v0_4.csv", index=False)

    compact_reps = {}
    for rep, rr in representation_results.items():
        compact_reps[rep] = {
            "counts": rr["counts"],
            "broad": {k: v for k, v in rr["broad"].items() if k != "cv_predictions"},
            "family": rr["family"],
        }
    result = {
        "schema": "historical-notation-blind-classifier-v0.4",
        "window": WINDOW, "stride": STRIDE, "minimum": MINIMUM,
        "character_policy": "frequency-rank canonicalised independently within each window",
        "ammerbach_audit": ammer_audit,
        "gabc_audit": gabc_audit,
        "representations": compact_reps,
        "selected_representation": selected,
        "external_gate_pass": unlocked,
        "family_gate_pass": family_unlocked,
        "voynich_opened": unlocked,
        "voynich": target_result["summary"] if target_result else None,
    }
    (args.out / "historical_notation_classifier_results_v0_4.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "ammerbach_intake_audit_v0_4.json").write_text(json.dumps(ammer_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Historical notation calibration v0.4 — result", "",
        f"**Selected Ammerbach representation:** `{selected}`",  
        f"**External recognition gate:** {'PASS' if unlocked else 'FAIL / ABSTAIN'}",  
        f"**Historical family gate:** {'PASS' if family_unlocked else 'FAIL / ABSTAIN'}",  
        f"**Voynich target opened:** {'yes' if unlocked else 'no'}", "",
        "## External results", "",
    ]
    for rep, rr in compact_reps.items():
        b = rr["broad"]
        lines += [
            f"### {rep}", "",
            f"- Counts: `{rr['counts']}`",
            f"- Ensemble ROC AUC: {b['metrics']['ensemble']['roc_auc']:.4f}",
            f"- Logistic ROC AUC: {b['metrics']['logistic']['roc_auc']:.4f}",
            f"- Forest ROC AUC: {b['metrics']['forest']['roc_auc']:.4f}",
            f"- Calibrated balanced accuracy: {b['balanced_accuracy_threshold']:.4f}",
            f"- Organ-tablature recall: {b['organ_recall_threshold']:.4f}",
            f"- Multiclass macro-F1: {rr['family']['macro_f1']:.4f}",
            f"- Broad gate: {'PASS' if b['gate_pass'] else 'FAIL'}", "",
        ]
    lines += ["## Voynich adjudication", ""]
    if target_result:
        s = target_result["summary"]
        lines += [
            f"- Mean historical-notation probability: {s['mean_historical_notation_probability']:.4f}",
            f"- Median probability: {s['median_historical_notation_probability']:.4f}",
            f"- Fraction above external threshold: {s['fraction_above_threshold']:.4f}",
            f"- Mean family probabilities: `{s['mean_family_probabilities']}`", "",
            "These are family-recognition outputs, not a decipherment or semantic identification.",
        ]
    else:
        lines += ["The external gate failed, so the sealed Voynich feature table was not passed to the classifiers."]
    (args.out / "HISTORICAL_NOTATION_CALIBRATION_RESULT_v0_4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"selected": selected, "external_gate": unlocked, "family_gate": family_unlocked, "voynich": result["voynich"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
