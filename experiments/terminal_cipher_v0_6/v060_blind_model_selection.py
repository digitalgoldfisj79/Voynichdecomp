#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import statistics
import sys
import zlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V05 = ROOT / "experiments" / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import recoverability_v050 as core
import v060_family_p_mode_blind as pblind

LENGTH = 384
CLASSES = ("mono", "P", "S", "T", "mixed", "generated", "notation", "ordinary", "none")
COUNTS = {"train": 24, "dev": 12, "test": 12}
TRANSCRIPTIONS = ("ZLZI", "TTLI", "VDRB-1")
RANDOM_STATE = 601733


def dense(values: Iterable[int]) -> np.ndarray:
    mapping: dict[int, int] = {}
    out: list[int] = []
    for raw in values:
        value = int(raw)
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return np.asarray(out, dtype=np.int32)


def entropy_counts(counts: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log2(p)).sum())


def ngram_entropy(seq: np.ndarray, width: int) -> float:
    if len(seq) < width:
        return 0.0
    counts = collections.Counter(tuple(int(x) for x in seq[i : i + width]) for i in range(len(seq) - width + 1))
    return entropy_counts(np.asarray(list(counts.values()), dtype=np.float64))


def mutual_information_lag(seq: np.ndarray, lag: int, alphabet: int) -> float:
    if lag <= 0 or len(seq) <= lag:
        return 0.0
    left = seq[:-lag]
    right = seq[lag:]
    joint = np.zeros((alphabet, alphabet), dtype=np.float64)
    np.add.at(joint, (left, right), 1.0)
    joint /= max(1.0, joint.sum())
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    nz = joint > 0
    denom = px[:, None] * py[None, :]
    value = float((joint[nz] * np.log2(joint[nz] / denom[nz])).sum())
    return value / max(math.log2(max(2, alphabet)), 1e-12)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p /= max(float(p.sum()), 1e-12)
    q /= max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    value = 0.0
    for x, y in ((p, m), (q, m)):
        mask = x > 0
        value += 0.5 * float((x[mask] * np.log2(x[mask] / m[mask])).sum())
    return value


def phase_vector(length: int, line_starts: list[int], period: int, reset: bool) -> np.ndarray:
    if not reset:
        return np.arange(length, dtype=np.int32) % period
    starts = sorted(set([0] + [int(x) for x in line_starts if 0 <= int(x) < length]))
    ends = starts[1:] + [length]
    phase = np.zeros(length, dtype=np.int32)
    for left, right in zip(starts, ends):
        phase[left:right] = np.arange(right - left, dtype=np.int32) % period
    return phase


def phase_features(seq: np.ndarray, phase: np.ndarray, period: int, alphabet: int) -> tuple[float, float, float]:
    joint = np.zeros((period, alphabet), dtype=np.float64)
    np.add.at(joint, (phase, seq), 1.0)
    total = max(float(joint.sum()), 1.0)
    pxy = joint / total
    pp = pxy.sum(axis=1)
    ps = pxy.sum(axis=0)
    denom = pp[:, None] * ps[None, :]
    nz = pxy > 0
    mi = float((pxy[nz] * np.log2(pxy[nz] / denom[nz])).sum())
    mi /= max(min(math.log2(max(2, period)), math.log2(max(2, alphabet))), 1e-12)

    profiles = []
    overall = joint.sum(axis=0)
    js_values = []
    for row in joint:
        if row.sum() <= 0:
            profile = np.zeros(alphabet, dtype=np.float64)
        else:
            profile = row / row.sum()
        profiles.append(np.sort(profile)[::-1])
        js_values.append(js_divergence(row, overall))
    matrix = np.asarray(profiles)
    centre = matrix.mean(axis=0)
    agreement = 1.0 - float(np.abs(matrix - centre).sum(axis=1).mean() / 2.0)
    return mi, agreement, float(np.mean(js_values))


def lz78_rate(seq: np.ndarray) -> float:
    dictionary: set[tuple[int, ...]] = set()
    current: tuple[int, ...] = ()
    phrases = 0
    for raw in seq:
        candidate = current + (int(raw),)
        if candidate in dictionary:
            current = candidate
        else:
            dictionary.add(candidate)
            phrases += 1
            current = ()
    if current:
        phrases += 1
    return phrases / max(1, len(seq))


def feature_names() -> list[str]:
    names = [
        "alphabet", "occupancy", "h1_norm", "h2_cond_norm", "h3_cond_norm",
        "collision", "transition_types", "bigram_types", "trigram_types",
        "lz78_rate", "zlib_ratio", "line_count_norm", "line_length_cv",
        "line_start_js", "recurrence_mean", "recurrence_std", "recurrence_q25",
        "recurrence_median", "recurrence_q75", "recurrence_max",
    ]
    names.extend(f"rank_mass_{i}" for i in range(10))
    for lag in range(1, 25):
        names.extend((f"lag_equal_{lag}", f"lag_mi_{lag}"))
    for period in range(2, 13):
        names.extend((
            f"global_mi_p{period}", f"global_sorted_p{period}", f"global_js_p{period}",
            f"line_mi_p{period}", f"line_sorted_p{period}", f"line_js_p{period}",
        ))
    return names


def extract_features(values: Iterable[int], line_starts: list[int]) -> np.ndarray:
    seq = dense(values)
    n = len(seq)
    if n == 0:
        return np.zeros(len(feature_names()), dtype=np.float64)
    alphabet = int(seq.max()) + 1
    counts = np.bincount(seq, minlength=alphabet).astype(np.float64)
    ranked = sorted((counts / n).tolist(), reverse=True)
    ranked = ranked[:10] + [0.0] * max(0, 10 - len(ranked))
    h1 = entropy_counts(counts)
    h2 = ngram_entropy(seq, 2)
    h3 = ngram_entropy(seq, 3)
    denom = max(math.log2(max(2, alphabet)), 1e-12)

    bigrams = {(int(seq[i]), int(seq[i + 1])) for i in range(max(0, n - 1))}
    trigrams = {(int(seq[i]), int(seq[i + 1]), int(seq[i + 2])) for i in range(max(0, n - 2))}
    transitions = collections.defaultdict(set)
    for left, right in zip(seq[:-1], seq[1:]):
        transitions[int(left)].add(int(right))
    transition_types = statistics.fmean(len(x) for x in transitions.values()) / max(1, alphabet) if transitions else 0.0

    packed = np.asarray(seq, dtype=np.uint16).tobytes()
    zratio = len(zlib.compress(packed, 9)) / max(1, len(packed))

    starts = sorted(set([0] + [int(x) for x in line_starts if 0 <= int(x) < n]))
    lengths = np.diff(np.asarray(starts + [n], dtype=np.int32)).astype(np.float64)
    line_cv = float(lengths.std() / max(lengths.mean(), 1e-12)) if len(lengths) else 0.0
    start_counts = np.bincount(seq[np.asarray(starts, dtype=np.int32)], minlength=alphabet)
    interior_mask = np.ones(n, dtype=bool)
    interior_mask[np.asarray(starts, dtype=np.int32)] = False
    interior_counts = np.bincount(seq[interior_mask], minlength=alphabet) if interior_mask.any() else counts
    start_js = js_divergence(start_counts, interior_counts)

    last: dict[int, int] = {}
    distances: list[float] = []
    for i, raw in enumerate(seq):
        value = int(raw)
        if value in last:
            distances.append((i - last[value]) / n)
        last[value] = i
    if distances:
        d = np.asarray(distances, dtype=np.float64)
        recurrence = [float(d.mean()), float(d.std()), float(np.quantile(d, 0.25)), float(np.median(d)), float(np.quantile(d, 0.75)), float(d.max())]
    else:
        recurrence = [0.0] * 6

    features: list[float] = [
        alphabet / n,
        len(set(int(x) for x in seq)) / n,
        h1 / denom,
        max(0.0, h2 - h1) / denom,
        max(0.0, h3 - h2) / denom,
        float((counts * (counts - 1)).sum() / max(1.0, n * (n - 1))),
        transition_types,
        len(bigrams) / max(1, n - 1),
        len(trigrams) / max(1, n - 2),
        lz78_rate(seq),
        zratio,
        len(starts) / n,
        line_cv,
        start_js,
        *recurrence,
        *ranked,
    ]
    for lag in range(1, 25):
        features.append(float(np.mean(seq[:-lag] == seq[lag:])) if n > lag else 0.0)
        features.append(mutual_information_lag(seq, lag, alphabet))
    for period in range(2, 13):
        features.extend(phase_features(seq, phase_vector(n, starts, period, False), period, alphabet))
        features.extend(phase_features(seq, phase_vector(n, starts, period, True), period, alphabet))
    return np.asarray(features, dtype=np.float64)


def random_line_starts(seed: int, length: int = LENGTH) -> list[int]:
    rng = random.Random(seed)
    starts = [0]
    cursor = 0
    while cursor < length:
        cursor += rng.randint(40, 72)
        if cursor < length:
            starts.append(cursor)
    return starts


def ensure_length(seq: list[int], language: core.LanguageData, rng: random.Random) -> list[int]:
    out = list(map(int, seq[:LENGTH]))
    while len(out) < LENGTH:
        out.append(core.weighted_choice(rng, language.probabilities))
    return out


def notation_generate(language: core.LanguageData, rng: random.Random) -> list[int]:
    a = max(8, len(language.alphabet))
    separator = 0
    pools = [
        list(range(1, min(a, 5))),
        list(range(min(a - 1, 5), min(a, 10))) or [1],
        list(range(min(a - 1, 10), min(a, 16))) or [2],
        list(range(min(a - 1, 16), a)) or [3],
    ]
    out: list[int] = []
    while len(out) < LENGTH:
        if out:
            out.append(separator)
        width = rng.randint(2, 5)
        for slot in range(width):
            pool = pools[slot % len(pools)]
            out.append(rng.choice(pool))
            if slot == 0 and rng.random() < 0.25:
                out.append(out[-1])
    return out[:LENGTH]


def none_generate(plain: list[int], language: core.LanguageData, rng: random.Random, variant: int) -> list[int]:
    if variant == 0:
        return [core.weighted_choice(rng, language.probabilities) for _ in range(LENGTH)]
    if variant == 1:
        out = list(plain[:LENGTH])
        rng.shuffle(out)
        return out
    motif = [rng.randrange(len(language.alphabet)) for _ in range(rng.randint(2, 7))]
    out = []
    while len(out) < LENGTH:
        piece = motif if (len(out) // len(motif)) % 2 == 0 else list(reversed(motif))
        out.extend(piece)
    return out[:LENGTH]


def block_permute(seq: list[int], rng: random.Random) -> list[int]:
    block = rng.choice((4, 5, 6, 7, 8))
    order = list(range(block))
    rng.shuffle(order)
    padded = list(seq)
    while len(padded) % block:
        padded.append(padded[-1] if padded else 0)
    out: list[int] = []
    for offset in range(0, len(padded), block):
        piece = padded[offset : offset + block]
        out.extend(piece[i] for i in order)
    return out[:LENGTH]


def make_example(label: str, language: core.LanguageData, split: str, index: int) -> tuple[list[int], list[int]]:
    chunks = core.source_chunks(language, split, LENGTH)
    if not chunks:
        raise RuntimeError(f"no chunks for {language.iso}/{split}")
    class_index = CLASSES.index(label)
    plain = list(chunks[(index + class_index * COUNTS[split]) % len(chunks)])
    seed = core.stable_seed("v060-model-selection", label, language.iso, split, index)
    rng = random.Random(seed)
    starts = random_line_starts(seed)

    if label == "ordinary":
        seq = plain
    elif label == "mono":
        seq = core.encrypt_sequence(plain, "mono", language, rng, parameter_mode=split).cipher
    elif label == "P":
        mode = pblind.MODES[index % len(pblind.MODES)]
        trial = pblind.make_trial(language, split, LENGTH, mode, index + 1000 * class_index)
        seq = trial.cipher
        starts = trial.line_starts
    elif label == "S":
        family = ("homophonic", "null_homophonic", "fractionated")[index % 3]
        seq = core.encrypt_sequence(plain, family, language, rng, parameter_mode=split).cipher
    elif label == "T":
        seq = core.encrypt_sequence(plain, "transposition", language, rng, parameter_mode=split).cipher
    elif label == "mixed":
        mode = pblind.MODES[index % len(pblind.MODES)]
        trial = pblind.make_trial(language, split, LENGTH, mode, index + 2000)
        seq = block_permute(trial.cipher, rng)
        starts = trial.line_starts
    elif label == "generated":
        family = core.CONTROL_FAMILIES[index % len(core.CONTROL_FAMILIES)]
        seq = core.generate_control(language, family, LENGTH, rng)
    elif label == "notation":
        seq = notation_generate(language, rng)
    elif label == "none":
        seq = none_generate(plain, language, rng, index % 3)
    else:
        raise ValueError(label)
    return ensure_length(list(seq), language, rng), starts


def build_dataset(languages: dict[str, core.LanguageData], split: str) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    count = COUNTS[split]
    for iso, language in sorted(languages.items()):
        for label in CLASSES:
            for index in range(count):
                seq, starts = make_example(label, language, split, index)
                features.append(extract_features(seq, starts))
                labels.append(label)
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=object)


def fit_isotonic(probs: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> list[IsotonicRegression]:
    models = []
    for index, name in enumerate(classes):
        target = (labels == name).astype(np.float64)
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(probs[:, index], target)
        models.append(model)
    return models


def calibrate(probs: np.ndarray, models: list[IsotonicRegression]) -> np.ndarray:
    adjusted = np.column_stack([model.predict(probs[:, i]) for i, model in enumerate(models)])
    adjusted = np.clip(adjusted, 1e-9, None)
    adjusted /= adjusted.sum(axis=1, keepdims=True)
    return adjusted


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, classes: np.ndarray, bins: int = 15) -> float:
    pred = classes[np.argmax(probs, axis=1)]
    confidence = np.max(probs, axis=1)
    correct = (pred == labels).astype(np.float64)
    total = len(labels)
    value = 0.0
    for left in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        right = left + 1.0 / bins
        mask = (confidence >= left) & (confidence < right if right < 1.0 else confidence <= right)
        if mask.any():
            value += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def choose_threshold(probs: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    p_index = int(np.where(classes == "P")[0][0])
    order = np.argsort(probs, axis=1)
    top = order[:, -1]
    margin = probs[np.arange(len(probs)), order[:, -1]] - probs[np.arange(len(probs)), order[:, -2]]
    structured = np.isin(labels, ("generated", "notation", "none"))
    positives = labels == "P"
    best = None
    for threshold in np.linspace(0.30, 0.95, 66):
        for min_margin in np.linspace(0.0, 0.50, 51):
            selected = (top == p_index) & (probs[:, p_index] >= threshold) & (margin >= min_margin)
            fpr = float(selected[structured].mean()) if structured.any() else 0.0
            recall = float(selected[positives].mean()) if positives.any() else 0.0
            if fpr <= 0.05 + 1e-12:
                candidate = (recall, -fpr, threshold, min_margin)
                if best is None or candidate > best[0]:
                    best = (candidate, {"probability": float(threshold), "margin": float(min_margin), "dev_p_recall": recall, "dev_structured_fpr": fpr})
    if best is None:
        return {"probability": 0.95, "margin": 0.50, "dev_p_recall": 0.0, "dev_structured_fpr": 1.0}
    return best[1]


def p_evidence(probs: np.ndarray, classes: np.ndarray, rule: dict[str, float]) -> np.ndarray:
    p_index = int(np.where(classes == "P")[0][0])
    order = np.argsort(probs, axis=1)
    margin = probs[np.arange(len(probs)), order[:, -1]] - probs[np.arange(len(probs)), order[:, -2]]
    return (order[:, -1] == p_index) & (probs[:, p_index] >= rule["probability"]) & (margin >= rule["margin"])


def synthetic_summary(classifier: ExtraTreesClassifier, calibrators: list[IsotonicRegression], x: np.ndarray, y: np.ndarray, rule: dict[str, float]) -> tuple[dict[str, Any], np.ndarray]:
    classes = classifier.classes_
    probs = calibrate(classifier.predict_proba(x), calibrators)
    binary = label_binarize(y, classes=classes)
    auc = float(roc_auc_score(binary, probs, average="macro", multi_class="ovr"))
    ece = expected_calibration_error(probs, y, classes)
    evidence = p_evidence(probs, classes, rule)
    structured = np.isin(y, ("generated", "notation", "none"))
    positives = y == "P"
    fpr = float(evidence[structured].mean()) if structured.any() else 0.0
    recall = float(evidence[positives].mean()) if positives.any() else 0.0
    precision = float((y[evidence] == "P").mean()) if evidence.any() else 0.0
    pred = classes[np.argmax(probs, axis=1)]
    matrix = confusion_matrix(y, pred, labels=classes)
    gate = auc >= 0.90 and ece <= 0.05 and fpr <= 0.05 and recall >= 0.80 and precision >= 0.90
    summary = {
        "macro_auc": auc,
        "ece": ece,
        "p_structured_fpr": fpr,
        "p_recall": recall,
        "p_precision": precision,
        "gate_pass": bool(gate),
        "classes": classes.tolist(),
        "confusion_matrix": matrix.tolist(),
    }
    return summary, probs


def folio_fold(folio: str) -> int:
    return hashlib.sha256(folio.encode("utf-8")).digest()[0] % 2


def clean_line(text: str) -> list[str]:
    tokens = ["".join(ch for ch in token if not ch.isspace()) for token in text.split()]
    tokens = [token for token in tokens if token]
    return list(" ".join(tokens))


def line_recurrence(chars: list[str]) -> list[int]:
    mapping: dict[str, int] = {}
    out = []
    for char in chars:
        if char not in mapping:
            mapping[char] = len(mapping)
        out.append(mapping[char])
    return out


def make_windows(lines: list[tuple[str, str, list[int]]]) -> list[tuple[list[int], list[int], dict[str, str]]]:
    windows = []
    buffer: list[int] = []
    starts: list[int] = []
    metadata: dict[str, str] | None = None
    for folio, section, line in lines:
        if not line:
            continue
        if not buffer:
            metadata = {"folio_start": folio, "section": section, "fold": str(folio_fold(folio))}
        starts.append(len(buffer))
        buffer.extend(line)
        buffer.append(65535)
        if len(buffer) >= LENGTH:
            windows.append((buffer[:LENGTH], [x for x in starts if x < LENGTH], dict(metadata or {})))
            buffer = []
            starts = []
            metadata = None
    return windows


def transcription_windows(data: dict[str, Any], section_map: dict[str, str], tid: str, recurrence: bool) -> tuple[list[tuple[list[int], list[int], dict[str, str]]], int]:
    raw_lines: list[tuple[str, str, list[str]]] = []
    symbols = set()
    for folio, page in sorted(data["pages"].items()):
        section = section_map.get(folio)
        if not section:
            continue
        for line_num, row in sorted(page.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 999999):
            text = row.get("t", {}).get(tid, "")
            if not text:
                continue
            chars = clean_line(text)
            if chars:
                raw_lines.append((folio, section, chars))
                symbols.update(chars)
    symbol_map = {symbol: index for index, symbol in enumerate(sorted(symbols))}
    grouped: dict[tuple[str, int], list[tuple[str, str, list[int]]]] = collections.defaultdict(list)
    for folio, section, chars in raw_lines:
        encoded = line_recurrence(chars) if recurrence else [symbol_map[ch] for ch in chars]
        grouped[(section, folio_fold(folio))].append((folio, section, encoded))
    windows = []
    for key in sorted(grouped):
        windows.extend(make_windows(grouped[key]))
    return windows, len(symbol_map)


def aggregate_voynich(rows: list[dict[str, Any]], classes: np.ndarray, rule: dict[str, float]) -> dict[str, Any]:
    if not rows:
        return {"windows": 0, "top_class": None, "p_rate": 0.0, "median_p": 0.0, "folds": {}, "sections": {}}
    p_index = int(np.where(classes == "P")[0][0])
    probs = np.asarray([row["probs"] for row in rows], dtype=np.float64)
    evidence = p_evidence(probs, classes, rule)
    top = classes[np.argmax(probs, axis=1)]
    aggregate_top = classes[int(np.argmax(probs.mean(axis=0)))]
    folds = {}
    sections = {}
    for fold in ("0", "1"):
        mask = np.asarray([row["fold"] == fold for row in rows])
        if mask.any():
            folds[fold] = {"windows": int(mask.sum()), "p_rate": float(evidence[mask].mean()), "median_p": float(np.median(probs[mask, p_index]))}
    for section in sorted(set(row["section"] for row in rows)):
        mask = np.asarray([row["section"] == section for row in rows])
        sections[section] = {
            "windows": int(mask.sum()),
            "p_rate": float(evidence[mask].mean()),
            "median_p": float(np.median(probs[mask, p_index])),
            "top_class": str(collections.Counter(top[mask]).most_common(1)[0][0]),
        }
    return {
        "windows": len(rows),
        "top_class": str(aggregate_top),
        "p_rate": float(evidence.mean()),
        "median_p": float(np.median(probs[:, p_index])),
        "class_fraction": {str(name): float(np.mean(top == name)) for name in classes},
        "folds": folds,
        "sections": sections,
    }


def run_voynich(classifier: ExtraTreesClassifier, calibrators: list[IsotonicRegression], rule: dict[str, float], languages: dict[str, core.LanguageData]) -> dict[str, Any]:
    data = json.loads((ROOT / "voynich_transcriptions_slim.json").read_text(encoding="utf-8"))
    section_payload = json.loads((ROOT / "voynich_section_map.json").read_text(encoding="utf-8"))
    section_map = section_payload["mapping"]
    classes = classifier.classes_
    representations = {}
    row_store = {}
    specs = [(tid, False, tid) for tid in TRANSCRIPTIONS] + [("ZLZI", True, "ZLZI-line-recurrence")]
    for tid, recurrence, name in specs:
        windows, alphabet_size = transcription_windows(data, section_map, tid, recurrence)
        if windows:
            x = np.asarray([extract_features(seq, starts) for seq, starts, _meta in windows])
            probs = calibrate(classifier.predict_proba(x), calibrators)
        else:
            probs = np.empty((0, len(classes)), dtype=np.float64)
        rows = []
        for (_seq, _starts, meta), probability in zip(windows, probs):
            rows.append({**meta, "probs": probability.tolist()})
        row_store[name] = rows
        aggregate = aggregate_voynich(rows, classes, rule)
        aggregate["alphabet_size_without_separator"] = alphabet_size
        aggregate["matching_language_alphabets"] = [iso for iso, language in languages.items() if len(language.alphabet) == alphabet_size + 1]
        representations[name] = aggregate

    raw_ok = all(
        representations[name]["top_class"] == "P"
        and representations[name]["p_rate"] >= 0.70
        and all(representations[name]["folds"].get(str(fold), {}).get("p_rate", 0.0) >= 0.60 for fold in (0, 1))
        for name in TRANSCRIPTIONS
    )
    recurrence_ok = (
        representations["ZLZI-line-recurrence"]["top_class"] == "P"
        and representations["ZLZI-line-recurrence"]["p_rate"] >= 0.60
    )
    common_sections = set.intersection(*(set(representations[name]["sections"]) for name in TRANSCRIPTIONS))
    qualifying_sections = []
    for section in sorted(common_sections):
        if all(
            representations[name]["sections"][section]["windows"] >= 8
            and representations[name]["sections"][section]["p_rate"] > 0.50
            for name in TRANSCRIPTIONS
        ):
            qualifying_sections.append(section)
    selected = bool(raw_ok and recurrence_ok and len(qualifying_sections) >= 4)
    compatibility = {
        "observed_historical_ring_order": False,
        "cardinality_match_by_representation": {
            name: representations[name]["matching_language_alphabets"] for name in representations
        },
        "direct_plaintext_solver_authorised": False,
        "reason": "EVA provides no independently observed circular glyph order; no glyph padding, dropping, merging or output-optimised ordering is permitted.",
    }
    return {
        "selection": "P_SELECTED" if selected else "ABSTAIN_OR_OUT_OF_FAMILY",
        "rule_pass": selected,
        "qualifying_sections": qualifying_sections,
        "representations": representations,
        "compatibility": compatibility,
    }


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/v060_blind_model_selection_result.json")
    languages = core.load_languages(V05 / "corpus_manifest_v050.json", ROOT / ".cache" / "v060-model-selection")
    print("V060_MODEL_SELECTION_BUILD train", flush=True)
    x_train, y_train = build_dataset(languages, "train")
    print("V060_MODEL_SELECTION_BUILD dev", flush=True)
    x_dev, y_dev = build_dataset(languages, "dev")
    print("V060_MODEL_SELECTION_BUILD test", flush=True)
    x_test, y_test = build_dataset(languages, "test")

    classifier = ExtraTreesClassifier(
        n_estimators=600,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    classifier.fit(x_train, y_train)
    dev_raw = classifier.predict_proba(x_dev)
    calibrators = fit_isotonic(dev_raw, y_dev, classifier.classes_)
    dev_probs = calibrate(dev_raw, calibrators)
    rule = choose_threshold(dev_probs, y_dev, classifier.classes_)
    test_summary, _test_probs = synthetic_summary(classifier, calibrators, x_test, y_test, rule)
    print("V060_MODEL_SELECTION_SYNTHETIC", json.dumps({"threshold": rule, "locked_test": test_summary}, sort_keys=True), flush=True)

    payload: dict[str, Any] = {
        "config": {
            "classes": list(CLASSES),
            "counts_per_language_class": COUNTS,
            "languages": sorted(languages),
            "length": LENGTH,
            "classifier": "ExtraTreesClassifier(n_estimators=600,min_samples_leaf=2,max_features=sqrt,class_weight=balanced)",
            "calibration": "one-vs-rest isotonic on dev, renormalized",
            "feature_count": len(feature_names()),
            "transcriptions": list(TRANSCRIPTIONS),
        },
        "threshold": rule,
        "synthetic_locked_test": test_summary,
        "feature_importance": [
            {"feature": feature_names()[int(i)], "importance": float(classifier.feature_importances_[int(i)])}
            for i in np.argsort(classifier.feature_importances_)[::-1][:25]
        ],
    }
    if test_summary["gate_pass"]:
        payload["voynich"] = run_voynich(classifier, calibrators, rule, languages)
        print("V060_MODEL_SELECTION_VOYNICH", json.dumps(payload["voynich"], sort_keys=True), flush=True)
    else:
        payload["voynich"] = {"status": "NOT_OPENED_SYNTHETIC_GATE_FAILED"}

    scientific = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["sha256"] = hashlib.sha256(scientific).hexdigest()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_MODEL_SELECTION_SHA256", payload["sha256"], flush=True)
    print("V060_MODEL_SELECTION_OUTPUT", str(output), flush=True)


if __name__ == "__main__":
    main()
