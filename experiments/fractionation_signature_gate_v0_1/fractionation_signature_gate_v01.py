#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

TRAIN_BLOCKS = (2, 3, 4, 6, 8)
TEST_BLOCKS = (5, 7, 9, 10, 11, 12)
CANDIDATE_BLOCKS = tuple(range(1, 13))
LENGTH = 384


def stable_seed(*parts: object) -> int:
    import hashlib
    blob = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def load_v050(repo: Path):
    path = repo / "experiments/recoverability_frontier_v0_5/recoverability_v050.py"
    spec = importlib.util.spec_from_file_location("recoverability_v050", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def nonspace_values(values: Sequence[int], space: int | None) -> list[int]:
    if space is None:
        return list(map(int, values))
    return [int(x) for x in values if int(x) != int(space)]


def make_pair_key(symbols: Sequence[int], rng: random.Random):
    symbols = sorted(set(map(int, symbols)))
    n = max(2, len(symbols))
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(cells)
    mapping = {s: cells[i] for i, s in enumerate(symbols)}
    return mapping, rows, cols


def make_surface(rows: int, cols: int, rng: random.Random, noisy: bool):
    row_pools: dict[int, list[int]] = {}
    col_pools: dict[int, list[int]] = {}
    cursor = 0
    for r in range(rows):
        k = 1 + int(noisy and rng.random() < 0.35)
        row_pools[r] = list(range(cursor, cursor + k))
        cursor += k
    for c in range(cols):
        k = 1 + int(noisy and rng.random() < 0.35)
        col_pools[c] = list(range(cursor, cursor + k))
        cursor += k
    if noisy and rows and cols:
        overlap_n = max(1, min(rows, cols) // 4)
        for c in rng.sample(range(cols), overlap_n):
            r = rng.randrange(rows)
            col_pools[c][0] = row_pools[r][0]
    nulls: list[int] = []
    if noisy:
        nulls = [cursor, cursor + 1]
    return row_pools, col_pools, nulls


def encode_two_stream(plain: Sequence[int], block: int, rng: random.Random, noisy: bool = False) -> list[int]:
    mapping, rows, cols = make_pair_key(plain, rng)
    row_pools, col_pools, nulls = make_surface(rows, cols, rng, noisy)
    row_seq: list[int] = []
    col_seq: list[int] = []
    for x in plain:
        r, c = mapping[int(x)]
        row_seq.append(rng.choice(row_pools[r]))
        col_seq.append(rng.choice(col_pools[c]))
    out: list[int] = []
    for start in range(0, len(plain), block):
        stop = min(len(plain), start + block)
        out.extend(row_seq[start:stop])
        out.extend(col_seq[start:stop])
    if noisy and nulls:
        expanded: list[int] = []
        for symbol in out:
            if rng.random() < 0.015:
                expanded.append(rng.choice(nulls))
            expanded.append(symbol)
            if rng.random() < 0.005:
                expanded.append(rng.choice(nulls))
        out = expanded
    return out


def encode_easy_bigraphic(plain: Sequence[int], rng: random.Random) -> list[int]:
    symbols = sorted(set(map(int, plain)))
    vocab = max(8, int(math.ceil(2.0 * math.sqrt(max(2, len(symbols))))))
    codebook = {s: (rng.randrange(vocab), rng.randrange(vocab)) for s in symbols}
    out: list[int] = []
    for x in plain:
        a, b = codebook[int(x)]
        out.extend((a, b))
    return out


def entropy(seq: Sequence[int]) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    n = len(seq)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def mutual_information(a: Sequence[int], b: Sequence[int]) -> float:
    n = min(len(a), len(b))
    if n <= 1:
        return 0.0
    ca, cb, cab = Counter(a[:n]), Counter(b[:n]), Counter(zip(a[:n], b[:n]))
    mi = 0.0
    for (x, y), cxy in cab.items():
        pxy = cxy / n
        px = ca[x] / n
        py = cb[y] / n
        mi += pxy * math.log2(pxy / (px * py))
    return mi


def js_divergence(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    na, nb = len(a), len(b)
    keys = set(ca) | set(cb)
    pa = {k: ca[k] / na for k in keys}
    pb = {k: cb[k] / nb for k in keys}
    m = {k: 0.5 * (pa[k] + pb[k]) for k in keys}
    def kl(p):
        return sum(v * math.log2(v / m[k]) for k, v in p.items() if v > 0 and m[k] > 0)
    return 0.5 * kl(pa) + 0.5 * kl(pb)


def role_score(seq: Sequence[int], block: int) -> float:
    left: list[int] = []
    right: list[int] = []
    width = 2 * block
    usable = len(seq) - (len(seq) % width)
    for start in range(0, usable, width):
        chunk = seq[start:start + width]
        left.extend(chunk[:block])
        right.extend(chunk[block:])
    return js_divergence(left, right)


def features(seq: Sequence[int]) -> list[float]:
    seq = list(map(int, seq))
    h = entropy(seq)
    vals: list[float] = [math.log1p(len(seq)), math.log1p(len(set(seq))), h]
    for lag in range(1, 13):
        vals.append(mutual_information(seq[:-lag], seq[lag:]) / h if len(seq) > lag and h > 0 else 0.0)
    scores = [role_score(seq, b) for b in CANDIDATE_BLOCKS]
    vals.extend(scores)
    vals.extend([max(scores), statistics.fmean(scores), statistics.pstdev(scores), float(CANDIDATE_BLOCKS[int(np.argmax(scores))])])
    return vals


def auc_with_null(y: np.ndarray, p: np.ndarray, seed: int, n_perm: int = 400):
    obs = float(roc_auc_score(y, p))
    rng = np.random.default_rng(seed)
    null = [float(roc_auc_score(rng.permutation(y), p)) for _ in range(n_perm)]
    sd = float(np.std(null, ddof=1))
    effect = obs - 0.5
    return {"auc": obs, "effect_over_chance": effect, "null_sd": sd, "effect_over_null_sd": effect / sd if sd > 0 else float("inf")}


def build_dataset(mod, languages, split: str, blocks: Sequence[int], replicates: int, negative: str):
    X: list[list[float]] = []
    y: list[int] = []
    meta: list[dict] = []
    for iso, lang in languages.items():
        chunks = mod.source_chunks(lang, split, LENGTH)
        for rep, chunk in enumerate(chunks[: min(replicates, len(chunks))]):
            plain = nonspace_values(chunk, lang.char_to_id.get(" "))
            if len(plain) < 64:
                continue
            b = blocks[rep % len(blocks)]
            noisy = bool(rep % 2)
            pos = encode_two_stream(plain, b, random.Random(stable_seed("pos", split, iso, rep, b, noisy)), noisy=noisy)
            X.append(features(pos)); y.append(1); meta.append({"iso": iso, "block": b, "class": "fractionated"})
            if negative == "easy":
                neg = encode_easy_bigraphic(plain, random.Random(stable_seed("easy", split, iso, rep, b, noisy)))
            elif negative == "twin":
                neg = encode_two_stream(plain, b, random.Random(stable_seed("twin", split, iso, rep, b, noisy)), noisy=noisy)
            else:
                raise ValueError(negative)
            X.append(features(neg)); y.append(0); meta.append({"iso": iso, "block": b, "class": negative})
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int), meta


def fit_eval(mod, languages, replicates: int, negative: str):
    Xtr, ytr, _ = build_dataset(mod, languages, "train", TRAIN_BLOCKS, replicates, negative)
    Xte, yte, meta = build_dataset(mod, languages, "test", TEST_BLOCKS, replicates, negative)
    clf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_leaf=4, class_weight="balanced", random_state=stable_seed("rf", negative) % (2**32 - 1), n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    overall = auc_with_null(yte, p, stable_seed("perm", negative) % (2**32 - 1))
    per_language = {}
    for iso in languages:
        idx = np.asarray([i for i, row in enumerate(meta) if row["iso"] == iso], dtype=int)
        if len(idx) and len(set(yte[idx])) == 2:
            per_language[iso] = auc_with_null(yte[idx], p[idx], stable_seed("perm", negative, iso) % (2**32 - 1), n_perm=200)
    return overall, per_language


def parameter_recovery(mod, languages, replicates: int):
    rows = []
    for iso, lang in languages.items():
        chunks = mod.source_chunks(lang, "test", LENGTH)
        for rep, chunk in enumerate(chunks[: min(replicates, len(chunks))]):
            plain = nonspace_values(chunk, lang.char_to_id.get(" "))
            b = TEST_BLOCKS[rep % len(TEST_BLOCKS)]
            for noisy in (False, True):
                seq = encode_two_stream(plain, b, random.Random(stable_seed("param", iso, rep, b, noisy)), noisy=noisy)
                scores = {cand: role_score(seq, cand) for cand in CANDIDATE_BLOCKS}
                pred = max(scores, key=scores.get)
                rows.append({"iso": iso, "true": b, "pred": pred, "noisy": noisy, "score": scores[pred]})
    def summarise(sub):
        return {"n": len(sub), "exact_rate": statistics.fmean(r["pred"] == r["true"] for r in sub), "within_one_rate": statistics.fmean(abs(r["pred"] - r["true"]) <= 1 for r in sub), "mean_best_jsd": statistics.fmean(r["score"] for r in sub)} if sub else {}
    return {"clean": summarise([r for r in rows if not r["noisy"]]), "noisy": summarise([r for r in rows if r["noisy"]]), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--replicates", type=int, default=32)
    args = ap.parse_args()
    mod = load_v050(args.repo)
    here = args.repo / "experiments/recoverability_frontier_v0_5"
    languages = mod.load_languages(here / "corpus_manifest_v050.json", args.repo / ".cache/ud-v050")
    easy, easy_lang = fit_eval(mod, languages, args.replicates, "easy")
    twin, twin_lang = fit_eval(mod, languages, args.replicates, "twin")
    param = parameter_recovery(mod, languages, args.replicates)
    easy_resolves = easy["auc"] >= 0.80 and easy["effect_over_null_sd"] >= 2.0
    twin_resolves = twin["auc"] >= 0.65 and twin["effect_over_null_sd"] >= 2.0
    decision = "GO_TO_VOYNICH" if (easy_resolves and twin_resolves) else "STOP_NON_IDENTIFIABLE"
    if not easy_resolves:
        decision = "STOP_DETECTOR_WEAK"
    payload = {
        "programme": "fractionation-signature-gate-v0.1",
        "design": {"languages": list(languages), "length": LENGTH, "train_blocks": TRAIN_BLOCKS, "test_blocks_unseen": TEST_BLOCKS, "replicates_per_language_split": args.replicates, "voynich_used": False},
        "easy_control": {"overall": easy, "per_language": easy_lang},
        "observational_twin": {"overall": twin, "per_language": twin_lang},
        "block_parameter_recovery": param,
        "gate": {"easy_resolves": easy_resolves, "twin_resolves": twin_resolves, "decision": decision, "rule": "GO only if easy AUC>=0.80 and hard-twin AUC>=0.65, with effect/null-SD>=2 for both"},
        "interpretation_constraint": "A hard-twin failure means surface statistics can recognize two-stream production but cannot attribute it specifically to coordinate/Polybius semantics."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("FRAC_GATE_EASY", json.dumps(easy, sort_keys=True))
    print("FRAC_GATE_TWIN", json.dumps(twin, sort_keys=True))
    print("FRAC_GATE_PARAM_CLEAN", json.dumps(param["clean"], sort_keys=True))
    print("FRAC_GATE_PARAM_NOISY", json.dumps(param["noisy"], sort_keys=True))
    print("FRAC_GATE_DECISION", decision)


if __name__ == "__main__":
    main()
