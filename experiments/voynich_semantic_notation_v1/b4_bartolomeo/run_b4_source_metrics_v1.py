#!/usr/bin/env python3
"""VSN-B4-v1 source-only audit/metrics runner.

This program MUST NOT read Voynich data. It operates only on the frozen
Bartolomeo transcription CSV. Until all required rows are resolved at A/B
confidence it returns BLOCKED_SOURCE_TRANSCRIPTION and does not run the
binding 10k controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

SEED_NAMESPACE = "VSN_B4_V1"
REQUIRED = [
    "group_id", "compound_norm", "text_conf",
    "syll1", "syll2", "syll3", "syll4",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entropy(values) -> float:
    vals = list(values)
    if not vals:
        return float("nan")
    c = Counter(vals)
    n = len(vals)
    return -sum((v / n) * math.log2(v / n) for v in c.values())


def conditional_entropy_next(strings: list[str]) -> float:
    pairs = Counter()
    prev = Counter()
    for s in strings:
        s = "^" + s + "$"
        for a, b in zip(s, s[1:]):
            pairs[(a, b)] += 1
            prev[a] += 1
    total = sum(pairs.values())
    if not total:
        return float("nan")
    out = 0.0
    for (a, b), n in pairs.items():
        p_ab = n / total
        p_b_a = n / prev[a]
        out -= p_ab * math.log2(p_b_a)
    return out


def lev1_location(a: str, b: str):
    """Return edit index for equal-length Hamming-1 or single ins/del; else None."""
    if abs(len(a) - len(b)) > 1:
        return None
    if len(a) == len(b):
        dif = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        return dif[0] if len(dif) == 1 else None
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    used = False
    loc = None
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
        elif not used:
            used = True; loc = j; j += 1
        else:
            return None
    if not used:
        loc = len(b) - 1
    return loc


def edit_bucket(loc: int, max_len: int) -> str:
    if max_len <= 1:
        return "internal"
    x = loc / max(1, max_len - 1)
    if x <= 0.25:
        return "prefix"
    if x >= 0.75:
        return "suffix"
    return "internal"


def edit_graph(strings: list[str]):
    deg = Counter({s: 0 for s in strings})
    buckets = Counter()
    pairs = 0
    for i, a in enumerate(strings):
        for b in strings[i + 1:]:
            loc = lev1_location(a, b)
            if loc is not None:
                pairs += 1
                deg[a] += 1; deg[b] += 1
                buckets[edit_bucket(loc, max(len(a), len(b)))] += 1
    return {
        "pairs": pairs,
        "mean_degree": mean(deg.values()) if deg else float("nan"),
        "isolated_fraction": (sum(v == 0 for v in deg.values()) / len(deg)) if deg else float("nan"),
        "edit_location": dict(buckets),
    }


def positional_entropies(strings: list[str], from_right=False):
    by_pos = defaultdict(list)
    for s in strings:
        seq = s[::-1] if from_right else s
        for i, ch in enumerate(seq):
            by_pos[i].append(ch)
    return {str(k): entropy(v) for k, v in sorted(by_pos.items())}


def weighted_mean_entropy(d: dict[str, float]) -> float:
    vals = [v for v in d.values() if not math.isnan(v)]
    return mean(vals) if vals else float("nan")


def complete_row(r):
    if r.get("text_conf") not in {"A", "B"}:
        return False
    for k in REQUIRED:
        if not r.get(k) or r[k].strip().upper() == "U":
            return False
    return True


def compute_metrics(rows):
    strings = [r["compound_norm"].replace(" ", "").lower() for r in rows]
    slots = [[r[f"syll{i}"].lower() for r in rows] for i in range(1, 5)]
    left = positional_entropies(strings, False)
    right = positional_entropies(strings, True)
    return {
        "n": len(strings),
        "mean_char_length": mean(map(len, strings)) if strings else float("nan"),
        "char_lengths": Counter(map(len, strings)),
        "slot_entropy_bits": [entropy(s) for s in slots],
        "initial_component_entropy_bits": entropy(slots[0]),
        "final_component_entropy_bits": entropy(slots[3]),
        "H_next_given_prev_bits": conditional_entropy_next(strings),
        "left_positional_entropy_bits": left,
        "right_positional_entropy_bits": right,
        "right_minus_left_mean_entropy_bits": weighted_mean_entropy(right) - weighted_mean_entropy(left),
        "edit1": edit_graph(strings),
        "component_reuse": dict(Counter(x for slot in slots for x in slot)),
    }


def deterministic_rng(label: str):
    seed = int.from_bytes(hashlib.sha256((SEED_NAMESPACE + "|" + label).encode()).digest()[:8], "big")
    return random.Random(seed), seed


def control_slot_shuffle(rows, rng):
    slots = [[r[f"syll{i}"].lower() for r in rows] for i in range(1, 5)]
    for s in slots:
        rng.shuffle(s)
    return ["".join(slots[i][j] for i in range(4)) for j in range(len(rows))]


def control_order_shuffle(rows, rng):
    out = []
    for r in rows:
        s = [r[f"syll{i}"].lower() for i in range(1, 5)]
        rng.shuffle(s)
        out.append("".join(s))
    return out


def summarize_control(strings):
    return {
        "H_next_given_prev_bits": conditional_entropy_next(strings),
        "right_minus_left_mean_entropy_bits": weighted_mean_entropy(positional_entropies(strings, True)) - weighted_mean_entropy(positional_entropies(strings, False)),
        "edit1_pairs": edit_graph(strings)["pairs"],
    }


def run_controls(rows, nrep):
    observed = compute_metrics(rows)
    results = {}
    for name, maker in [("slot_marginal_shuffle", control_slot_shuffle), ("within_codeword_order_shuffle", control_order_shuffle)]:
        vals = []
        seeds = []
        for i in range(nrep):
            rng, seed = deterministic_rng(f"{name}|{i}")
            seeds.append(seed)
            vals.append(summarize_control(maker(rows, rng)))
        results[name] = {
            "replicates": nrep,
            "first_seed": seeds[0] if seeds else None,
            "metrics_mean": {k: mean(v[k] for v in vals) for k in vals[0]} if vals else {},
            "observed": {
                "H_next_given_prev_bits": observed["H_next_given_prev_bits"],
                "right_minus_left_mean_entropy_bits": observed["right_minus_left_mean_entropy_bits"],
                "edit1_pairs": observed["edit1"]["pairs"],
            },
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path)
    ap.add_argument("--output", type=Path, default=Path("b4_source_audit.json"))
    ap.add_argument("--controls", type=int, default=10000)
    ap.add_argument("--require-complete", action="store_true")
    args = ap.parse_args()

    with args.csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    complete = [r for r in rows if complete_row(r)]
    unresolved = [r["group_id"] for r in rows if not complete_row(r)]

    out = {
        "namespace": "VSN-B4-v1",
        "seed_namespace": SEED_NAMESPACE,
        "input": str(args.csv_path),
        "input_sha256": sha256_file(args.csv_path),
        "total_rows": len(rows),
        "complete_AB_rows": len(complete),
        "unresolved_rows": unresolved,
        "voynich_data_accessed": False,
    }

    if complete:
        out["draft_metrics_complete_rows_only"] = compute_metrics(complete)

    if unresolved:
        out["status"] = "BLOCKED_SOURCE_TRANSCRIPTION"
        out["binding_controls_run"] = False
        args.output.write_text(json.dumps(out, indent=2, sort_keys=True, default=lambda x: dict(x) if isinstance(x, Counter) else x), encoding="utf-8")
        print(json.dumps(out, indent=2, default=str))
        if args.require_complete:
            raise SystemExit(3)
        return

    out["status"] = "SOURCE_COMPLETE"
    out["binding_metrics"] = compute_metrics(rows)
    out["binding_controls"] = run_controls(rows, args.controls)
    out["binding_controls_run"] = True
    args.output.write_text(json.dumps(out, indent=2, sort_keys=True, default=lambda x: dict(x) if isinstance(x, Counter) else x), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
