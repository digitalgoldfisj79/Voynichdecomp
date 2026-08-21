#!/usr/bin/env python3
"""SVT v0.1: order-free stateful variable-length cipher calibration.

Synthetic-only by construction. This module contains no Voynich loader.

The cipher maps each plaintext unit, under a bounded state schedule, to an
opaque surface codeword of length 1..3. Codeword boundaries, state schedule,
codebooks and plaintext are hidden from the joint solver. Visible symbols have
no circular order.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import mono_solver_v051 as mono

CONFIG = json.loads((HERE / "CONFIG_FROZEN.json").read_text(encoding="utf-8"))
MAX_CODE_LENGTH = int(CONFIG["max_code_length"])
SURFACE_A = int(CONFIG["surface_alphabet_size"])
LENGTH_PRIOR = {int(k): float(v) for k, v in CONFIG["length_prior"].items()}
BEAM_WIDTH = int(CONFIG["beam_width"])
NEW_MAP_PENALTY = float(CONFIG["new_mapping_penalty"])
NEW_BRANCH = int(CONFIG["candidate_plain_symbols_per_new_codeword"])
UNIGRAM_WEIGHT = float(CONFIG["score_unigram_weight"])
PERIOD_PENALTY = float(CONFIG["period_penalty"])
MODES = tuple(CONFIG["modes"])
DEV_PERIODS = tuple(int(x) for x in CONFIG["development_periods"])
LOCKED_PERIODS = tuple(int(x) for x in CONFIG["locked_periods"])
SEED_NS = str(CONFIG["seed_namespace"])


def stable_seed(*parts: object) -> int:
    blob = "|".join([SEED_NS, *(str(x) for x in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def weighted_choice(rng: random.Random, pairs: Sequence[tuple[Any, float]]) -> Any:
    total = sum(weight for _, weight in pairs)
    needle = rng.random() * total
    cumulative = 0.0
    for value, weight in pairs:
        cumulative += weight
        if needle <= cumulative:
            return value
    return pairs[-1][0]


def make_plain_line_starts(length: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    starts = [0]
    cursor = 0
    while cursor < length:
        cursor += rng.randint(24, 40)
        if cursor < length:
            starts.append(cursor)
    return starts


def line_index_for_plain(pos: int, starts: Sequence[int]) -> tuple[int, int]:
    line = 0
    for idx, start in enumerate(starts):
        if start <= pos:
            line = idx
        else:
            break
    return line, pos - starts[line]


def state_for_plain(pos: int, period: int, mode: str, plain_line_starts: Sequence[int]) -> int:
    if mode == "periodic":
        return pos % period
    if mode == "line_reset":
        _, within = line_index_for_plain(pos, plain_line_starts)
        return within % period
    raise ValueError(mode)


def all_surface_words(max_len: int = MAX_CODE_LENGTH) -> list[tuple[int, ...]]:
    words: list[tuple[int, ...]] = []
    frontier = [()]
    for _length in range(1, max_len + 1):
        nxt: list[tuple[int, ...]] = []
        for prefix in frontier:
            for symbol in range(SURFACE_A):
                word = prefix + (symbol,)
                words.append(word)
                nxt.append(word)
        frontier = nxt
    return words


SURFACE_WORDS = all_surface_words()


def make_codebooks(alphabet_size: int, period: int, seed: int) -> list[dict[int, tuple[int, ...]]]:
    """Fresh injective codebook per state; cross-state reuse is allowed."""
    books: list[dict[int, tuple[int, ...]]] = []
    for state in range(period):
        rng = random.Random(stable_seed("codebook", seed, state))
        by_len: dict[int, list[tuple[int, ...]]] = {
            length: [word for word in SURFACE_WORDS if len(word) == length]
            for length in range(1, MAX_CODE_LENGTH + 1)
        }
        for values in by_len.values():
            rng.shuffle(values)
        cursor = {length: 0 for length in by_len}
        used: set[tuple[int, ...]] = set()
        book: dict[int, tuple[int, ...]] = {}
        for plain in range(alphabet_size):
            for _attempt in range(1000):
                length = int(weighted_choice(rng, sorted(LENGTH_PRIOR.items())))
                values = by_len[length]
                if cursor[length] >= len(values):
                    rng.shuffle(values)
                    cursor[length] = 0
                word = values[cursor[length]]
                cursor[length] += 1
                if word not in used:
                    used.add(word)
                    book[plain] = word
                    break
            else:
                raise RuntimeError("failed to sample injective state codebook")
        books.append(book)
    return books


@dataclasses.dataclass(frozen=True)
class Trial:
    iso: str
    split: str
    length: int
    mode: str
    period: int
    replicate: int
    seed: int
    plain: tuple[int, ...]
    cipher: tuple[int, ...]
    plain_line_starts: tuple[int, ...]
    surface_line_starts: tuple[int, ...]
    truth_boundaries: tuple[int, ...]


def make_trial(
    language: core.LanguageData,
    split: str,
    length: int,
    mode: str,
    period: int,
    replicate: int,
) -> Trial:
    chunks = core.source_chunks(language, split, length)
    if not chunks:
        raise RuntimeError(f"no source chunks for {language.iso}/{split}/{length}")
    plain = tuple(chunks[replicate % len(chunks)])
    seed = stable_seed(language.iso, split, length, mode, period, replicate)
    plain_line_starts = make_plain_line_starts(len(plain), stable_seed("lines", seed))
    books = make_codebooks(len(language.alphabet), period, seed)
    cipher: list[int] = []
    boundaries: list[int] = []
    surface_line_starts = [0]
    next_plain_line = 1
    for pos, value in enumerate(plain):
        if next_plain_line < len(plain_line_starts) and pos == plain_line_starts[next_plain_line]:
            surface_line_starts.append(len(cipher))
            next_plain_line += 1
        state = state_for_plain(pos, period, mode, plain_line_starts)
        cipher.extend(books[state][int(value)])
        boundaries.append(len(cipher))
    return Trial(
        iso=language.iso,
        split=split,
        length=length,
        mode=mode,
        period=period,
        replicate=replicate,
        seed=seed,
        plain=plain,
        cipher=tuple(cipher),
        plain_line_starts=tuple(plain_line_starts),
        surface_line_starts=tuple(surface_line_starts),
        truth_boundaries=tuple(boundaries),
    )


def surface_line_ends(trial: Trial) -> tuple[int, ...]:
    return tuple(list(trial.surface_line_starts[1:]) + [len(trial.cipher)])


def surface_line_at(pos: int, starts: Sequence[int]) -> int:
    line = 0
    for idx, start in enumerate(starts):
        if start <= pos:
            line = idx
        else:
            break
    return line


@dataclasses.dataclass
class Hypothesis:
    score: float
    i: int
    j: int
    line_units: int
    history: tuple[int, ...]
    mapping: dict[tuple[int, tuple[int, ...]], int]
    inverse: dict[tuple[int, int], tuple[int, ...]]
    decoded: tuple[int, ...]
    boundaries: tuple[int, ...]


def lm_increment(
    history: Sequence[int],
    symbol: int,
    trigram: np.ndarray,
    unigram: np.ndarray,
) -> float:
    score = UNIGRAM_WEIGHT * float(unigram[symbol])
    if len(history) >= 2:
        score += float(trigram[int(history[-2]), int(history[-1]), symbol])
    else:
        score += float(unigram[symbol])
    return score


def top_plain_candidates(
    history: Sequence[int],
    alphabet_size: int,
    trigram: np.ndarray,
    unigram: np.ndarray,
    limit: int,
) -> list[int]:
    scored = [
        (lm_increment(history, symbol, trigram, unigram), symbol)
        for symbol in range(alphabet_size)
    ]
    scored.sort(reverse=True)
    return [symbol for _, symbol in scored[:limit]]


def current_state(h: Hypothesis, period: int, mode: str) -> int:
    if mode == "periodic":
        return h.j % period
    if mode == "line_reset":
        return h.line_units % period
    raise ValueError(mode)


def can_take_segment(trial: Trial, pos: int, length: int) -> bool:
    if pos + length > len(trial.cipher):
        return False
    line = surface_line_at(pos, trial.surface_line_starts)
    return pos + length <= surface_line_ends(trial)[line]


def line_units_after(trial: Trial, old_i: int, new_i: int, old_units: int) -> int:
    old_line = surface_line_at(old_i, trial.surface_line_starts)
    if new_i >= len(trial.cipher):
        return old_units + 1
    new_line = surface_line_at(new_i, trial.surface_line_starts)
    if new_line != old_line:
        return 0
    return old_units + 1


def path_rank(h: Hypothesis) -> float:
    # Beam pruning only. Final model selection uses raw penalised score.
    return h.score / max(1.0, math.sqrt(h.j + 1.0))


def allowed_lengths_from_truth(trial: Trial, pos: int) -> tuple[int, ...]:
    for boundary in trial.truth_boundaries:
        if boundary > pos:
            return (boundary - pos,)
    return ()


def solve_candidate(
    trial: Trial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    mode: str,
    period: int,
    oracle_boundaries: bool,
    beam_width: int = BEAM_WIDTH,
) -> dict[str, Any]:
    trigram, unigram = model
    alphabet_size = len(language.alphabet)
    frontier = [
        Hypothesis(
            score=0.0,
            i=0,
            j=0,
            line_units=0,
            history=(),
            mapping={},
            inverse={},
            decoded=(),
            boundaries=(),
        )
    ]
    completed: list[Hypothesis] = []
    max_steps = max(4, len(trial.cipher) + 2)

    for _step in range(max_steps):
        expanded: list[Hypothesis] = []
        for h in frontier:
            if h.i == len(trial.cipher):
                completed.append(h)
                continue
            state = current_state(h, period, mode)
            lengths: Iterable[int]
            if oracle_boundaries:
                lengths = allowed_lengths_from_truth(trial, h.i)
            else:
                lengths = range(1, MAX_CODE_LENGTH + 1)
            for length in lengths:
                if not can_take_segment(trial, h.i, int(length)):
                    continue
                segment = tuple(trial.cipher[h.i : h.i + int(length)])
                key = (state, segment)
                known = h.mapping.get(key)
                if known is not None:
                    symbols = [known]
                    is_new = False
                else:
                    symbols = top_plain_candidates(
                        h.history,
                        alphabet_size,
                        trigram,
                        unigram,
                        max(NEW_BRANCH * 2, NEW_BRANCH),
                    )
                    symbols = [s for s in symbols if (state, s) not in h.inverse][:NEW_BRANCH]
                    is_new = True
                for symbol in symbols:
                    mapping = h.mapping
                    inverse = h.inverse
                    score = h.score + lm_increment(h.history, symbol, trigram, unigram)
                    score += math.log(LENGTH_PRIOR[int(length)])
                    if is_new:
                        mapping = dict(mapping)
                        inverse = dict(inverse)
                        mapping[key] = symbol
                        inverse[(state, symbol)] = segment
                        score -= NEW_MAP_PENALTY
                    new_i = h.i + int(length)
                    new_line_units = line_units_after(trial, h.i, new_i, h.line_units)
                    expanded.append(
                        Hypothesis(
                            score=score,
                            i=new_i,
                            j=h.j + 1,
                            line_units=new_line_units,
                            history=(h.history + (symbol,))[-2:],
                            mapping=mapping,
                            inverse=inverse,
                            decoded=h.decoded + (symbol,),
                            boundaries=h.boundaries + (new_i,),
                        )
                    )
        if not expanded:
            break
        # Deduplicate exact structural states before truncation.
        best_by_signature: dict[tuple[Any, ...], Hypothesis] = {}
        for h in expanded:
            signature = (
                h.i,
                h.j % max(1, period),
                h.line_units % max(1, period),
                h.history,
                tuple(sorted(h.mapping.items())),
            )
            prior = best_by_signature.get(signature)
            if prior is None or h.score > prior.score:
                best_by_signature[signature] = h
        frontier = sorted(best_by_signature.values(), key=path_rank, reverse=True)[:beam_width]
        if frontier and all(h.i == len(trial.cipher) for h in frontier):
            completed.extend(frontier)
            break

    if not completed:
        return {
            "ok": False,
            "mode": mode,
            "period": period,
            "reason": "no_complete_path",
        }
    best = max(completed, key=lambda h: h.score)
    complexity = PERIOD_PENALTY * max(0, period - 1) * math.log(max(2, len(best.decoded)))
    return {
        "ok": True,
        "mode": mode,
        "period": period,
        "raw_score": float(best.score),
        "score": float(best.score - complexity),
        "decoded": list(best.decoded),
        "boundaries": list(best.boundaries),
        "mapping_size": len(best.mapping),
    }


def boundary_f1(truth: Sequence[int], predicted: Sequence[int]) -> float:
    a = set(int(x) for x in truth)
    b = set(int(x) for x in predicted)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    tp = len(a & b)
    precision = tp / len(b)
    recall = tp / len(a)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def recovery(truth: Sequence[int], predicted: Sequence[int]) -> float:
    return float(mono.fast_accuracy(list(truth), list(predicted)))


def solve_trial(
    trial: Trial,
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    stage: str,
) -> dict[str, Any]:
    if stage == "oracle_boundaries":
        candidates = [(trial.mode, trial.period, True)]
    elif stage == "oracle_schedule":
        candidates = [(trial.mode, trial.period, False)]
    elif stage in ("joint", "locked"):
        periods = DEV_PERIODS if stage == "joint" else LOCKED_PERIODS
        candidates = [(mode, period, False) for mode in MODES for period in periods]
    else:
        raise ValueError(stage)

    rows = [
        solve_candidate(trial, language, model, mode, period, oracle_boundaries)
        for mode, period, oracle_boundaries in candidates
    ]
    legal = [row for row in rows if row.get("ok")]
    if not legal:
        return {
            "ok": False,
            "iso": trial.iso,
            "stage": stage,
            "true_mode": trial.mode,
            "true_period": trial.period,
            "replicate": trial.replicate,
        }
    selected = max(legal, key=lambda row: float(row["score"]))
    return {
        "ok": True,
        "iso": trial.iso,
        "stage": stage,
        "split": trial.split,
        "length": trial.length,
        "replicate": trial.replicate,
        "true_mode": trial.mode,
        "true_period": trial.period,
        "selected_mode": selected["mode"],
        "selected_period": selected["period"],
        "mode_correct": selected["mode"] == trial.mode,
        "period_correct": selected["period"] == trial.period,
        "recovery": recovery(trial.plain, selected["decoded"]),
        "boundary_f1": boundary_f1(trial.truth_boundaries, selected["boundaries"]),
        "decoded_length": len(selected["decoded"]),
        "cipher_length": len(trial.cipher),
        "mapping_size": selected["mapping_size"],
        "score": selected["score"],
    }


def summarize(rows: Sequence[dict[str, Any]], stage: str) -> dict[str, Any]:
    legal = [row for row in rows if row.get("ok")]
    if not legal:
        return {"stage": stage, "trials": len(rows), "gate_pass": False, "reason": "no_legal_trials"}
    rec = [float(row["recovery"]) for row in legal]
    bnd = [float(row["boundary_f1"]) for row in legal]
    result = {
        "stage": stage,
        "trials": len(rows),
        "legal_trials": len(legal),
        "mean_recovery": statistics.fmean(rec),
        "median_recovery": statistics.median(rec),
        "minimum_recovery": min(rec),
        "mean_boundary_f1": statistics.fmean(bnd),
        "mode_accuracy": statistics.fmean(bool(row["mode_correct"]) for row in legal),
        "period_accuracy": statistics.fmean(bool(row["period_correct"]) for row in legal),
        "trials_ge_070": sum(value >= 0.70 for value in rec),
        "trials_ge_080": sum(value >= 0.80 for value in rec),
    }
    n = len(rows)
    if stage == "oracle_boundaries":
        result["gate_pass"] = (
            len(legal) == n
            and result["mean_recovery"] >= 0.90
            and result["median_recovery"] >= 0.95
            and result["trials_ge_080"] >= math.ceil(0.875 * n)
        )
    elif stage == "oracle_schedule":
        result["gate_pass"] = (
            len(legal) == n
            and result["mean_recovery"] >= 0.80
            and result["median_recovery"] >= 0.90
            and result["mean_boundary_f1"] >= 0.85
            and result["trials_ge_070"] >= math.ceil(0.875 * n)
        )
    elif stage == "joint":
        result["gate_pass"] = (
            len(legal) == n
            and result["mean_recovery"] >= 0.75
            and result["median_recovery"] >= 0.90
            and result["mean_boundary_f1"] >= 0.80
            and result["trials_ge_070"] >= math.ceil(0.875 * n)
            and result["mode_accuracy"] >= 0.875
            and result["period_accuracy"] >= 0.75
        )
    else:
        result["gate_pass"] = (
            len(legal) == n
            and result["mean_recovery"] >= 0.75
            and result["median_recovery"] >= 0.90
            and result["mean_boundary_f1"] >= 0.80
            and result["trials_ge_070"] >= math.ceil(0.875 * n)
            and result["mode_accuracy"] >= 0.875
            and result["period_accuracy"] >= 0.75
        )
    return result


def run_stage(
    repo: Path,
    stage: str,
    iso: str,
    length: int,
    replicates: int,
    output: Path,
) -> dict[str, Any]:
    if stage == "locked" and (HERE / "DEVELOPMENT_PASS.json").exists() is False:
        raise RuntimeError("locked stage blocked: DEVELOPMENT_PASS.json is absent")
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        repo / ".cache" / "svt-v01-corpora",
    )
    if iso not in languages:
        raise KeyError(f"unknown language {iso}; available={sorted(languages)}")
    language = languages[iso]
    model = mono.build_language_model(language)
    split = "test" if stage == "locked" else "dev"
    periods = LOCKED_PERIODS if stage == "locked" else DEV_PERIODS
    trials: list[Trial] = []
    for replicate in range(replicates):
        mode = MODES[replicate % len(MODES)]
        period = periods[replicate % len(periods)]
        trials.append(make_trial(language, split, length, mode, period, replicate))
    rows = [solve_trial(trial, language, model, stage) for trial in trials]
    summary = summarize(rows, stage)
    payload = {
        "programme": "SVT-v0.1",
        "config_sha256": hashlib.sha256((HERE / "CONFIG_FROZEN.json").read_bytes()).hexdigest(),
        "stage": stage,
        "iso": iso,
        "length": length,
        "rows": rows,
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stage", choices=("oracle_boundaries", "oracle_schedule", "joint", "locked"), required=True)
    parser.add_argument("--iso", default="de")
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_stage(args.repo, args.stage, args.iso, args.length, args.replicates, args.output)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    if not result["summary"].get("gate_pass"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
