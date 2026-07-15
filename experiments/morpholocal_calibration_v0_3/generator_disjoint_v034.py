#!/usr/bin/env python3
"""Generator-disjoint development validation for the v0.3.3 sequence test."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import statistics
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Sequence

import remediation_runtime_v033 as remediation

base = remediation.base

POSITIVE_FAMILIES = (
    "prf_global",
    "rotor_currier",
    "feedback_global_null",
    "line_keyed_currier_null",
)
CONTROL_FAMILIES = (
    "ordered_hmm",
    "motif_grammar",
    "topic_fsm",
    "copy_mutate_latent",
)
CORPORA = {
    "greek_general": "Paper/Cipher_paper/greek_corpus_parsed.pkl",
    "greek_dmm": "Paper/Cipher_paper/greek_dmm_corpus.pkl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def independent_key(
    rng: random.Random,
    sizes: Sequence[int],
    null_count: int,
    n_cells: int,
    n_payload_units: int,
) -> tuple[int, ...]:
    cells = list(range(n_cells))
    rng.shuffle(cells)
    assignment = [-1] * n_cells
    offset = 0
    for unit, size in enumerate(sizes):
        for cell in cells[offset : offset + int(size)]:
            assignment[cell] = unit
        offset += int(size)
    for cell in cells[offset : offset + null_count]:
        assignment[cell] = n_payload_units
    offset += null_count
    if offset != n_cells or any(value < 0 for value in assignment):
        raise RuntimeError("invalid independent key")
    return tuple(assignment)


def load_words(repo: Path, corpus: str) -> list[str]:
    path = repo / CORPORA[corpus]
    payload = pickle.load(path.open("rb"))
    words = [str(word) for word in payload["all_words"] if str(word).strip()]
    if len(words) < 1000:
        raise RuntimeError(f"corpus too small: {corpus}")
    return words


def corpus_lines(module: Any, words: Sequence[str], plan, seed: int, null_count: int) -> list[list[int]]:
    offset = stable_int("corpus-offset", seed) % len(words)
    cursor = 0
    null_rate = 0.055 if null_count else 0.0
    rng = random.Random(seed ^ 0xC04F05)
    lines: list[list[int]] = []
    for _doc, _quire, _section, _currier, _is_test, line_lengths in plan:
        for length in line_lengths:
            row: list[int] = []
            for _ in range(length):
                if null_count and rng.random() < null_rate:
                    row.append(12)
                    continue
                word = words[(offset + cursor) % len(words)]
                cursor += 1
                row.append(int(module.classify_word(word)))
            lines.append(row)
    return lines


def ordered_control_lines(plan, family: str, seed: int) -> list[list[int]]:
    rng = random.Random(seed ^ 0x0C017201)
    lines: list[list[int]] = []
    previous_by_doc: dict[int, list[int]] = {}
    motifs = [
        [rng.randrange(12) for _ in range(rng.randrange(3, 7))]
        for _ in range(8)
    ]
    successors = [rng.randrange(12) for _ in range(12)]
    for doc, _quire, _section, _currier, _is_test, line_lengths in plan:
        topic = random.Random(seed ^ (doc * 0x9E3779B1)).sample(range(12), 4)
        for line_no, length in enumerate(line_lengths):
            if family == "ordered_hmm":
                state = rng.randrange(12)
                row = [state]
                for _ in range(1, length):
                    draw = rng.random()
                    if draw < 0.58:
                        state = state
                    elif draw < 0.90:
                        state = successors[state]
                    else:
                        state = rng.randrange(12)
                    row.append(state)
            elif family == "motif_grammar":
                motif = motifs[(doc + line_no + rng.randrange(len(motifs))) % len(motifs)]
                row = []
                for index in range(length):
                    value = motif[index % len(motif)]
                    if rng.random() < 0.10:
                        value = rng.randrange(12)
                    row.append(value)
            elif family == "topic_fsm":
                state = topic[(doc + line_no) % len(topic)]
                row = [state]
                for _ in range(1, length):
                    if rng.random() < 0.86:
                        state = topic[(topic.index(state) + 1) % len(topic)]
                    else:
                        state = rng.randrange(12)
                    row.append(state)
            elif family == "copy_mutate_latent":
                prior = previous_by_doc.get(doc)
                if prior is None:
                    row = [rng.randrange(12) for _ in range(length)]
                else:
                    row = [prior[index % len(prior)] for index in range(length)]
                    for index in range(length):
                        if rng.random() < 0.14:
                            row[index] = rng.randrange(12)
                previous_by_doc[doc] = list(row)
            else:
                raise ValueError(family)
            lines.append(row)
    return lines


def family_spec(family: str) -> tuple[str, int, str, str]:
    if family == "prf_global":
        return "global", 0, "unequal", "prf"
    if family == "rotor_currier":
        return "currier", 0, "balanced", "rotor"
    if family == "feedback_global_null":
        return "global", 2, "unequal", "feedback"
    if family == "line_keyed_currier_null":
        return "currier", 2, "balanced", "line_keyed"
    raise ValueError(family)


def choose_surface_cell(
    assignment: Sequence[int],
    unit: int,
    mechanism: str,
    seed: int,
    label: str,
    doc: int,
    line: int,
    occurrence: int,
    previous_unit: int | None,
    previous_cell: int | None,
) -> int:
    candidates = [cell for cell, value in enumerate(assignment) if int(value) == int(unit)]
    if not candidates:
        raise RuntimeError(f"unit {unit} has no cells")
    if mechanism == "prf":
        slot = stable_int(seed, label, unit, occurrence, "prf") % len(candidates)
    elif mechanism == "rotor":
        offset = stable_int(seed, label, unit, "rotor") % len(candidates)
        slot = (occurrence + offset + doc + line) % len(candidates)
    elif mechanism == "feedback":
        feedback = -1 if previous_unit is None else int(previous_unit)
        slot = stable_int(seed, label, unit, feedback, occurrence // 2) % len(candidates)
    elif mechanism == "line_keyed":
        slot = stable_int(seed, label, unit, doc, line, occurrence) % len(candidates)
    else:
        raise ValueError(mechanism)
    cell = int(sorted(candidates)[slot])
    if mechanism == "feedback" and previous_unit == unit and previous_cell in candidates:
        if stable_int(seed, doc, line, occurrence, "repeat") % 5 == 0:
            cell = int(previous_cell)
    return cell


def independent_token(registry: Any, seed: int, cell: int, is_test: bool, serial: int) -> tuple[int, str]:
    use_heldout = is_test and stable_int(seed, cell, serial, "heldout") % 20 < 7
    candidates = registry.heldout_token_indexes[cell] if use_heldout else registry.train_token_indexes[cell]
    if not candidates:
        candidates = tuple(range(len(registry.token_names[cell])))
    index = int(candidates[stable_int(seed, cell, serial, "token") % len(candidates)])
    return index, str(registry.token_names[cell][index])


def render_lines(
    module: Any,
    registry: Any,
    plan,
    latent_lines: Sequence[Sequence[int]],
    seed: int,
    key_scheme: str,
    null_count: int,
    size_profile: str,
    mechanism: str,
    expose_truth: bool,
):
    rng = random.Random(seed ^ 0x4B4559)
    sizes = module.class_sizes(size_profile, len(registry.cells) - null_count)
    labels = ("GLOBAL",) if key_scheme == "global" else ("A", "B")
    keys = {
        label: independent_key(rng, sizes, null_count, len(registry.cells), 12)
        for label in labels
    }
    occurrence: defaultdict[tuple[str, int], int] = defaultdict(int)
    events = []
    line_cursor = 0
    serial = 0
    for doc, quire, section, currier, is_test, line_lengths in plan:
        for line_no, length in enumerate(line_lengths):
            latent = list(latent_lines[line_cursor])
            line_cursor += 1
            if len(latent) != length:
                raise RuntimeError("latent line length mismatch")
            previous_unit = None
            previous_cell = None
            for position_index, unit in enumerate(latent):
                position = "FIRST" if position_index == 0 else "LAST" if position_index == length - 1 else "MID"
                label = "GLOBAL" if key_scheme == "global" else currier
                count = occurrence[(label, int(unit))]
                cell = choose_surface_cell(
                    keys[label], int(unit), mechanism, seed, label, doc, line_no,
                    count, previous_unit, previous_cell,
                )
                occurrence[(label, int(unit))] += 1
                token_index, token = independent_token(registry, seed, cell, is_test, serial)
                events.append(module.Event(
                    doc=doc,
                    quire=quire,
                    line=line_no,
                    section=section,
                    currier=currier,
                    position=position,
                    test=is_test,
                    cell=cell,
                    token_index=token_index,
                    token=token,
                    length=len(token),
                    true_unit=int(unit) if expose_truth else None,
                ))
                serial += 1
                previous_unit = int(unit)
                previous_cell = int(cell)
    return events, keys, tuple(int(value) for value in sizes)


def load_assets(repo: Path):
    remediation.install()
    gr, module = base.load_v02(repo)
    record_path, ci_path = module.locate_data(repo)
    registry = module.build_surface_registry(pickle.load(record_path.open("rb")))
    external = module.build_external_models(pickle.load(ci_path.open("rb")))
    return gr, module, registry, external


def run_task(repo_text: str, cfg: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    repo = Path(repo_text)
    gr, module, registry, external = load_assets(repo)
    seed = int(task["seed"])
    plan = module.document_plan(
        registry, seed, n_docs=int(task["n_docs"]), tokens_per_doc=int(task["tokens_per_doc"])
    )
    if task["trial_type"] == "positive":
        family = str(task["family"])
        key_scheme, null_count, size_profile, mechanism = family_spec(family)
        words = load_words(repo, str(task["corpus"]))
        latent = corpus_lines(module, words, plan, seed, null_count)
        events, keys, sizes = render_lines(
            module, registry, plan, latent, seed, key_scheme, null_count,
            size_profile, mechanism, True,
        )
        result = base.score_trial_v03(
            module, gr, events, registry, external, seed ^ 0xC1F304,
            "beam", cfg, None,
        )
        result.update({
            "trial_type": "positive",
            "family": family,
            "corpus": task["corpus"],
            "generator_key_scheme": key_scheme,
            "generator_null_count": null_count,
            "generator_size_profile": size_profile,
            "generator_mechanism": mechanism,
            "generator_key_hash": hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest(),
            "generator_class_sizes": list(sizes),
        })
    else:
        family = str(task["family"])
        latent = ordered_control_lines(plan, family, seed)
        surface_mechanisms = ("prf", "rotor", "feedback", "line_keyed")
        mechanism = surface_mechanisms[int(task["replicate"]) % len(surface_mechanisms)]
        key_scheme = "global" if int(task["replicate"]) % 2 == 0 else "currier"
        events, keys, sizes = render_lines(
            module, registry, plan, latent, seed, key_scheme, 0,
            "unequal" if key_scheme == "global" else "balanced",
            mechanism, False,
        )
        result = base.score_trial_v03(
            module, gr, events, registry, external, seed ^ 0xF00D34,
            "beam", cfg, None,
        )
        result.update({
            "trial_type": "control",
            "family": family,
            "corpus": None,
            "generator_key_scheme": key_scheme,
            "generator_null_count": 0,
            "generator_mechanism": mechanism,
            "generator_key_hash": hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest(),
            "generator_class_sizes": list(sizes),
        })
    sequence = result["sequence_randomization"]
    result.update({
        "task_id": task["task_id"],
        "replicate": int(task["replicate"]),
        "seed": seed,
        "n_docs_requested": int(task["n_docs"]),
        "tokens_per_doc_requested": int(task["tokens_per_doc"]),
        "detected": bool(sequence["p_value"] <= 0.05 and sequence["advantage_bits"] > 0.0),
    })
    return result


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)) ** 0.5) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in results if row["trial_type"] == "positive"]
    controls = [row for row in results if row["trial_type"] == "control"]
    tp = sum(bool(row["detected"]) for row in positives)
    fp = sum(bool(row["detected"]) for row in controls)

    def grouped(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
        buckets: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key))].append(row)
        return {
            value: {
                "detected": sum(bool(row["detected"]) for row in bucket),
                "trials": len(bucket),
                "rate": sum(bool(row["detected"]) for row in bucket) / len(bucket),
                "median_p": statistics.median(float(row["sequence_randomization"]["p_value"]) for row in bucket),
                "median_effect_bpt": statistics.median(float(row["sequence_randomization"]["advantage_bits_per_transition"]) for row in bucket),
            }
            for value, bucket in sorted(buckets.items())
        }

    mechanism_groups = grouped(positives, "family")
    control_groups = grouped(controls, "family")
    escalation = (
        tp / max(1, len(positives)) >= 0.65
        and fp / max(1, len(controls)) <= 0.20
        and sum(row["rate"] >= 0.50 for row in mechanism_groups.values()) >= 3
        and all(row["rate"] <= 0.50 for row in control_groups.values())
    )
    return {
        "positive": {
            "detected": tp,
            "trials": len(positives),
            "sensitivity": tp / max(1, len(positives)),
            "wilson95": wilson(tp, len(positives)),
        },
        "control": {
            "false_positives": fp,
            "trials": len(controls),
            "false_positive_rate": fp / max(1, len(controls)),
            "specificity": 1.0 - fp / max(1, len(controls)),
            "wilson95_fpr": wilson(fp, len(controls)),
        },
        "positive_family": mechanism_groups,
        "positive_corpus": grouped(positives, "corpus"),
        "control_family": control_groups,
        "smoke_escalation_gate": bool(escalation),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--positive-replicates", type=int, default=2)
    parser.add_argument("--control-replicates", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3403403)
    parser.add_argument("--n-docs", type=int, default=12)
    parser.add_argument("--tokens-per-doc", type=int, default=180)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.positive_replicates < 1 or args.control_replicates < 1:
        raise SystemExit("replicates must be positive")
    cfg = json.loads(args.config.read_text())
    cfg.setdefault("sequence_randomizations", 199)
    tasks: list[dict[str, Any]] = []
    serial = 0
    for corpus in CORPORA:
        for family in POSITIVE_FAMILIES:
            for replicate in range(args.positive_replicates):
                tasks.append({
                    "task_id": f"P-{corpus}-{family}-{replicate}",
                    "trial_type": "positive",
                    "corpus": corpus,
                    "family": family,
                    "replicate": replicate,
                    "seed": args.seed + 100000 + serial * 104729,
                    "n_docs": args.n_docs,
                    "tokens_per_doc": args.tokens_per_doc,
                })
                serial += 1
    for family_index, family in enumerate(CONTROL_FAMILIES):
        for replicate in range(args.control_replicates):
            tasks.append({
                "task_id": f"C-{family}-{replicate}",
                "trial_type": "control",
                "corpus": None,
                "family": family,
                "replicate": replicate,
                "seed": args.seed + 900000 + (family_index * args.control_replicates + replicate) * 130363,
                "n_docs": args.n_docs,
                "tokens_per_doc": args.tokens_per_doc,
            })
    started = time.time()
    results: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(run_task, str(args.repo), cfg, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            task = futures[future]
            sequence = result["sequence_randomization"]
            print(
                f"V034_PROGRESS completed={completed}/{len(tasks)} id={task['task_id']} "
                f"detected={int(result['detected'])} p={sequence['p_value']:.6g} "
                f"effect={sequence['advantage_bits_per_transition']:.6g} "
                f"elapsed={time.time()-started:.1f}s",
                flush=True,
            )
    results.sort(key=lambda row: str(row["task_id"]))
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip()
    source_paths = [
        args.repo / "experiments/morpholocal_calibration_v0_3/generator_disjoint_v034.py",
        args.repo / "experiments/morpholocal_calibration_v0_3/remediation_runtime_v033.py",
        args.repo / "experiments/morpholocal_calibration_v0_3/remediation_runtime_v032.py",
        args.repo / "experiments/morpholocal_calibration_v0_3/remediation_runtime.py",
        args.repo / "experiments/morpholocal_calibration_v0_3/tournament_runner.py",
    ]
    corpus_paths = {name: args.repo / relative for name, relative in CORPORA.items()}
    payload = {
        "programme": "morpholocal-calibration-v0.3.4-generator-disjoint-smoke",
        "formal": False,
        "primary_metric": "v0.3.3_sequence_randomization_detection",
        "decision": {"alpha": 0.05, "positive_effect_required": True},
        "config": cfg,
        "seed": args.seed,
        "n_docs": args.n_docs,
        "tokens_per_doc": args.tokens_per_doc,
        "git_commit": commit,
        "source_sha256": {str(path.relative_to(args.repo)): sha256_file(path) for path in source_paths},
        "corpus_sha256": {name: sha256_file(path) for name, path in corpus_paths.items()},
        "summary": summarize(results),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print("V034_SUMMARY", json.dumps(payload["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
