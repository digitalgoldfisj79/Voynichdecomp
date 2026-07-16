#!/usr/bin/env python3
"""Final permitted Family G2 amended blind carrier solver.

This module preserves the frozen 2,935-rule inventory, data construction,
matched-null operating point and gates from v060_family_g_stage_g2.py.  It
implements only the amendment frozen in V060_FAMILY_G_G2_FINAL_AMENDMENT.md:
a 128-candidate invariant shortlist, 100k x 8 mono screening, null-calibrated
raw mono language evidence, and a 0.05 invariant auxiliary weight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import recoverability_v050 as core
import mono_solver_v051 as mono
import v060_family_g_stage_g1 as g1
import v060_family_g_stage_g2 as base

SHORTLIST_IDENTITY = 4
SHORTLIST_INVARIANT = 128
SCREEN_MONO_ITERATIONS = 100_000
SCREEN_MONO_RESTARTS = 8
INVARIANT_WEIGHT = 0.05


def amended_screen_mono_candidate(
    values: list[int],
    language: core.LanguageData,
    identity_model: tuple[np.ndarray, np.ndarray],
    reference: base.ReferenceStats,
    seed: int,
) -> dict[str, Any]:
    trigram, unigram = identity_model
    array = np.asarray(values, dtype=np.int32)
    initial = mono.frequency_key(values, language)
    solved_key, solved_score = mono.anneal_mono(
        array,
        initial,
        trigram,
        unigram,
        SCREEN_MONO_ITERATIONS,
        SCREEN_MONO_RESTARTS,
        int(seed & 0x7FFFFFFFFFFFFFFF),
    )
    active = len(set(values))
    penalty_per_char = (
        0.5 * max(1, active - 1) * math.log(max(2, len(values))) / len(values)
    )
    mono_average = float(solved_score) / len(values)
    # The matched 256-null maximum already calibrates key search and the full
    # 2,935-rule multiplicity.  The raw language z-score is therefore the
    # detection/ranking evidence; MDL remains a diagnostic and tie-breaker.
    mono_evidence = base.z(
        mono_average,
        reference.identity_mean,
        reference.identity_std,
    )
    mdl_adjusted_evidence = base.z(
        mono_average - penalty_per_char,
        reference.identity_mean,
        reference.identity_std,
    )
    prediction = solved_key[array].astype(np.int32).tolist()
    return {
        "mono_average": mono_average,
        "mono_penalty_per_char": penalty_per_char,
        "mono_evidence": mono_evidence,
        "mdl_adjusted_evidence": mdl_adjusted_evidence,
        "key": solved_key,
        "prediction": prediction,
    }


def amended_solve_cover(
    layout: g1.CoverLayout,
    inventory: list[base.Candidate],
    language: core.LanguageData,
    identity_model: tuple[np.ndarray, np.ndarray],
    reference: base.ReferenceStats,
    seed: int,
    refine_prediction: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    screened: list[dict[str, Any]] = []
    for index, candidate in enumerate(inventory):
        extracted = base.extract_candidate(layout, candidate)
        if len(extracted) != base.PAYLOAD_LENGTH:
            continue
        features = base.candidate_features(extracted, identity_model, reference)
        screened.append(
            {
                "index": index,
                "candidate": candidate,
                "extracted": extracted,
                "features": features,
            }
        )
    if len(screened) != len(inventory):
        raise RuntimeError(
            f"only {len(screened)} of {len(inventory)} frozen candidates had capacity"
        )

    identity_rank = sorted(
        screened, key=lambda row: row["features"]["identity_z"], reverse=True
    )[:SHORTLIST_IDENTITY]
    invariant_rank = sorted(
        screened,
        key=lambda row: row["features"]["invariant_score"],
        reverse=True,
    )[:SHORTLIST_INVARIANT]
    shortlist: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in identity_rank + invariant_rank:
        if row["index"] not in seen:
            shortlist.append(row)
            seen.add(row["index"])

    refined: list[dict[str, Any]] = []
    for row in shortlist:
        mono_result = amended_screen_mono_candidate(
            row["extracted"],
            language,
            identity_model,
            reference,
            core.stable_seed("v060-g2-final-screen-mono", seed, row["index"]),
        )
        identity_evidence = row["features"]["identity_z"]
        mono_evidence = mono_result["mono_evidence"]
        if mono_evidence > identity_evidence:
            selected_arm = "mono"
            primary = mono_evidence
            prediction = mono_result["prediction"]
        else:
            selected_arm = "plaintext"
            primary = identity_evidence
            prediction = list(row["extracted"])
        evidence = primary + INVARIANT_WEIGHT * row["features"]["invariant_score"]
        refined.append(
            row
            | {
                "mono": mono_result,
                "selected_arm": selected_arm,
                "prediction": prediction,
                "evidence": evidence,
            }
        )

    # Primary evidence selects the rule.  If evidence is numerically tied, the
    # smaller key-description penalty wins, preserving the frozen MDL role.
    best = max(
        refined,
        key=lambda row: (
            row["evidence"],
            -row["mono"]["mono_penalty_per_char"],
            -row["index"],
        ),
    )

    if refine_prediction and best["selected_arm"] == "mono":
        trigram, unigram = identity_model
        array = np.asarray(best["extracted"], dtype=np.int32)
        final_key, final_score = mono.anneal_mono(
            array,
            best["mono"]["key"],
            trigram,
            unigram,
            base.FINAL_MONO_ITERATIONS,
            base.FINAL_MONO_RESTARTS,
            int(
                core.stable_seed(
                    "v060-g2-final-amended-mono", seed, best["index"]
                )
                & 0x7FFFFFFFFFFFFFFF
            ),
        )
        best["prediction"] = final_key[array].astype(np.int32).tolist()
        best["final_mono_average"] = float(final_score) / len(best["extracted"])

    candidate = best["candidate"]
    return {
        "candidate_index": best["index"],
        "carrier": candidate.carrier,
        "parameters": candidate.parameter_dict(),
        "selected_arm": best["selected_arm"],
        "evidence": float(best["evidence"]),
        "prediction": best["prediction"],
        "features": best["features"],
        "mono_diagnostics": {
            "mono_average": best["mono"]["mono_average"],
            "mono_penalty_per_char": best["mono"]["mono_penalty_per_char"],
            "mono_evidence": best["mono"]["mono_evidence"],
            "mdl_adjusted_evidence": best["mono"]["mdl_adjusted_evidence"],
        },
        "shortlist_size": len(shortlist),
        "elapsed_seconds": time.perf_counter() - started,
    }


def configure_base() -> None:
    base.SHORTLIST_IDENTITY = SHORTLIST_IDENTITY
    base.SHORTLIST_INVARIANT = SHORTLIST_INVARIANT
    base.SCREEN_MONO_ITERATIONS = SCREEN_MONO_ITERATIONS
    base.SCREEN_MONO_RESTARTS = SCREEN_MONO_RESTARTS
    base.solve_cover = amended_solve_cover


def encrypted_execution_smoke() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator", choices=g1.COVER_GENERATORS, default="markov2")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--execution-smoke", action="store_true")
    args = parser.parse_args()

    root = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v060-family-g2-final-smoke",
    )
    language = languages["en"]
    identity_model = mono.build_language_model(language)
    reference = base.build_reference_stats(language, identity_model)
    inventory = base.candidate_inventory()
    manifest = base.inventory_manifest(inventory)
    if manifest["total"] != 2935:
        raise RuntimeError("frozen candidate inventory changed")

    chunks = core.source_chunks(language, "dev", base.PAYLOAD_LENGTH)
    selected: tuple[str, int, list[int]] | None = None
    for carrier in g1.CARRIER_CLASSES:
        for replicate in range(4):
            encrypted = (
                g1.parameter_index(args.generator, replicate)
                + g1.CARRIER_CLASSES.index(carrier)
            ) % 2 == 1
            if encrypted:
                chunk_index = (
                    g1.COVER_GENERATORS.index(args.generator) * 16
                    + g1.CARRIER_CLASSES.index(carrier) * 4
                    + replicate
                )
                selected = (carrier, replicate, list(chunks[chunk_index]))
                break
        if selected is not None:
            break
    if selected is None:
        raise RuntimeError("no deterministic encrypted smoke cell found")

    carrier, replicate, plaintext = selected
    layout, truth = base.embed_payload_cover(
        language, args.generator, carrier, replicate, plaintext
    )
    if not truth["encrypted"]:
        raise RuntimeError("smoke payload is not encrypted")
    solved = amended_solve_cover(
        layout,
        inventory,
        language,
        identity_model,
        reference,
        truth["seed"],
        refine_prediction=True,
    )
    selected_parameters = base.normalise_parameters(
        solved["carrier"], solved["parameters"]
    )
    payload_row = truth | solved
    payload_row["carrier_correct"] = solved["carrier"] == truth["true_carrier"]
    payload_row["parameters_correct"] = (
        payload_row["carrier_correct"]
        and selected_parameters == truth["true_parameters"]
    )
    payload_row["recovery"] = mono.fast_accuracy(plaintext, solved["prediction"])
    payload_row["selected_status_correct"] = solved["selected_arm"] == "mono"
    payload_row.pop("payload", None)
    payload_row.pop("prediction", None)

    null_layout, null_metadata = base.make_null_cover(
        language, args.generator, carrier, replicate, 0
    )
    null_solved = amended_solve_cover(
        null_layout,
        inventory,
        language,
        identity_model,
        reference,
        null_metadata["seed"],
        refine_prediction=False,
    )
    null_solved.pop("prediction", None)
    null_row = null_metadata | null_solved

    result = {
        "config": {
            "execution_smoke": True,
            "encrypted_payload": True,
            "shortlist_identity": SHORTLIST_IDENTITY,
            "shortlist_invariant": SHORTLIST_INVARIANT,
            "screen_mono_iterations": SCREEN_MONO_ITERATIONS,
            "screen_mono_restarts": SCREEN_MONO_RESTARTS,
            "invariant_weight": INVARIANT_WEIGHT,
        },
        "inventory": {
            "total": manifest["total"],
            "counts": manifest["counts"],
            "sha256": manifest["sha256"],
        },
        "payload_row": payload_row,
        "null_row": null_row,
    }
    scientific = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["sha256"] = hashlib.sha256(scientific).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("V060_G2_FINAL_INVENTORY", json.dumps(result["inventory"], sort_keys=True), flush=True)
    print("V060_G2_FINAL_SMOKE_PAYLOAD", json.dumps(payload_row, sort_keys=True), flush=True)
    print("V060_G2_FINAL_SMOKE_NULL", json.dumps(null_row, sort_keys=True), flush=True)
    print("V060_G2_FINAL_SMOKE_SHA256", result["sha256"], flush=True)


def main() -> None:
    configure_base()
    if "--execution-smoke" in sys.argv:
        encrypted_execution_smoke()
    else:
        base.main()


if __name__ == "__main__":
    main()
