#!/usr/bin/env python3
"""Torch-free CPU phase for the frozen final v0.6 Family S3 evaluation."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import requests

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import v060_family_s_stage_s1 as s1
import v060_family_s_stage_s2 as s2
import v060_family_s_stage_s3 as s3
from v060_supabase_checkpoint_transport import DEFAULT_SIGNER_URL, signed_url

PHASE1_OBJECT = "v060/s3/evaluation/dev/phase1.json"
PHASE2_OBJECT = "v060/s3/evaluation/dev/phase2.json"
TARGET_LENGTH = 384


def json_bytes(payload: dict[str, Any], pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def download_json(object_path: str, signer_url: str) -> dict[str, Any]:
    response = requests.get(
        signed_url(signer_url, "download", object_path), timeout=600
    )
    response.raise_for_status()
    return response.json()


def upload_json(object_path: str, payload: dict[str, Any], signer_url: str) -> None:
    response = requests.put(
        signed_url(signer_url, "upload", object_path),
        data=json_bytes(payload, pretty=True),
        headers={"Content-Type": "application/json"},
        timeout=600,
    )
    response.raise_for_status()


def load_language(repo: Path) -> core.LanguageData:
    root = repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        root / "corpus_manifest_v050.json",
        repo / ".cache" / "v060-family-s3-neural-final",
    )
    return languages["en"]


def pieces_from_lengths(cipher: list[int], lengths: list[int]) -> list[str]:
    text = s3.visible_text(cipher)
    pieces: list[str] = []
    cursor = 0
    for width in lengths:
        right = cursor + int(width)
        pieces.append(text[cursor:right])
        cursor = right
    if cursor != len(text):
        raise RuntimeError("segmentation lengths do not reconstruct ciphertext")
    return pieces


def solve_one(
    trial: s1.SegmentationTrial,
    phase1_row: dict[str, Any],
    inventory: list[tuple[int, ...]],
    unit_model: tuple[Any, Any, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    invalid_candidates = 0
    for rank, segmentation in enumerate(phase1_row["segmentations"]):
        lengths = [int(value) for value in segmentation["lengths"]]
        decoded = s3.decode_candidate(
            pieces_from_lengths(trial.cipher, lengths),
            inventory,
            unit_model,
            trial.seed,
            f"neural-screen-rank-{rank}",
            700000,
            50,
        )
        if decoded is None:
            invalid_candidates += 1
            continue
        candidates.append(
            {
                "rank": rank,
                "lengths": lengths,
                "boundary_posterior_score": float(segmentation["score"]),
                "combined_score": float(decoded["combined_score"]),
            }
        )
    if not candidates:
        direct = [int(value) for value in phase1_row["direct_plaintext"]]
        return {
            "replicate": trial.replicate,
            "lattice_available": False,
            "lattice_abstention_reason": "all_eight_paths_exceed_frozen_inventory",
            "selected_rank": None,
            "selected_lengths": None,
            "selected_screen_combined_score": None,
            "lattice_plaintext": direct,
            "lattice_final_combined_score": None,
            "valid_screen_candidates": 0,
            "invalid_screen_candidates": invalid_candidates,
            "elapsed_seconds": time.perf_counter() - started,
        }
    candidates.sort(key=lambda row: (-row["combined_score"], row["rank"]))
    selected = candidates[0]
    final = s3.decode_candidate(
        pieces_from_lengths(trial.cipher, selected["lengths"]),
        inventory,
        unit_model,
        trial.seed,
        f"neural-final-rank-{selected['rank']}",
        700000,
        200,
    )
    if final is None:
        raise RuntimeError(f"final lattice candidate invalid for replicate {trial.replicate}")
    return {
        "replicate": trial.replicate,
        "lattice_available": True,
        "lattice_abstention_reason": None,
        "selected_rank": selected["rank"],
        "selected_lengths": selected["lengths"],
        "selected_screen_combined_score": selected["combined_score"],
        "lattice_plaintext": final["plaintext"],
        "lattice_final_combined_score": float(final["combined_score"]),
        "valid_screen_candidates": len(candidates),
        "invalid_screen_candidates": invalid_candidates,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    repo = Path(sys.argv[1])
    signer_url = DEFAULT_SIGNER_URL
    started = time.perf_counter()
    phase1_payload = download_json(PHASE1_OBJECT, signer_url)
    language = load_language(repo)
    inventory = s1.candidate_inventory(language)
    unit_model = s2.build_unit_model(language, inventory)
    trials = [
        s1.make_trial(language, "dev", TARGET_LENGTH, replicate)
        for replicate in range(16)
    ]
    phase1_by_replicate = {
        int(row["replicate"]): row for row in phase1_payload["rows"]
    }

    def run_one(trial: s1.SegmentationTrial) -> dict[str, Any]:
        row = solve_one(
            trial,
            phase1_by_replicate[trial.replicate],
            inventory,
            unit_model,
        )
        print(
            "V060_S3_PHASE2_TRIAL",
            json.dumps(
                {
                    "replicate": row["replicate"],
                    "lattice_available": row["lattice_available"],
                    "selected_rank": row["selected_rank"],
                    "valid_screen_candidates": row["valid_screen_candidates"],
                    "invalid_screen_candidates": row["invalid_screen_candidates"],
                    "elapsed_seconds": row["elapsed_seconds"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        rows = list(executor.map(run_one, trials))
    rows.sort(key=lambda row: int(row["replicate"]))
    payload: dict[str, Any] = {
        "phase": 2,
        "split": "dev",
        "phase1_sha256": phase1_payload["sha256"],
        "rows": rows,
        "lattice_available_count": sum(bool(row["lattice_available"]) for row in rows),
        "lattice_abstention_count": sum(not bool(row["lattice_available"]) for row in rows),
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload["sha256"] = hashlib.sha256(json_bytes(payload)).hexdigest()
    upload_json(PHASE2_OBJECT, payload, signer_url)
    print(
        "V060_S3_PHASE2_COMPLETE",
        json.dumps(
            {
                "object_path": PHASE2_OBJECT,
                "sha256": payload["sha256"],
                "trials": len(rows),
                "lattice_available_count": payload["lattice_available_count"],
                "lattice_abstention_count": payload["lattice_abstention_count"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
