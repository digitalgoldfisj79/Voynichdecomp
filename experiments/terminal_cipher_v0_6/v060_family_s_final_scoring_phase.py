#!/usr/bin/env python3
"""Final abstention-aware scoring for the frozen v0.6 Family S3 development run."""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch

import v060_family_s_neural_final_evaluate as ev
import v060_family_s_stage_s1 as s1
import v060_family_s_stage_s2 as s2
import v060_family_s_stage_s3 as s3


def run(repo: Path) -> None:
    started = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("final scoring requires CUDA")
    signer_url = ev.DEFAULT_SIGNER_URL
    phase1_payload = ev.download_json(ev.PHASE1_OBJECT, signer_url)
    phase2_payload = ev.download_json(ev.PHASE2_OBJECT, signer_url)
    with tempfile.TemporaryDirectory(prefix="v060-s3-final-score-") as directory:
        models, manifests = ev.load_models(device, Path(directory), signer_url)
        language = ev.load_language(repo)
        inventory = s1.candidate_inventory(language)
        unit_model = s2.build_unit_model(language, inventory)
        calibration = phase1_payload["calibration"]
        phase1_by_replicate = {
            int(row["replicate"]): row for row in phase1_payload["rows"]
        }
        phase2_by_replicate = {
            int(row["replicate"]): row for row in phase2_payload["rows"]
        }
        rows = []
        for replicate in range(16):
            trial = s1.make_trial(language, "dev", ev.TARGET_LENGTH, replicate)
            p1 = phase1_by_replicate[replicate]
            p2 = phase2_by_replicate[replicate]
            direct = [int(value) for value in p1["direct_plaintext"]]
            lattice_available = bool(p2["lattice_available"])
            source, line_flags, source_padding = ev.trial_tensors(trial, device)
            memories, _boundary_probability = ev.encode_ensemble(
                models, source, line_flags, source_padding
            )
            if lattice_available:
                lattice = [int(value) for value in p2["lattice_plaintext"]]
                neural_scores = ev.teacher_scores(
                    models, memories, source_padding, [direct, lattice]
                )
                direct_features = ev.candidate_features(
                    direct, neural_scores[0], inventory, unit_model
                )
                lattice_features = ev.candidate_features(
                    lattice, neural_scores[1], inventory, unit_model
                )
                direct_logit = ev.calibrated_logit(direct_features, calibration)
                lattice_logit = ev.calibrated_logit(lattice_features, calibration)
                if lattice_logit > direct_logit:
                    selected_name = "lattice"
                    selected = lattice
                    selected_logit = lattice_logit
                else:
                    selected_name = "direct"
                    selected = direct
                    selected_logit = direct_logit
                lattice_length = len(lattice)
                lattice_accuracy = ev.plaintext_accuracy(trial.plain, lattice)
            else:
                neural_score = ev.teacher_scores(
                    models, memories, source_padding, [direct]
                )[0]
                direct_features = ev.candidate_features(
                    direct, neural_score, inventory, unit_model
                )
                direct_logit = ev.calibrated_logit(direct_features, calibration)
                lattice_logit = None
                selected_name = "direct"
                selected = direct
                selected_logit = direct_logit
                lattice_length = None
                lattice_accuracy = None
            top_lengths = [
                int(value) for value in p1["segmentations"][0]["lengths"]
            ]
            predicted_boundaries = ev.boundaries_from_lengths(top_lengths)
            boundary_f1 = s3.boundary_f1(trial.boundaries, predicted_boundaries)
            row = {
                "iso": "en",
                "split": "dev",
                "replicate": replicate,
                "cipher_length": len(trial.cipher),
                "true_units": len(trial.units),
                "top_boundary_units": len(top_lengths),
                "boundary_f1": boundary_f1,
                "lattice_available": lattice_available,
                "lattice_abstention_reason": p2.get("lattice_abstention_reason"),
                "selected_hypothesis": selected_name,
                "selected_calibrated_logit": selected_logit,
                "direct_calibrated_logit": direct_logit,
                "lattice_calibrated_logit": lattice_logit,
                "direct_length": len(direct),
                "lattice_length": lattice_length,
                "selected_length": len(selected),
                "direct_accuracy": ev.plaintext_accuracy(trial.plain, direct),
                "lattice_accuracy": lattice_accuracy,
                "plaintext_accuracy": ev.plaintext_accuracy(trial.plain, selected),
                "plaintext_exact": selected == trial.plain,
            }
            rows.append(row)
            print("V060_S3_FINAL_TRIAL", json.dumps(row, sort_keys=True), flush=True)
        summary = ev.summarize(rows)
        summary["lattice"] = {
            "available_count": sum(row["lattice_available"] for row in rows),
            "abstention_count": sum(not row["lattice_available"] for row in rows),
        }
        payload = {
            "config": {
                "split": "dev",
                "length": ev.TARGET_LENGTH,
                "replicates": 16,
                "beam_width": ev.BEAM_WIDTH,
                "lattice_paths": ev.LATTICE_PATHS,
                "screen_iterations": 700000,
                "screen_restarts": 50,
                "final_iterations": 700000,
                "final_restarts": 200,
                "checkpoint_sha256": {
                    str(seed): manifests[index]["original_sha256"]
                    for index, seed in enumerate((1731, 1732))
                },
                "phase1_sha256": phase1_payload["sha256"],
                "phase2_sha256": phase2_payload["sha256"],
            },
            "calibration": calibration,
            "rows": rows,
            "summary": summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        payload["sha256"] = hashlib.sha256(ev.json_bytes(payload)).hexdigest()
        ev.upload_bytes(ev.FINAL_OBJECT, ev.json_bytes(payload, pretty=True), signer_url)
        print("V060_S3_FINAL_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
        print("V060_S3_FINAL_SHA256", payload["sha256"], flush=True)
        print("V060_S3_FINAL_RESULT", json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1]))
