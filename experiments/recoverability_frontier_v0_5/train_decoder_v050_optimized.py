#!/usr/bin/env python3
"""Optimized key-invariant launcher for v0.5.0 full decoder runs."""
from __future__ import annotations

import statistics
from typing import Any

import torch
from rapidfuzz.distance import Levenshtein

import recoverability_v050 as core
import train_decoder_v050 as base


def fast_edit_distance(a, b):
    return int(Levenshtein.distance(a, b))


core.edit_distance = fast_edit_distance


class RecurrenceDataset(base.SyntheticDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        tag_count = 2 if self.known_family else 1
        tags = row["source"][:tag_count]
        surface = row["source"][tag_count:]
        mapping = {}
        canonical = []
        for token in surface:
            raw = int(token) - base.SURFACE_OFFSET
            if raw not in mapping:
                mapping[raw] = len(mapping)
            canonical.append(base.SURFACE_OFFSET + mapping[raw])
        if len(mapping) >= base.MAX_SURFACE_SYMBOLS:
            raise RuntimeError("recurrence vocabulary overflow")
        row["source"] = [*tags, *canonical]
        row["distinct_surface_symbols"] = len(mapping)
        return row


@torch.no_grad()
def optimized_evaluate(model, loader, device, threshold, max_decode):
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        _, _, class_logits = model.encode(source)
        probabilities = torch.sigmoid(class_logits).cpu().tolist()
        predictions: dict[int, list[int]] = {}

        declared_positive_indices = [
            index
            for index, (probability, meta) in enumerate(zip(probabilities, batch["meta"]))
            if probability >= threshold and bool(meta["message"])
        ]
        by_length: dict[int, list[int]] = {}
        for index in declared_positive_indices:
            by_length.setdefault(int(batch["meta"][index]["length"]), []).append(index)
        for length, indices in by_length.items():
            subset = source[torch.tensor(indices, device=device)]
            _, generated = model.greedy(subset, min(max_decode, length + 16))
            for index, values in zip(indices, generated.cpu().tolist()):
                predictions[index] = base.strip_generated(values)

        for index, (probability, meta) in enumerate(zip(probabilities, batch["meta"])):
            declared = probability >= threshold
            positive = bool(meta["message"])
            accuracy = 0.0
            exact = False
            if positive and declared:
                truth = base.local_to_output(
                    loader.dataset.languages[meta["iso"]],
                    meta["plain_local"],
                    loader.dataset.vocab,
                )
                decoded = predictions.get(index, [])
                accuracy = core.character_accuracy(truth, decoded)
                exact = truth == decoded
            rows.append({
                "message": positive,
                "declared_message": declared,
                "probability": float(probability),
                "accuracy": float(accuracy),
                "exact": bool(exact),
                "iso": meta["iso"],
                "family": meta["family"],
                "noise": meta["noise"],
                "length": meta["length"],
                "control_family": meta["control_family"],
            })

    positives = [row for row in rows if row["message"]]
    controls = [row for row in rows if not row["message"]]
    detected = [row for row in positives if row["declared_message"]]
    by_family = {}
    for family in loader.dataset.families:
        subset = [row for row in positives if row["family"] == family]
        by_family[family] = {
            "trials": len(subset),
            "sensitivity": sum(row["declared_message"] for row in subset) / max(1, len(subset)),
            "mean_accuracy_all": statistics.fmean(row["accuracy"] for row in subset),
            "exact_rate_all": statistics.fmean(float(row["exact"]) for row in subset),
        }
    return {
        "threshold": threshold,
        "positive_trials": len(positives),
        "control_trials": len(controls),
        "sensitivity": len(detected) / max(1, len(positives)),
        "false_positive_rate": sum(row["declared_message"] for row in controls) / max(1, len(controls)),
        "mean_accuracy_all_positives": statistics.fmean(row["accuracy"] for row in positives),
        "mean_accuracy_detected": statistics.fmean(row["accuracy"] for row in detected) if detected else 0.0,
        "exact_rate_all_positives": statistics.fmean(float(row["exact"]) for row in positives),
        "by_family": by_family,
        "rows": rows,
    }


base.SyntheticDataset = RecurrenceDataset
base.evaluate = optimized_evaluate

if __name__ == "__main__":
    base.main()
