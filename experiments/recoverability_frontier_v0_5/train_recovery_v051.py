#!/usr/bin/env python3
"""v0.5.1 recovery-only recurrence Transformer.

This runner removes the non-identifiable messagehood classifier from the primary
objective.  It trains on natural-source ciphertext/plaintext pairs, decodes all
examples, calibrates free-decoding confidence on development data, and tests on
both natural held-out text and generated latent sequences.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader
from rapidfuzz.distance import Levenshtein

import recoverability_v050 as core
import train_decoder_v050 as base

PAD = base.PAD
BOS = base.BOS
EOS = base.EOS


def fast_edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    return int(Levenshtein.distance(a, b))


core.edit_distance = fast_edit_distance


class RecoveryDataset(base.SyntheticDataset):
    """Natural positives plus generated latent sequences with retained targets."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        natural = index < self.positives
        local_index = index if natural else index - self.positives
        iso, family, length, noise, replicate = self._cell(local_index)
        language = self.languages[iso]
        rng = random.Random(core.stable_seed("recovery-v051", self.seed, self.split, natural, index))

        if natural:
            pool = self.chunks[(iso, length)]
            plain = list(pool[replicate % len(pool)])
            source_type = "natural"
            control_family = None
        else:
            control_family = self.control_families[replicate % len(self.control_families)]
            plain = self._control_plain(language, iso, control_family, length, rng)
            source_type = "generated"

        packet = core.encrypt_sequence(
            plain,
            family,
            language,
            rng,
            parameter_mode="test" if self.split == "test" else self.split,
        )
        packet = core.apply_noise(packet, noise, rng)
        if packet.max_symbol >= base.MAX_SURFACE_SYMBOLS:
            raise RuntimeError(f"surface symbol overflow: {packet.max_symbol}")

        # First-occurrence recurrence canonicalisation removes arbitrary key IDs.
        mapping: dict[int, int] = {}
        canonical: list[int] = []
        for raw in packet.cipher:
            raw = int(raw)
            if raw not in mapping:
                mapping[raw] = len(mapping)
            canonical.append(base.SURFACE_OFFSET + mapping[raw])
        if len(mapping) >= base.MAX_SURFACE_SYMBOLS:
            raise RuntimeError("recurrence vocabulary overflow")

        source = [self.vocab.language_tags[iso]]
        if self.known_family:
            source.append(self.vocab.family_tags[family])
        source.extend(canonical)

        target = [BOS]
        target.extend(base.local_to_output(language, plain, self.vocab))
        target.append(EOS)

        return {
            "source": source,
            "target": target,
            "message": 1.0,
            "iso": iso,
            "family": family,
            "length": length,
            "noise": noise,
            "control_family": control_family,
            "source_type": source_type,
            "plain_local": plain,
            "alphabet": language.alphabet,
            "distinct_surface_symbols": len(mapping),
        }


def collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out = base.collate(rows)
    out["positive_mask"] = torch.ones(len(rows), dtype=torch.bool)
    return out


def train_epoch(
    model: base.RecoverabilityTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    use_amp = device.type == "cuda"
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        positive = batch["positive_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, seq_logits = model(source, target[:, :-1])
            loss = base.sequence_loss(seq_logits, target, positive)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
    return {"sequence_loss": statistics.fmean(losses)}


@torch.no_grad()
def greedy_with_confidence(
    model: base.RecoverabilityTransformer,
    source: torch.Tensor,
    max_length: int,
) -> tuple[list[list[int]], list[float]]:
    memory, pad, _ = model.encode(source)
    generated = torch.full((source.shape[0], 1), BOS, dtype=torch.long, device=source.device)
    finished = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
    log_sums = torch.zeros(source.shape[0], dtype=torch.float32, device=source.device)
    counts = torch.zeros(source.shape[0], dtype=torch.float32, device=source.device)

    for _ in range(max_length):
        logits = model.decode_teacher(memory, pad, generated)[:, -1]
        log_probs = logits.log_softmax(dim=-1)
        next_token = log_probs.argmax(dim=-1)
        active = ~finished
        chosen = log_probs.gather(1, next_token[:, None]).squeeze(1)
        log_sums += chosen.float() * active.float()
        counts += active.float()
        generated = torch.cat((generated, next_token[:, None]), dim=1)
        finished |= next_token.eq(EOS)
        if bool(finished.all()):
            break

    confidences = torch.exp(log_sums / counts.clamp_min(1.0)).cpu().tolist()
    predictions = [base.strip_generated(values) for values in generated.cpu().tolist()]
    return predictions, [float(value) for value in confidences]


@torch.no_grad()
def evaluate(
    model: base.RecoverabilityTransformer,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        by_length: dict[int, list[int]] = defaultdict(list)
        for index, meta in enumerate(batch["meta"]):
            by_length[int(meta["length"])].append(index)

        decoded: dict[int, list[int]] = {}
        confidence: dict[int, float] = {}
        for length, indices in by_length.items():
            select = torch.tensor(indices, device=device)
            predictions, confidences = greedy_with_confidence(model, source[select], length + 16)
            for index, prediction, score in zip(indices, predictions, confidences):
                decoded[index] = prediction
                confidence[index] = score

        for index, meta in enumerate(batch["meta"]):
            truth = base.local_to_output(
                loader.dataset.languages[meta["iso"]],
                meta["plain_local"],
                loader.dataset.vocab,
            )
            prediction = decoded[index]
            accuracy = core.character_accuracy(truth, prediction)
            rows.append({
                "iso": meta["iso"],
                "family": meta["family"],
                "length": int(meta["length"]),
                "noise": float(meta["noise"]),
                "source_type": meta["source_type"],
                "control_family": meta["control_family"],
                "accuracy": float(accuracy),
                "exact": truth == prediction,
                "confidence": confidence[index],
                "truth_length": len(truth),
                "prediction_length": len(prediction),
            })
    return rows


def select_confidence_thresholds(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    candidates = sorted({float(row["confidence"]) for row in rows}, reverse=True)

    def best_for(target: float) -> dict[str, float]:
        best = {"threshold": 1.0, "coverage": 0.0, "mean_accuracy": 0.0, "target": target}
        for threshold in candidates:
            selected = [row for row in rows if row["confidence"] >= threshold]
            if not selected:
                continue
            mean_accuracy = statistics.fmean(row["accuracy"] for row in selected)
            coverage = len(selected) / len(rows)
            if mean_accuracy >= target and coverage > best["coverage"]:
                best = {
                    "threshold": threshold,
                    "coverage": coverage,
                    "mean_accuracy": mean_accuracy,
                    "target": target,
                }
        return best

    return {"recovered": best_for(0.70), "partial": best_for(0.30)}


def summarize(rows: Sequence[dict[str, Any]], thresholds: dict[str, Any]) -> dict[str, Any]:
    recovered_threshold = float(thresholds["recovered"]["threshold"])
    partial_threshold = float(thresholds["partial"]["threshold"])

    enriched = []
    for row in rows:
        item = dict(row)
        if item["confidence"] >= recovered_threshold:
            item["status"] = "RECOVERED"
        elif item["confidence"] >= partial_threshold:
            item["status"] = "PARTIAL"
        else:
            item["status"] = "LOW_RECOVERY_CONFIDENCE"
        enriched.append(item)

    def group(field: str) -> dict[str, Any]:
        values: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in enriched:
            values[str(row[field])].append(row)
        return {
            key: {
                "trials": len(subset),
                "mean_accuracy": statistics.fmean(row["accuracy"] for row in subset),
                "exact_rate": statistics.fmean(float(row["exact"]) for row in subset),
                "mean_confidence": statistics.fmean(row["confidence"] for row in subset),
                "recovered_rate": statistics.fmean(float(row["status"] == "RECOVERED") for row in subset),
                "partial_or_better_rate": statistics.fmean(float(row["status"] != "LOW_RECOVERY_CONFIDENCE") for row in subset),
            }
            for key, subset in sorted(values.items())
        }

    recovered = [row for row in enriched if row["status"] == "RECOVERED"]
    partial = [row for row in enriched if row["status"] == "PARTIAL"]
    summary = {
        "trials": len(enriched),
        "mean_accuracy": statistics.fmean(row["accuracy"] for row in enriched),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in enriched),
        "mean_confidence": statistics.fmean(row["confidence"] for row in enriched),
        "recovered_rate": len(recovered) / len(enriched),
        "recovered_empirical_accuracy": statistics.fmean(row["accuracy"] for row in recovered) if recovered else 0.0,
        "partial_rate": len(partial) / len(enriched),
        "partial_empirical_accuracy": statistics.fmean(row["accuracy"] for row in partial) if partial else 0.0,
        "by_family": group("family"),
        "by_language": group("iso"),
        "by_length": group("length"),
        "by_noise": group("noise"),
        "by_source_type": group("source_type"),
        "rows": enriched,
    }
    return summary


def emit_artifact(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    encoded = base64.b64encode(compressed).decode("ascii")
    part_chars = 60000
    parts = [encoded[index:index + part_chars] for index in range(0, len(encoded), part_chars)]
    metadata = {
        "format": "gzip+base64",
        "raw_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "compressed_bytes": len(compressed),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "encoded_chars": len(encoded),
        "parts": len(parts),
    }
    print("V051_ARTIFACT_META " + json.dumps(metadata, sort_keys=True), flush=True)
    for index, part in enumerate(parts):
        print(f"V051_ARTIFACT_PART {index:04d}/{len(parts):04d} {part}", flush=True)
    print("V051_ARTIFACT_END " + metadata["raw_sha256"], flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("known", "blind"), required=True)
    parser.add_argument("--seed", type=int, default=505101)
    parser.add_argument("--train", type=int, default=120000)
    parser.add_argument("--dev-natural", type=int, default=4320)
    parser.add_argument("--dev-generated", type=int, default=4320)
    parser.add_argument("--test-natural", type=int, default=8640)
    parser.add_argument("--test-generated", type=int, default=8640)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    root = args.repo / "experiments/recoverability_frontier_v0_5"
    languages = core.load_languages(root / "corpus_manifest_v050.json", args.repo / ".cache/ud-v050")
    if args.smoke:
        languages = {iso: languages[iso] for iso in ("en", "tr")}
        args.train = min(args.train, 2048)
        args.dev_natural = min(args.dev_natural, 128)
        args.dev_generated = min(args.dev_generated, 128)
        args.test_natural = min(args.test_natural, 128)
        args.test_generated = min(args.test_generated, 128)
        args.epochs = min(args.epochs, 1)
        args.d_model = min(args.d_model, 128)
        args.heads = min(args.heads, 4)
        args.encoder_layers = min(args.encoder_layers, 2)
        args.decoder_layers = min(args.decoder_layers, 2)
        args.ff = min(args.ff, 384)
        args.batch_size = min(args.batch_size, 16)

    known = args.mode == "known"
    vocab = base.build_vocab(languages)
    datasets = {
        "train": RecoveryDataset(languages, vocab, "train", args.train, 0, known, args.seed, args.smoke),
        "dev": RecoveryDataset(languages, vocab, "dev", args.dev_natural, args.dev_generated, known, args.seed, args.smoke),
        "test": RecoveryDataset(languages, vocab, "test", args.test_natural, args.test_generated, known, args.seed, args.smoke),
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
            persistent_workers=args.workers > 0,
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = base.RecoverabilityTransformer(
        input_vocab=vocab.input_vocab_size,
        output_vocab=vocab.output_vocab_size,
        d_model=args.d_model,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        ff=args.ff,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        metrics = train_epoch(model, loaders["train"], optimizer, scaler, device)
        metrics["epoch"] = epoch
        metrics["elapsed_seconds"] = time.time() - started
        history.append(metrics)
        print("V051_TRAIN", args.mode, json.dumps(metrics, sort_keys=True), flush=True)

    dev_rows = evaluate(model, loaders["dev"], device)
    thresholds = select_confidence_thresholds(dev_rows)
    print("V051_CONFIDENCE", args.mode, json.dumps(thresholds, sort_keys=True), flush=True)

    test_rows = evaluate(model, loaders["test"], device)
    test = summarize(test_rows, thresholds)
    families_over_50 = sum(
        row["mean_accuracy"] >= 0.50
        for row in test["by_family"].values()
    )
    gate = {
        "three_families_over_50": families_over_50 >= 3,
        "recovered_calibrated": (
            test["recovered_rate"] == 0.0
            or test["recovered_empirical_accuracy"] >= 0.70
        ),
    }
    gate["pass"] = all(gate.values())

    payload = {
        "programme": "recoverability-frontier-v0.5.1-recovery-only",
        "mode": args.mode,
        "device": str(device),
        "config": vars(args),
        "history": history,
        "development_confidence": thresholds,
        "test": test,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)

    public_test = {key: value for key, value in test.items() if key != "rows"}
    print("V051_TEST", args.mode, json.dumps(public_test, sort_keys=True), flush=True)
    print("V051_GATE", args.mode, json.dumps(gate, sort_keys=True), flush=True)
    emit_artifact(args.output)


if __name__ == "__main__":
    main()
