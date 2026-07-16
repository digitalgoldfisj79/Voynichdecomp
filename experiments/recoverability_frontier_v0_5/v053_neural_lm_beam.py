#!/usr/bin/env python3
"""v0.5.3 train-only neural-LM beam search for fresh-key homophonic ciphers."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import recoverability_v050 as core
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono


class CharLSTM(nn.Module):
    def __init__(self, alphabet_size: int, embedding: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.bos_id = alphabet_size
        self.embedding = nn.Embedding(alphabet_size + 1, embedding)
        self.lstm = nn.LSTM(
            embedding,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden, alphabet_size)

    def forward(self, values: torch.Tensor, hidden=None):
        encoded = self.embedding(values)
        output, hidden = self.lstm(encoded, hidden)
        return self.output(output), hidden

    def step(self, values: torch.Tensor, hidden):
        logits, hidden = self.forward(values[:, None], hidden)
        return logits[:, 0], hidden


@dataclass
class BeamResult:
    mapping: list[int]
    score: float
    prediction: list[int]
    elapsed_seconds: float


def deterministic_segments(stream: list[int], length: int, count: int, seed: int) -> list[list[int]]:
    if len(stream) <= length:
        raise RuntimeError("training stream is too short")
    rng = random.Random(seed)
    starts = [rng.randrange(0, len(stream) - length) for _ in range(count)]
    return [stream[start : start + length] for start in starts]


def train_lm(
    model: CharLSTM,
    language: core.LanguageData,
    device: torch.device,
    steps: int,
    batch_size: int,
    sequence_length: int,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    stream = language.train_stream
    bos = model.bos_id
    history: list[dict[str, float]] = []
    rng = random.Random(seed)

    for step in range(1, steps + 1):
        starts = [rng.randrange(0, len(stream) - sequence_length) for _ in range(batch_size)]
        targets = torch.tensor(
            [stream[start : start + sequence_length] for start in starts],
            dtype=torch.long,
            device=device,
        )
        inputs = torch.empty_like(targets)
        inputs[:, 0] = bos
        inputs[:, 1:] = targets[:, :-1]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, model.alphabet_size), targets.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % max(1, steps // 20) == 0 or step == 1:
            item = {"step": float(step), "loss": float(loss.detach().cpu())}
            history.append(item)
            print("V053_BEAM_LM_TRAIN", json.dumps(item, sort_keys=True), flush=True)
    return history


def first_positions(cipher: list[int], symbol_count: int) -> list[int]:
    positions = [-1] * symbol_count
    for index, symbol in enumerate(cipher):
        if positions[symbol] < 0:
            positions[symbol] = index
    if any(value < 0 for value in positions):
        raise RuntimeError("canonical cipher contains missing symbol IDs")
    return positions


def expand_beam(
    model: CharLSTM,
    cipher: torch.Tensor,
    symbol_index: int,
    segment_start: int,
    segment_end: int,
    scores: torch.Tensor,
    mappings: torch.Tensor,
    remaining: torch.Tensor,
    last_tokens: torch.Tensor,
    hidden: tuple[torch.Tensor, torch.Tensor] | None,
    beam_width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    parent_indices: list[int] = []
    labels: list[int] = []
    remaining_cpu = remaining.detach().cpu().numpy()
    for parent, counts in enumerate(remaining_cpu):
        for label in np.flatnonzero(counts > 0):
            parent_indices.append(parent)
            labels.append(int(label))
    if not parent_indices:
        raise RuntimeError("beam has no legal label assignment")

    parents = torch.tensor(parent_indices, dtype=torch.long, device=device)
    candidate_labels = torch.tensor(labels, dtype=torch.long, device=device)
    candidate_scores = scores.index_select(0, parents).clone()
    candidate_mappings = mappings.index_select(0, parents).clone()
    candidate_remaining = remaining.index_select(0, parents).clone()
    candidate_last = last_tokens.index_select(0, parents).clone()
    candidate_mappings[:, symbol_index] = candidate_labels
    candidate_remaining[
        torch.arange(candidate_labels.shape[0], device=device), candidate_labels
    ] -= 1

    if hidden is None:
        layers = model.lstm.num_layers
        hidden_size = model.lstm.hidden_size
        candidate_hidden = (
            torch.zeros(layers, len(parent_indices), hidden_size, device=device),
            torch.zeros(layers, len(parent_indices), hidden_size, device=device),
        )
    else:
        candidate_hidden = (
            hidden[0].index_select(1, parents).contiguous(),
            hidden[1].index_select(1, parents).contiguous(),
        )

    cipher_segment = cipher[segment_start:segment_end]
    if cipher_segment.numel() > 0:
        target_segment = candidate_mappings.index_select(1, cipher_segment)
        for position in range(target_segment.shape[1]):
            logits, candidate_hidden = model.step(candidate_last, candidate_hidden)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            target = target_segment[:, position]
            candidate_scores += log_probs.gather(1, target[:, None])[:, 0]
            candidate_last = target

    keep = min(beam_width, candidate_scores.shape[0])
    best_scores, best_indices = torch.topk(candidate_scores, k=keep, largest=True, sorted=True)
    best_mappings = candidate_mappings.index_select(0, best_indices)
    best_remaining = candidate_remaining.index_select(0, best_indices)
    best_last = candidate_last.index_select(0, best_indices)
    best_hidden = (
        candidate_hidden[0].index_select(1, best_indices).contiguous(),
        candidate_hidden[1].index_select(1, best_indices).contiguous(),
    )
    return best_scores, best_mappings, best_remaining, best_last, best_hidden


@torch.inference_mode()
def decipher_trial(
    model: CharLSTM,
    trial: dict[str, Any],
    beam_width: int,
    device: torch.device,
) -> BeamResult:
    started = time.perf_counter()
    model.eval()
    cipher_values = list(map(int, trial["cipher"]))
    symbol_count = len(trial["inferred_labels"])
    positions = first_positions(cipher_values, symbol_count)
    inventory = np.bincount(
        np.asarray(trial["inferred_labels"], dtype=np.int32),
        minlength=model.alphabet_size,
    ).astype(np.int16)
    cipher = torch.tensor(cipher_values, dtype=torch.long, device=device)
    mappings = torch.full((1, symbol_count), -1, dtype=torch.long, device=device)
    remaining = torch.tensor(inventory[None, :], dtype=torch.int16, device=device)
    scores = torch.zeros(1, dtype=torch.float32, device=device)
    last_tokens = torch.full((1,), model.bos_id, dtype=torch.long, device=device)
    hidden = None
    segment_start = 0

    for symbol_index in range(symbol_count):
        segment_end = positions[symbol_index + 1] if symbol_index + 1 < symbol_count else len(cipher_values)
        scores, mappings, remaining, last_tokens, hidden = expand_beam(
            model,
            cipher,
            symbol_index,
            segment_start,
            segment_end,
            scores,
            mappings,
            remaining,
            last_tokens,
            hidden,
            beam_width,
            device,
        )
        segment_start = segment_end

    best_mapping = mappings[0]
    prediction = best_mapping.index_select(0, cipher).detach().cpu().tolist()
    return BeamResult(
        mapping=best_mapping.detach().cpu().tolist(),
        score=float(scores[0].detach().cpu()),
        prediction=prediction,
        elapsed_seconds=time.perf_counter() - started,
    )


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accuracies = [float(row["accuracy"]) for row in rows]
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "at_least_70_rate": statistics.fmean(value >= 0.70 for value in accuracies),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in accuracies),
        "at_least_95_rate": statistics.fmean(value >= 0.95 for value in accuracies),
        "mean_seconds": statistics.fmean(float(row["elapsed_seconds"]) for row in rows),
    }


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--length", type=int, default=384)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embedding", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beam-widths", default="128,512,2048")
    parser.add_argument("--seed", type=int, default=53031)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        torch.set_num_threads(max(1, min(32, torch.get_num_threads())))
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    model = CharLSTM(
        len(language.alphabet),
        args.embedding,
        args.hidden,
        args.layers,
        args.dropout,
    ).to(device)
    training_history = train_lm(
        model,
        language,
        device,
        args.train_steps,
        args.batch_size,
        args.train_length,
        args.learning_rate,
        args.seed,
    )
    trials = [
        homophonic.make_trial(language, args.split, args.length, args.offset + replicate)
        for replicate in range(args.replicates)
    ]
    widths = [int(value) for value in args.beam_widths.split(",") if value]
    candidates: list[dict[str, Any]] = []
    for width in widths:
        rows: list[dict[str, Any]] = []
        for completed, trial in enumerate(trials, start=1):
            result = decipher_trial(model, trial, width, device)
            accuracy = mono.fast_accuracy(trial["plain"], result.prediction)
            row = {
                "replicate": int(trial["replicate"]),
                "beam_width": width,
                "accuracy": accuracy,
                "exact": result.prediction == trial["plain"],
                "score": result.score,
                "elapsed_seconds": result.elapsed_seconds,
                "inventory_overlap": float(trial["inventory_overlap"]),
            }
            rows.append(row)
            print(
                "V053_BEAM_TRIAL",
                json.dumps(row, sort_keys=True),
                flush=True,
            )
        item = {"beam_width": width, "summary": summary(rows), "rows": rows}
        candidates.append(item)
        print(
            "V053_BEAM_CANDIDATE",
            json.dumps({"beam_width": width, "summary": item["summary"]}, sort_keys=True),
            flush=True,
        )

    eligible = [
        item
        for item in candidates
        if item["summary"]["mean_accuracy"] >= 0.70
        and item["summary"]["median_accuracy"] >= 0.90
        and item["summary"]["at_least_70_rate"] >= 0.875
    ]
    selected = min(eligible, key=lambda item: item["beam_width"]) if eligible else None
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.3-neural-lm-beam",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates": args.replicates,
        "model": {
            "layers": args.layers,
            "embedding": args.embedding,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "train_length": args.train_length,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "training_history": training_history,
        "candidates": candidates,
        "selected_beam_width": None if selected is None else selected["beam_width"],
        "development_gate_pass": selected is not None,
        "device": str(device),
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V053_BEAM_SELECTED", payload["selected_beam_width"], flush=True)
    print("V053_BEAM_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
