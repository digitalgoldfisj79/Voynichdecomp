#!/usr/bin/env python3
"""Train-only masked-word Transformer for v0.5.4 nomenclator A1."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

import recoverability_v050 as core
import mono_solver_v051 as mono
import v054_nomenclator_stage_a as stage

PAD = 0
UNK = 1
MASK = 2
BOUNDARY = 3
FIRST_WORD = 4


class MaskedWordTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_length: int,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocabulary_size, d_model, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocabulary_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length = values.shape
        if length > self.max_length:
            raise RuntimeError(f"word sequence {length} exceeds max {self.max_length}")
        positions = torch.arange(length, device=values.device)
        encoded = self.token_embedding(values) + self.position_embedding(positions)[None, :, :]
        padding_mask = values.eq(PAD)
        encoded = self.encoder(encoded, src_key_padding_mask=padding_mask)
        return self.output(self.norm(encoded))


def build_vocabulary(
    language: core.LanguageData, vocabulary_words: int
) -> tuple[dict[tuple[int, ...], int], list[tuple[int, ...] | None], list[int]]:
    counts = collections.Counter(language.train_words)
    words = [word for word, _count in counts.most_common(vocabulary_words)]
    word_to_id = {word: index + FIRST_WORD for index, word in enumerate(words)}
    id_to_word: list[tuple[int, ...] | None] = [None] * (len(words) + FIRST_WORD)
    for word, value in word_to_id.items():
        id_to_word[value] = word
    stream: list[int] = []
    for text in language.texts["train"]:
        if stream:
            stream.append(BOUNDARY)
        for raw in text.split():
            word = stage.encode_word(language, raw)
            if word:
                stream.append(word_to_id.get(word, UNK))
    return word_to_id, id_to_word, stream


def random_mask_batch(
    stream: list[int],
    batch_size: int,
    sequence_length: int,
    mask_rate: float,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for _ in range(batch_size):
        start = rng.randrange(0, len(stream) - sequence_length)
        target = list(stream[start : start + sequence_length])
        source = list(target)
        masked = 0
        for index, value in enumerate(source):
            if value not in (PAD, BOUNDARY) and rng.random() < mask_rate:
                source[index] = MASK
                masked += 1
        if masked == 0:
            eligible = [i for i, value in enumerate(source) if value not in (PAD, BOUNDARY)]
            if eligible:
                source[rng.choice(eligible)] = MASK
        training_target = [
            value if source[index] == MASK else -100
            for index, value in enumerate(target)
        ]
        inputs.append(source)
        targets.append(training_target)
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train_model(
    model: MaskedWordTransformer,
    stream: list[int],
    device: torch.device,
    steps: int,
    batch_size: int,
    sequence_length: int,
    mask_rate: float,
    learning_rate: float,
    seed: int,
) -> list[dict[str, float]]:
    model.train()
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history: list[dict[str, float]] = []
    for step_index in range(1, steps + 1):
        source, target = random_mask_batch(
            stream, batch_size, sequence_length, mask_rate, rng
        )
        source = source.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = model(source)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=-100
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step_index % max(1, steps // 20) == 0 or step_index == 1:
            item = {"step": float(step_index), "loss": float(loss.detach().cpu())}
            history.append(item)
            print("V054_MASK_TRAIN", json.dumps(item, sort_keys=True), flush=True)
    return history


def units_to_model_input(
    units: list[tuple[str, Any]], word_to_id: dict[tuple[int, ...], int]
) -> tuple[list[int], dict[int, list[int]]]:
    values: list[int] = []
    code_positions: dict[int, list[int]] = collections.defaultdict(list)
    for kind, value in units:
        if kind == "word":
            values.append(word_to_id.get(value, UNK))
        else:
            code_positions[int(value)].append(len(values))
            values.append(MASK)
    return values, dict(code_positions)


def infer_assignment(
    model: MaskedWordTransformer,
    values: list[int],
    code_positions: dict[int, list[int]],
    candidate_words: tuple[tuple[int, ...], ...],
    word_to_id: dict[tuple[int, ...], int],
    device: torch.device,
    iterations: int,
) -> dict[int, tuple[int, ...]]:
    code_symbols = sorted(code_positions)
    if not code_symbols:
        return {}
    candidate_ids = np.asarray(
        [word_to_id.get(word, UNK) for word in candidate_words], dtype=np.int64
    )
    current_values = list(values)
    assignment: dict[int, tuple[int, ...]] = {}
    model.eval()
    for _ in range(iterations):
        masked_values = list(current_values)
        for positions in code_positions.values():
            for position in positions:
                masked_values[position] = MASK
        tensor = torch.tensor([masked_values], dtype=torch.long, device=device)
        with torch.inference_mode():
            logits = model(tensor)[0]
            log_probs = F.log_softmax(logits.float(), dim=-1).detach().cpu().numpy()
        scores = np.empty((len(code_symbols), len(candidate_words)), dtype=np.float64)
        for row, symbol in enumerate(code_symbols):
            positions = code_positions[symbol]
            scores[row] = log_probs[np.asarray(positions)[:, None], candidate_ids[None, :]].sum(axis=0)
        rows, columns = linear_sum_assignment(-scores)
        new_assignment = {
            code_symbols[int(row)]: candidate_words[int(column)]
            for row, column in zip(rows, columns)
        }
        if new_assignment == assignment:
            assignment = new_assignment
            break
        assignment = new_assignment
        current_values = list(values)
        for symbol, positions in code_positions.items():
            word_id = word_to_id.get(assignment[symbol], UNK)
            for position in positions:
                current_values[position] = word_id
    return assignment


def condition_rows(
    model: MaskedWordTransformer,
    language: core.LanguageData,
    word_to_id: dict[tuple[int, ...], int],
    device: torch.device,
    length: int,
    pool_size: int,
    codebook_size: int,
    replicates: int,
    iterations: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    word_model = stage.build_word_model(language, candidate_pool_size=pool_size)
    trials = [
        stage.make_trial(language, word_model, "dev", length, replicate, codebook_size)
        for replicate in range(replicates)
    ]
    rows: list[dict[str, Any]] = []
    for trial in trials:
        units = stage.parse_word_units(
            trial, trial.char_to_plain, language.char_to_id[" "]
        )
        values, positions = units_to_model_input(units, word_to_id)
        assignment = infer_assignment(
            model,
            values,
            positions,
            word_model.candidate_words,
            word_to_id,
            device,
            iterations,
        )
        expanded = stage.expand_surface(trial, trial.char_to_plain, assignment)
        mapping_accuracy, occurrence_accuracy = stage.code_metrics(trial, assignment)
        row = {
            "replicate": trial.replicate,
            "observed_code_symbols": len(trial.code_symbols),
            "observed_code_occurrences": sum(
                symbol in trial.code_to_word for symbol in trial.surface
            ),
            "code_mapping_accuracy": mapping_accuracy,
            "code_occurrence_accuracy": occurrence_accuracy,
            "expanded_accuracy": mono.fast_accuracy(trial.plain, expanded),
        }
        rows.append(row)
        print(
            "V054_MASK_TRIAL",
            json.dumps(
                {
                    "length": length,
                    "pool": pool_size,
                    "codebook": codebook_size,
                    **row,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        "trials": len(rows),
        "mean_code_mapping_accuracy": statistics.fmean(
            row["code_mapping_accuracy"] for row in rows
        ),
        "mean_code_occurrence_accuracy": statistics.fmean(
            row["code_occurrence_accuracy"] for row in rows
        ),
        "mean_expanded_accuracy": statistics.fmean(
            row["expanded_accuracy"] for row in rows
        ),
        "mean_observed_code_symbols": statistics.fmean(
            row["observed_code_symbols"] for row in rows
        ),
        "mean_observed_code_occurrences": statistics.fmean(
            row["observed_code_occurrences"] for row in rows
        ),
    }
    summary["gate_pass"] = (
        summary["mean_code_mapping_accuracy"] >= 0.80
        and summary["mean_code_occurrence_accuracy"] >= 0.80
        and summary["mean_expanded_accuracy"] >= 0.90
        and summary["mean_observed_code_symbols"] >= 8.0
    )
    return rows, summary


def canonical_sha(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iso", default="en")
    parser.add_argument("--vocabulary-words", type=int, default=6000)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=192)
    parser.add_argument("--mask-rate", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--refinement-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=54041)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    word_to_id, _id_to_word, train_stream = build_vocabulary(
        language, args.vocabulary_words
    )
    vocabulary_size = max(word_to_id.values(), default=FIRST_WORD - 1) + 1
    model = MaskedWordTransformer(
        vocabulary_size,
        args.max_length,
        args.d_model,
        args.heads,
        args.layers,
        args.feedforward,
        args.dropout,
    ).to(device)
    history = train_model(
        model,
        train_stream,
        device,
        args.train_steps,
        args.batch_size,
        args.sequence_length,
        args.mask_rate,
        args.learning_rate,
        args.seed,
    )

    conditions = [
        (384, 32, 16),
        (384, 64, 24),
        (384, 96, 24),
        (768, 32, 16),
        (768, 64, 24),
        (768, 96, 24),
        (1536, 32, 16),
        (1536, 64, 24),
        (1536, 96, 24),
    ]
    results: list[dict[str, Any]] = []
    for length, pool, codebook in conditions:
        rows, summary = condition_rows(
            model,
            language,
            word_to_id,
            device,
            length,
            pool,
            codebook,
            args.replicates,
            args.refinement_iterations,
        )
        item = {
            "length": length,
            "candidate_pool": pool,
            "codebook_size": codebook,
            "summary": summary,
            "rows": rows,
        }
        results.append(item)
        print(
            "V054_MASK_CONDITION",
            json.dumps({key: value for key, value in item.items() if key != "rows"}, sort_keys=True),
            flush=True,
        )
    passing = [item for item in results if item["summary"]["gate_pass"]]
    selected = (
        min(passing, key=lambda item: (item["length"], item["candidate_pool"]))
        if passing
        else None
    )
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.4-a1-masked-word-transformer",
        "iso": args.iso,
        "model": {
            "vocabulary_words": args.vocabulary_words,
            "max_length": args.max_length,
            "d_model": args.d_model,
            "heads": args.heads,
            "layers": args.layers,
            "feedforward": args.feedforward,
            "dropout": args.dropout,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "mask_rate": args.mask_rate,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "training_history": history,
        "conditions": results,
        "selected_condition": None
        if selected is None
        else {
            "length": selected["length"],
            "candidate_pool": selected["candidate_pool"],
            "codebook_size": selected["codebook_size"],
        },
        "development_gate_pass": selected is not None,
        "device": str(device),
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V054_MASK_SELECTED", json.dumps(payload["selected_condition"]), flush=True)
    print("V054_MASK_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
