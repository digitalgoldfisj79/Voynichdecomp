#!/usr/bin/env python3
"""v0.5.3 shared-code-pool attention-LSTM positive control.

This arm intentionally exposes stable code-pool identities across examples and
is not eligible as fresh-key decipherment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import recoverability_v050 as core
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono


class SharedPoolLSTM(nn.Module):
    def __init__(
        self,
        symbol_vocab: int,
        alphabet_size: int,
        embedding: int,
        hidden: int,
        layers: int,
        attention_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.embedding = nn.Embedding(symbol_vocab, embedding)
        self.lstm = nn.LSTM(
            embedding,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            hidden * 2,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.output = nn.Linear(hidden * 2, alphabet_size)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.embedding(values)
        recurrent, _ = self.lstm(encoded)
        attended, _ = self.attention(recurrent, recurrent, recurrent, need_weights=False)
        return self.output(self.norm(recurrent + attended))


def build_pool(language: core.LanguageData) -> tuple[list[list[int]], int]:
    pools: list[list[int]] = []
    cursor = 0
    for probability in language.probabilities:
        count = homophonic.multiplicity(float(probability))
        pools.append(list(range(cursor, cursor + count)))
        cursor += count
    return pools, cursor


def encrypt_shared_pool(
    plain: list[int],
    pools: list[list[int]],
    rng: random.Random,
) -> list[int]:
    active: list[list[int]] = []
    for pool in pools:
        subset_size = rng.randint(1, len(pool))
        active.append(rng.sample(pool, subset_size))
    return [rng.choice(active[int(value)]) for value in plain]


def random_batch(
    language: core.LanguageData,
    pools: list[list[int]],
    batch_size: int,
    length: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    stream = language.train_stream
    cipher_rows: list[list[int]] = []
    plain_rows: list[list[int]] = []
    for _ in range(batch_size):
        start = rng.randrange(0, len(stream) - length)
        plain = list(stream[start : start + length])
        cipher_rows.append(encrypt_shared_pool(plain, pools, rng))
        plain_rows.append(plain)
    return (
        torch.tensor(cipher_rows, dtype=torch.long),
        torch.tensor(plain_rows, dtype=torch.long),
    )


def train_model(
    model: SharedPoolLSTM,
    language: core.LanguageData,
    pools: list[list[int]],
    device: torch.device,
    steps: int,
    batch_size: int,
    length: int,
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
    for step in range(1, steps + 1):
        cipher, target = random_batch(language, pools, batch_size, length, rng)
        cipher = cipher.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(cipher)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % max(1, steps // 20) == 0 or step == 1:
            item = {"step": float(step), "loss": float(loss.detach().cpu())}
            history.append(item)
            print("V053_SHARED_TRAIN", json.dumps(item, sort_keys=True), flush=True)
    return history


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
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--embedding", type=int, default=192)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=53033)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    pools, symbol_vocab = build_pool(language)
    model = SharedPoolLSTM(
        symbol_vocab,
        len(language.alphabet),
        args.embedding,
        args.hidden,
        args.layers,
        args.attention_heads,
        args.dropout,
    ).to(device)
    training_history = train_model(
        model,
        language,
        pools,
        device,
        args.train_steps,
        args.batch_size,
        args.length,
        args.learning_rate,
        args.seed,
    )

    chunks = core.source_chunks(language, args.split, args.length)
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for local_replicate in range(args.replicates):
            replicate = args.offset + local_replicate
            plain = list(chunks[replicate % len(chunks)])
            seed = core.stable_seed(
                "v053-shared-pool", args.split, args.iso, args.length, replicate
            )
            cipher = encrypt_shared_pool(plain, pools, random.Random(seed))
            values = torch.tensor([cipher], dtype=torch.long, device=device)
            prediction = model(values)[0].argmax(dim=-1).detach().cpu().tolist()
            accuracy = mono.fast_accuracy(plain, prediction)
            row = {
                "replicate": replicate,
                "seed": seed,
                "accuracy": accuracy,
                "exact": prediction == plain,
            }
            rows.append(row)
            print("V053_SHARED_TRIAL", json.dumps(row, sort_keys=True), flush=True)

    accuracies = [float(row["accuracy"]) for row in rows]
    summary = {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(accuracies),
        "median_accuracy": statistics.median(accuracies),
        "exact_rate": statistics.fmean(float(row["exact"]) for row in rows),
        "at_least_95_rate": statistics.fmean(value >= 0.95 for value in accuracies),
    }
    gate = {"positive_control_95_percent_pass": summary["mean_accuracy"] >= 0.95}
    gate["pass"] = all(gate.values())
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.3-shared-pool-positive-control",
        "eligible_as_fresh_key": False,
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates": args.replicates,
        "symbol_pool_size": symbol_vocab,
        "model": {
            "embedding": args.embedding,
            "hidden": args.hidden,
            "layers": args.layers,
            "attention_heads": args.attention_heads,
            "dropout": args.dropout,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "training_history": training_history,
        "summary": summary,
        "gate": gate,
        "rows": rows,
        "device": str(device),
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V053_SHARED_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    print("V053_SHARED_GATE", json.dumps(gate, sort_keys=True), flush=True)
    print("V053_SHARED_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
