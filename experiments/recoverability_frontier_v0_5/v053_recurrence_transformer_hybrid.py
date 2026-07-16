#!/usr/bin/env python3
"""v0.5.3 fresh-key recurrence Transformer plus classical hybrid refinement."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

import recoverability_v050 as core
import homophonic_solver_v052 as homophonic
import mono_solver_v051 as mono
from homophonic_confirm_v052_quadgram import build_quadgram_model


class RecurrenceTransformer(nn.Module):
    def __init__(
        self,
        recurrence_vocab: int,
        alphabet_size: int,
        max_length: int,
        d_model: int,
        heads: int,
        encoder_layers: int,
        decoder_layers: int,
        feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.alphabet_size = alphabet_size
        self.max_length = max_length
        self.recurrence_embedding = nn.Embedding(recurrence_vocab, d_model)
        self.source_position = nn.Embedding(max_length, d_model)
        self.target_queries = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, alphabet_size)

    def forward(self, recurrence: torch.Tensor) -> torch.Tensor:
        batch, length = recurrence.shape
        if length > self.max_length:
            raise RuntimeError("sequence exceeds configured maximum")
        positions = torch.arange(length, device=recurrence.device)
        source = self.recurrence_embedding(recurrence) + self.source_position(positions)[None, :, :]
        memory = self.encoder(source)
        queries = self.target_queries(positions)[None, :, :].expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.output(self.output_norm(decoded))


def load_cryptool_namespace(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    needle = (
        "        if selected < 0:\n"
        "            state, selected = rng_int(state, alphabet_size)\n"
        "        key[key_index] = selected\n"
        "        distribution[selected] += 1\n"
    )
    replacement = (
        "        if selected < 0:\n"
        "            state, selected = rng_int(state, alphabet_size)\n"
        "        selected = int(selected)\n"
        "        key[key_index] = selected\n"
        "        distribution[selected] += 1\n"
    )
    if source.count(needle) != 1:
        raise RuntimeError("CrypTool port cast site mismatch")
    patched = source.replace(needle, replacement)
    namespace: dict[str, Any] = {
        "__name__": "v053_hybrid_cryptool_library",
        "__file__": str(path),
    }
    exec(compile(patched, str(path), "exec"), namespace)
    return namespace


def random_train_batch(
    language: core.LanguageData,
    batch_size: int,
    length: int,
    recurrence_vocab: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    stream = language.train_stream
    recurrence_rows: list[list[int]] = []
    plaintext_rows: list[list[int]] = []
    for _ in range(batch_size):
        start = rng.randrange(0, len(stream) - length)
        plain = list(stream[start : start + length])
        seed = rng.getrandbits(63)
        packet = core.encrypt_sequence(
            plain, "homophonic", language, random.Random(seed), parameter_mode="train"
        )
        recurrence, _canonical_to_raw = homophonic.canonicalize_with_inverse(packet.cipher)
        encoded = [value + 1 for value in recurrence]
        if max(encoded, default=0) >= recurrence_vocab:
            raise RuntimeError("recurrence vocabulary too small")
        recurrence_rows.append(encoded)
        plaintext_rows.append(plain)
    return (
        torch.tensor(recurrence_rows, dtype=torch.long),
        torch.tensor(plaintext_rows, dtype=torch.long),
    )


def train_model(
    model: RecurrenceTransformer,
    language: core.LanguageData,
    device: torch.device,
    recurrence_vocab: int,
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
        recurrence, target = random_train_batch(
            language, batch_size, length, recurrence_vocab, rng
        )
        recurrence = recurrence.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(recurrence)
            loss = F.cross_entropy(logits.reshape(-1, model.alphabet_size), target.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % max(1, steps // 25) == 0 or step == 1:
            item = {"step": float(step), "loss": float(loss.detach().cpu())}
            history.append(item)
            print("V053_RECURRENCE_TRAIN", json.dumps(item, sort_keys=True), flush=True)
    return history


def symbol_label_scores(
    logits: torch.Tensor,
    cipher: list[int],
    symbol_count: int,
) -> np.ndarray:
    log_probs = F.log_softmax(logits.float(), dim=-1).detach().cpu().numpy()
    scores = np.zeros((symbol_count, log_probs.shape[-1]), dtype=np.float64)
    counts = np.zeros(symbol_count, dtype=np.int32)
    for position, symbol in enumerate(cipher):
        scores[int(symbol)] += log_probs[position]
        counts[int(symbol)] += 1
    scores /= np.maximum(counts[:, None], 1)
    return scores


def assignment_from_scores(
    scores: np.ndarray,
    slot_labels: np.ndarray,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    expanded = scores[:, slot_labels]
    if noise is not None:
        expanded = expanded + noise
    rows, columns = linear_sum_assignment(-expanded)
    mapping = np.empty(scores.shape[0], dtype=np.int32)
    for row, column in zip(rows, columns):
        mapping[int(row)] = int(slot_labels[int(column)])
    return mapping


def posterior_seeds(
    scores: np.ndarray,
    slot_labels: np.ndarray,
    count: int,
    seed: int,
) -> list[np.ndarray]:
    seeds: list[np.ndarray] = []
    seen: set[bytes] = set()
    base = assignment_from_scores(scores, slot_labels)
    seeds.append(base)
    seen.add(base.tobytes())
    rng = np.random.default_rng(seed)
    temperature_grid = (0.03, 0.06, 0.10, 0.16)
    attempts = 0
    while len(seeds) < count and attempts < count * 20:
        temperature = temperature_grid[attempts % len(temperature_grid)]
        gumbel = rng.gumbel(size=(scores.shape[0], len(slot_labels))) * temperature
        mapping = assignment_from_scores(scores, slot_labels, gumbel)
        signature = mapping.tobytes()
        if signature not in seen:
            seen.add(signature)
            seeds.append(mapping)
        attempts += 1
    return seeds


def hybrid_refine(
    seeds: list[np.ndarray],
    trial: dict[str, Any],
    language: core.LanguageData,
    model: tuple[np.ndarray, np.ndarray],
    cryptool: dict[str, Any],
    steps: int,
    target_acceptance: float,
    workers: int,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    cipher_values = list(map(int, trial["cipher"]))
    cipher = np.asarray(cipher_values, dtype=np.int32)
    positions, offsets, rare_order = cryptool["build_positions"](cipher_values)
    _minimum, maximum, proposal_cdf = cryptool["distribution_arrays"](language)

    def refine(item: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, float, float]:
        index, start_key = item
        key, score, temperature, _mutations = cryptool["cryptool_style_single_run"](
            cipher,
            start_key.astype(np.int32, copy=True),
            model[0],
            model[1],
            positions,
            offsets,
            rare_order,
            maximum,
            proposal_cdf,
            steps,
            target_acceptance,
            50,
            0,
            int(core.stable_seed("v053-hybrid", trial["seed"], index) & 0x7FFFFFFFFFFFFFFF),
        )
        return index, key, float(score), float(temperature)

    results: list[tuple[int, np.ndarray, float, float]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results.extend(executor.map(refine, list(enumerate(seeds))))
    best = max(results, key=lambda item: item[2])
    diagnostics = [
        {"seed_index": float(index), "score": score, "start_temperature": temperature}
        for index, _key, score, temperature in results
    ]
    return best[1], best[2], diagnostics


def accuracy_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row[field]) for row in rows]
    return {
        "trials": len(rows),
        "mean_accuracy": statistics.fmean(values),
        "median_accuracy": statistics.median(values),
        "exact_rate": statistics.fmean(value == 1.0 for value in values),
        "at_least_70_rate": statistics.fmean(value >= 0.70 for value in values),
        "at_least_90_rate": statistics.fmean(value >= 0.90 for value in values),
        "at_least_95_rate": statistics.fmean(value >= 0.95 for value in values),
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
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--posterior-seeds", type=int, default=32)
    parser.add_argument("--hybrid-steps", type=int, default=250_000)
    parser.add_argument("--hybrid-workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=53032)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment = args.repo / "experiments" / "recoverability_frontier_v0_5"
    languages = core.load_languages(
        experiment / "corpus_manifest_v050.json",
        args.repo / ".cache" / "v050_corpora",
    )
    language = languages[args.iso]
    maximum_symbols = sum(
        homophonic.multiplicity(float(probability)) for probability in language.probabilities
    )
    recurrence_vocab = maximum_symbols + 2
    model = RecurrenceTransformer(
        recurrence_vocab=recurrence_vocab,
        alphabet_size=len(language.alphabet),
        max_length=args.length,
        d_model=args.d_model,
        heads=args.heads,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        feedforward=args.feedforward,
        dropout=args.dropout,
    ).to(device)
    training_history = train_model(
        model,
        language,
        device,
        recurrence_vocab,
        args.train_steps,
        args.batch_size,
        args.length,
        args.learning_rate,
        args.seed,
    )
    model.eval()
    cryptool = load_cryptool_namespace(experiment / "cryptool_homophonic_port_v052.py")
    quadgram = build_quadgram_model(language)
    trials = [
        homophonic.make_trial(language, args.split, args.length, args.offset + replicate)
        for replicate in range(args.replicates)
    ]

    # Compile the classical refinement kernel on the first trial.
    compile_trial = trials[0]
    compile_cipher_values = list(map(int, compile_trial["cipher"]))
    compile_cipher = np.asarray(compile_cipher_values, dtype=np.int32)
    compile_key = homophonic.frequency_slot_key(
        compile_cipher_values,
        compile_trial["inferred_labels"],
        compile_trial["expected_slot_probabilities"],
    )
    compile_positions, compile_offsets, compile_rare = cryptool["build_positions"](
        compile_cipher_values
    )
    _minimum, compile_maximum, compile_cdf = cryptool["distribution_arrays"](language)
    cryptool["cryptool_style_single_run"](
        compile_cipher,
        compile_key,
        quadgram[0],
        quadgram[1],
        compile_positions,
        compile_offsets,
        compile_rare,
        compile_maximum,
        compile_cdf,
        10,
        0.05,
        2,
        0,
        1,
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for completed, trial in enumerate(trials, start=1):
            started = time.perf_counter()
            recurrence = torch.tensor(
                [[value + 1 for value in trial["cipher"]]],
                dtype=torch.long,
                device=device,
            )
            logits = model(recurrence)[0]
            greedy = logits.argmax(dim=-1).detach().cpu().tolist()
            scores = symbol_label_scores(
                logits, list(map(int, trial["cipher"])), len(trial["inferred_labels"])
            )
            slot_labels = np.asarray(trial["inferred_labels"], dtype=np.int32)
            constrained_key = assignment_from_scores(scores, slot_labels)
            cipher_array = np.asarray(trial["cipher"], dtype=np.int32)
            constrained = constrained_key[cipher_array].tolist()
            seeds = posterior_seeds(
                scores,
                slot_labels,
                args.posterior_seeds,
                core.stable_seed("v053-posterior-seeds", trial["seed"]),
            )
            hybrid_key, hybrid_score, hybrid_diagnostics = hybrid_refine(
                seeds,
                trial,
                language,
                quadgram,
                cryptool,
                args.hybrid_steps,
                0.05,
                args.hybrid_workers,
            )
            hybrid = hybrid_key[cipher_array].tolist()
            row = {
                "replicate": int(trial["replicate"]),
                "inventory_overlap": float(trial["inventory_overlap"]),
                "greedy_accuracy": mono.fast_accuracy(trial["plain"], greedy),
                "constrained_accuracy": mono.fast_accuracy(trial["plain"], constrained),
                "hybrid_accuracy": mono.fast_accuracy(trial["plain"], hybrid),
                "greedy_exact": greedy == trial["plain"],
                "constrained_exact": constrained == trial["plain"],
                "hybrid_exact": hybrid == trial["plain"],
                "hybrid_score": hybrid_score,
                "posterior_seed_count": len(seeds),
                "hybrid_diagnostics": hybrid_diagnostics,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print("V053_RECURRENCE_TRIAL", json.dumps(row, sort_keys=True), flush=True)

    summaries = {
        "greedy": accuracy_summary(rows, "greedy_accuracy"),
        "constrained": accuracy_summary(rows, "constrained_accuracy"),
        "hybrid": accuracy_summary(rows, "hybrid_accuracy"),
    }
    eligible_modes = [
        mode
        for mode, summary in summaries.items()
        if summary["mean_accuracy"] >= 0.70
        and summary["median_accuracy"] >= 0.90
        and summary["at_least_70_rate"] >= 0.875
    ]
    payload: dict[str, Any] = {
        "programme": "recoverability-frontier-v0.5.3-recurrence-transformer-hybrid",
        "iso": args.iso,
        "split": args.split,
        "length": args.length,
        "offset": args.offset,
        "replicates": args.replicates,
        "model": {
            "d_model": args.d_model,
            "heads": args.heads,
            "encoder_layers": args.encoder_layers,
            "decoder_layers": args.decoder_layers,
            "feedforward": args.feedforward,
            "dropout": args.dropout,
            "train_steps": args.train_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "fresh_key_training": True,
        },
        "hybrid": {
            "posterior_seeds": args.posterior_seeds,
            "steps_per_seed": args.hybrid_steps,
            "target_initial_acceptance": 0.05,
        },
        "training_history": training_history,
        "summaries": summaries,
        "eligible_modes": eligible_modes,
        "development_gate_pass": bool(eligible_modes),
        "rows": rows,
        "device": str(device),
    }
    payload["scientific_sha256"] = canonical_sha(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("V053_RECURRENCE_SUMMARIES", json.dumps(summaries, sort_keys=True), flush=True)
    print("V053_RECURRENCE_ELIGIBLE", json.dumps(eligible_modes), flush=True)
    print("V053_RECURRENCE_SHA256", payload["scientific_sha256"], flush=True)


if __name__ == "__main__":
    main()
