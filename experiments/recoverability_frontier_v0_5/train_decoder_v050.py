#!/usr/bin/env python3
"""Train and evaluate v0.5.0 synthetic recoverability decoders from scratch."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import recoverability_v050 as core

PAD = 0
BOS = 1
EOS = 2
MAX_SURFACE_SYMBOLS = 1024
SURFACE_OFFSET = 3


@dataclass(frozen=True)
class Vocabulary:
    chars: tuple[str, ...]
    char_to_token: dict[str, int]
    token_to_char: dict[int, str]
    input_vocab_size: int
    output_vocab_size: int
    language_tags: dict[str, int]
    family_tags: dict[str, int]


def build_vocab(languages: dict[str, core.LanguageData]) -> Vocabulary:
    chars = tuple(sorted({ch for language in languages.values() for ch in language.alphabet}))
    char_to_token = {ch: i + 3 for i, ch in enumerate(chars)}
    token_to_char = {i + 3: ch for i, ch in enumerate(chars)}
    cursor = SURFACE_OFFSET + MAX_SURFACE_SYMBOLS
    language_tags = {}
    for iso in sorted(languages):
        language_tags[iso] = cursor
        cursor += 1
    family_tags = {}
    for family in core.FAMILIES:
        family_tags[family] = cursor
        cursor += 1
    return Vocabulary(
        chars=chars,
        char_to_token=char_to_token,
        token_to_char=token_to_char,
        input_vocab_size=cursor,
        output_vocab_size=3 + len(chars),
        language_tags=language_tags,
        family_tags=family_tags,
    )


def local_to_output(language: core.LanguageData, values: Sequence[int], vocab: Vocabulary) -> list[int]:
    return [vocab.char_to_token[language.alphabet[x]] for x in values if 0 <= x < len(language.alphabet)]


def build_markov(language: core.LanguageData) -> dict[tuple[int, int], tuple[int, ...]]:
    followers: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    stream = language.train_stream
    for a, b, c in zip(stream, stream[1:], stream[2:]):
        followers[(a, b)].append(c)
    return {key: tuple(values) for key, values in followers.items()}


def build_slot_parts(language: core.LanguageData) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], list[tuple[int, ...]]]:
    words = language.train_words
    if not words:
        return ([(0,)], [(0,)], [(0,)])
    unique = list(dict.fromkeys(words))[:4096]
    prefixes = [word[: max(1, len(word) // 3)] for word in unique if word]
    cores = [word[max(0, len(word) // 3) : max(1, 2 * len(word) // 3)] for word in unique if word]
    suffixes = [word[max(0, 2 * len(word) // 3) :] for word in unique if word]
    return prefixes, cores, suffixes


class SyntheticDataset(Dataset):
    def __init__(
        self,
        languages: dict[str, core.LanguageData],
        vocab: Vocabulary,
        split: str,
        positives: int,
        controls: int,
        known_family: bool,
        seed: int,
        smoke: bool = False,
    ):
        self.languages = languages
        self.vocab = vocab
        self.split = split
        self.positives = positives
        self.controls = controls
        self.known_family = known_family
        self.seed = seed
        self.isos = tuple(sorted(languages))
        self.families = core.FAMILIES if not smoke else ("mono", "homophonic", "polyalphabetic", "transposition")
        self.lengths = core.LENGTHS if not smoke else (64, 96)
        self.noises = core.NOISE_LEVELS if not smoke else (0.0, 0.01)
        self.control_families = core.CONTROL_FAMILIES
        self.chunks: dict[tuple[str, int], list[list[int]]] = {}
        for iso, language in languages.items():
            for length in self.lengths:
                values = core.source_chunks(language, split, length)
                if not values:
                    raise RuntimeError(f"no source chunks for {iso}/{split}/{length}")
                self.chunks[(iso, length)] = values
        self.markov = {iso: build_markov(language) for iso, language in languages.items()}
        self.slots = {iso: build_slot_parts(language) for iso, language in languages.items()}

    def __len__(self) -> int:
        return self.positives + self.controls

    def _cell(self, index: int) -> tuple[str, str, int, float, int]:
        cells = len(self.isos) * len(self.families) * len(self.lengths) * len(self.noises)
        cell_index = index % cells
        replicate = index // cells
        noise = self.noises[cell_index % len(self.noises)]
        cell_index //= len(self.noises)
        length = self.lengths[cell_index % len(self.lengths)]
        cell_index //= len(self.lengths)
        family = self.families[cell_index % len(self.families)]
        cell_index //= len(self.families)
        iso = self.isos[cell_index % len(self.isos)]
        return iso, family, length, noise, replicate

    def _markov_control(self, language: core.LanguageData, iso: str, length: int, rng: random.Random) -> list[int]:
        stream = language.train_stream
        start = rng.randrange(max(1, len(stream) - 2))
        out = [stream[start], stream[start + 1]]
        followers = self.markov[iso]
        while len(out) < length:
            choices = followers.get((out[-2], out[-1]))
            out.append(rng.choice(choices) if choices else core.weighted_choice(rng, language.probabilities))
        return out[:length]

    def _slot_control(self, language: core.LanguageData, iso: str, length: int, rng: random.Random) -> list[int]:
        prefixes, cores, suffixes = self.slots[iso]
        space = language.char_to_id.get(" ", 0)
        out: list[int] = []
        while len(out) < length:
            word = list(rng.choice(prefixes)) + list(rng.choice(cores)) + list(rng.choice(suffixes))
            if len(word) > 2:
                word = word[: rng.randint(2, min(14, len(word)))]
            if out:
                out.append(space)
            out.extend(word)
        return out[:length]

    def _control_plain(
        self,
        language: core.LanguageData,
        iso: str,
        control_family: str,
        length: int,
        rng: random.Random,
    ) -> list[int]:
        if control_family == "markov2":
            return self._markov_control(language, iso, length, rng)
        if control_family == "slot":
            return self._slot_control(language, iso, length, rng)
        if control_family == "motif":
            return core.motif_generate(language, length, rng)
        if control_family == "copy_mutate":
            return core.copy_mutate_generate(language, length, rng)
        raise ValueError(control_family)

    def __getitem__(self, index: int) -> dict[str, Any]:
        positive = index < self.positives
        local_index = index if positive else index - self.positives
        iso, family, length, noise, replicate = self._cell(local_index)
        language = self.languages[iso]
        rng = random.Random(core.stable_seed("decoder-v050", self.seed, self.split, positive, index))

        if positive:
            pool = self.chunks[(iso, length)]
            plain = list(pool[replicate % len(pool)])
            control_family = None
        else:
            control_family = self.control_families[replicate % len(self.control_families)]
            plain = self._control_plain(language, iso, control_family, length, rng)

        packet = core.encrypt_sequence(
            plain,
            family,
            language,
            rng,
            parameter_mode="test" if self.split == "test" else self.split,
        )
        packet = core.apply_noise(packet, noise, rng)
        if packet.max_symbol >= MAX_SURFACE_SYMBOLS:
            raise RuntimeError(f"surface symbol overflow: {packet.max_symbol}")

        source = [self.vocab.language_tags[iso]]
        if self.known_family:
            source.append(self.vocab.family_tags[family])
        source.extend(SURFACE_OFFSET + x for x in packet.cipher)

        target = [BOS]
        if positive:
            target.extend(local_to_output(language, plain, self.vocab))
        target.append(EOS)

        return {
            "source": source,
            "target": target,
            "message": 1.0 if positive else 0.0,
            "iso": iso,
            "family": family,
            "length": length,
            "noise": noise,
            "control_family": control_family,
            "plain_local": plain if positive else [],
            "alphabet": language.alphabet,
        }


def collate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_source = max(len(row["source"]) for row in rows)
    max_target = max(len(row["target"]) for row in rows)
    source = torch.full((len(rows), max_source), PAD, dtype=torch.long)
    target = torch.full((len(rows), max_target), PAD, dtype=torch.long)
    message = torch.tensor([row["message"] for row in rows], dtype=torch.float32)
    positive_mask = message.bool()
    for i, row in enumerate(rows):
        source[i, : len(row["source"])] = torch.tensor(row["source"], dtype=torch.long)
        target[i, : len(row["target"])] = torch.tensor(row["target"], dtype=torch.long)
    return {
        "source": source,
        "target": target,
        "message": message,
        "positive_mask": positive_mask,
        "meta": list(rows),
    }


class RecoverabilityTransformer(nn.Module):
    def __init__(
        self,
        input_vocab: int,
        output_vocab: int,
        d_model: int,
        heads: int,
        encoder_layers: int,
        decoder_layers: int,
        ff: int,
        dropout: float,
        max_positions: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_embedding = nn.Embedding(input_vocab, d_model, padding_idx=PAD)
        self.output_embedding = nn.Embedding(output_vocab, d_model, padding_idx=PAD)
        self.input_position = nn.Embedding(max_positions, d_model)
        self.output_position = nn.Embedding(max_positions, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.classifier = nn.Linear(d_model, 1)
        self.output = nn.Linear(d_model, output_vocab)

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_mask = source.eq(PAD)
        positions = torch.arange(source.shape[1], device=source.device).unsqueeze(0)
        x = self.input_embedding(source) * math.sqrt(self.d_model) + self.input_position(positions)
        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1)
        pooled = (memory * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        logits = self.classifier(pooled).squeeze(-1)
        return memory, pad_mask, logits

    def decode_teacher(
        self,
        memory: torch.Tensor,
        memory_pad: torch.Tensor,
        target_input: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(target_input.shape[1], device=target_input.device).unsqueeze(0)
        y = self.output_embedding(target_input) * math.sqrt(self.d_model) + self.output_position(positions)
        causal = nn.Transformer.generate_square_subsequent_mask(target_input.shape[1], device=target_input.device)
        decoded = self.decoder(
            y,
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target_input.eq(PAD),
            memory_key_padding_mask=memory_pad,
        )
        return self.output(decoded)

    def forward(self, source: torch.Tensor, target_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        memory, pad, classification = self.encode(source)
        return classification, self.decode_teacher(memory, pad, target_input)

    @torch.no_grad()
    def greedy(self, source: torch.Tensor, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        memory, pad, classification = self.encode(source)
        generated = torch.full((source.shape[0], 1), BOS, dtype=torch.long, device=source.device)
        finished = torch.zeros(source.shape[0], dtype=torch.bool, device=source.device)
        for _ in range(max_length):
            logits = self.decode_teacher(memory, pad, generated)
            next_token = logits[:, -1].argmax(dim=-1)
            generated = torch.cat((generated, next_token[:, None]), dim=1)
            finished |= next_token.eq(EOS)
            if bool(finished.all()):
                break
        return classification, generated


def sequence_loss(logits: torch.Tensor, target: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
    if not bool(positive_mask.any()):
        return logits.sum() * 0.0
    labels = target[:, 1:].clone()
    labels[labels.eq(PAD)] = -100
    labels[~positive_mask] = -100
    return nn.functional.cross_entropy(
        logits[:, : labels.shape[1]].reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def train_epoch(
    model: RecoverabilityTransformer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    classification_weight: float,
    grad_clip: float,
) -> dict[str, float]:
    model.train()
    losses = []
    class_losses = []
    seq_losses = []
    use_amp = device.type == "cuda"
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        message = batch["message"].to(device, non_blocking=True)
        positive = batch["positive_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            class_logits, seq_logits = model(source, target[:, :-1])
            class_loss = nn.functional.binary_cross_entropy_with_logits(class_logits, message)
            seq_loss = sequence_loss(seq_logits, target, positive)
            loss = classification_weight * class_loss + seq_loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach()))
        class_losses.append(float(class_loss.detach()))
        seq_losses.append(float(seq_loss.detach()))
    return {
        "loss": statistics.fmean(losses),
        "classification_loss": statistics.fmean(class_losses),
        "sequence_loss": statistics.fmean(seq_losses),
    }


@torch.no_grad()
def classifier_scores(
    model: RecoverabilityTransformer,
    loader: DataLoader,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        _, _, logits = model.encode(source)
        probabilities = torch.sigmoid(logits).cpu().tolist()
        for probability, meta in zip(probabilities, batch["meta"]):
            rows.append({
                "probability": float(probability),
                "message": bool(meta["message"]),
                "iso": meta["iso"],
                "family": meta["family"],
                "noise": meta["noise"],
                "length": meta["length"],
                "control_family": meta["control_family"],
            })
    return rows


def select_threshold(rows: Sequence[dict[str, Any]], max_fpr: float = 0.05) -> dict[str, float]:
    candidates = sorted({float(row["probability"]) for row in rows}, reverse=True)
    best = {"threshold": 1.0, "sensitivity": 0.0, "fpr": 0.0}
    positives = [row for row in rows if row["message"]]
    controls = [row for row in rows if not row["message"]]
    for threshold in candidates:
        sensitivity = sum(row["probability"] >= threshold for row in positives) / max(1, len(positives))
        fpr = sum(row["probability"] >= threshold for row in controls) / max(1, len(controls))
        if fpr <= max_fpr and (sensitivity, -fpr, threshold) > (
            best["sensitivity"], -best["fpr"], best["threshold"]
        ):
            best = {"threshold": threshold, "sensitivity": sensitivity, "fpr": fpr}
    return best


def strip_generated(values: Sequence[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        value = int(value)
        if value == BOS:
            continue
        if value == EOS:
            break
        if value != PAD:
            out.append(value)
    return out


@torch.no_grad()
def evaluate(
    model: RecoverabilityTransformer,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    max_decode: int,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        source = batch["source"].to(device, non_blocking=True)
        class_logits, generated = model.greedy(source, max_decode)
        probabilities = torch.sigmoid(class_logits).cpu().tolist()
        generated = generated.cpu().tolist()
        for probability, prediction, meta in zip(probabilities, generated, batch["meta"]):
            declared = probability >= threshold
            positive = bool(meta["message"])
            accuracy = 0.0
            exact = False
            if positive and declared:
                truth = local_to_output(
                    loader.dataset.languages[meta["iso"]],
                    meta["plain_local"],
                    loader.dataset.vocab,
                )
                decoded = strip_generated(prediction)
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


def save_checkpoint(path: Path, model: nn.Module, vocab: Vocabulary, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "vocab": {
                "chars": vocab.chars,
                "language_tags": vocab.language_tags,
                "family_tags": vocab.family_tags,
            },
            "config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=("known", "blind"), required=True)
    parser.add_argument("--seed", type=int, default=505050)
    parser.add_argument("--train-positives", type=int, default=120000)
    parser.add_argument("--train-controls", type=int, default=120000)
    parser.add_argument("--dev-positives", type=int, default=8640)
    parser.add_argument("--dev-controls", type=int, default=8640)
    parser.add_argument("--test-positives", type=int, default=8640)
    parser.add_argument("--test-controls", type=int, default=8640)
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
        args.train_positives = min(args.train_positives, 1024)
        args.train_controls = min(args.train_controls, 1024)
        args.dev_positives = min(args.dev_positives, 128)
        args.dev_controls = min(args.dev_controls, 128)
        args.test_positives = min(args.test_positives, 128)
        args.test_controls = min(args.test_controls, 128)
        args.epochs = min(args.epochs, 1)
        args.d_model = min(args.d_model, 128)
        args.heads = min(args.heads, 4)
        args.encoder_layers = min(args.encoder_layers, 2)
        args.decoder_layers = min(args.decoder_layers, 2)
        args.ff = min(args.ff, 384)
        args.batch_size = min(args.batch_size, 16)

    known = args.mode == "known"
    vocab = build_vocab(languages)
    datasets = {
        split: SyntheticDataset(
            languages,
            vocab,
            split,
            positives=getattr(args, f"{split}_positives"),
            controls=getattr(args, f"{split}_controls"),
            known_family=known,
            seed=args.seed,
            smoke=args.smoke,
        )
        for split in ("train", "dev", "test")
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
    model = RecoverabilityTransformer(
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
        metrics = train_epoch(model, loaders["train"], optimizer, scaler, device, 1.0, 1.0)
        metrics["epoch"] = epoch
        metrics["elapsed_seconds"] = time.time() - started
        history.append(metrics)
        print("V050_TRAIN", args.mode, json.dumps(metrics, sort_keys=True), flush=True)

    dev_rows = classifier_scores(model, loaders["dev"], device)
    threshold = select_threshold(dev_rows, max_fpr=0.05)
    print("V050_THRESHOLD", args.mode, json.dumps(threshold, sort_keys=True), flush=True)
    max_decode = max(datasets["test"].lengths) + 16
    test = evaluate(model, loaders["test"], device, threshold["threshold"], max_decode)
    gate_families = sum(row["mean_accuracy_all"] >= 0.70 for row in test["by_family"].values())
    gate = {
        "sensitivity_pass": test["sensitivity"] >= 0.80,
        "fpr_pass": test["false_positive_rate"] <= 0.05,
        "five_families_accuracy_pass": gate_families >= 5,
    }
    gate["pass"] = all(gate.values())
    payload = {
        "programme": "recoverability-frontier-v0.5.0-learned-decoder",
        "mode": args.mode,
        "device": str(device),
        "config": vars(args),
        "history": history,
        "development_threshold": threshold,
        "test": test,
        "gate": gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    if args.checkpoint:
        save_checkpoint(args.checkpoint, model, vocab, vars(args))
    print("V050_TEST", args.mode, json.dumps({k: v for k, v in test.items() if k != "rows"}, sort_keys=True), flush=True)
    print("V050_GATE", args.mode, json.dumps(gate, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
