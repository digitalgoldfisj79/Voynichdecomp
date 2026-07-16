#!/usr/bin/env python3
"""Shared neural components for the final v0.6 Family S3 amendment."""
from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
V05 = HERE.parent / "recoverability_frontier_v0_5"
if str(V05) not in sys.path:
    sys.path.insert(0, str(V05))

import recoverability_v050 as core
import v060_family_s_stage_s1 as s1


@dataclass
class Example:
    source: list[int]
    line_flags: list[int]
    boundaries: list[float]
    target: list[int]


def canonical_first_occurrence(values: Sequence[int]) -> list[int]:
    mapping: dict[int, int] = {}
    out: list[int] = []
    for value in values:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        out.append(mapping[value])
    return out


def unitise_with_spans(
    plain: list[int], inventory: list[tuple[int, ...]]
) -> list[tuple[tuple[int, ...], int, int]]:
    lookup = set(inventory)
    out: list[tuple[tuple[int, ...], int, int]] = []
    index = 0
    while index < len(plain):
        selected: tuple[int, ...] | None = None
        for width in (3, 2):
            if index + width <= len(plain):
                candidate = tuple(plain[index : index + width])
                if candidate in lookup:
                    selected = candidate
                    break
        if selected is None:
            selected = (plain[index],)
        out.append((selected, index, index + len(selected)))
        index += len(selected)
    return out


def sample_unique_code(
    rng: random.Random, used: set[tuple[int, ...]]
) -> tuple[int, ...]:
    value = rng.random()
    width = 1 if value < 0.20 else (2 if value < 0.65 else 3)
    for _ in range(10000):
        candidate = tuple(rng.randrange(10) for _ in range(width))
        if candidate not in used:
            used.add(candidate)
            return candidate
    for fallback_width in (2, 3, 1):
        for number in range(10 ** fallback_width):
            digits = []
            current = number
            for _ in range(fallback_width):
                digits.append(current % 10)
                current //= 10
            candidate = tuple(reversed(digits))
            if candidate not in used:
                used.add(candidate)
                return candidate
    raise RuntimeError("visible code space exhausted")


def make_line_starts(rng: random.Random, length: int) -> list[int]:
    starts = [0]
    cursor = 0
    while cursor < length:
        cursor += rng.randint(40, 72)
        if cursor < length:
            starts.append(cursor)
    return starts


class SyntheticGenerator:
    def __init__(
        self,
        language: core.LanguageData,
        seed: int,
        plaintext_length: int = 384,
    ) -> None:
        self.language = language
        self.inventory = s1.candidate_inventory(language)
        self.rng = random.Random(seed)
        self.length = plaintext_length
        self.stream = language.train_stream
        if len(self.stream) <= self.length:
            raise RuntimeError("training stream too short")

    def sample(self) -> Example:
        start = self.rng.randrange(len(self.stream) - self.length)
        plain = list(self.stream[start : start + self.length])
        units = unitise_with_spans(plain, self.inventory)
        active = sorted({unit for unit, _left, _right in units})
        used: set[tuple[int, ...]] = set()
        codebook = {unit: sample_unique_code(self.rng, used) for unit in active}
        visible = list(range(10))
        self.rng.shuffle(visible)
        line_starts = make_line_starts(self.rng, self.length)

        raw_cipher: list[int] = []
        boundary_positions: list[int] = []
        source_line_positions: list[int] = []
        line_index = 0
        for unit, left, right in units:
            while line_index < len(line_starts) and line_starts[line_index] < left:
                line_index += 1
            code_start = len(raw_cipher)
            code = tuple(visible[digit] for digit in codebook[unit])
            raw_cipher.extend(code)
            boundary_positions.append(len(raw_cipher) - 1)
            for line_start in line_starts:
                if left <= line_start < right:
                    source_line_positions.append(code_start)

        canonical = canonical_first_occurrence(raw_cipher)
        source = [value + 1 for value in canonical]  # zero is padding
        boundaries = [0.0] * len(source)
        for position in boundary_positions:
            boundaries[position] = 1.0
        line_flags = [0] * len(source)
        for position in source_line_positions:
            if 0 <= position < len(line_flags):
                line_flags[position] = 1
        return Example(source, line_flags, boundaries, plain)


def collate(examples: list[Example], device: torch.device) -> dict[str, torch.Tensor]:
    batch = len(examples)
    source_length = max(len(example.source) for example in examples)
    target_length = len(examples[0].target)
    source = torch.zeros((batch, source_length), dtype=torch.long)
    line_flags = torch.zeros((batch, source_length), dtype=torch.long)
    boundary = torch.zeros((batch, source_length), dtype=torch.float32)
    source_padding = torch.ones((batch, source_length), dtype=torch.bool)
    target = torch.zeros((batch, target_length), dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.source)
        source[row, :length] = torch.tensor(example.source, dtype=torch.long)
        line_flags[row, :length] = torch.tensor(example.line_flags, dtype=torch.long)
        boundary[row, :length] = torch.tensor(example.boundaries, dtype=torch.float32)
        source_padding[row, :length] = False
        target[row] = torch.tensor(example.target, dtype=torch.long)
    return {
        "source": source.to(device, non_blocking=True),
        "line_flags": line_flags.to(device, non_blocking=True),
        "boundary": boundary.to(device, non_blocking=True),
        "source_padding": source_padding.to(device, non_blocking=True),
        "target": target.to(device, non_blocking=True),
    }


class NeuralTransducer(nn.Module):
    def __init__(
        self,
        alphabet_size: int,
        d_model: int = 384,
        nhead: int = 8,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        dim_feedforward: int = 1536,
        max_source: int = 1200,
        max_target: int = 384,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.alphabet_size = alphabet_size
        self.d_model = d_model
        self.bos_id = alphabet_size
        self.source_embedding = nn.Embedding(11, d_model, padding_idx=0)
        self.line_embedding = nn.Embedding(2, d_model)
        self.source_position = nn.Embedding(max_source, d_model)
        self.target_embedding = nn.Embedding(alphabet_size + 1, d_model)
        self.target_position = nn.Embedding(max_target, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, alphabet_size)
        self.boundary = nn.Linear(d_model, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        source: torch.Tensor,
        line_flags: torch.Tensor,
        source_padding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length = source.shape
        positions = torch.arange(length, device=source.device).unsqueeze(0).expand(batch, -1)
        embedded = (
            self.source_embedding(source)
            + self.line_embedding(line_flags)
            + self.source_position(positions)
        ) * math.sqrt(self.d_model)
        memory = self.encoder(embedded, src_key_padding_mask=source_padding)
        memory = self.encoder_norm(memory)
        boundary_logits = self.boundary(memory).squeeze(-1)
        return memory, boundary_logits

    def decode(
        self,
        decoder_input: torch.Tensor,
        memory: torch.Tensor,
        source_padding: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = decoder_input.shape
        positions = torch.arange(length, device=decoder_input.device).unsqueeze(0).expand(batch, -1)
        embedded = (
            self.target_embedding(decoder_input)
            + self.target_position(positions)
        ) * math.sqrt(self.d_model)
        causal = torch.triu(
            torch.full((length, length), float("-inf"), device=decoder_input.device),
            diagonal=1,
        )
        decoded = self.decoder(
            embedded,
            memory,
            tgt_mask=causal,
            memory_key_padding_mask=source_padding,
        )
        return self.output(self.decoder_norm(decoded))

    def forward(
        self,
        source: torch.Tensor,
        line_flags: torch.Tensor,
        source_padding: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, boundary_logits = self.encode(source, line_flags, source_padding)
        bos = torch.full(
            (target.shape[0], 1), self.bos_id, dtype=torch.long, device=target.device
        )
        decoder_input = torch.cat([bos, target[:, :-1]], dim=1)
        logits = self.decode(decoder_input, memory, source_padding)
        return logits, boundary_logits


def model_config(alphabet_size: int) -> dict[str, int | float]:
    return {
        "alphabet_size": alphabet_size,
        "d_model": 384,
        "nhead": 8,
        "encoder_layers": 6,
        "decoder_layers": 6,
        "dim_feedforward": 1536,
        "max_source": 1200,
        "max_target": 384,
        "dropout": 0.10,
    }
