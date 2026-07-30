#!/usr/bin/env python3
"""Frozen text representations for the compression-transfer programme."""
from __future__ import annotations

import re
import struct
import unicodedata
from typing import Iterable

_WS_RE = re.compile(r"\s+")
TOKEN_BOUNDARY = 0x110000


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def collapse_whitespace(text: str) -> str:
    return _WS_RE.sub(" ", normalize_text(text)).strip()


def _pack_u32(values: Iterable[int]) -> bytes:
    out = bytearray()
    for value in values:
        if not (0 <= value <= 0xFFFFFFFF):
            raise ValueError(f"value out of uint32 range: {value}")
        out.extend(struct.pack(">I", value))
    return bytes(out)


def encode_representation(text: str, representation: str) -> bytes:
    norm = collapse_whitespace(text)
    if representation == "surface_utf8":
        return norm.encode("utf-8")
    if representation == "codepoint_u32_ws":
        return _pack_u32(ord(ch) for ch in norm)
    if representation == "codepoint_u32_nospace":
        return _pack_u32(ord(ch) for ch in norm if not ch.isspace())
    if representation == "token_recurrence_u32":
        registry: dict[str, int] = {}
        next_id = 1
        seq: list[int] = []
        for token in norm.split(" ") if norm else []:
            if token not in registry:
                registry[token] = next_id
                next_id += 1
            seq.extend((registry[token], TOKEN_BOUNDARY))
        return _pack_u32(seq)
    if representation == "char_recurrence_u32":
        registry: dict[str, int] = {}
        next_id = 1
        seq: list[int] = []
        for ch in norm:
            if ch.isspace():
                seq.append(TOKEN_BOUNDARY)
                continue
            if ch not in registry:
                registry[ch] = next_id
                next_id += 1
            seq.append(registry[ch])
        return _pack_u32(seq)
    if representation == "token_length_u32":
        return _pack_u32(v for token in norm.split(" ") for v in (len(token), TOKEN_BOUNDARY))
    raise KeyError(f"unknown representation: {representation}")


def chunk_text(text: str, representation: str, units: int, stride: int | None = None) -> list[str]:
    if units <= 0:
        raise ValueError("units must be positive")
    stride = units if stride is None else stride
    if stride <= 0:
        raise ValueError("stride must be positive")
    norm = collapse_whitespace(text)
    chunks: list[str] = []
    if representation in {"token_recurrence_u32", "token_length_u32"}:
        tokens = norm.split(" ") if norm else []
        token_window = max(1, units // 2)
        token_stride = max(1, stride // 2)
        for start in range(0, max(0, len(tokens) - token_window + 1), token_stride):
            chunks.append(" ".join(tokens[start:start + token_window]))
        return chunks
    sequence = norm if representation != "codepoint_u32_nospace" else "".join(ch for ch in norm if not ch.isspace())
    for start in range(0, max(0, len(sequence) - units + 1), stride):
        chunks.append(sequence[start:start + units])
    return chunks
