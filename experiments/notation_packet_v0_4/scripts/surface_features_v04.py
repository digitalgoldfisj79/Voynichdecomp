#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12


def entropy(values: Iterable[object]) -> float:
    c = Counter(values)
    n = sum(c.values())
    if n <= 0:
        return 0.0
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v)


def conditional_entropy(pairs: Iterable[tuple[object, object]]) -> float:
    by = defaultdict(Counter)
    n = 0
    for ctx, val in pairs:
        by[ctx][val] += 1
        n += 1
    if n <= 0:
        return 0.0
    out = 0.0
    for counts in by.values():
        m = sum(counts.values())
        out += (m / n) * entropy(list(counts.elements()))
    return out


def mutual_information(xs: Sequence[object], ys: Sequence[object]) -> float:
    if not xs or len(xs) != len(ys):
        return 0.0
    return max(0.0, entropy(xs) + entropy(ys) - entropy(list(zip(xs, ys))))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > EPS else 0.0


def autocorr_1(values: Sequence[float]) -> float:
    if len(values) < 3:
        return 0.0
    a = np.asarray(values[:-1], dtype=float)
    b = np.asarray(values[1:], dtype=float)
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def lcs(a: str, b: str) -> int:
    return lcp(a[::-1], b[::-1])


def _char_markov(events: Sequence[str]) -> tuple[float, float, float, int]:
    chars: list[str] = []
    bigrams: list[tuple[str, str]] = []
    trigrams: list[tuple[tuple[str, str], str]] = []
    for event in events:
        s = str(event)
        chars.extend(s)
        bigrams.extend(zip(s[:-1], s[1:]))
        trigrams.extend(((s[i], s[i + 1]), s[i + 2]) for i in range(max(0, len(s) - 2)))
    h0 = entropy(chars)
    h1 = conditional_entropy(bigrams)
    h2 = conditional_entropy(trigrams)
    return h0, h1, h2, len(set(chars))


def _event_markov(events: Sequence[str]) -> tuple[float, float]:
    h0 = entropy(events)
    h1 = conditional_entropy(zip(events[:-1], events[1:]))
    return h0, h1


def extract_surface_features(events: Sequence[str]) -> dict[str, float]:
    ev = [str(x) for x in events if str(x) != ""]
    n = len(ev)
    if n == 0:
        raise ValueError("empty event sequence")

    lengths = [len(x) for x in ev]
    chars = [ch for x in ev for ch in x]
    first = [x[0] for x in ev if x]
    last = [x[-1] for x in ev if x]
    ec = Counter(ev)
    h0, h1, h2, char_inventory = _char_markov(ev)
    eh0, eh1 = _event_markov(ev)

    pos_labels: list[str] = []
    pos_chars: list[str] = []
    for x in ev:
        if not x:
            continue
        for i, ch in enumerate(x):
            if i == 0:
                p = "I"
            elif i == len(x) - 1:
                p = "F"
            else:
                p = "M"
            pos_labels.append(p)
            pos_chars.append(ch)

    adjacent = list(zip(ev[:-1], ev[1:]))
    prefix_overlap = [safe_div(lcp(a, b), min(len(a), len(b))) for a, b in adjacent if min(len(a), len(b))]
    suffix_overlap = [safe_div(lcs(a, b), min(len(a), len(b))) for a, b in adjacent if min(len(a), len(b))]
    distinct_per_event = [len(set(x)) for x in ev]

    alnum = sum(ch.isalnum() for ch in chars)
    digits = sum(ch.isdigit() for ch in chars)
    uppers = sum(ch.isupper() for ch in chars)
    punct = len(chars) - alnum
    total_chars = max(1, len(chars))

    top_event_mass = max(ec.values()) / n
    hapax = sum(v == 1 for v in ec.values()) / max(1, len(ec))
    same_adj = sum(a == b for a, b in adjacent) / max(1, len(adjacent))
    len_change = sum(a != b for a, b in zip(lengths[:-1], lengths[1:])) / max(1, n - 1)

    features = {
        "n_events": float(n),
        "n_chars": float(len(chars)),
        "mean_event_len": float(np.mean(lengths)),
        "sd_event_len": float(np.std(lengths)),
        "cv_event_len": safe_div(float(np.std(lengths)), float(np.mean(lengths))),
        "median_event_len": float(np.median(lengths)),
        "max_event_len": float(max(lengths)),
        "single_char_frac": sum(x == 1 for x in lengths) / n,
        "length_entropy": entropy(lengths),
        "length_autocorr1": autocorr_1(lengths),
        "length_change_rate": len_change,
        "char_inventory": float(char_inventory),
        "event_inventory": float(len(ec)),
        "type_token_ratio": len(ec) / n,
        "hapax_type_frac": hapax,
        "top_event_mass": top_event_mass,
        "same_adjacent_event_rate": same_adj,
        "char_H0": h0,
        "char_H1": h1,
        "char_H2": h2,
        "char_RED1": safe_div(h0 - h1, h0),
        "char_RED2": safe_div(h1 - h2, h1),
        "char_H0_norm": safe_div(h0, math.log2(max(2, char_inventory))),
        "char_H1_norm": safe_div(h1, math.log2(max(2, char_inventory))),
        "event_H0": eh0,
        "event_H1": eh1,
        "event_RED1": safe_div(eh0 - eh1, eh0),
        "first_char_H": entropy(first),
        "last_char_H": entropy(last),
        "first_inventory": float(len(set(first))),
        "last_inventory": float(len(set(last))),
        "first_last_MI": mutual_information(first, last),
        "position_char_MI": mutual_information(pos_labels, pos_chars),
        "mean_prefix_overlap": float(np.mean(prefix_overlap)) if prefix_overlap else 0.0,
        "mean_suffix_overlap": float(np.mean(suffix_overlap)) if suffix_overlap else 0.0,
        "mean_distinct_chars_per_event": float(np.mean(distinct_per_event)),
        "distinct_char_ratio": safe_div(float(np.mean(distinct_per_event)), float(np.mean(lengths))),
        "digit_frac": digits / total_chars,
        "uppercase_frac": uppers / total_chars,
        "punct_frac": punct / total_chars,
        "space_like_boundary_rate": 1.0 / max(1.0, float(np.mean(lengths))),
    }
    return features


def windows(events: Sequence[str], window: int = 160, stride: int = 80, minimum: int = 64):
    n = len(events)
    if n < minimum:
        return
    if n <= window:
        yield 0, n, list(events)
        return
    starts = list(range(0, n - window + 1, stride))
    if starts[-1] != n - window:
        starts.append(n - window)
    for start in starts:
        yield start, start + window, list(events[start : start + window])


def parse_gabc_body(body: str) -> tuple[list[str], list[str]]:
    notation = [x for x in re.findall(r"\(([^()]*)\)", body) if x and not re.fullmatch(r"[cf]\d", x)]
    lyrics = re.sub(r"\([^()]*\)", " ", body)
    words = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", lyrics)
    return notation, words


def canonicalize_events(events: Sequence[str]) -> list[str]:
    """Map characters to a deterministic frequency-rank alphabet.

    This removes transcription-specific glyph identities while preserving equality,
    event boundaries, within-event order, and character frequency rank.
    """
    counts = Counter(ch for event in events for ch in str(event))
    ordered = sorted(counts, key=lambda ch: (-counts[ch], ch))
    alphabet = []
    for start, end in [(0x61,0x7A),(0x3B1,0x3C9),(0x430,0x44F),(0x561,0x586),(0x10D0,0x10F0)]:
        for cp in range(start, end + 1):
            c = chr(cp)
            if c.isalpha() and c.islower():
                alphabet.append(c)
    if len(ordered) > len(alphabet):
        alphabet.extend(chr(0xE000 + i) for i in range(len(ordered) - len(alphabet)))
    mapping = {ch: alphabet[i] for i, ch in enumerate(ordered)}
    return [''.join(mapping[ch] for ch in str(event)) for event in events]
