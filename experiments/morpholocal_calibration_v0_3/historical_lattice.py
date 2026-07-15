#!/usr/bin/env python3
"""Explicit historical mixed-unit lattice benchmark for v0.3 development."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("v03_fast_lattice", HERE / "tournament_fast.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot import tournament_fast.py")
fast = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fast
spec.loader.exec_module(fast)
base = fast.base
ALPHA = 0.5
WORD_RE = re.compile(r"[^a-z]+")


@dataclass(frozen=True)
class Unit:
    id: int
    kind: str
    output: str


@dataclass(frozen=True)
class Inventory:
    units: tuple[Unit, ...]
    boundary_id: int
    hash: str


@dataclass(frozen=True)
class Event:
    cell: int
    true_unit: int
    line: int
    position: str
    test: bool


@dataclass(frozen=True)
class NGramModel:
    order: int
    vocabulary: int
    counts: dict[tuple[int, ...], int]
    contexts: dict[tuple[int, ...], int]

    def score(self, sequence: Sequence[int]) -> float:
        padded = [self.vocabulary] * (self.order - 1) + list(sequence)
        bits = 0.0
        support = self.vocabulary + 1
        for i in range(self.order - 1, len(padded)):
            context = tuple(padded[i - self.order + 1:i])
            gram = context + (padded[i],)
            bits -= math.log2(
                (self.counts.get(gram, 0) + ALPHA)
                / (self.contexts.get(context, 0) + ALPHA * support)
            )
        return bits


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_word(word: str) -> str:
    return WORD_RE.sub("", word.lower())


def load_words(repo: Path) -> list[str]:
    _, module = base.load_v02(repo)
    _, ci_path = module.locate_data(repo)
    ci = pickle.load(ci_path.open("rb"))
    return [w for w in (normalize_word(str(x)) for x in ci["all_words"]) if w]


def partitions(words: Sequence[str]):
    a, b = int(len(words) * .60), int(len(words) * .80)
    return {"lm_train": list(words[:a]), "development": list(words[a:b]), "formal_reserve": list(words[b:])}


def build_inventory(words: Sequence[str], n_syllables=16, n_words=16) -> Inventory:
    chars = Counter("".join(words))
    characters = sorted(chars, key=lambda ch: (-chars[ch], ch))
    word_counts = Counter(words)
    whole_words = [w for w, _ in sorted(word_counts.items(), key=lambda x: (-x[1], x[0])) if len(w) >= 2][:n_words]
    substrings = Counter()
    for word in words:
        seen = set()
        for length in (2, 3, 4):
            for start in range(len(word) - length + 1):
                value = word[start:start + length]
                if value not in seen:
                    substrings[value] += 1
                    seen.add(value)
    forbidden = set(characters) | set(whole_words)
    syllables = []
    for value, count in sorted(substrings.items(), key=lambda x: (-x[1] * (len(x[0]) - 1), -len(x[0]), x[0])):
        if count >= 5 and value not in forbidden:
            syllables.append(value)
            forbidden.add(value)
            if len(syllables) == n_syllables:
                break
    units = [Unit(i, "letter", value) for i, value in enumerate(characters)]
    units += [Unit(len(units) + i, "syllable", value) for i, value in enumerate(syllables)]
    units += [Unit(len(units) + i, "word", value) for i, value in enumerate(whole_words)]
    boundary_id = len(units)
    units.append(Unit(boundary_id, "boundary", " "))
    records = [asdict(unit) for unit in units]
    return Inventory(tuple(units), boundary_id, canonical_hash(records))


def encode_word(word: str, inventory: Inventory, profile="balanced") -> list[int]:
    penalties = {
        "word_heavy": {"word": -1.2, "syllable": -.3, "letter": .5},
        "balanced": {"word": -.4, "syllable": -.15, "letter": 0.0},
        "letter_heavy": {"word": 1.0, "syllable": .4, "letter": 0.0},
    }[profile]
    matches = defaultdict(list)
    for unit in inventory.units:
        if unit.kind == "boundary":
            continue
        if unit.kind == "word":
            if word == unit.output:
                matches[0].append(unit)
            continue
        start = word.find(unit.output)
        while start >= 0:
            matches[start].append(unit)
            start = word.find(unit.output, start + 1)
    best = [None] * (len(word) + 1)
    best[len(word)] = (0.0, ())
    for position in range(len(word) - 1, -1, -1):
        choices = []
        for unit in matches[position]:
            end = position + len(unit.output)
            if end <= len(word) and best[end] is not None:
                cost = 1.0 + penalties[unit.kind] - .08 * max(0, len(unit.output) - 1)
                choices.append((cost + best[end][0], (unit.id,) + best[end][1]))
        if not choices:
            raise RuntimeError(f"cannot encode {word!r} at {position}")
        best[position] = min(choices, key=lambda x: (x[0], x[1]))
    return list(best[0][1])


def encode_words(words: Sequence[str], inventory: Inventory, profile="balanced") -> list[int]:
    result = []
    for word in words:
        result.extend(encode_word(word, inventory, profile))
        result.append(inventory.boundary_id)
    return result


def reconstruct(sequence: Sequence[int], inventory: Inventory) -> str:
    return "".join(inventory.units[int(x)].output for x in sequence).strip()


def build_ngram(sequence: Sequence[int], order: int, vocabulary: int) -> NGramModel:
    counts, contexts = Counter(), Counter()
    padded = [vocabulary] * (order - 1) + list(sequence)
    for i in range(order - 1, len(padded)):
        context = tuple(padded[i - order + 1:i])
        counts[context + (padded[i],)] += 1
        contexts[context] += 1
    return NGramModel(order, vocabulary, dict(counts), dict(contexts))


def transition_model(sequence: Sequence[int], n_units: int):
    pair = np.full((n_units, n_units), .25)
    uni = np.full(n_units, .25)
    for unit in sequence:
        uni[unit] += 1
    for a, b in zip(sequence, sequence[1:]):
        pair[a, b] += 1
    return pair / pair.sum(axis=1, keepdims=True), uni / uni.sum()


def allocate_homophones(sequence: Sequence[int], n_units: int, n_surface: int, profile: str):
    if n_surface < n_units:
        raise ValueError("surface alphabet smaller than unit inventory")
    counts = np.ones(n_units, dtype=int)
    frequencies = np.bincount(np.asarray(sequence), minlength=n_units).astype(float)
    weights = np.ones(n_units) if profile == "balanced" else np.sqrt(frequencies + 1)
    for _ in range(n_surface - n_units):
        counts[int(np.argmin(counts / weights))] += 1
    return counts


def make_key(seed: int, counts: Sequence[int]):
    mapping = [unit for unit, count in enumerate(counts) for _ in range(int(count))]
    random.Random(seed).shuffle(mapping)
    return tuple(mapping)


def encipher(sequence: Sequence[int], key: Sequence[int], seed: int, policy: str, test=False):
    rng = random.Random(seed)
    candidates = defaultdict(list)
    for cell, unit in enumerate(key):
        candidates[int(unit)].append(cell)
    context_weights = {
        (unit, position): [0.5 + 1.5 * rng.random() for _ in cells]
        for unit, cells in candidates.items() for position in ("FIRST", "MID", "LAST")
    }
    cycle = defaultdict(int)
    events = []
    cursor, line_id = 0, 0
    while cursor < len(sequence):
        length = rng.randint(7, 14)
        line = sequence[cursor:cursor + length]
        previous_unit = previous_cell = None
        for index, unit in enumerate(line):
            position = "FIRST" if index == 0 else "LAST" if index == len(line) - 1 else "MID"
            cells = candidates[int(unit)]
            if policy == "iid_uniform":
                cell = cells[rng.randrange(len(cells))]
            elif policy == "frequency_weighted":
                cell = rng.choices(cells, weights=context_weights[(int(unit), position)], k=1)[0]
            elif policy == "cyclic":
                cell = sorted(cells)[cycle[int(unit)] % len(cells)]
                cycle[int(unit)] += 1
            elif policy == "sticky_line_reset":
                if previous_unit == unit and previous_cell in cells and rng.random() < .75:
                    cell = previous_cell
                else:
                    cell = rng.choices(cells, weights=context_weights[(int(unit), position)], k=1)[0]
            else:
                raise ValueError(policy)
            events.append(Event(int(cell), int(unit), line_id, position, bool(test)))
            previous_unit, previous_cell = unit, cell
        cursor += length
        line_id += 1
    return events


def cell_statistics(events: Sequence[Event], n_surface: int):
    pair = np.zeros((n_surface, n_surface))
    uni = np.zeros(n_surface)
    previous_line = previous_cell = None
    for event in events:
        uni[event.cell] += 1
        if event.line == previous_line and previous_cell is not None:
            pair[previous_cell, event.cell] += 1
        previous_line, previous_cell = event.line, event.cell
    return pair, uni


def objective(pair, uni, mapping, transition, stationary):
    mapping = np.asarray(mapping, dtype=int)
    return float(
        (pair * np.log2(np.clip(transition[mapping[:, None], mapping[None, :]], 1e-300, None))).sum()
        + .25 * (uni * np.log2(np.clip(stationary[mapping], 1e-300, None))).sum()
    )


def initial_mapping(uni, counts, stationary):
    cells = list(np.argsort(-uni, kind="stable"))
    units = [int(unit) for unit in np.argsort(-stationary, kind="stable") for _ in range(int(counts[int(unit)]))]
    mapping = np.empty(len(cells), dtype=int)
    for cell, unit in zip(cells, units):
        mapping[int(cell)] = unit
    return mapping


def anneal(pair, uni, counts, transition, stationary, seed, steps, restarts):
    best = None
    for restart in range(restarts):
        rng = random.Random(seed ^ (restart * 0x9E3779B1))
        mapping = initial_mapping(uni, counts, stationary)
        for _ in range(len(mapping) * 2):
            a, b = rng.sample(range(len(mapping)), 2)
            mapping[a], mapping[b] = mapping[b], mapping[a]
        score = objective(pair, uni, mapping, transition, stationary)
        local = (score, mapping.copy())
        for step in range(steps):
            a, b = rng.sample(range(len(mapping)), 2)
            if mapping[a] == mapping[b]:
                continue
            mapping[a], mapping[b] = mapping[b], mapping[a]
            candidate = objective(pair, uni, mapping, transition, stationary)
            fraction = step / max(1, steps - 1)
            temperature = 2 * (.01 / 2) ** fraction
            delta = candidate - score
            if delta >= 0 or rng.random() < math.exp(max(-700, delta / temperature)):
                score = candidate
                if (score, tuple(mapping)) > (local[0], tuple(local[1])):
                    local = (score, mapping.copy())
            else:
                mapping[a], mapping[b] = mapping[b], mapping[a]
        if best is None or (local[0], tuple(local[1])) > (best[0], tuple(best[1])):
            best = local
    return tuple(int(x) for x in best[1]), float(best[0])


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def run_fixture(repo: Path, seed: int, output: Path, steps: int, restarts: int):
    words = load_words(repo)
    split = partitions(words)
    inventory = build_inventory(split["lm_train"])
    lm_sequence = encode_words(split["lm_train"], inventory)
    transition, stationary = transition_model(lm_sequence, len(inventory.units))
    models = {order: build_ngram(lm_sequence, order, len(inventory.units)) for order in (3, 5, 6)}
    plaintext = split["development"][:1800]
    train_units = encode_words(plaintext[:1400], inventory)
    test_units = encode_words(plaintext[1400:], inventory)
    counts = allocate_homophones(train_units, len(inventory.units), max(80, len(inventory.units) + 24), "unequal")
    key = make_key(seed, counts)
    train_events = encipher(train_units, key, seed ^ 0x1111, "frequency_weighted")
    test_events = encipher(test_units, key, seed ^ 0x2222, "frequency_weighted", True)
    pair, uni = cell_statistics(train_events, len(key))
    fitted, score = anneal(pair, uni, counts, transition, stationary, seed ^ 0x3333, steps, restarts)
    predicted_units = [fitted[event.cell] for event in test_events]
    oracle_units = [key[event.cell] for event in test_events]
    truth_text = reconstruct(test_units, inventory)
    oracle_text = reconstruct(oracle_units, inventory)
    predicted_text = reconstruct(predicted_units, inventory)
    if oracle_text != truth_text:
        raise RuntimeError("oracle reconstruction is not lossless")
    report = {
        "programme": "morpholocal-calibration-v0.3-historical-lattice-development",
        "formal": False,
        "seed": seed,
        "corpus": {
            "words": len(words),
            "partitions": {name: len(values) for name, values in split.items()},
            "sha256": hashlib.sha256("\n".join(words).encode()).hexdigest(),
        },
        "inventory": {
            "units": len(inventory.units),
            "letters": sum(unit.kind == "letter" for unit in inventory.units),
            "syllables": sum(unit.kind == "syllable" for unit in inventory.units),
            "words": sum(unit.kind == "word" for unit in inventory.units),
            "boundary_id": inventory.boundary_id,
            "hash": inventory.hash,
            "records": [asdict(unit) for unit in inventory.units],
        },
        "language_models": {
            str(order): {"ngrams": len(model.counts), "sample_bits": model.score(lm_sequence[:10000])}
            for order, model in models.items()
        },
        "trial": {
            "surface_symbols": len(key),
            "train_units": len(train_units),
            "test_units": len(test_units),
            "oracle_ter": levenshtein(oracle_text, truth_text) / max(1, len(truth_text)),
            "mapping_accuracy": sum(a == b for a, b in zip(fitted, key)) / len(key),
            "latent_unit_error": sum(a != b for a, b in zip(predicted_units, test_units)) / max(1, len(test_units)),
            "character_ter": levenshtein(predicted_text, truth_text) / max(1, len(truth_text)),
            "bigram_objective": score,
            "true_5gram_bits": models[5].score(test_units),
            "predicted_5gram_bits": models[5].score(predicted_units),
        },
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=base.DEFAULT_REPO)
    parser.add_argument("--seed", type=int, default=3030317)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_fixture(args.repo, args.seed, args.output, args.steps, args.restarts)


if __name__ == "__main__":
    main()
