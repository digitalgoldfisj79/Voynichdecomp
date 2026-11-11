#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

SEED = 20260714
ALPHABET = "abcdefghiklmnopqrstuxyzw"
POSITIONS = ["F1", "F2", "F3", "F4", "F5", "F6", "L1", "L2", "L3"]
N_FOLDS = 5
RESTARTS = 5
STEPS = 700
NULLS_PRIMARY = 100
NULLS_SENSITIVITY = 39
TIE_TOLERANCE = 0.05

LANGUAGE_SOURCES = {
    "latin": [
        "Corpora/Historical_texts/Secreta_Secretorum_LAT",
        "Corpora/Historical_texts/Picatrix",
    ],
    "italian": ["Corpora/Historical_texts/Rettorica"],
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quire_for_folio(folio: str) -> str:
    match = re.match(r"f(\d+)", folio)
    number = int(match.group(1)) if match else 999
    cuts = [8, 16, 22, 32, 38, 42, 50, 58, 66, 73, 84, 86, 90, 96, 103, 116]
    for i, cut in enumerate(cuts):
        if number <= cut:
            return f"Q{i+1}"
    return "UNK"


def normalise_plaintext(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower().replace("j", "i").replace("v", "u")
    return "".join(character for character in text if character in ALPHABET)


def load_vms(repository: Path, transcriber: str) -> list[dict]:
    data = json.loads((repository / "voynich_transcriptions_slim.json").read_text())
    sections = json.loads((repository / "voynich_section_map.json").read_text())["mapping"]
    rows = []
    folios = sorted(
        data["pages"],
        key=lambda x: (int(re.match(r"f(\d+)", x).group(1)) if re.match(r"f(\d+)", x) else 999, x),
    )
    for folio in folios:
        if sections.get(folio) == "text-only":
            continue
        line_numbers = sorted(
            data["pages"][folio],
            key=lambda x: int(x) if str(x).isdigit() else 9999,
        )
        for line_number in line_numbers:
            text = data["pages"][folio][line_number].get("t", {}).get(transcriber, "")
            tokens = [token.lower() for token in text.split()]
            tokens = [token for token in tokens if re.fullmatch("[a-z]+", token)]
            if tokens:
                rows.append(
                    {
                        "folio": folio,
                        "quire": quire_for_folio(folio),
                        "section": sections.get(folio, "unknown"),
                        "line": str(line_number),
                        "tokens": tokens,
                    }
                )
    return rows


def extract_position(rows: list[dict], position: str) -> list[dict]:
    from_left = position.startswith("F")
    offset = int(position[1:])
    output = []
    for row in rows:
        characters = []
        for token in row["tokens"]:
            if len(token) < offset:
                continue
            characters.append(token[offset - 1] if from_left else token[-offset])
        if len(characters) >= 3:
            output.append({**{k: row[k] for k in ("folio", "quire", "section", "line")}, "chars": characters})
    return output


def split_folds(stream: list[dict], n_folds: int = N_FOLDS) -> list[set[str]]:
    counts = Counter()
    for row in stream:
        counts[row["quire"]] += len(row["chars"])
    bins = [(0, set()) for _ in range(n_folds)]
    for quire, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        index = min(range(n_folds), key=lambda i: bins[i][0])
        total, members = bins[index]
        members = set(members)
        members.add(quire)
        bins[index] = (total + count, members)
    return [members for _, members in bins]


def stream_subset(stream: list[dict], quires: set[str], include: bool) -> list[list[str]]:
    return [row["chars"] for row in stream if (row["quire"] in quires) == include]


def counts_for_lines(lines: list[list[str]], symbols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    index = {symbol: i for i, symbol in enumerate(symbols)}
    unigram = np.zeros(len(symbols), dtype=float)
    bigram = np.zeros((len(symbols), len(symbols)), dtype=float)
    for line in lines:
        ids = [index[character] for character in line if character in index]
        for i in ids:
            unigram[i] += 1
        for a, b in zip(ids, ids[1:]):
            bigram[a, b] += 1
    return unigram, bigram


@dataclass
class LanguageModel:
    name: str
    alphabet: str
    logp1: np.ndarray
    logp2: np.ndarray
    logp3: np.ndarray

    @classmethod
    def build(cls, name: str, text: str, alpha: float = 0.1) -> "LanguageModel":
        alphabet = ALPHABET
        index = {character: i for i, character in enumerate(alphabet)}
        k = len(alphabet)
        c1 = np.full(k, alpha, dtype=float)
        c2 = np.full((k, k), alpha, dtype=float)
        c3 = np.full((k, k, k), alpha, dtype=float)
        ids = [index[character] for character in text if character in index]
        for i in ids:
            c1[i] += 1
        for a, b in zip(ids, ids[1:]):
            c2[a, b] += 1
        for a, b, c in zip(ids, ids[1:], ids[2:]):
            c3[a, b, c] += 1
        p1 = c1 / c1.sum()
        p2 = c2 / c2.sum(axis=1, keepdims=True)
        p3 = c3 / c3.sum(axis=2, keepdims=True)
        return cls(name, alphabet, np.log2(p1), np.log2(p2), np.log2(p3))

    def objective(self, unigram: np.ndarray, bigram: np.ndarray, permutation: np.ndarray) -> float:
        used = len(unigram)
        mapped = permutation[:used]
        score1 = float((unigram * self.logp1[mapped]).sum() / max(1.0, unigram.sum()))
        score2 = float(
            (bigram * self.logp2[np.ix_(mapped, mapped)]).sum() / max(1.0, bigram.sum())
        )
        return score1 + 2.0 * score2

    def trigram_bpc(self, lines: list[list[str]], symbols: list[str], mapping: dict[str, str]) -> float:
        target_index = {character: i for i, character in enumerate(self.alphabet)}
        total_logp, count = 0.0, 0
        for line in lines:
            mapped = [target_index[mapping[c]] for c in line if c in mapping]
            for a, b, c in zip(mapped, mapped[1:], mapped[2:]):
                total_logp += self.logp3[a, b, c]
                count += 1
        return -total_logp / count if count else float("inf")

    def unigram_bpc(self, lines: list[list[str]], mapping: dict[str, str]) -> float:
        target_index = {character: i for i, character in enumerate(self.alphabet)}
        total_logp, count = 0.0, 0
        for line in lines:
            for character in line:
                if character in mapping:
                    total_logp += self.logp1[target_index[mapping[character]]]
                    count += 1
        return -total_logp / count if count else float("inf")


def optimise_mapping(
    train_lines: list[list[str]], language: LanguageModel, seed: int,
    restarts: int = RESTARTS, steps: int = STEPS,
) -> tuple[dict[str, str], float]:
    symbols = sorted(set(character for line in train_lines for character in line))
    if len(symbols) > len(ALPHABET):
        raise ValueError(f"symbol inventory {len(symbols)} exceeds Trithemian alphabet")
    unigram, bigram = counts_for_lines(train_lines, symbols)
    symbol_order = list(np.argsort(-unigram))
    language_order = list(np.argsort(-np.exp2(language.logp1)))
    initial = np.array(language_order, dtype=int)
    # Rank-match the used assignments to observed symbol frequencies.
    ranked = np.empty(len(symbols), dtype=int)
    for rank, symbol_index in enumerate(symbol_order):
        ranked[symbol_index] = language_order[rank]
    initial[: len(symbols)] = ranked
    unused = [x for x in range(len(ALPHABET)) if x not in set(ranked.tolist())]
    initial[len(symbols):] = unused

    best_permutation = initial.copy()
    best_score = language.objective(unigram, bigram, best_permutation)
    rng = random.Random(seed)

    for restart in range(restarts):
        permutation = initial.copy()
        if restart:
            for _ in range(2 + restart):
                a, b = rng.sample(range(len(ALPHABET)), 2)
                permutation[a], permutation[b] = permutation[b], permutation[a]
        score = language.objective(unigram, bigram, permutation)
        local_best, local_score = permutation.copy(), score
        for step in range(steps):
            a = rng.randrange(len(ALPHABET))
            b = rng.randrange(len(ALPHABET) - 1)
            if b >= a:
                b += 1
            if a >= len(symbols) and b >= len(symbols):
                continue
            permutation[a], permutation[b] = permutation[b], permutation[a]
            proposed = language.objective(unigram, bigram, permutation)
            fraction = step / max(1, steps - 1)
            temperature = 0.20 * (0.002 / 0.20) ** fraction
            delta = proposed - score
            if delta >= 0 or rng.random() < math.exp(delta / max(temperature, 1e-9)):
                score = proposed
                if score > local_score:
                    local_best, local_score = permutation.copy(), score
            else:
                permutation[a], permutation[b] = permutation[b], permutation[a]
        if local_score > best_score:
            best_permutation, best_score = local_best, local_score

    mapping = {
        symbol: language.alphabet[int(best_permutation[i])]
        for i, symbol in enumerate(symbols)
    }
    return mapping, best_score


def evaluate_stream(
    stream: list[dict], language_models: dict[str, LanguageModel], seed: int,
    planted_mapping: dict[str, str] | None = None,
) -> dict:
    folds = split_folds(stream)
    fold_rows = []
    learned_mappings = []
    mapping_accuracies = []

    for fold_index, heldout_quires in enumerate(folds):
        train_lines = stream_subset(stream, heldout_quires, include=False)
        test_lines = stream_subset(stream, heldout_quires, include=True)
        candidates = []
        for language_index, language in enumerate(language_models.values()):
            mapping, objective = optimise_mapping(
                train_lines, language,
                seed + fold_index * 10000 + language_index * 1000,
            )
            train_bpc = language.trigram_bpc(train_lines, sorted(mapping), mapping)
            candidates.append((train_bpc, language.name, mapping, objective))
        candidates.sort(key=lambda row: (row[0], row[1]))
        _, language_name, mapping, objective = candidates[0]
        language = language_models[language_name]
        trigram_bpc = language.trigram_bpc(test_lines, sorted(mapping), mapping)
        unigram_bpc = language.unigram_bpc(test_lines, mapping)
        gain = unigram_bpc - trigram_bpc
        fold_rows.append(
            {
                "fold": fold_index,
                "heldout_quires": ",".join(sorted(heldout_quires)),
                "language": language_name,
                "trigram_bpc": trigram_bpc,
                "unigram_bpc": unigram_bpc,
                "gain": gain,
                "training_objective": objective,
                "mapping": mapping,
            }
        )
        learned_mappings.append((language_name, mapping))
        if planted_mapping:
            common = sorted(set(mapping) & set(planted_mapping))
            accuracy = (
                sum(mapping[symbol] == planted_mapping[symbol] for symbol in common) / len(common)
                if common else 0.0
            )
            mapping_accuracies.append(accuracy)

    agreements = []
    for (language_a, mapping_a), (language_b, mapping_b) in combinations(learned_mappings, 2):
        if language_a != language_b:
            agreements.append(0.0)
            continue
        common = sorted(set(mapping_a) & set(mapping_b))
        agreements.append(
            sum(mapping_a[symbol] == mapping_b[symbol] for symbol in common) / len(common)
            if common else 0.0
        )

    return {
        "median_gain": float(np.median([row["gain"] for row in fold_rows])),
        "median_trigram_bpc": float(np.median([row["trigram_bpc"] for row in fold_rows])),
        "median_unigram_bpc": float(np.median([row["unigram_bpc"] for row in fold_rows])),
        "mapping_agreement": float(np.median(agreements)) if agreements else 0.0,
        "mapping_accuracy": float(np.median(mapping_accuracies)) if mapping_accuracies else None,
        "folds": fold_rows,
        "n_characters": sum(len(row["chars"]) for row in stream),
        "inventory": sorted(set(character for row in stream for character in row["chars"])),
    }


def shuffle_within_quire(stream: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_quire = defaultdict(list)
    for index, row in enumerate(stream):
        by_quire[row["quire"]].append(index)
    output = [{**row, "chars": list(row["chars"])} for row in stream]
    for indices in by_quire.values():
        pool = [character for index in indices for character in output[index]["chars"]]
        rng.shuffle(pool)
        cursor = 0
        for index in indices:
            length = len(output[index]["chars"])
            output[index]["chars"] = pool[cursor:cursor + length]
            cursor += length
    return output


def synthetic_control(template: list[dict], plaintext: str, seed: int) -> tuple[list[dict], dict[str, str]]:
    needed = sum(len(row["chars"]) for row in template)
    repetitions = needed // len(plaintext) + 1
    source = (plaintext * repetitions)[:needed]
    rng = random.Random(seed)
    symbols = list(ALPHABET)
    rng.shuffle(symbols)
    plaintext_to_symbol = {letter: symbols[i] for i, letter in enumerate(ALPHABET)}
    planted = {symbol: letter for letter, symbol in plaintext_to_symbol.items()}
    output, cursor = [], 0
    for row in template:
        length = len(row["chars"])
        characters = [plaintext_to_symbol[letter] for letter in source[cursor:cursor + length]]
        cursor += length
        output.append({**{k: row[k] for k in ("folio", "quire", "section", "line")}, "chars": characters})
    return output, planted


def evaluate_positions(rows: list[dict], language_models: dict[str, LanguageModel], seed: int) -> dict[str, dict]:
    results = {}
    for position_index, position in enumerate(POSITIONS):
        stream = extract_position(rows, position)
        results[position] = evaluate_stream(stream, language_models, seed + position_index * 100000)
    return results


def one_null(rows: list[dict], language_models: dict[str, LanguageModel], seed: int) -> dict:
    gains = {}
    for position_index, position in enumerate(POSITIONS):
        stream = extract_position(rows, position)
        null_stream = shuffle_within_quire(stream, seed + position_index * 10000)
        gains[position] = evaluate_stream(
            null_stream, language_models, seed + position_index * 100000
        )["median_gain"]
    return {"seed": seed, "max_gain": max(gains.values()), "f2_gain": gains["F2"], "gains": gains}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    here = Path(__file__).resolve().parent
    repository = here.parents[1]
    output = Path("/tmp/piiv_fixed_position")
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir()
    shutil.copy2(here / "PROTOCOL.md", output / "PROTOCOL.md")
    shutil.copy2(here / "run.py", output / "run.py")

    hermes = output / "hermes"
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/hermesj/R_Voynich_Stats.git", str(hermes)],
        check=True,
    )

    language_texts = {}
    for language, paths in LANGUAGE_SOURCES.items():
        language_texts[language] = "".join(
            normalise_plaintext((hermes / path).read_text(errors="ignore")) for path in paths
        )
    language_models = {
        language: LanguageModel.build(language, text)
        for language, text in language_texts.items()
    }

    primary_rows = load_vms(repository, "ZLZI")
    sensitivity_rows = {name: load_vms(repository, name) for name in ["ZLZB", "TTIA"]}
    primary_results = evaluate_positions(primary_rows, language_models, SEED)

    # Positive calibration on the exact F2 line/quire template.
    f2_template = extract_position(primary_rows, "F2")
    positive_results = {}
    for i, language in enumerate(["latin", "italian"]):
        synthetic, planted = synthetic_control(f2_template, language_texts[language], SEED + 700000 + i)
        positive_results[language] = evaluate_stream(
            synthetic, language_models, SEED + 710000 + i, planted_mapping=planted
        )

    positive_pass = all(
        row["mapping_accuracy"] is not None
        and row["mapping_accuracy"] >= 0.70
        and row["median_gain"] > 0
        for row in positive_results.values()
    )

    # Full family-wise null for the primary transcription.
    null_seeds = [SEED + 1000000 + i for i in range(NULLS_PRIMARY)]
    primary_nulls = Parallel(n_jobs=-1, verbose=10)(
        delayed(one_null)(primary_rows, language_models, seed) for seed in null_seeds
    )
    observed_f2_gain = primary_results["F2"]["median_gain"]
    family_p = (1 + sum(row["max_gain"] >= observed_f2_gain for row in primary_nulls)) / (1 + len(primary_nulls))
    nominal_p = (1 + sum(row["f2_gain"] >= observed_f2_gain for row in primary_nulls)) / (1 + len(primary_nulls))

    sensitivity_results = {}
    for transcriber_index, (transcriber, rows) in enumerate(sensitivity_rows.items()):
        observed = evaluate_positions(rows, language_models, SEED + 2000000 + transcriber_index * 100000)
        seeds = [SEED + 3000000 + transcriber_index * 100000 + i for i in range(NULLS_SENSITIVITY)]
        nulls = Parallel(n_jobs=-1, verbose=5)(
            delayed(one_null)(rows, language_models, seed) for seed in seeds
        )
        f2_gain = observed["F2"]["median_gain"]
        p = (1 + sum(row["f2_gain"] >= f2_gain for row in nulls)) / (1 + len(nulls))
        sensitivity_results[transcriber] = {"positions": observed, "f2_nominal_p": p, "nulls": nulls}

    best_gain = max(row["median_gain"] for row in primary_results.values())
    primary_best_or_tied = primary_results["F2"]["median_gain"] >= best_gain - TIE_TOLERANCE
    sensitivity_pass = True
    for transcriber, result in sensitivity_results.items():
        best = max(row["median_gain"] for row in result["positions"].values())
        f2_tied = result["positions"]["F2"]["median_gain"] >= best - TIE_TOLERANCE
        if not (f2_tied and result["f2_nominal_p"] <= 0.05):
            sensitivity_pass = False

    weaker_positive_gain = min(row["median_gain"] for row in positive_results.values())
    gain_close_enough = observed_f2_gain >= weaker_positive_gain - 0.50
    pass_conditions = {
        "positive_calibration": positive_pass,
        "f2_best_or_tied": primary_best_or_tied,
        "familywise_p_le_0_05": family_p <= 0.05,
        "mapping_agreement_ge_0_70": primary_results["F2"]["mapping_agreement"] >= 0.70,
        "alternate_transcriptions": sensitivity_pass,
        "gain_within_0_50_of_positive": gain_close_enough,
    }

    if not positive_pass:
        verdict = "UNRESOLVED_CALIBRATION_FAILED"
    elif all(pass_conditions.values()):
        verdict = "PASS_EXACT_FIXED_POSITION_PAYLOAD"
    else:
        verdict = "FAIL_EXACT_FIXED_POSITION_PAYLOAD"

    position_rows = []
    for position, row in primary_results.items():
        position_rows.append(
            {
                "transcriber": "ZLZI",
                "position": position,
                "median_gain": row["median_gain"],
                "median_trigram_bpc": row["median_trigram_bpc"],
                "median_unigram_bpc": row["median_unigram_bpc"],
                "mapping_agreement": row["mapping_agreement"],
                "n_characters": row["n_characters"],
                "inventory_size": len(row["inventory"]),
            }
        )
    for transcriber, result in sensitivity_results.items():
        for position, row in result["positions"].items():
            position_rows.append(
                {
                    "transcriber": transcriber,
                    "position": position,
                    "median_gain": row["median_gain"],
                    "median_trigram_bpc": row["median_trigram_bpc"],
                    "median_unigram_bpc": row["median_unigram_bpc"],
                    "mapping_agreement": row["mapping_agreement"],
                    "n_characters": row["n_characters"],
                    "inventory_size": len(row["inventory"]),
                }
            )
    write_csv(output / "POSITION_RESULTS.csv", position_rows)
    write_csv(
        output / "PRIMARY_NULLS.csv",
        [{"seed": row["seed"], "max_gain": row["max_gain"], "f2_gain": row["f2_gain"]} for row in primary_nulls],
    )

    provenance = {
        "seed": SEED,
        "protocol_sha256": sha256(output / "PROTOCOL.md"),
        "voynichdecomp_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip(),
        "hermes_commit": subprocess.check_output(
            ["git", "-C", str(hermes), "rev-parse", "HEAD"], text=True
        ).strip(),
        "language_lengths": {language: len(text) for language, text in language_texts.items()},
        "restarts": RESTARTS,
        "steps": STEPS,
        "primary_nulls": NULLS_PRIMARY,
        "sensitivity_nulls": NULLS_SENSITIVITY,
        "davis_labels_used": False,
        "voynich_language_model_fit": False,
    }
    (output / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    calibration = {"positive_controls": positive_results, "positive_pass": positive_pass}
    (output / "CALIBRATION.json").write_text(json.dumps(calibration, indent=2))

    final = {
        "status": "COMPLETE",
        "formal_verdict": verdict,
        "primary_transcription": "ZLZI",
        "f2_gain": observed_f2_gain,
        "f2_familywise_p": family_p,
        "f2_nominal_p": nominal_p,
        "f2_mapping_agreement": primary_results["F2"]["mapping_agreement"],
        "best_position": max(primary_results, key=lambda p: primary_results[p]["median_gain"]),
        "best_gain": best_gain,
        "positive_controls": {
            language: {
                "gain": row["median_gain"],
                "mapping_accuracy": row["mapping_accuracy"],
                "mapping_agreement": row["mapping_agreement"],
            }
            for language, row in positive_results.items()
        },
        "pass_conditions": pass_conditions,
        "sensitivity": {
            transcriber: {
                "f2_gain": result["positions"]["F2"]["median_gain"],
                "f2_nominal_p": result["f2_nominal_p"],
                "best_position": max(result["positions"], key=lambda p: result["positions"][p]["median_gain"]),
            }
            for transcriber, result in sensitivity_results.items()
        },
        "scope": "Global one-glyph-per-word fixed-position payload under a monoalphabetic substitution; F2 is the literal Book IV prediction.",
        "non_exclusions": [
            "changing payload positions",
            "polyalphabetic or scribe-specific mappings",
            "multi-glyph payload units",
            "insertions, deletions, nulls or sparse messages",
            "an independent post-encryption surface realiser",
        ],
    }
    (output / "FINAL_RESULT.json").write_text(json.dumps(final, indent=2))

    report = [
        "# PIIV-FIXED-POSITION results",
        "",
        f"**Formal verdict: {verdict}**",
        "",
        f"- F2 held-out language gain: **{observed_f2_gain:.4f} bits/character**",
        f"- F2 family-wise permutation p: **{family_p:.4f}**",
        f"- F2 nominal permutation p: **{nominal_p:.4f}**",
        f"- F2 mapping agreement: **{primary_results['F2']['mapping_agreement']:.3f}**",
        f"- Best observed position: **{final['best_position']}** ({best_gain:.4f})",
        "",
        "## Position table",
        "",
        "| Position | Gain | Trigram BPC | Mapping agreement | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for position in POSITIONS:
        row = primary_results[position]
        report.append(
            f"| {position} | {row['median_gain']:.4f} | {row['median_trigram_bpc']:.4f} | "
            f"{row['mapping_agreement']:.3f} | {row['n_characters']} |"
        )
    report.extend([
        "",
        "## Decision conditions",
        "",
    ])
    for name, passed in pass_conditions.items():
        report.append(f"- `{name}`: **{passed}**")
    report.extend([
        "",
        "The result is bounded to a globally fixed, one-character-per-token payload position under one monoalphabetic mapping. It does not test changing positions, polyalphabetic keys, multi-character payload units, sparse payloads, or a separate surface-realisation layer.",
    ])
    (output / "EXECUTIVE_SUMMARY.md").write_text("\n".join(report) + "\n")

    shutil.rmtree(hermes)
    checksums = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256(path)}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    archive = shutil.make_archive("/tmp/PIIV_FIXED_POSITION_2026-07-14", "zip", "/tmp", output.name)
    print(json.dumps({"final": final, "archive": archive}, indent=2), flush=True)


if __name__ == "__main__":
    main()
