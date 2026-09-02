#!/usr/bin/env python3
"""Fractionation-composition falsification programme v0.1.

Primary question: can a symbol-renaming-invariant detector distinguish bounded
coordinate/component fractionation with regrouping from slot-like and other
expanded controls after matching token lengths, symbol counts, and edge-position
restrictions? Voynich is not touched by this script; it is a synthetic gate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import statistics
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

LANGS = ("en", "de", "fi", "tr", "he", "ar")
BLOCKS = (1, 2, 3, 4, 5, 6, 7, 8)
PRIMARY_POSITIVE = ("frac_pair", "frac_block_token", "frac_block_stream", "frac_homophonic")
PRIMARY_CONTROL = ("slot_control", "expanded_mono", "expanded_transposition", "markov_control")


def stable_seed(*parts: object) -> int:
    blob = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    out: list[str] = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in ("L", "M", "N"):
            out.append(ch)
        elif ch in ("'", "’"):
            out.append("'")
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Voynichdecomp-fractionation-v01"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def parse_conllu_texts(raw: bytes) -> list[str]:
    texts: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("# text = "):
            value = normalize_text(line[len("# text = "):])
            if value:
                texts.append(value)
    return texts


@dataclass
class Language:
    iso: str
    alphabet: tuple[str, ...]
    char_to_id: dict[str, int]
    train_words: list[list[int]]
    dev_words: list[list[int]]
    test_words: list[list[int]]


def load_languages(manifest_path: Path, cache_dir: Path) -> dict[str, Language]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_texts: dict[str, dict[str, list[str]]] = {}
    train_counts: dict[str, collections.Counter[str]] = {}

    for corpus in manifest["corpora"]:
        iso = corpus["iso"]
        raw_texts[iso] = {}
        base = f"https://raw.githubusercontent.com/{corpus['repo']}/{corpus['commit']}"
        for split in ("train", "dev", "test"):
            filename = f"{corpus['prefix']}-ud-{split}.conllu"
            path = cache_dir / filename
            if path.exists():
                raw = path.read_bytes()
            else:
                raw = fetch(f"{base}/{filename}")
                path.write_bytes(raw)
            actual = hashlib.sha256(raw).hexdigest()
            expected = corpus["files"][split]["sha256"]
            if actual != expected:
                raise RuntimeError(f"hash mismatch {iso}/{split}: {actual} != {expected}")
            raw_texts[iso][split] = parse_conllu_texts(raw)
        counts: collections.Counter[str] = collections.Counter()
        for text in raw_texts[iso]["train"]:
            counts.update(ch for ch in text if ch != " ")
        train_counts[iso] = counts

    languages: dict[str, Language] = {}
    for corpus in manifest["corpora"]:
        iso = corpus["iso"]
        alphabet = tuple(ch for ch, n in sorted(train_counts[iso].items(), key=lambda kv: (-kv[1], kv[0])) if n >= 2)
        char_to_id = {ch: i for i, ch in enumerate(alphabet)}

        def encode_words(split: str) -> list[list[int]]:
            words: list[list[int]] = []
            for text in raw_texts[iso][split]:
                for word in text.split():
                    enc = [char_to_id[ch] for ch in word if ch in char_to_id]
                    if len(enc) >= 2:
                        words.append(enc)
            return words

        languages[iso] = Language(
            iso=iso,
            alphabet=alphabet,
            char_to_id=char_to_id,
            train_words=encode_words("train"),
            dev_words=encode_words("dev"),
            test_words=encode_words("test"),
        )
    return languages


def sample_word_chunk(words: Sequence[Sequence[int]], target_letters: int, rng: random.Random) -> list[list[int]]:
    if not words:
        raise RuntimeError("no words")
    start = rng.randrange(len(words))
    out: list[list[int]] = []
    total = 0
    i = start
    while total < target_letters:
        word = list(words[i % len(words)])
        if word:
            out.append(word)
            total += len(word)
        i += 1
        if len(out) > 5000:
            break
    return out


def permuted_labels(rng: random.Random, n: int, offset: int = 0) -> list[int]:
    vals = list(range(offset, offset + n))
    rng.shuffle(vals)
    return vals


@dataclass
class CoordCodec:
    columns: int
    rows: int
    row_choices: list[list[int]]
    col_choices: list[list[int]]
    max_symbol: int


def build_coord_codec(a: int, rng: random.Random, homophonic: bool = False) -> CoordCodec:
    columns = int(math.ceil(math.sqrt(a)))
    rows = int(math.ceil(a / columns))
    if not homophonic:
        row_labels = permuted_labels(rng, rows, 0)
        col_labels = permuted_labels(rng, columns, rows)
        return CoordCodec(columns, rows, [[x] for x in row_labels], [[x] for x in col_labels], rows + columns - 1)

    row_choices: list[list[int]] = []
    col_choices: list[list[int]] = []
    cursor = 0
    for _ in range(rows):
        k = 2 if rng.random() < 0.65 else 1
        row_choices.append(list(range(cursor, cursor + k)))
        cursor += k
    for _ in range(columns):
        k = 2 if rng.random() < 0.65 else 1
        col_choices.append(list(range(cursor, cursor + k)))
        cursor += k
    overlap_pairs = max(1, min(rows, columns) // 3)
    for _ in range(overlap_pairs):
        r = rng.randrange(rows)
        c = rng.randrange(columns)
        shared = rng.choice(row_choices[r])
        if shared not in col_choices[c]:
            col_choices[c].append(shared)
    all_labels = sorted({x for xs in row_choices + col_choices for x in xs})
    relabel = permuted_labels(rng, len(all_labels), 0)
    mapping = dict(zip(all_labels, relabel))
    row_choices = [[mapping[x] for x in xs] for xs in row_choices]
    col_choices = [[mapping[x] for x in xs] for xs in col_choices]
    return CoordCodec(columns, rows, row_choices, col_choices, len(all_labels) - 1)


def components(word: Sequence[int], codec: CoordCodec, rng: random.Random) -> tuple[list[int], list[int]]:
    rs: list[int] = []
    cs: list[int] = []
    for x in word:
        r, c = divmod(int(x), codec.columns)
        rs.append(rng.choice(codec.row_choices[r]))
        cs.append(rng.choice(codec.col_choices[c]))
    return rs, cs


def interleave_blocks(rs: Sequence[int], cs: Sequence[int], block: int) -> list[int]:
    out: list[int] = []
    for start in range(0, len(rs), block):
        out.extend(rs[start:start + block])
        out.extend(cs[start:start + block])
    return out


def encrypt_fractionated(words: Sequence[Sequence[int]], a: int, rng: random.Random, family: str, block: int) -> list[list[int]]:
    hom = family == "frac_homophonic"
    codec = build_coord_codec(a, rng, homophonic=hom)
    if family == "frac_pair":
        block = 1
    if family in ("frac_pair", "frac_block_token", "frac_homophonic"):
        out: list[list[int]] = []
        for word in words:
            rs, cs = components(word, codec, rng)
            out.append(interleave_blocks(rs, cs, block))
        return out
    if family == "frac_block_stream":
        flat_plain = [x for word in words for x in word]
        rs, cs = components(flat_plain, codec, rng)
        flat_cipher = interleave_blocks(rs, cs, block)
        out: list[list[int]] = []
        cursor = 0
        for word in words:
            n = 2 * len(word)
            out.append(flat_cipher[cursor:cursor + n])
            cursor += n
        if cursor != len(flat_cipher):
            raise AssertionError("stream repartition mismatch")
        return out
    raise ValueError(family)


def expanded_mono(words: Sequence[Sequence[int]], a: int, rng: random.Random, transpose: bool = False) -> list[list[int]]:
    choices: list[list[int]] = []
    cursor = 0
    for _ in range(a):
        k = 2 if rng.random() < 0.7 else 1
        choices.append(list(range(cursor, cursor + k)))
        cursor += k
    labels = permuted_labels(rng, cursor)
    remap = {i: labels[i] for i in range(cursor)}
    choices = [[remap[x] for x in xs] for xs in choices]
    out: list[list[int]] = []
    for word in words:
        token: list[int] = []
        for x in word:
            token.extend((rng.choice(choices[x]), rng.choice(choices[x])))
        if transpose and len(token) >= 4:
            block = rng.choice((4, 6, 8, 10))
            perm = list(range(block))
            rng.shuffle(perm)
            trans: list[int] = []
            for start in range(0, len(token), block):
                piece = token[start:start + block]
                if len(piece) == block:
                    trans.extend(piece[i] for i in perm)
                else:
                    trans.extend(piece)
            token = trans
        out.append(token)
    return out


def slot_control(words: Sequence[Sequence[int]], rng: random.Random, alphabet_size: int = 16) -> list[list[int]]:
    labels = permuted_labels(rng, alphabet_size)
    prefix = labels[:6]
    core = labels[4:12]
    suffix = labels[9:]
    templates = [(0.75, 0.18), (0.55, 0.35), (0.35, 0.55), (0.18, 0.75)]
    out: list[list[int]] = []
    prev_core = rng.choice(core)
    for word in words:
        n = 2 * len(word)
        token: list[int] = []
        t = rng.randrange(len(templates))
        for pos in range(n):
            frac = pos / max(1, n - 1)
            if frac < templates[t][0] / 2:
                pool = prefix
            elif frac > 1 - templates[t][1] / 2:
                pool = suffix
            else:
                pool = core
            if pool is core and rng.random() < 0.28:
                symbol = prev_core
            else:
                symbol = rng.choice(pool)
            token.append(symbol)
            if symbol in core:
                prev_core = symbol
        out.append(token)
    return out


def markov_control(words: Sequence[Sequence[int]], rng: random.Random, alphabet_size: int = 16) -> list[list[int]]:
    trans: list[list[float]] = []
    for i in range(alphabet_size):
        weights = [0.2 + rng.random() for _ in range(alphabet_size)]
        weights[i] += 2.5
        s = sum(weights)
        trans.append([x / s for x in weights])

    def choose(weights: Sequence[float]) -> int:
        u = rng.random()
        c = 0.0
        for i, w in enumerate(weights):
            c += w
            if u <= c:
                return i
        return len(weights) - 1

    out: list[list[int]] = []
    state = rng.randrange(alphabet_size)
    for word in words:
        token: list[int] = []
        for _ in range(2 * len(word)):
            state = choose(trans[state])
            token.append(state)
        out.append(token)
    return out


def entropy_from_counts(counts: collections.Counter[int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        p = n / total
        h -= p * math.log2(p)
    return h


def normalized_symbol_role_mi(symbols: Sequence[int], roles: Sequence[int]) -> float:
    if len(symbols) != len(roles) or not symbols:
        return 0.0
    joint: collections.Counter[tuple[int, int]] = collections.Counter(zip(symbols, roles))
    sc = collections.Counter(symbols)
    rc = collections.Counter(roles)
    total = len(symbols)
    hr = entropy_from_counts(rc)
    if hr <= 1e-12:
        return 0.0
    mi = 0.0
    for (s, r), n in joint.items():
        p = n / total
        ps = sc[s] / total
        pr = rc[r] / total
        mi += p * math.log2(p / (ps * pr))
    return max(0.0, mi / hr)


def flatten(tokens: Sequence[Sequence[int]]) -> list[int]:
    return [x for token in tokens for x in token]


def role_score_token(tokens: Sequence[Sequence[int]], block: int) -> float:
    symbols: list[int] = []
    roles: list[int] = []
    for token in tokens:
        for pos, symbol in enumerate(token):
            symbols.append(symbol)
            roles.append((pos // block) & 1)
    return normalized_symbol_role_mi(symbols, roles)


def role_score_stream(tokens: Sequence[Sequence[int]], block: int) -> float:
    symbols = flatten(tokens)
    roles = [((pos // block) & 1) for pos in range(len(symbols))]
    return normalized_symbol_role_mi(symbols, roles)


def phase_peak(tokens: Sequence[Sequence[int]]) -> tuple[float, str, int, dict[str, float]]:
    scores: dict[str, float] = {}
    best = (-1.0, "", -1)
    for b in BLOCKS:
        for mode, fn in (("token", role_score_token), ("stream", role_score_stream)):
            s = fn(tokens, b)
            scores[f"{mode}_b{b}"] = s
            if s > best[0]:
                best = (s, mode, b)
    return best[0], best[1], best[2], scores


def edge_stratum(n: int, pos: int, edge: int = 2) -> str:
    if pos < edge:
        return f"L{pos}"
    r = n - 1 - pos
    if r < edge:
        return f"R{r}"
    frac = pos / max(1, n - 1)
    if frac < 1 / 3:
        return "I1"
    if frac > 2 / 3:
        return "I3"
    return "I2"


def matched_shuffle(tokens: Sequence[Sequence[int]], rng: random.Random) -> list[list[int]]:
    pools: dict[str, list[int]] = collections.defaultdict(list)
    locations: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for ti, token in enumerate(tokens):
        for pi, symbol in enumerate(token):
            key = edge_stratum(len(token), pi)
            pools[key].append(symbol)
            locations[key].append((ti, pi))
    out = [list(token) for token in tokens]
    for key, vals in pools.items():
        shuffled = list(vals)
        rng.shuffle(shuffled)
        for (ti, pi), symbol in zip(locations[key], shuffled):
            out[ti][pi] = symbol
    return out


def evaluate_sample(tokens: Sequence[Sequence[int]], rng: random.Random, null_reps: int) -> dict[str, object]:
    observed, mode, block, all_scores = phase_peak(tokens)
    null_scores: list[float] = []
    for _ in range(null_reps):
        null = matched_shuffle(tokens, rng)
        null_scores.append(phase_peak(null)[0])
    mean_null = statistics.fmean(null_scores)
    sd_null = statistics.stdev(null_scores) if len(null_scores) > 1 else 0.0
    residual = observed - mean_null
    z = residual / sd_null if sd_null > 1e-12 else (999.0 if residual > 0 else 0.0)
    empirical_p = (1 + sum(x >= observed for x in null_scores)) / (1 + len(null_scores))
    return {
        "observed": observed,
        "best_mode": mode,
        "best_block": block,
        "null_mean": mean_null,
        "null_sd": sd_null,
        "residual": residual,
        "z": z,
        "empirical_p": empirical_p,
        "scores": all_scores,
    }


def make_family(words: Sequence[Sequence[int]], a: int, family: str, rng: random.Random, block: int) -> list[list[int]]:
    if family in PRIMARY_POSITIVE:
        return encrypt_fractionated(words, a, rng, family, block)
    if family == "slot_control":
        return slot_control(words, rng)
    if family == "expanded_mono":
        return expanded_mono(words, a, rng, transpose=False)
    if family == "expanded_transposition":
        return expanded_mono(words, a, rng, transpose=True)
    if family == "markov_control":
        return markov_control(words, rng)
    raise ValueError(family)


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_family: dict[str, object] = {}
    for family in PRIMARY_POSITIVE + PRIMARY_CONTROL:
        rr = [r for r in rows if r["family"] == family]
        zs = [float(r["z"]) for r in rr]
        residuals = [float(r["residual"]) for r in rr]
        null_sds = [float(r["null_sd"]) for r in rr]
        by_family[family] = {
            "n": len(rr),
            "mean_z": statistics.fmean(zs),
            "median_z": statistics.median(zs),
            "min_z": min(zs),
            "max_z": max(zs),
            "mean_residual": statistics.fmean(residuals),
            "mean_null_sd": statistics.fmean(null_sds),
            "rate_z_ge_3": statistics.fmean(float(z >= 3.0) for z in zs),
            "rate_p_le_0_01": statistics.fmean(float(float(r["empirical_p"]) <= 0.01) for r in rr),
        }
    pos = [float(r["z"]) for r in rows if r["family"] in PRIMARY_POSITIVE]
    neg = [float(r["z"]) for r in rows if r["family"] in PRIMARY_CONTROL]
    pos_rate = statistics.fmean(float(z >= 3.0) for z in pos)
    neg_rate = statistics.fmean(float(z >= 3.0) for z in neg)
    pos_mean = statistics.fmean(pos)
    neg_mean = statistics.fmean(neg)
    neg_sd = statistics.stdev(neg) if len(neg) > 1 else 0.0
    separation = pos_mean - neg_mean
    separation_in_control_sd = separation / neg_sd if neg_sd > 1e-12 else 999.0
    gate = pos_rate >= 0.90 and neg_rate <= 0.10 and separation_in_control_sd >= 2.0
    return {
        "by_family": by_family,
        "aggregate": {
            "positive_rate_z_ge_3": pos_rate,
            "control_rate_z_ge_3": neg_rate,
            "positive_mean_z": pos_mean,
            "control_mean_z": neg_mean,
            "control_z_sd": neg_sd,
            "mean_z_separation": separation,
            "separation_in_control_sd": separation_in_control_sd,
        },
        "gate": {
            "criteria": {
                "positive_rate_z_ge_3_at_least": 0.90,
                "control_rate_z_ge_3_at_most": 0.10,
                "mean_separation_at_least_control_sd": 2.0,
            },
            "decision": "GO_TO_VOYNICH" if gate else "STOP_NON_IDENTIFIABLE",
        },
    }


def run(languages: dict[str, Language], reps: int, target_letters: int, null_reps: int, split: str) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    families = PRIMARY_POSITIVE + PRIMARY_CONTROL
    for iso in LANGS:
        lang = languages[iso]
        words_source = lang.dev_words if split == "dev" else lang.test_words
        for rep in range(reps):
            chunk_rng = random.Random(stable_seed("frac-v01", split, iso, rep, "chunk"))
            words = sample_word_chunk(words_source, target_letters, chunk_rng)
            for family in families:
                block = 1 if family == "frac_pair" else 2 + (stable_seed("frac-v01", split, iso, rep, family, "block") % 7)
                rng = random.Random(stable_seed("frac-v01", split, iso, rep, family, "cipher"))
                tokens = make_family(words, len(lang.alphabet), family, rng, int(block))
                eval_rng = random.Random(stable_seed("frac-v01", split, iso, rep, family, "null"))
                ev = evaluate_sample(tokens, eval_rng, null_reps)
                rows.append({
                    "split": split,
                    "iso": iso,
                    "rep": rep,
                    "family": family,
                    "block": int(block),
                    "plain_words": len(words),
                    "plain_letters": sum(len(w) for w in words),
                    "cipher_symbols": sum(len(t) for t in tokens),
                    "cipher_alphabet": len(set(flatten(tokens))),
                    **ev,
                })
    return {"rows": rows, "summary": summarize(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--target-letters", type=int, default=600)
    ap.add_argument("--null-reps", type=int, default=99)
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    args = ap.parse_args()

    manifest = args.repo / "experiments/recoverability_frontier_v0_5/corpus_manifest_v050.json"
    languages = load_languages(manifest, args.repo / ".cache/ud-v050")
    result = run(languages, args.reps, args.target_letters, args.null_reps, args.split)
    payload = {
        "programme": "fractionation-composition-v0.1-synthetic-gate",
        "split": args.split,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "parameters": {
            "languages": list(LANGS),
            "blocks": list(BLOCKS),
            "reps": args.reps,
            "target_letters": args.target_letters,
            "null_reps": args.null_reps,
            "edge_null": "L0,L1,R0,R1 + interior thirds; exact within-stratum symbol permutation",
        },
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("FRACTIONATION_V01_GATE", json.dumps(result["summary"]["gate"], sort_keys=True))
    print("FRACTIONATION_V01_AGG", json.dumps(result["summary"]["aggregate"], sort_keys=True))
    for family, row in result["summary"]["by_family"].items():
        print("FRACTIONATION_V01_FAMILY", family, json.dumps(row, sort_keys=True))
    print("FRACTIONATION_V01_SHA256", hashlib.sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
