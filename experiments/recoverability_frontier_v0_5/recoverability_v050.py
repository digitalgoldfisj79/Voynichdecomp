#!/usr/bin/env python3
"""Recoverability frontier v0.5.0 core corpus, cipher DSL and oracle runner."""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Sequence

FAMILIES = (
    "mono",
    "homophonic",
    "null_homophonic",
    "polyalphabetic",
    "feedback",
    "nomenclator",
    "transposition",
    "fractionated",
)
CONTROL_FAMILIES = ("markov2", "motif", "copy_mutate", "slot")
LENGTHS = (96, 192, 384)
NOISE_LEVELS = (0.0, 0.01, 0.03)
SPLITS = ("train", "dev", "test")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_seed(*parts: object) -> int:
    blob = "|".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    out: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category[0] in ("L", "M", "N"):
            out.append(ch)
        elif ch in ("'", "’"):
            out.append("'")
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Voynichdecomp-v050"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


def parse_conllu_texts(raw: bytes) -> list[str]:
    texts: list[str] = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("# text = "):
            value = normalize_text(line[len("# text = "):])
            if value:
                texts.append(value)
    return texts


@dataclasses.dataclass
class LanguageData:
    iso: str
    language: str
    alphabet: tuple[str, ...]
    char_to_id: dict[str, int]
    probabilities: tuple[float, ...]
    texts: dict[str, list[str]]
    encoded_sentences: dict[str, list[list[int]]]
    train_stream: list[int]
    train_words: list[tuple[int, ...]]
    top_words: list[tuple[int, ...]]

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in normalize_text(text) if ch in self.char_to_id]

    def decode(self, values: Sequence[int]) -> str:
        return "".join(self.alphabet[x] if 0 <= int(x) < len(self.alphabet) else "�" for x in values)


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_languages(manifest_path: Path, cache_dir: Path) -> dict[str, LanguageData]:
    manifest = load_manifest(manifest_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts_by_iso: dict[str, dict[str, list[str]]] = {}

    for corpus in manifest["corpora"]:
        iso = corpus["iso"]
        texts_by_iso[iso] = {}
        base = f"https://raw.githubusercontent.com/{corpus['repo']}/{corpus['commit']}"
        for split in SPLITS:
            filename = f"{corpus['prefix']}-ud-{split}.conllu"
            path = cache_dir / filename
            if path.exists():
                raw = path.read_bytes()
            else:
                raw = fetch(f"{base}/{filename}")
                path.write_bytes(raw)
            expected = corpus["files"][split]["sha256"]
            actual = sha256_bytes(raw)
            if actual != expected:
                raise RuntimeError(f"corpus hash mismatch {iso}/{split}: {actual} != {expected}")
            texts_by_iso[iso][split] = parse_conllu_texts(raw)

    all_train_chars = collections.Counter()
    per_iso_counts: dict[str, collections.Counter[str]] = {}
    for corpus in manifest["corpora"]:
        iso = corpus["iso"]
        counts: collections.Counter[str] = collections.Counter()
        for text in texts_by_iso[iso]["train"]:
            counts.update(text)
        per_iso_counts[iso] = counts
        all_train_chars.update(counts)

    global_order = [
        ch for ch, count in sorted(all_train_chars.items(), key=lambda x: (-x[1], x[0]))
        if count >= 2
    ]
    languages: dict[str, LanguageData] = {}
    for corpus in manifest["corpora"]:
        iso = corpus["iso"]
        local_chars = {ch for ch, count in per_iso_counts[iso].items() if count >= 2}
        alphabet = tuple(ch for ch in global_order if ch in local_chars)
        if " " not in alphabet:
            alphabet = (" ",) + alphabet
        char_to_id = {ch: i for i, ch in enumerate(alphabet)}

        encoded: dict[str, list[list[int]]] = {}
        for split in SPLITS:
            encoded[split] = [
                [char_to_id[ch] for ch in text if ch in char_to_id]
                for text in texts_by_iso[iso][split]
            ]
            encoded[split] = [x for x in encoded[split] if x]

        train_stream: list[int] = []
        space = char_to_id.get(" ")
        for sentence in encoded["train"]:
            if train_stream and space is not None:
                train_stream.append(space)
            train_stream.extend(sentence)

        counts = collections.Counter(train_stream)
        total = sum(counts.values())
        probabilities = tuple(
            (counts[i] + 0.5) / (total + 0.5 * len(alphabet))
            for i in range(len(alphabet))
        )

        words: list[tuple[int, ...]] = []
        for text in texts_by_iso[iso]["train"]:
            for word in text.split():
                values = tuple(char_to_id[ch] for ch in word if ch in char_to_id)
                if values:
                    words.append(values)
        word_counts = collections.Counter(words)
        top_words = [word for word, _ in word_counts.most_common(128)]

        languages[iso] = LanguageData(
            iso=iso,
            language=corpus["language"],
            alphabet=alphabet,
            char_to_id=char_to_id,
            probabilities=probabilities,
            texts=texts_by_iso[iso],
            encoded_sentences=encoded,
            train_stream=train_stream,
            train_words=words,
            top_words=top_words,
        )
    return languages


def source_chunks(language: LanguageData, split: str, length: int) -> list[list[int]]:
    """Make deterministic non-overlapping source chunks for factorial crossing."""
    chunks: list[list[int]] = []
    current: list[int] = []
    space = language.char_to_id.get(" ")
    for sentence in language.encoded_sentences[split]:
        if current and space is not None:
            current.append(space)
        current.extend(sentence)
        while len(current) >= length:
            chunks.append(current[:length])
            current = current[length:]
    return chunks


def weighted_choice(rng: random.Random, probabilities: Sequence[float]) -> int:
    value = rng.random()
    cumulative = 0.0
    for i, probability in enumerate(probabilities):
        cumulative += probability
        if value <= cumulative:
            return i
    return len(probabilities) - 1


@dataclasses.dataclass
class CipherPacket:
    family: str
    cipher: list[int]
    metadata: dict[str, Any]
    max_symbol: int


def _random_permutation(rng: random.Random, n: int) -> list[int]:
    values = list(range(n))
    rng.shuffle(values)
    return values


def encrypt_sequence(
    plain: Sequence[int],
    family: str,
    language: LanguageData,
    rng: random.Random,
    parameter_mode: str = "train",
) -> CipherPacket:
    a = len(language.alphabet)
    values = list(map(int, plain))

    if family == "mono":
        mapping = _random_permutation(rng, a)
        inverse = [0] * a
        for p, c in enumerate(mapping):
            inverse[c] = p
        return CipherPacket(family, [mapping[x] for x in values], {"inverse": inverse}, a - 1)

    if family in ("homophonic", "null_homophonic"):
        symbol_lists: list[list[int]] = []
        cursor = 0
        for probability in language.probabilities:
            k = 1 + min(3, int(round(3.5 * math.sqrt(max(probability, 0.0)))))
            symbols = list(range(cursor, cursor + k))
            cursor += k
            symbol_lists.append(symbols)
        permutation = _random_permutation(rng, cursor)
        symbol_lists = [[permutation[x] for x in row] for row in symbol_lists]
        inverse: dict[int, int] = {}
        for p, row in enumerate(symbol_lists):
            for symbol in row:
                inverse[symbol] = p
        cipher = [rng.choice(symbol_lists[x]) for x in values]
        nulls: list[int] = []
        if family == "null_homophonic":
            null_count = max(2, a // 8)
            nulls = list(range(cursor, cursor + null_count))
            cursor += null_count
            rate = 0.06 if parameter_mode != "test" else 0.075
            expanded: list[int] = []
            for symbol in cipher:
                if rng.random() < rate:
                    expanded.append(rng.choice(nulls))
                expanded.append(symbol)
                if rng.random() < rate / 3:
                    expanded.append(rng.choice(nulls))
            cipher = expanded
        return CipherPacket(
            family,
            cipher,
            {"inverse": inverse, "nulls": nulls},
            max(cursor - 1, max(cipher, default=0)),
        )

    if family == "polyalphabetic":
        period_pool = (2, 3, 4, 5) if parameter_mode != "test" else (6, 7, 8)
        period = rng.choice(period_pool)
        shifts = [rng.randrange(1, a) for _ in range(period)]
        cipher = [(x + shifts[i % period]) % a for i, x in enumerate(values)]
        return CipherPacket(family, cipher, {"shifts": shifts, "alphabet": a}, a - 1)

    if family == "feedback":
        key = rng.randrange(1, a)
        state = rng.randrange(a)
        initial = state
        cipher: list[int] = []
        for x in values:
            y = (x + state + key) % a
            cipher.append(y)
            state = x
        return CipherPacket(
            family,
            cipher,
            {"key": key, "initial_state": initial, "alphabet": a},
            a - 1,
        )

    if family == "nomenclator":
        mapping = _random_permutation(rng, a)
        inverse = [0] * a
        for p, c in enumerate(mapping):
            inverse[c] = p
        number_codes = 24 if parameter_mode != "test" else 32
        selected = language.top_words[: min(number_codes, len(language.top_words))]
        word_to_code = {word: a + i for i, word in enumerate(selected)}
        code_to_word = {a + i: list(word) for i, word in enumerate(selected)}
        space = language.char_to_id.get(" ")
        cipher: list[int] = []
        words: list[list[int]] = []
        current: list[int] = []
        for x in values:
            if x == space:
                words.append(current)
                words.append([space])
                current = []
            else:
                current.append(x)
        words.append(current)
        for token in words:
            if token == [space]:
                cipher.append(mapping[space])
            elif tuple(token) in word_to_code:
                cipher.append(word_to_code[tuple(token)])
            else:
                cipher.extend(mapping[x] for x in token)
        return CipherPacket(
            family,
            cipher,
            {"inverse": inverse, "code_to_word": code_to_word},
            a + len(selected) - 1,
        )

    if family == "transposition":
        mapping = _random_permutation(rng, a)
        inverse = [0] * a
        for p, c in enumerate(mapping):
            inverse[c] = p
        block_pool = (4, 5, 6) if parameter_mode != "test" else (7, 8)
        block = rng.choice(block_pool)
        permutation = _random_permutation(rng, block)
        pad = a
        substituted = [mapping[x] for x in values]
        original_length = len(substituted)
        while len(substituted) % block:
            substituted.append(pad)
        cipher: list[int] = []
        for offset in range(0, len(substituted), block):
            piece = substituted[offset : offset + block]
            cipher.extend(piece[i] for i in permutation)
        return CipherPacket(
            family,
            cipher,
            {
                "inverse": inverse,
                "block": block,
                "permutation": permutation,
                "pad": pad,
                "original_length": original_length,
            },
            a,
        )

    if family == "fractionated":
        columns = int(math.ceil(math.sqrt(a)))
        rows = int(math.ceil(a / columns))
        row_symbols = _random_permutation(rng, rows)
        col_symbols = [rows + x for x in _random_permutation(rng, columns)]
        row_inverse = {symbol: i for i, symbol in enumerate(row_symbols)}
        col_inverse = {symbol: i for i, symbol in enumerate(col_symbols)}
        cipher: list[int] = []
        for x in values:
            row, col = divmod(x, columns)
            cipher.extend((row_symbols[row], col_symbols[col]))
        return CipherPacket(
            family,
            cipher,
            {
                "row_inverse": row_inverse,
                "col_inverse": col_inverse,
                "columns": columns,
                "alphabet": a,
            },
            rows + columns - 1,
        )

    raise ValueError(family)


def oracle_decrypt(packet: CipherPacket) -> list[int]:
    family = packet.family
    cipher = packet.cipher
    meta = packet.metadata

    if family in ("mono", "homophonic", "null_homophonic"):
        inverse = meta["inverse"]
        nulls = set(meta.get("nulls", []))
        out: list[int] = []
        for symbol in cipher:
            if symbol in nulls:
                continue
            if isinstance(inverse, list):
                out.append(inverse[symbol] if 0 <= symbol < len(inverse) else -1)
            else:
                out.append(int(inverse.get(symbol, -1)))
        return out

    if family == "polyalphabetic":
        a = int(meta["alphabet"])
        shifts = meta["shifts"]
        return [(x - shifts[i % len(shifts)]) % a if 0 <= x < a else -1 for i, x in enumerate(cipher)]

    if family == "feedback":
        a = int(meta["alphabet"])
        key = int(meta["key"])
        state = int(meta["initial_state"])
        out: list[int] = []
        for symbol in cipher:
            if not 0 <= symbol < a:
                out.append(-1)
                continue
            value = (symbol - state - key) % a
            out.append(value)
            state = value
        return out

    if family == "nomenclator":
        inverse = meta["inverse"]
        code_to_word = {int(k): v for k, v in meta["code_to_word"].items()}
        out: list[int] = []
        for symbol in cipher:
            if symbol in code_to_word:
                out.extend(code_to_word[symbol])
            elif 0 <= symbol < len(inverse):
                out.append(inverse[symbol])
            else:
                out.append(-1)
        return out

    if family == "transposition":
        block = int(meta["block"])
        permutation = list(meta["permutation"])
        inverse_map = list(meta["inverse"])
        unpermuted: list[int] = []
        for offset in range(0, len(cipher), block):
            piece = cipher[offset : offset + block]
            if len(piece) != block:
                unpermuted.extend([-1] * len(piece))
                continue
            original = [0] * block
            for out_index, source_index in enumerate(permutation):
                original[source_index] = piece[out_index]
            unpermuted.extend(original)
        unpermuted = unpermuted[: int(meta["original_length"])]
        return [inverse_map[x] if 0 <= x < len(inverse_map) else -1 for x in unpermuted]

    if family == "fractionated":
        rows = {int(k): int(v) for k, v in meta["row_inverse"].items()}
        cols = {int(k): int(v) for k, v in meta["col_inverse"].items()}
        columns = int(meta["columns"])
        a = int(meta["alphabet"])
        out: list[int] = []
        for i in range(0, len(cipher), 2):
            if i + 1 >= len(cipher):
                out.append(-1)
                break
            row = rows.get(cipher[i])
            col = cols.get(cipher[i + 1])
            if row is None or col is None:
                out.append(-1)
            else:
                value = row * columns + col
                out.append(value if value < a else -1)
        return out

    raise ValueError(family)


def apply_noise(packet: CipherPacket, level: float, rng: random.Random) -> CipherPacket:
    if level <= 0:
        return packet
    out: list[int] = []
    max_symbol = max(packet.max_symbol, max(packet.cipher, default=0))
    for symbol in packet.cipher:
        if rng.random() < level / 3:
            continue
        if rng.random() < level / 3:
            symbol = rng.randrange(max_symbol + 1)
        out.append(symbol)
        if rng.random() < level / 3:
            out.append(rng.randrange(max_symbol + 1))
    return CipherPacket(packet.family, out, packet.metadata, packet.max_symbol)


def markov2_generate(language: LanguageData, length: int, rng: random.Random) -> list[int]:
    stream = language.train_stream
    followers: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
    for a, b, c in zip(stream, stream[1:], stream[2:]):
        followers[(a, b)].append(c)
    if len(stream) < 3:
        return [weighted_choice(rng, language.probabilities) for _ in range(length)]
    start = rng.randrange(len(stream) - 2)
    out = [stream[start], stream[start + 1]]
    while len(out) < length:
        choices = followers.get((out[-2], out[-1]))
        out.append(rng.choice(choices) if choices else weighted_choice(rng, language.probabilities))
    return out[:length]


def motif_generate(language: LanguageData, length: int, rng: random.Random) -> list[int]:
    stream = language.train_stream
    motifs = []
    for _ in range(12):
        width = rng.randint(3, 12)
        start = rng.randrange(max(1, len(stream) - width))
        motifs.append(stream[start : start + width])
    out: list[int] = []
    while len(out) < length:
        motif = list(rng.choice(motifs))
        for i in range(len(motif)):
            if rng.random() < 0.08:
                motif[i] = weighted_choice(rng, language.probabilities)
        out.extend(motif)
    return out[:length]


def copy_mutate_generate(language: LanguageData, length: int, rng: random.Random) -> list[int]:
    stream = language.train_stream
    start = rng.randrange(max(1, len(stream) - length))
    out = list(stream[start : start + length])
    for i in range(len(out)):
        if rng.random() < 0.12:
            out[i] = weighted_choice(rng, language.probabilities)
    return out


def slot_generate(language: LanguageData, length: int, rng: random.Random) -> list[int]:
    space = language.char_to_id.get(" ", 0)
    words = language.train_words or [(weighted_choice(rng, language.probabilities),)]
    sample_size = min(128, len(words))
    prefixes = [word[: max(1, len(word) // 3)] for word in rng.sample(words, sample_size)]
    cores = [word[max(0, len(word) // 3) : max(1, 2 * len(word) // 3)] for word in rng.sample(words, sample_size)]
    suffixes = [word[max(0, 2 * len(word) // 3) :] for word in rng.sample(words, sample_size)]
    out: list[int] = []
    while len(out) < length:
        word = list(rng.choice(prefixes)) + list(rng.choice(cores)) + list(rng.choice(suffixes))
        word = word[: rng.randint(2, max(2, min(14, len(word))))]
        if out:
            out.append(space)
        out.extend(word)
    return out[:length]


def generate_control(language: LanguageData, family: str, length: int, rng: random.Random) -> list[int]:
    if family == "markov2":
        return markov2_generate(language, length, rng)
    if family == "motif":
        return motif_generate(language, length, rng)
    if family == "copy_mutate":
        return copy_mutate_generate(language, length, rng)
    if family == "slot":
        return slot_generate(language, length, rng)
    raise ValueError(family)


def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def character_accuracy(truth: Sequence[int], prediction: Sequence[int]) -> float:
    if not truth:
        return 1.0 if not prediction else 0.0
    return max(0.0, 1.0 - edit_distance(truth, prediction) / max(len(truth), len(prediction), 1))


def run_oracle(
    languages: dict[str, LanguageData],
    replicates: int,
    lengths: Sequence[int],
    noise_levels: Sequence[float],
    families: Sequence[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for iso, language in languages.items():
        for length in lengths:
            chunks = source_chunks(language, "test", length)
            if len(chunks) < replicates:
                raise RuntimeError(f"insufficient test chunks {iso}/{length}: {len(chunks)}")
            selected = chunks[:replicates]
            for family in families:
                for noise in noise_levels:
                    for replicate, plain in enumerate(selected):
                        rng = random.Random(stable_seed("v050", iso, length, family, noise, replicate))
                        packet = encrypt_sequence(plain, family, language, rng, parameter_mode="test")
                        noisy = apply_noise(packet, noise, rng)
                        prediction = oracle_decrypt(noisy)
                        rows.append({
                            "iso": iso,
                            "family": family,
                            "length": length,
                            "noise": noise,
                            "replicate": replicate,
                            "plain_length": len(plain),
                            "cipher_length": len(noisy.cipher),
                            "accuracy": character_accuracy(plain, prediction),
                            "exact": plain == prediction,
                        })
    grouped: dict[str, Any] = {}
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        by_noise = {}
        for noise in noise_levels:
            subset = [row for row in family_rows if row["noise"] == noise]
            by_noise[str(noise)] = {
                "trials": len(subset),
                "mean_accuracy": statistics.fmean(row["accuracy"] for row in subset),
                "min_accuracy": min(row["accuracy"] for row in subset),
                "exact_rate": statistics.fmean(float(row["exact"]) for row in subset),
            }
        grouped[family] = by_noise
    noiseless_pass = all(grouped[family]["0.0"]["mean_accuracy"] >= 0.999 for family in families)
    return {
        "rows": rows,
        "summary": grouped,
        "gate": {
            "noiseless_all_families_at_least_0_999": noiseless_pass,
            "decision": "GO_TO_LEARNED_SMOKE" if noiseless_pass else "STOP_IMPLEMENTATION_DEFECT",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    here = args.repo / "experiments/recoverability_frontier_v0_5"
    languages = load_languages(here / "corpus_manifest_v050.json", args.repo / ".cache/ud-v050")
    if args.smoke:
        result = run_oracle(
            {key: languages[key] for key in ("en", "tr")},
            replicates=2,
            lengths=(96,),
            noise_levels=(0.0, 0.01),
            families=FAMILIES,
        )
    else:
        result = run_oracle(
            languages,
            replicates=args.replicates,
            lengths=LENGTHS,
            noise_levels=NOISE_LEVELS,
            families=FAMILIES,
        )
    payload = {
        "programme": "recoverability-frontier-v0.5.0-channel-oracle",
        "manifest_sha256": sha256_bytes((here / "corpus_manifest_v050.json").read_bytes()),
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, args.output)
    print("V050_ORACLE_SHA256", sha256_bytes(args.output.read_bytes()), flush=True)
    print("V050_ORACLE_GATE", json.dumps(result["gate"], sort_keys=True), flush=True)
    for family, summary in result["summary"].items():
        print(
            "V050_ORACLE_FAMILY",
            family,
            "noiseless",
            f"{summary['0.0']['mean_accuracy']:.6f}",
            "noise1",
            f"{summary.get('0.01', {}).get('mean_accuracy', float('nan')):.6f}",
            "noise3",
            f"{summary.get('0.03', {}).get('mean_accuracy', float('nan')):.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
