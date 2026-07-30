#!/usr/bin/env python3
"""Build the frozen Stage-2 surface-class panel without computing scores.

The source inventory and transformation rules are frozen in
``data/STAGE2_SOURCE_LOCK.json``. Voynich data are never read.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import random
import re
import shutil
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Sequence

MANIFEST_FIELDS = [
    "corpus_id", "document_id", "split", "class_label", "language", "family",
    "path", "sha256", "encoding", "license", "author_id", "work_id",
    "date_band", "notes",
]
WS = re.compile(r"\s+")
LETTER = re.compile(r"[a-z]+")
SOURCE_LIMIT = 24576
NULL_RATE = 0.055
TOKEN_REGISTRY_SIZE = 512


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return WS.sub(" ", text).strip() + "\n"


def base26(n: int, width: int = 3) -> str:
    chars = []
    for _ in range(width):
        chars.append(chr(97 + n % 26))
        n //= 26
    return "".join(reversed(chars))


TOKENS = tuple("q" + base26(i) for i in range(TOKEN_REGISTRY_SIZE))
ALPHABET = tuple("abcdefghijklmnopqrstuvwxyz")


def words(text: str) -> list[str]:
    return LETTER.findall(text.lower())


def source_text(text: str) -> str:
    norm = normalize(text)
    if len(norm) < SOURCE_LIMIT:
        raise ValueError(f"source shorter than {SOURCE_LIMIT}: {len(norm)}")
    return norm[:SOURCE_LIMIT].strip() + "\n"


def letter_stream(text: str) -> list[str]:
    out: list[str] = []
    for word in words(text):
        out.extend(word)
        out.append("|")
    return out


def render_tokens(seq: Sequence[str]) -> str:
    out: list[str] = []
    for item in seq:
        if item == "|":
            if out and out[-1] != "\n":
                out.append("\n")
        else:
            out.append(item)
            out.append(" ")
    return "".join(out).strip() + "\n"


def fresh_pool(rng: random.Random, n: int) -> list[str]:
    if n > len(TOKENS):
        raise ValueError("token registry exhausted")
    pool = list(TOKENS)
    rng.shuffle(pool)
    return pool[:n]


def mono_render(text: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    pool = fresh_pool(rng, 26)
    mapping = dict(zip(ALPHABET, pool))
    seq = [x if x == "|" else mapping[x] for x in letter_stream(text)]
    return render_tokens(seq), {"mapping": mapping}


def homophonic_render(text: str, rng: random.Random, add_nulls: bool = False) -> tuple[str, dict[str, Any]]:
    freq_order = "etaoinshrdlucmfwypvbgkjqxz"
    counts = {ch: 2 + max(0, 4 - freq_order.index(ch) // 5) for ch in ALPHABET}
    pool = fresh_pool(rng, sum(counts.values()) + 16)
    cursor = 0
    mapping: dict[str, list[str]] = {}
    for ch in ALPHABET:
        mapping[ch] = pool[cursor:cursor + counts[ch]]
        cursor += counts[ch]
    nulls = pool[cursor:cursor + 8]
    seq: list[str] = []
    previous = ""
    for item in letter_stream(text):
        if item == "|":
            seq.append(item)
            continue
        choices = mapping[item]
        slot = (rng.randrange(len(choices)) + (ord(previous) if previous else 0)) % len(choices)
        seq.append(choices[slot])
        if add_nulls and rng.random() < NULL_RATE:
            seq.append(nulls[rng.randrange(len(nulls))])
        previous = item
    return render_tokens(seq), {
        "mapping": mapping,
        "null_tokens": nulls,
        "null_rate": NULL_RATE if add_nulls else 0.0,
    }


def global_common_words(source_rows: Sequence[dict[str, str]], stage1_root: Path) -> list[str]:
    counter: collections.Counter[str] = collections.Counter()
    for row in source_rows:
        counter.update(words((stage1_root / row["path"]).read_text(encoding=row.get("encoding") or "utf-8")))
    return [w for w, _ in counter.most_common(96) if len(w) >= 3][:64]


def nomenclator_render(text: str, rng: random.Random, common: Sequence[str]) -> tuple[str, dict[str, Any]]:
    pool = fresh_pool(rng, 26 + len(common) + 8)
    char_map = dict(zip(ALPHABET, pool[:26]))
    word_map = dict(zip(common, pool[26:26 + len(common)]))
    nulls = pool[26 + len(common):]
    seq: list[str] = []
    for word in words(text):
        if word in word_map:
            seq.append(word_map[word])
        else:
            seq.extend(char_map[ch] for ch in word)
        if rng.random() < 0.025:
            seq.append(nulls[rng.randrange(len(nulls))])
        seq.append("|")
    return render_tokens(seq), {
        "character_mapping": char_map,
        "word_mapping": word_map,
        "null_tokens": nulls,
    }


def transpose_blocks(tokens: list[str], width: int, permutation: Sequence[int]) -> list[str]:
    out: list[str] = []
    for start in range(0, len(tokens), width):
        block = tokens[start:start + width]
        out.extend((block[i] for i in permutation) if len(block) == width else block)
    return out


def substitution_transposition_render(text: str, rng: random.Random) -> tuple[str, dict[str, Any]]:
    mono, truth = mono_render(text, rng)
    toks = mono.split()
    width = rng.randint(8, 16)
    perm = list(range(width))
    rng.shuffle(perm)
    return " ".join(transpose_blocks(toks, width, perm)) + "\n", {
        **truth,
        "width": width,
        "permutation": perm,
    }


def family_p_render(text: str, rng: random.Random, split: str, ordinal: int) -> tuple[str, dict[str, Any]]:
    pool = fresh_pool(rng, 26 * 12)
    maps = [dict(zip(ALPHABET, pool[i * 26:(i + 1) * 26])) for i in range(12)]
    stream = letter_stream(text)
    seq: list[str] = []
    state = 0
    since_change = 0
    if split == "train":
        mode = "periodic_independent"
        period = 2 + ordinal % 3
    elif split == "dev":
        mode = "line_reset_periodic"
        period = 5 + ordinal % 3
    else:
        mode = "irregular_state_change" if ordinal % 2 == 0 else "alberti_rotation"
        period = 7 + ordinal % 6
    for position, item in enumerate(stream):
        if item == "|":
            seq.append(item)
            if mode == "line_reset_periodic":
                state = 0
            continue
        if mode in {"periodic_independent", "line_reset_periodic"}:
            state = position % period
        elif mode == "irregular_state_change":
            if since_change >= rng.randint(16, 96):
                state = rng.randrange(12)
                since_change = 0
            since_change += 1
        elif position and position % period == 0:
            state = (state + 1 + rng.randrange(3)) % 12
        seq.append(maps[state][item])
    return render_tokens(seq), {"mode": mode, "period": period, "mappings": maps}


def polygraphic_render(text: str, rng: random.Random, split: str) -> tuple[str, dict[str, Any]]:
    letters = [x for x in letter_stream(text) if x != "|"]
    pool = fresh_pool(rng, 400)
    seq: list[str] = []
    codebook: dict[str, list[str]] = {}
    i = 0
    mode = "fixed_digraph" if split == "train" else "mixed_character_digraph" if split == "dev" else "variable_one_to_three"
    while i < len(letters):
        if mode == "fixed_digraph":
            width = 2
        elif mode == "mixed_character_digraph":
            width = 2 if rng.random() < 0.7 else 1
        else:
            width = rng.choice((1, 2, 3))
        unit = "".join(letters[i:i + width])
        if unit not in codebook:
            index = stable_seed("poly", unit, len(codebook)) % len(pool)
            count = 2 if mode == "variable_one_to_three" and stable_seed(unit, "fraction") % 4 == 0 else 1
            codebook[unit] = [pool[(index + j) % len(pool)] for j in range(count)]
        seq.extend(codebook[unit])
        if (i // max(1, width)) % 24 == 23:
            seq.append("|")
        i += width
    return render_tokens(seq), {"mode": mode, "codebook": codebook}


def polygraphia_render(text: str, rng: random.Random, split: str) -> tuple[str, dict[str, Any]]:
    cycle = 4 if split == "train" else 7 if split == "dev" else 11
    pool = fresh_pool(rng, 26 * cycle)
    tables = [dict(zip(ALPHABET, pool[i * 26:(i + 1) * 26])) for i in range(cycle)]
    seq: list[str] = []
    position = 0
    for item in letter_stream(text):
        if item == "|":
            seq.append(item)
            continue
        seq.append(tables[position % cycle][item])
        position += 1
    return render_tokens(seq), {"cycle": cycle, "tables": tables}


def structured_generator(length_tokens: int, rng: random.Random, split: str, ordinal: int) -> tuple[str, dict[str, Any]]:
    alphabet = list(TOKENS[:32])
    mode = ("ordered_hmm" if ordinal % 2 == 0 else "motif_grammar") if split == "train" else "topic_fsm" if split == "dev" else "copy_mutate_latent"
    seq: list[str] = []
    state = rng.randrange(12)
    motif = [rng.randrange(12) for _ in range(rng.randint(3, 7))]
    topic = rng.sample(range(12), 4)
    prior: list[int] | None = None
    line_length = 48
    while len(seq) < length_tokens:
        line: list[int] = []
        if mode == "ordered_hmm":
            for _ in range(line_length):
                draw = rng.random()
                if draw < 0.58:
                    pass
                elif draw < 0.90:
                    state = (state * 5 + 3) % 12
                else:
                    state = rng.randrange(12)
                line.append(state)
        elif mode == "motif_grammar":
            for j in range(line_length):
                value = motif[j % len(motif)]
                if rng.random() < 0.10:
                    value = rng.randrange(12)
                line.append(value)
        elif mode == "topic_fsm":
            state = topic[len(seq) % len(topic)]
            for _ in range(line_length):
                if state not in topic:
                    state = topic[0]
                state = topic[(topic.index(state) + 1) % len(topic)] if rng.random() < 0.86 else rng.randrange(12)
                line.append(state)
        else:
            if prior is None:
                line = [rng.randrange(12) for _ in range(line_length)]
            else:
                line = list(prior)
                for j in range(len(line)):
                    if rng.random() < 0.14:
                        line[j] = rng.randrange(12)
            prior = list(line)
        seq.extend(alphabet[v] for v in line)
        seq.append("|")
    return render_tokens(seq[:length_tokens]), {"mode": mode, "alphabet": alphabet}


def token_shuffle(text: str, rng: random.Random) -> str:
    toks = text.split()
    rng.shuffle(toks)
    return " ".join(toks) + "\n"


def block_shuffle(text: str, rng: random.Random, width: int = 32) -> str:
    toks = text.split()
    blocks = [toks[i:i + width] for i in range(0, len(toks), width)]
    rng.shuffle(blocks)
    return " ".join(x for block in blocks for x in block) + "\n"


def markov_generate(text: str, rng: random.Random, n: int) -> str:
    toks = text.split()
    nexts: dict[str, list[str]] = collections.defaultdict(list)
    for a, b in zip(toks, toks[1:]):
        nexts[a].append(b)
    current = toks[0]
    out = [current]
    for _ in range(n - 1):
        choices = nexts.get(current)
        current = choices[rng.randrange(len(choices))] if choices else toks[rng.randrange(len(toks))]
        out.append(current)
    return " ".join(out) + "\n"


def sampled_shingles(text: str, width: int = 5, cap: int = 6000) -> set[tuple[str, ...]]:
    toks = text.split()
    if len(toks) < width:
        return {tuple(toks)}
    n = len(toks) - width + 1
    stride = max(1, n // cap)
    return {tuple(toks[i:i + width]) for i in range(0, n, stride)}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_document(
    output_dir: Path,
    rows: list[dict[str, str]],
    metadata: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    *, family: str, document_id: str, split: str, text: str, language: str,
    license_text: str, author_id: str, work_id: str, date_band: str,
    notes: str, source_id: str, truth: dict[str, Any] | None = None,
) -> None:
    text = normalize(text)
    encoded = text.encode("utf-8")
    path = output_dir / "documents" / f"{document_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    rows.append({
        "corpus_id": family,
        "document_id": document_id,
        "split": split,
        "class_label": family,
        "language": language,
        "family": family,
        "path": f"documents/{document_id}.txt",
        "sha256": sha256_bytes(encoded),
        "encoding": "utf-8",
        "license": license_text,
        "author_id": author_id,
        "work_id": work_id,
        "date_band": date_band,
        "notes": notes,
    })
    metadata.append({
        "family": family,
        "document_id": document_id,
        "split": split,
        "source_id": source_id,
        "normalized_characters": len(text),
        "tokens": len(text.split()),
        "sha256": sha256_bytes(encoded),
        "notes": notes,
    })
    if truth is not None:
        truths.append({
            "family": family,
            "document_id": document_id,
            "split": split,
            "source_id": source_id,
            "truth": truth,
            "truth_sha256": sha256_bytes(canonical_json(truth)),
        })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--gibberish-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    stage1 = args.stage1.resolve()
    gib_repo = args.gibberish_repo.resolve()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "documents").mkdir(parents=True)

    lock_path = repo / "experiments/compression_transfer_v0_1/data/STAGE2_SOURCE_LOCK.json"
    source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    stage1_freeze = json.loads((stage1 / "STAGE1_ACQUISITION_FREEZE.json").read_text(encoding="utf-8"))
    expected_stage1 = source_lock["source_assets"]["stage1_acquisition_freeze_payload_sha256"]
    if stage1_freeze["freeze_payload_sha256"] != expected_stage1:
        raise ValueError("Stage-1 acquisition freeze mismatch")

    stage1_rows = load_manifest(stage1 / "formal_stage1_manifest.csv")
    english = sorted((r for r in stage1_rows if r["corpus_id"] == "english"), key=lambda r: r["document_id"])
    if len(english) != 12:
        raise ValueError(f"expected 12 frozen English documents, got {len(english)}")
    common = global_common_words(english, stage1)

    manifest: list[dict[str, str]] = []
    metadata: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    synthetic_families = [
        "plaintext", "monoalphabetic", "homophonic", "nomenclator",
        "substitution_transposition", "family_p", "null_bearing_cipher",
        "polygraphic_fractionating", "structured_generator",
        "polygraphia_table", "matched_null",
    ]

    for ordinal, row in enumerate(english):
        source_id = row["document_id"]
        raw = (stage1 / row["path"]).read_text(encoding=row.get("encoding") or "utf-8")
        src = source_text(raw)
        for family in synthetic_families:
            seed = stable_seed("stage2", family, source_id)
            rng = random.Random(seed)
            if family == "plaintext":
                rendered, truth = src, {"source_sha256": sha256_bytes(src.encode())}
            elif family == "monoalphabetic":
                rendered, truth = mono_render(src, rng)
            elif family == "homophonic":
                rendered, truth = homophonic_render(src, rng)
            elif family == "nomenclator":
                rendered, truth = nomenclator_render(src, rng, common)
            elif family == "substitution_transposition":
                rendered, truth = substitution_transposition_render(src, rng)
            elif family == "family_p":
                rendered, truth = family_p_render(src, rng, row["split"], ordinal)
            elif family == "null_bearing_cipher":
                rendered, truth = homophonic_render(src, rng, add_nulls=True)
                truth["mode"] = "feedback_homophonic_with_nulls" if ordinal % 2 == 0 else "line_keyed_homophonic_with_nulls"
            elif family == "polygraphic_fractionating":
                rendered, truth = polygraphic_render(src, rng, row["split"])
            elif family == "structured_generator":
                rendered, truth = structured_generator(6000, rng, row["split"], ordinal)
            elif family == "polygraphia_table":
                rendered, truth = polygraphia_render(src, rng, row["split"])
            elif family == "matched_null":
                base, base_truth = family_p_render(src, random.Random(stable_seed("nullbase", source_id)), row["split"], ordinal)
                if row["split"] == "train":
                    mode = "token_shuffle" if ordinal % 2 == 0 else "block_shuffle"
                elif row["split"] == "dev":
                    mode = "first_order_markov"
                else:
                    mode = "token_shuffle" if ordinal % 2 == 0 else "first_order_markov"
                rendered = token_shuffle(base, rng) if mode == "token_shuffle" else block_shuffle(base, rng) if mode == "block_shuffle" else markov_generate(base, rng, len(base.split()))
                truth = {"mode": mode, "base_family": "family_p", "base_truth_sha256": sha256_bytes(canonical_json(base_truth))}
            else:
                raise AssertionError(family)

            if len(rendered) < 4096 or len(rendered.split()) < 2048:
                raise ValueError(f"{family}/{source_id} lacks formal geometry: chars={len(rendered)}, tokens={len(rendered.split())}")
            document_id = f"{family}__{source_id}"
            add_document(
                output, manifest, metadata, truths,
                family=family,
                document_id=document_id,
                split=row["split"],
                text=rendered,
                language="English" if family == "plaintext" else "encoded-English" if family not in {"structured_generator", "matched_null"} else "none",
                license_text=row["license"] + "; deterministic derived research surface",
                author_id=row.get("author_id", ""),
                work_id=row.get("work_id", ""),
                date_band=row.get("date_band", ""),
                notes=f"source={source_id}; seed={seed}; key_sha256={sha256_bytes(canonical_json(truth))}",
                source_id=source_id,
                truth=truth,
            )

    corema_path = repo / "experiments/corema_recoverability_v0_6/results/corema_recipe_rows_v0_6.csv.gz"
    by_ms: dict[str, list[str]] = collections.defaultdict(list)
    with gzip.open(corema_path, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_ms[row["manuscript"]].append(row["text"])
    eligible = []
    for ms, parts in sorted(by_ms.items()):
        text = normalize("\n".join(parts))
        if len(text) >= 16384 and len(text.split()) >= 2048:
            eligible.append((ms, text[:SOURCE_LIMIT]))
    if len(eligible) < 12:
        raise ValueError(f"only {len(eligible)} eligible CoReMA manuscripts")
    selected = eligible[:12]
    for i, (ms, text) in enumerate(selected):
        split = "train" if i < 8 else "dev" if i < 10 else "test"
        add_document(
            output, manifest, metadata, truths,
            family="corema_procedural",
            document_id=f"corema_procedural__{ms}",
            split=split,
            text=text,
            language="historical_German_Latin_mixed",
            license_text="CoReMA source-derived research representation; see repository provenance",
            author_id=ms,
            work_id=ms,
            date_band="medieval_early_modern",
            notes=f"complete manuscript-grouped lexical recipe stream; manuscript={ms}",
            source_id=ms,
        )

    archive = gib_repo / source_lock["source_assets"]["human_meaningless_archive"]
    extracted = output / "_gibberish_source"
    with zipfile.ZipFile(archive) as z:
        z.extractall(extracted)
    samples: list[tuple[str, str, int]] = []
    for path in sorted(extracted.rglob("*.txt")):
        text = normalize(path.read_text(encoding="utf-8", errors="replace"))
        samples.append((path.stem, text, len(text.split())))
    if len(samples) < 38:
        raise ValueError(f"expected at least 38 human samples, got {len(samples)}")
    bins: list[list[tuple[str, str, int]]] = [[] for _ in range(4)]
    totals = [0, 0, 0, 0]
    for sample in sorted(samples, key=lambda x: (-x[2], x[0])):
        target = min(range(4), key=lambda i: (totals[i], i))
        bins[target].append(sample)
        totals[target] += sample[2]
    split_by_bin = ("train", "train", "dev", "test")
    for i, group in enumerate(bins):
        participants = sorted(item[0] for item in group)
        text = "\n<COMPLETE_PARTICIPANT_BOUNDARY>\n".join(item[1].strip() for item in sorted(group))
        if len(text) < 4096 or len(text.split()) < 2048:
            raise ValueError(f"human composite {i} lacks formal geometry: chars={len(text)}, tokens={len(text.split())}")
        add_document(
            output, manifest, metadata, truths,
            family="human_meaningless",
            document_id=f"human_meaningless__composite_{i:02d}",
            split=split_by_bin[i],
            text=text,
            language="self_selected",
            license_text="Modified MIT License; danielgaskell/voynich d076a7d081f35098fa405928239595afd2e75927",
            author_id="|".join(participants),
            work_id=f"human_composite_{i:02d}",
            date_band="modern_control",
            notes=f"participant-disjoint complete-sample composite; participants={'|'.join(participants)}",
            source_id=f"composite_{i:02d}",
        )
    shutil.rmtree(extracted)

    manifest.sort(key=lambda r: (r["corpus_id"], r["split"], r["document_id"]))
    metadata.sort(key=lambda r: (r["family"], r["split"], r["document_id"]))
    truths.sort(key=lambda r: (r["family"], r["split"], r["document_id"]))

    counts = collections.Counter(r["corpus_id"] for r in manifest)
    split_counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for row in manifest:
        split_counts[row["corpus_id"]][row["split"]] += 1
    errors: list[str] = []
    required = source_lock["surface_families"]
    if sorted(counts) != sorted(required):
        errors.append(f"family inventory mismatch: {sorted(counts)}")
    for family in required:
        if not {"train", "dev", "test"}.issubset(split_counts[family]):
            errors.append(f"{family}: missing split")
    if len({r["document_id"] for r in manifest}) != len(manifest):
        errors.append("duplicate document IDs")
    for row in manifest:
        path = output / row["path"]
        text = path.read_text(encoding="utf-8")
        if len(text) < 4096:
            errors.append(f"{row['document_id']}: <4096 characters")
        if len(text.split()) < 2048:
            errors.append(f"{row['document_id']}: <2048 tokens for recurrence geometry")
        if sha256_file(path) != row["sha256"]:
            errors.append(f"{row['document_id']}: hash mismatch")

    exact: dict[str, list[str]] = collections.defaultdict(list)
    for row in manifest:
        exact[row["sha256"]].append(row["document_id"])
    exact_groups = [sorted(v) for v in exact.values() if len(v) > 1]
    if exact_groups:
        errors.append("exact duplicate derived documents")
    signatures = {
        r["document_id"]: sampled_shingles((output / r["path"]).read_text(encoding="utf-8"))
        for r in manifest
    }
    near = []
    for family in required:
        ids = sorted(r["document_id"] for r in manifest if r["corpus_id"] == family)
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                a, b = signatures[left], signatures[right]
                score = len(a & b) / len(a | b) if a | b else 1.0
                if score >= 0.50:
                    near.append({"family": family, "left": left, "right": right, "jaccard": score})
    if near:
        errors.append("within-family near duplicate candidates >=0.50")

    source_splits: dict[str, set[str]] = collections.defaultdict(set)
    for row in metadata:
        if row["family"] not in {"corema_procedural", "human_meaningless"}:
            source_splits[row["source_id"]].add(row["split"])
    leakage = {k: sorted(v) for k, v in source_splits.items() if len(v) > 1}
    if leakage:
        errors.append(f"source split leakage: {leakage}")

    manifest_path = output / "formal_stage2_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest)
    metadata_path = output / "stage2_metadata.jsonl"
    metadata_path.write_text("\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in metadata) + "\n", encoding="utf-8")
    truth_path = output / "stage2_truth_private.jsonl"
    truth_path.write_text("\n".join(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in truths) + "\n", encoding="utf-8")
    duplicate_path = output / "STAGE2_DUPLICATE_SCREEN.json"
    duplicate_path.write_text(json.dumps({"exact_groups": exact_groups, "near_pairs_ge_0_50": near}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    qualification = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "n_documents": len(manifest),
        "documents_per_family": dict(sorted(counts.items())),
        "split_counts": {k: dict(v) for k, v in sorted(split_counts.items())},
        "human_composite_token_totals": totals,
        "corema_manuscripts": [ms for ms, _ in selected],
        "generator_disjoint_rules": source_lock["generator_disjoint_rules"],
        "voynich_loaded_or_scored": False,
    }
    source_payload = {
        "source_lock_sha256": sha256_file(lock_path),
        "stage1_freeze_payload_sha256": stage1_freeze["freeze_payload_sha256"],
        "human_archive_sha256": sha256_file(archive),
        "corema_recipe_rows_sha256": sha256_file(corema_path),
        "manifest_sha256": sha256_file(manifest_path),
        "metadata_sha256": sha256_file(metadata_path),
        "truth_sha256": sha256_file(truth_path),
        "duplicate_screen_sha256": sha256_file(duplicate_path),
        "qualification": qualification,
    }
    source_payload["freeze_payload_sha256"] = sha256_bytes(canonical_json(source_payload))
    (output / "STAGE2_ACQUISITION_FREEZE.json").write_text(json.dumps(source_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(source_payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
