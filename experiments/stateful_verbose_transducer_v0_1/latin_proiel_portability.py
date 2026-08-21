#!/usr/bin/env python3
from __future__ import annotations

import collections
import hashlib
from pathlib import Path

import svt_v02 as svt

REPO = "UniversalDependencies/UD_Latin-PROIEL"
COMMIT = "bc36b0223deeaa86d1a5aa48d464770863c0fc7b"
PREFIX = "la_proiel"
BLOB_SHA = {
    "train": "1a02fc3f95f9a2d64249dbadb6877706759a96d5",
    "dev": "e9857ad6d660c34329ddfbd59d4a1037665603e9",
    "test": "d32ce9d3d3c3bcb149166e53e0a38476ec3afaa8",
}


def git_blob_sha(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def load_latin(cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    base = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}"
    for split in svt.core.SPLITS:
        fn = f"{PREFIX}-ud-{split}.conllu"
        p = cache_dir / fn
        raw = p.read_bytes() if p.exists() else svt.core.fetch(f"{base}/{fn}")
        if not p.exists():
            p.write_bytes(raw)
        actual = git_blob_sha(raw)
        if actual != BLOB_SHA[split]:
            raise RuntimeError(f"Latin PROIEL blob mismatch {split}: {actual} != {BLOB_SHA[split]}")
        texts[split] = svt.core.parse_conllu_texts(raw)

    counts = collections.Counter()
    for text in texts["train"]:
        counts.update(text)
    alphabet = tuple(ch for ch, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if n >= 2)
    if " " not in alphabet:
        alphabet = (" ",) + alphabet
    char_to_id = {ch: i for i, ch in enumerate(alphabet)}

    encoded = {}
    for split in svt.core.SPLITS:
        encoded[split] = [[char_to_id[ch] for ch in text if ch in char_to_id] for text in texts[split]]
        encoded[split] = [x for x in encoded[split] if x]

    train_stream = []
    space = char_to_id.get(" ")
    for sentence in encoded["train"]:
        if train_stream and space is not None:
            train_stream.append(space)
        train_stream.extend(sentence)
    id_counts = collections.Counter(train_stream)
    total = sum(id_counts.values())
    probabilities = tuple((id_counts[i] + 0.5) / (total + 0.5 * len(alphabet)) for i in range(len(alphabet)))

    words = []
    for text in texts["train"]:
        for word in text.split():
            vals = tuple(char_to_id[ch] for ch in word if ch in char_to_id)
            if vals:
                words.append(vals)
    wc = collections.Counter(words)
    top_words = [word for word, _ in wc.most_common(128)]

    language = svt.core.LanguageData(
        iso="la", language="Latin (PROIEL)", alphabet=alphabet, char_to_id=char_to_id,
        probabilities=probabilities, texts=texts, encoded_sentences=encoded,
        train_stream=train_stream, train_words=words, top_words=top_words,
    )
    return language, svt.mono.build_language_model(language)
