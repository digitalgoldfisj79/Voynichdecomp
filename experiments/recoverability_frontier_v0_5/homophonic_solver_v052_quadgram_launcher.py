#!/usr/bin/env python3
"""Run the frozen flexible v0.5.2 solver with a dense quadgram objective."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def quadgram_score_key(
    cipher: np.ndarray,
    key: np.ndarray,
    quadgram_logp: np.ndarray,
    unigram_logp: np.ndarray,
) -> float:
    length = cipher.shape[0]
    if length == 0:
        return -1e300
    score = 0.0
    prefix = 3 if length >= 3 else length
    for index in range(prefix):
        score += 0.12 * unigram_logp[key[cipher[index]]]
    for index in range(3, length):
        first = key[cipher[index - 3]]
        second = key[cipher[index - 2]]
        third = key[cipher[index - 1]]
        fourth = key[cipher[index]]
        score += quadgram_logp[first, second, third, fourth]
        score += 0.12 * unigram_logp[fourth]
    return score


def build_quadgram_model(language, alpha: float = 0.05):
    alphabet_size = len(language.alphabet)
    counts = np.full(
        (alphabet_size, alphabet_size, alphabet_size, alphabet_size),
        alpha,
        dtype=np.float32,
    )
    contexts = np.full(
        (alphabet_size, alphabet_size, alphabet_size),
        alpha * alphabet_size,
        dtype=np.float32,
    )
    stream = language.train_stream
    for first, second, third, fourth in zip(stream, stream[1:], stream[2:], stream[3:]):
        counts[first, second, third, fourth] += 1.0
        contexts[first, second, third] += 1.0
    counts /= contexts[:, :, :, None]
    np.log(counts, out=counts)
    unigram = np.log(np.asarray(language.probabilities, dtype=np.float64))
    return counts, unigram


source_path = Path(__file__).with_name("homophonic_solver_v052_flexible.py")
source = source_path.read_text(encoding="utf-8")

reassignment_needle = "                old_label = int(key[first])\n                if new_label != old_label"
reassignment_replacement = (
    "                first = int(first)\n"
    "                old_label = int(key[first])\n"
    "                new_label = int(new_label)\n"
    "                if new_label != old_label"
)
if source.count(reassignment_needle) != 3:
    raise RuntimeError("reassignment site mismatch")
patched = source.replace(reassignment_needle, reassignment_replacement)

anneal_needle = (
    "            if not changed:\n"
    "                temperature *= cooling\n"
    "                continue\n\n"
    "            candidate_score"
)
anneal_replacement = (
    "            if not changed:\n"
    "                temperature *= cooling\n"
    "                continue\n\n"
    "            first = int(first)\n"
    "            second = int(second)\n"
    "            old_label = int(old_label)\n"
    "            new_label = int(new_label)\n"
    "            candidate_score"
)
if patched.count(anneal_needle) != 1:
    raise RuntimeError("annealing site mismatch")
patched = patched.replace(anneal_needle, anneal_replacement)

polish_needle = (
    "            if not changed:\n"
    "                continue\n"
    "            candidate_score"
)
polish_replacement = (
    "            if not changed:\n"
    "                continue\n"
    "            first = int(first)\n"
    "            second = int(second)\n"
    "            old_label = int(old_label)\n"
    "            new_label = int(new_label)\n"
    "            candidate_score"
)
if patched.count(polish_needle) != 1:
    raise RuntimeError("polishing site mismatch")
patched = patched.replace(polish_needle, polish_replacement)

main_needle = 'if __name__ == "__main__":\n    main()\n'
main_replacement = '''if __name__ == "__main__":
    mono.build_language_model = QUADGRAM_BUILD
    mono.score_key = QUADGRAM_SCORE
    _original_make_trial = fixed.make_trial
    def _offset_make_trial(language, split, length, replicate):
        if split == "test":
            replicate = replicate + 20
        return _original_make_trial(language, split, length, replicate)
    fixed.make_trial = _offset_make_trial
    main()
'''
if patched.count(main_needle) != 1:
    raise RuntimeError("main site mismatch")
patched = patched.replace(main_needle, main_replacement)

print("V052Q_PATCH", {
    "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
    "quadgram_alpha": 0.05,
    "test_replicate_offset": 20,
}, flush=True)
namespace = {
    "__name__": "__main__",
    "__file__": str(source_path),
    "QUADGRAM_BUILD": build_quadgram_model,
    "QUADGRAM_SCORE": quadgram_score_key,
}
exec(compile(patched, str(source_path), "exec"), namespace)
