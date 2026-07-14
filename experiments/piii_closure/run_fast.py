#!/usr/bin/env python3
"""Performance-only wrapper. It changes no protocol, feature, threshold, or seed."""
from collections import Counter

import numpy as np
import piii_core


def linear_mattr(tokens, window):
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    counts = Counter(tokens[:window])
    distinct = len(counts)
    total = distinct / window
    number = 1
    for i in range(window, len(tokens)):
        outgoing = tokens[i-window]
        counts[outgoing] -= 1
        if counts[outgoing] == 0:
            del counts[outgoing]
            distinct -= 1
        incoming = tokens[i]
        if counts.get(incoming, 0) == 0:
            distinct += 1
        counts[incoming] += 1
        total += distinct / window
        number += 1
    return total / number


piii_core.mattr = linear_mattr
_original_chunks = piii_core.chunks
_chunk_cache = {}


def cached_chunks(corpus, target=120):
    key = (id(corpus), target)
    if key not in _chunk_cache:
        _chunk_cache[key] = _original_chunks(corpus, target)
    return _chunk_cache[key]


piii_core.chunks = cached_chunks

import run

if __name__ == '__main__':
    run.main()
