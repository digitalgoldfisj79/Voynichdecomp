#!/usr/bin/env python3
"""Exact open-path solver for small production-unit graphs.

This intentionally solves an UNDIRECTED proximity problem. It does not infer
reading direction. Reversal-equivalent paths are collapsed.
"""
from __future__ import annotations

from itertools import permutations
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Edge = Tuple[str, str]


def canon_edge(a: str, b: str) -> Edge:
    return (a, b) if a < b else (b, a)


def zscore_edges(channel: Mapping[Edge, float]) -> Dict[Edge, float]:
    vals = list(channel.values())
    mu = mean(vals)
    sd = stdev(vals)
    if sd == 0:
        raise ValueError("degenerate channel: zero edge-score variance")
    return {canon_edge(*e): (v - mu) / sd for e, v in channel.items()}


def consensus_edges(channels: Iterable[Mapping[Edge, float]]) -> Dict[Edge, float]:
    zs = [zscore_edges(c) for c in channels]
    keys = set.intersection(*(set(z) for z in zs))
    return {e: mean(z[e] for z in zs) for e in keys}


def path_score(path: Sequence[str], edges: Mapping[Edge, float]) -> float:
    return sum(edges[canon_edge(a, b)] for a, b in zip(path, path[1:]))


def exact_open_paths(units: Sequence[str], edges: Mapping[Edge, float]) -> List[Tuple[float, Tuple[str, ...]]]:
    """Return every reversal-unique open path, sorted best first."""
    out = []
    for p in permutations(units):
        if p[0] >= p[-1]:  # collapse reverse-equivalent paths
            continue
        out.append((path_score(p, edges), p))
    return sorted(out, reverse=True)


def adjacency_set(path: Sequence[str]) -> set[Edge]:
    return {canon_edge(a, b) for a, b in zip(path, path[1:])}


def adjacency_recall(recovered: Sequence[str], truth: Sequence[str]) -> float:
    t = adjacency_set(truth)
    return len(adjacency_set(recovered) & t) / len(t)


if __name__ == "__main__":
    print("Library module: load frozen pair scores, z-standardize within channel, then call exact_open_paths().")
