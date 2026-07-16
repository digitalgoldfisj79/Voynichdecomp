#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import v070_source_transfer_mdl as programme

_original_context = programme.mp.get_context
_original_build = programme.build_source_registry


def corrected_get_assets(repo: Path):
    cached = programme._PROCESS_CACHE.get("assets")
    if cached is not None:
        return cached
    gr, module, registry, old_external = programme.gen.load_assets(repo)
    raw_external, source_meta = _original_build(module, repo)
    source_external = type(old_external)(
        transition=raw_external["transition"],
        stationary=raw_external["stationary"],
        source_hash=raw_external["source_hash"],
    )
    cached = (gr, module, registry, source_external, source_meta)
    programme._PROCESS_CACHE["assets"] = cached
    return cached


def corrected_ordered_control_lines(plan, family: str, seed: int):
    rng = random.Random(seed ^ 0x0C017201)
    lines = []
    previous_by_doc = {}
    motifs = [[rng.randrange(12) for _ in range(rng.randrange(3, 7))] for _ in range(8)]
    successors = [rng.randrange(12) for _ in range(12)]
    for doc, _quire, _section, _currier, _is_test, line_lengths in plan:
        topic = random.Random(seed ^ (doc * 0x9E3779B1)).sample(range(12), 4)
        for line_no, length in enumerate(line_lengths):
            if family == "ordered_hmm":
                state = rng.randrange(12)
                row = [state]
                for _ in range(1, length):
                    draw = rng.random()
                    if draw < 0.58:
                        state = state
                    elif draw < 0.90:
                        state = successors[state]
                    else:
                        state = rng.randrange(12)
                    row.append(state)
            elif family == "motif_grammar":
                motif = motifs[(doc + line_no + rng.randrange(len(motifs))) % len(motifs)]
                row = []
                for index in range(length):
                    value = motif[index % len(motif)]
                    if rng.random() < 0.10:
                        value = rng.randrange(12)
                    row.append(value)
            elif family == "topic_fsm":
                state = topic[(doc + line_no) % len(topic)]
                row = [state]
                for _ in range(1, length):
                    if rng.random() < 0.86:
                        if state in topic:
                            state = topic[(topic.index(state) + 1) % len(topic)]
                        else:
                            state = topic[rng.randrange(len(topic))]
                    else:
                        state = rng.randrange(12)
                    row.append(state)
            elif family == "copy_mutate_latent":
                prior = previous_by_doc.get(doc)
                if prior is None:
                    row = [rng.randrange(12) for _ in range(length)]
                else:
                    row = [prior[index % len(prior)] for index in range(length)]
                    for index in range(length):
                        if rng.random() < 0.14:
                            row[index] = rng.randrange(12)
                previous_by_doc[doc] = list(row)
            else:
                raise ValueError(family)
            lines.append(row)
    return lines


programme.get_assets = corrected_get_assets
programme.gen.ordered_control_lines = corrected_ordered_control_lines
programme.mp.get_context = lambda _method=None: _original_context("fork")

if __name__ == "__main__":
    programme.main()
