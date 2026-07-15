#!/usr/bin/env python3
"""Transparent pre-execution patch for the v0.3.4 topic-control state reentry."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import generator_disjoint_v034 as impl

_ORIGINAL = impl.ordered_control_lines


def fixed_ordered_control_lines(plan, family: str, seed: int):
    if family != "topic_fsm":
        return _ORIGINAL(plan, family, seed)
    rng = random.Random(seed ^ 0x0C017201)
    lines: list[list[int]] = []
    for doc, _quire, _section, _currier, _is_test, line_lengths in plan:
        topic = random.Random(seed ^ (doc * 0x9E3779B1)).sample(range(12), 4)
        for line_no, length in enumerate(line_lengths):
            state = topic[(doc + line_no) % len(topic)]
            row = [state]
            for _ in range(1, length):
                if state not in topic:
                    state = topic[rng.randrange(len(topic))]
                if rng.random() < 0.86:
                    state = topic[(topic.index(state) + 1) % len(topic)]
                else:
                    state = rng.randrange(12)
                row.append(state)
            lines.append(row)
    return lines


def output_path(argv: list[str]) -> Path:
    index = argv.index("--output")
    return Path(argv[index + 1])


impl.ordered_control_lines = fixed_ordered_control_lines

if __name__ == "__main__":
    target = output_path(sys.argv)
    impl.main()
    payload = json.loads(target.read_text())
    patch = Path(__file__)
    payload.setdefault("source_sha256", {})[str(patch.relative_to(Path.cwd()))] = hashlib.sha256(patch.read_bytes()).hexdigest()
    temporary = target.with_suffix(target.suffix + ".patch.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
