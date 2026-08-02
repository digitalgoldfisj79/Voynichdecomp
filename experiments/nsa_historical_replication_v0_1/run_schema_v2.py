#!/usr/bin/env python3
"""Schema adapter for the frozen v0.1 runner.

The first target run established that the corpus schema is pages→line→t→ZLZI.
This adapter changes acquisition only; all frozen Stage-1 statistics, thresholds,
labels, seed, and panel remain in run.py.
"""
from __future__ import annotations
import run as frozen


def zlzi_text(folio_obj):
    parts = []
    if not isinstance(folio_obj, dict):
        return ''
    def line_key(item):
        key = str(item[0])
        return (0, int(key)) if key.isdigit() else (1, key)
    for _, line in sorted(folio_obj.items(), key=line_key):
        if not isinstance(line, dict):
            continue
        transcriptions = line.get('t', {})
        if isinstance(transcriptions, dict):
            text = transcriptions.get('ZLZI', '')
            if isinstance(text, str) and text:
                parts.append(text)
    return '\n'.join(parts)


frozen.raw_text = zlzi_text
frozen.main()
