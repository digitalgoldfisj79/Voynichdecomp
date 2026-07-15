#!/usr/bin/env python3
"""Key-label-invariant launcher for the v0.5.0 learned decoder.

Cipher symbols are renumbered by order of first occurrence independently for
every sample. Language and optional family tags are preserved. This prevents a
model from treating arbitrary integer labels as stable cryptographic meaning
and exposes recurrence/equality structure across unseen keys.
"""
from __future__ import annotations

import train_decoder_v050 as base


class RecurrenceDataset(base.SyntheticDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        tag_count = 2 if self.known_family else 1
        tags = row["source"][:tag_count]
        surface = row["source"][tag_count:]
        mapping = {}
        canonical = []
        for token in surface:
            raw = int(token) - base.SURFACE_OFFSET
            if raw not in mapping:
                mapping[raw] = len(mapping)
            canonical.append(base.SURFACE_OFFSET + mapping[raw])
        if len(mapping) >= base.MAX_SURFACE_SYMBOLS:
            raise RuntimeError("recurrence vocabulary overflow")
        row["source"] = [*tags, *canonical]
        row["distinct_surface_symbols"] = len(mapping)
        return row


base.SyntheticDataset = RecurrenceDataset

if __name__ == "__main__":
    base.main()
