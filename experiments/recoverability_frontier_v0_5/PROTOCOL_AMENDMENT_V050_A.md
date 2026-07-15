# Recoverability frontier v0.5.0 — protocol amendment A

Date: 2026-07-15

This amendment records implementation details fixed before the two full learned-model jobs were launched.

## Source-chunk independence

Source sentences are concatenated and divided into deterministic non-overlapping chunks within each corpus partition. A source chunk is then crossed factorially with cipher families and noise levels. Therefore ciphertext trials are not independent at the row level; the independent source unit is the underlying chunk. Final confidence intervals and any formal comparison must cluster by source chunk.

No source sentence crosses train, development or test boundaries.

## Key-label invariance

Raw synthetic cipher symbol integers have no stable meaning across independently sampled keys. Before learned decoding, each ciphertext is therefore canonically renumbered by order of first occurrence. This recurrence encoding preserves symbol equality, repetition and order while removing arbitrary numeric labels.

The recurrence implementation was committed as:

- `19d9b4f10a192839759efa6cec21f424158a596f`

The optimized full-run launcher was committed as:

- `54257e91338ca0ca9ec6350c802b0ba38d3dab28`

Both commits preceded the full learned-model executions.

## Exact metric acceleration

Pure-Python Levenshtein distance was replaced by the exact `rapidfuzz` implementation. This changes runtime only, not the metric or scientific result.

## Execution records

Channel-oracle full run:

- Hugging Face job: `Digitalgoldfish79/6a57f115b1669a49bf075fd7`
- result JSON SHA-256: `b9bb2bbb7af8a5c5e38e6e41b0f0dd0676cfd687aa7a67f95cf590bfd81be4ea`
- gate: passed; every noiseless cipher family recovered at 100% mean character accuracy.

Full family-known learned decoder:

- Hugging Face job: `Digitalgoldfish79/6a57f1a885d9643ce16d5648`
- hardware: `a100-large`
- status at amendment: running.

Full blind-family learned decoder:

- Hugging Face job: `Digitalgoldfish79/6a57f1b085d9643ce16d564a`
- hardware: `a100-large`
- status at amendment: running.

## Scientific boundary

This remains a development pilot. The cipher implementations and control generators share one codebase. A later locked validation requires independently authored cipher and control implementations and a separate test repository.
