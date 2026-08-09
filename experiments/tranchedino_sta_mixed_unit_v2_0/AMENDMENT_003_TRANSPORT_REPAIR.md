# Amendment 003 — transport repair

Date: 2026-08-09
Status: execution-only; no scientific data were scored by either failed launch.

The first remote Stage-A1 launch failed while defining Numba cache functions. The second failed while base64-decoding the transported Paduan count model because manually chunked `model.part2` contained one extra character (8,001 rather than 8,000 bytes). Neither launch reconstructed the Paduan model or generated a T20/H20 score.

The local source payload was re-hashed by chunk. Parts 0, 1 and 3 matched exactly. The corrupted part 2 was replaced operationally by two independently copied 4,000-byte files, `model.part2a` and `model.part2b`, from the unchanged local payload. Remote execution must concatenate `part0 + part1 + part2a + part2b + part3` and verify the compressed-byte SHA-256 `b3f56ce629172cb3825b2312b608fed149dbadd2a553dda5b2c401f54642bc8f` before decompression.

No model counts, smoothing, score definition, key geometry, seed, threshold, folio split or null definition changes.
