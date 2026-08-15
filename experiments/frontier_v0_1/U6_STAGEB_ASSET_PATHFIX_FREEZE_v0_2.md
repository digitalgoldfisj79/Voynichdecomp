# U6-v0.2 Stage-B asset-path correction freeze

Date: 2026-08-15
Branch: `experiment/voynich-frontier-programme-v0.1-20260814`

## Reason

The sealed Stage-B runner twice stopped before calibration because `u6_stageb.py` attempted to rediscover normalized crop files by assuming the manifest `id` equals the crop filename stem. On the authoritative full-corpus asset this resolves only 86 of the frozen 9,620 calibration words. The authoritative `results/corpus_crop_manifest.jsonl` already supplies the exact normalized crop `path` for every `(folio, word_index)` row.

This is an execution/data-addressing defect, not a scientific calibration failure.

## Frozen correction

For the rerun only, the runner applies two deterministic source substitutions to the committed `u6_stageb.py` before execution:

1. Preserve the manifest `path` field when constructing the 9,620-row calibration-word table.
2. Replace filename-stem rediscovery with exact `data_root / manifest.path` resolution, requiring:
   - all 9,620 frozen rows to survive the unchanged `(folio, word_index)` key-set SHA-256 gate;
   - all manifest IDs in the selected population to be unique;
   - every manifest path to be relative and non-traversing;
   - every referenced crop file to exist;
   - exactly 9,620 resolved crop paths.

The runner records the original and patched Stage-B source SHA-256 values in its execution status.

## Scientific invariants

Unchanged:

- frozen word population: 9,620 words / 107 folios;
- frozen key-set SHA-256: `c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84`;
- pair skeleton SHA-256: `7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4`;
- U6 external encoder SHA-256: `54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0`;
- visual scalar: raw cosine of L2-normalized 128-D U6 embeddings;
- nuisance nulls, physical synthetic sources, beta grid, folds, random seeds, development/confirmation repetition counts, model specification, thresholds, and qualification gate;
- true retained-vs-switched target labels remain sealed and are neither loaded nor read by Stage B.

No partial sample, imputation, re-registration, learned repair, threshold change, or post-hoc calibration adjustment is permitted.
