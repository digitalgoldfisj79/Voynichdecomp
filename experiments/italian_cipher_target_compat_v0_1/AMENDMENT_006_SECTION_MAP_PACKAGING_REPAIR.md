# Target Compatibility Amendment 006 — frozen section-map packaging repair

Date: 2026-08-15
Status: **pre-target-scoring implementation repair**.

GitHub Actions run `31907486122` completed the full external-only target-stage calibration successfully and froze:

- external status `PASS`;
- all three primary groups (`HOM`, `NOM`, `BIGRAM`) positive-control qualified;
- ACTIVE set = all 11 preregistered primary mechanism variants;
- global plaintext-null threshold = `1.6165962624929362`.

Only after that PASS did the workflow download the pinned RF-EVA and STA target files. It then stopped before any target parser, integrity test, entropy calculation, distance calculation, section analysis, or verdict because the workflow attempted to copy `voynich_section_map.json` from a repository path not present on this branch:

`cp: cannot stat 'experiments/structured_specificity_hostile_v0_2/voynich_section_map.json'`

Thus target **bytes were acquired**, as allowed by the frozen gate, but **no target statistic was computed or exposed**. The scientific protocol, external calibration, ACTIVE set, global null threshold, target representation, window size, metrics, and verdict rules remain frozen.

The exact previously frozen section map already exists in immutable GitHub Actions artifact `9249108075` from run `31893199016`, the same artifact used for the BVGS immutable-snapshot repair. Its required SHA-256 was frozen before target access in `SOURCE_MANIFEST.json`:

`a5d2a9e7aec3d3511ff00de828a17abd2d2255d065c70940ba72ed8abc753cb3`

The workflow is repaired only to extract that exact `voynich_section_map.json` from the already-downloaded immutable artifact, verify the pre-existing hash, and place it in `target_inputs/`. No new section classification, target-dependent mapping, target threshold, or scientific choice is introduced.

Because run `31907486122` exposed no target entropy or compatibility result, this packaging repair does not constitute post-result tuning. The next eligible run must reproduce the already-frozen external calibration before target scoring proceeds.
