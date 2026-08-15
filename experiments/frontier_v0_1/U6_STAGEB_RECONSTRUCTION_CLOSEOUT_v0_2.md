# U6-v0.2 Stage-B reconstruction closeout

**Date:** 2026-08-15  
**Branch:** `experiment/voynich-frontier-programme-v0.1-20260814`  
**Formal verdict:** `FAIL_RECONSTRUCTION__ABSTAIN_UNRESOLVED`

## Bottom line

U6-v0.2 passed its external writer-instrument qualification, but the inherited VTPS Stage-B calibration could not be made admissible because the exact frozen 9,620-word image population could not be reconstructed under the already-frozen registration QC gate.

This is **not** a Stage-B calibration failure. Stage B never ran. The real retained-vs-switched Voynich target labels were never opened or read, so there is no licensed inference for or against a physical production-state echo.

## Frozen scientific population

- 9,620 calibration words across 107 folios.
- Frozen `(folio, word_index)` key-set SHA-256: `c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84`.
- Frozen pair-skeleton SHA-256: `7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4`.
- Qualified U6 encoder SHA-256: `54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0`.
- Partial-sample substitution, imputation, threshold relaxation, or post-hoc learned repair were forbidden.

## Reconstruction result

The authoritative recovery reproduced 104 of 107 required folios. Three rectos failed the frozen registration requirement:

| Folio | Correct Yale canvas | Matches | Inliers | Inlier ratio | Median reproj. | Frozen ratio gate |
|---|---|---:|---:|---:|---:|---:|
| f32r | 1006136 | 1,455 | 793 | **0.545017** | 1.601 px | 0.55 |
| f39r | 1006150 | 2,417 | 1,031 | **0.426562** | 1.746 px | 0.55 |
| f40r | 1006152 | 2,630 | 1,407 | **0.534981** | 1.672 px | 0.55 |

All three had large absolute inlier counts and low reprojection errors, but the preregistered gate is conjunctive; their inlier ratios therefore fail.

## Candidate-selection audit

The frozen pipeline revision was `fabb7413736a13497f80cb2fd930b3cf5681d371`. Its `register_folio` implementation ranks candidates lexicographically by `(passed, inliers)`: any passing candidate outranks every failing candidate. The independent six-candidate rerun returned:

- f32r: 0 passing candidates;
- f39r: 0 passing candidates;
- f40r: 0 passing candidates.

Hence there is no candidate-selection bug or alternate already-permitted Yale canvas that rescues the three pages. The correct page is the best match in each case.

## Asset audit

Several potential non-scientific recovery routes were exhausted:

- The full-corpus manifest exists and preserves the correct 37,886-word corpus identity, but most full-corpus rows do not retain crop paths.
- Persisted PNGs in `Digitalgoldfish79/vdino3-crops` are not an addressable copy of the required full-corpus calibration images; the surviving crop manifest is the later eight-folio pharmaceutical shard.
- Full-corpus DINOv3 embeddings are present, but they cannot substitute for pixels because U6-v0.2 requires its own writer-sensitive encoder applied to image crops.
- The saved `register/reg_all.jsonl` ledger was overwritten by a later small run and does not retain the original full-corpus homographies.
- Re-registration with the unchanged pipeline reproduces the three QC failures above.

## Scientific firewall

Throughout recovery and diagnostics:

- `target_opened = false`
- `true_retention_read = false`
- no real retained-vs-switched labels were loaded into Stage B
- no Stage-B nuisance FPR or synthetic-power result was computed

Therefore `FAIL_VTPS_CALIBRATION` would be the wrong label. The correct stopping state is an **asset/reconstruction failure with abstention**.

## Stop rule

Do not run U6-v0.2 Stage B on 104/107 folios and do not lower the 0.55 registration threshold. Either would modify a gate after observing which pages fail it.

A future attempt is scientifically legitimate only if it is a separately preregistered reconstruction experiment or if the exact original full-corpus pixel/homography assets are recovered. Such a future experiment cannot retroactively change this v0.2 closeout.
