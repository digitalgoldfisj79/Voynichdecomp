# U6 / VTPS Stage-B registration sensitivity protocol v0.3

Date frozen: 2026-08-15
Status at freeze: TARGET SEALED. `target_opened=false`; `true_retention_read=false`.

## Purpose

This is a new secondary reconstruction-sensitivity experiment initiated after the v0.2 primary reconstruction gate failed on f32r, f39r and f40r. It does **not** revise or overwrite the v0.2 primary result, which remains `FAIL_RECONSTRUCTION__ABSTAIN_UNRESOLVED` under the inherited `REG_MIN_INLIER_RATIO=0.55` rule.

The scientific question is whether the inherited 0.55 SIFT inlier-ratio admission cutoff, an engineering QC threshold external to VTPS, is solely preventing an otherwise exact Stage-B calibration. No retained-vs-switched Voynich target labels have been opened. The experiment must complete Stage B before any target opening can be considered.

## Frozen invariants

Everything downstream of page registration remains exactly frozen from U6-v0.2:

- external writer encoder SHA256: `54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0`;
- Stage-B pair skeleton SHA256: `7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4`;
- calibration population: 9,620 normalized word crops on 107 folios;
- `(folio,word_index)` keyset SHA256: `c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84`;
- pipeline revision: `fabb7413736a13497f80cb2fd930b3cf5681d371`;
- SIFT/CLAHE/USAC registration computation, candidate ordering, homographies, box transforms and crop generation;
- Stage-B synthetic sources, nuisance sources, random seed, dev/confirm repetitions, gain selection, nuisance FPR <= 0.05 gate, and beta=0.50 power >= 0.80 gate;
- no page-subspace subtraction on this first U6-v0.2 Stage-B use.

Forbidden target fields remain unreadable by the Stage-B runner, including `mid_retain`, `suffix_retain`, `exact_retain`, `prev_midfix`, `cur_midfix`, `prev_suffix`, and `cur_suffix`.

## Only changed variable: registration admission

The selected registration solution returned by the frozen pipeline is not recomputed or tuned. The admission decision is assessed under the following rules.

### R0 — inherited primary rule

Original pipeline `passed` value: inliers >= 50, inlier ratio >= 0.55, median reprojection <= 3 px, and plausible homography. This rule is retained as the primary reference and is expected from already observed QC to remain incomplete.

### R1 — ratio-free geometry rule (principal sensitivity rule)

Admit the frozen selected registration iff all of the following hold:

1. the selected Yale canvas label is the exact folio label (`f32r` -> `32r`, etc.);
2. absolute inliers >= the frozen `REG_MIN_INLIERS=50`;
3. median reprojection <= the frozen `REG_MAX_MEDIAN_REPROJ_PX=3.0`;
4. the frozen homography independently passes the pipeline's own `_canvas_plausible` predicate when evaluated against the selected Yale derivative;
5. all other frozen registration and crop-generation machinery is unchanged.

The inlier-ratio criterion is the **only** removed criterion. No replacement numerical threshold is introduced.

### R2 — descriptive fixed ratio grid

For sensitivity description, retain conditions 1-4 above and additionally require inlier ratio >= each of the following fixed values: 0.55, 0.50, 0.45, 0.40. These thresholds are not used to select a preferred result and do not supersede R1.

## Population gate

A rule is Stage-B-admissible only if it yields the exact full population: all 107 frozen folios and all 9,620 frozen normalized word crops, with the frozen keyset hash and exact generated crop IDs matching the historical manifest. No 104/107 or other partial analysis is allowed.

If R1 fails to reconstruct the exact population, this sensitivity experiment stops unresolved without opening the target.

## Registration-realization equivalence

Admission thresholds do not alter the frozen registration solution itself. The runner will nevertheless hash the selected realization `(folio, service_id, H_deriv)` across all 107 folios. If two complete rules have the same realization hash, one Stage-B run is sufficient for those equivalent rules. If complete rules produce distinct realization hashes, Stage B must be run separately for each distinct realization before any robustness statement.

## Stage-B decision

The Stage-B calibration code is unchanged from v0.2. PASS requires every prespecified nuisance FPR <= 0.05 and power >= 0.80 for every prespecified physical synthetic source at beta <= 0.50 under the frozen confirmation run.

- If Stage B fails: `FAIL_CALIBRATION__ABSTAIN_UNRESOLVED`; target remains sealed.
- If Stage B passes: calibration is qualified under this secondary reconstruction sensitivity analysis. Only then may a separate target-opening step be considered.

The v0.2 primary result remains reported alongside any v0.3 sensitivity result. A v0.3 PASS is evidence that the earlier stop was caused by the inherited registration admission convention; it is not retroactively a primary v0.2 PASS.

## Anti-tuning rule

No registration thresholds, image model parameters, synthetic source definitions, nuisance controls, power criterion, or Stage-B decision rule may be changed after this file is committed and before the v0.3 result is recorded. The true VTPS retained/switched target labels remain sealed throughout.