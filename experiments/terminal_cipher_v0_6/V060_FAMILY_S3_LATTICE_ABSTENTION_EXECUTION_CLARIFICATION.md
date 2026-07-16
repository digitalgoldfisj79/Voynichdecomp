# v0.6 Family S3 — lattice abstention execution clarification

Date: 2026-07-16

Status: **RECORDED BEFORE ANY DEVELOPMENT PLAINTEXT RECOVERY OR TRUTH METRIC WAS CALCULATED.**

The first valid phase-2 execution reached replicate 0 and established a structural condition: each of its eight frozen boundary-posterior segmentations contained more distinct visible code groups than the fixed 63-unit S2 plaintext inventory. The unchanged `decode_candidate` function therefore returned `None` for every path, as required by its pre-existing bounded mapping rule. The job aborted before producing a trial result or calculating plaintext recovery, boundary F1, or any gate statistic.

The registered final amendment contains two candidate arms: direct neural decoding and lattice-refined decoding. A lattice path that cannot be mapped into the frozen inventory is not a legal hypothesis. Therefore:

- each of the eight paths is still attempted exactly once at the registered screen budget;
- paths with vocabulary larger than the frozen inventory remain invalid and are not truncated, merged, expanded or rescored;
- if at least one path is valid, selection and 200-restart refinement proceed exactly as frozen;
- if no path is valid, the lattice arm abstains for that trial;
- the direct neural beam candidate remains available and is selected by default because there is no competing legal lattice hypothesis;
- lattice abstention does not itself alter any gate threshold or count as success.

For transport through the existing phase-3 scorer, an abstaining lattice row carries an exact copy of the direct candidate plus `lattice_available=false`. This guarantees an exact feature and logit tie, which the already-frozen tie rule resolves to `direct`. The copy is not represented as an independently recovered lattice plaintext in the final scientific report.

This clarification does not add segmentations, enlarge the unit inventory, change code-length priors, alter search budgets, inspect truth, or improve a failed lattice path. It only prevents a structurally absent optional arm from aborting evaluation of the registered direct arm.