# Amendment 005 — Analysis-family semantics fix

Date: 2026-08-08
Programme: Alpine–Venetian Corridor Illustration Programme v0.1
Run: `corridor_v01_20260808_run01`

## Pre-outcome implementation defect

The original `corridor_programme.py::analyse()` grouped family-level secondary tests by `object_class` (`plant`, `zodiac`, etc.). This conflicts with the frozen protocol, which defines the independent feature/representation families as:

1. structure-oriented image embedding;
2. blind structured visual-description/text embedding;
3. explicit geometry/morphology.

The protocol's convergence requirement (>=3 corridor-positive independent families, >=2 BH q<0.05) therefore applies to **representation arms**, not to visual classes.

This defect was identified before any corridor-to-VMS similarity score was computed (`vms_similarity_computed=false`).

## Correction

- Primary manuscript composite remains equal-weight over available calibrated representation arms.
- Family-level convergence tests group `corridor_scores` by `arm`, not `object_class`.
- Visual-class (`object_class`) effects are still computed and reported as diagnostic/sensitivity results, but cannot satisfy the independent-family convergence rule.
- Missing arms remain missing; they are not zero-filled.
- The frozen p/q thresholds, seed, manuscript-level permutation unit and matching remain unchanged.

No candidate, control, image, VMS reference, match, threshold or outcome is changed by this amendment.