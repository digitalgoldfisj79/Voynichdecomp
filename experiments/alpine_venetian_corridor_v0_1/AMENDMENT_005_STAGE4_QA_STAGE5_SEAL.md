# Amendment 005 — Stage 4 crop QA and Stage 5 object seal

Date: 2026-08-08
Programme: Alpine–Venetian Corridor v0.1
Run: `corridor_v01_20260808_run01`

## Timing and firewall

This amendment records a pre-similarity quality-control step. At the time this amendment is frozen, no corridor-to-Voynich similarity score has been computed or inspected.

The first Stage 4 IIIF triage attempt (`6a77418bda2af92a634ef696`) is invalid because the parser assumed the model output was always a JSON object and failed on bare JSON lists. It ended cancelled and contributes no objects.

The corrected deterministic triage (`6a7742603e1f34a7e32bee12`) accepts either a bare list or `{objects:[...]}` and otherwise preserves the same sealed sources, deterministic page selection, class vocabulary, seed, and blind prompt. It completed successfully and generated 106 candidate crops.

A second independent blind QA pass (`6a7748253e1f34a7e32bee94`) saw only the isolated crop, proposed visual class, and neutral first-pass description. It had no access to manuscript geography, corridor/control status, dates, Voynich reference images, or similarity scores. Promotion required all three: class consistency, sufficiently tight crop, and substantive non-text visual content.

Final QA counts:

- usable: 47
- spurious: 57
- bad crop/source fetch: 2
- inference/parse errors: 0

Only the 47 `usable` objects may enter Stage 5. Rejected objects cannot be reinstated after outcome inspection.

## Stage 5 object seal

The canonical sorted JSON manifest of the 47 usable objects has SHA-256:

`1ce617085f8083d1205e8e96df278d9065a6f983558ec6251940c93ac1c68b59`

It contains objects from 8 manuscripts/candidates. Frozen class counts are:

- `architecture_cartography`: 9
- `diagram_geometry`: 15
- `plant`: 14
- `root`: 2
- `flower`: 3
- `zodiac`: 1
- `bath_human`: 1
- `other_relevant`: 2

Missing manuscript/class combinations remain missing. They are not scored as zero and may not be replaced by post-outcome hand-selection.

## Stage 5 scoring restrictions

1. Strict class matching remains mandatory.
2. Crops, not full-page RGB images, are the pixel unit.
3. The inferential unit remains manuscript, never crop.
4. Visual classes are subdomains, not the three independent convergence families. The three feature families are image/DINO, blind text-description, and explicit geometry/morphology.
5. A feature family may be absent for a manuscript. Missing remains missing.
6. The pixel arm must pass the frozen manuscript/institution confound gate before supporting H1.
7. VMS reference classes must themselves be reviewed/frozen before a class is scored. No cross-class surrogate reference is allowed.
8. Venetian chart/atlas clustering remains a mandatory sensitivity analysis.

This amendment changes quality-control implementation only; it does not alter the frozen geography, chronology, endpoint, significance threshold, convergence rule, or interpretation ladder.