# Amendment 008 — Stage 5 non-pixel visual class-support stop

Date: 2026-08-08
Run: `corridor_v01_20260808_run01`
Timing: after the DINO confound gate failed, but before any corridor-to-Voynich text-description or geometry/morphology similarity was computed.

## Reason for stop

The frozen protocol requires strict visual-class matching. In the final 78-object corpus, the manuscript-level support by geography is:

| Class | Corridor manuscripts | Control manuscripts | Total manuscripts | Number of possible geography label assignments at fixed group sizes |
|---|---:|---:|---:|---:|
| architecture_cartography | 7 | 2 | 9 | C(9,7)=36 |
| diagram_geometry | 6 | 2 | 8 | C(8,6)=28 |
| plant | 1 | 3 | 4 | C(4,1)=4 |
| flower | 1 | 2 | 3 | 3 |
| bath_human | 0 | 1 | 1 | non-comparative |
| other_relevant | 0 | 2 | 2 | non-comparative |
| root | 0 | 1 | 1 | non-comparative |
| zodiac | 0 | 1 | 1 | non-comparative |

The preregistered confirmatory threshold is two-sided `p < 0.01` under manuscript-level permutation.

For the best-supported matched class, architecture/cartography, even the most extreme possible **one-sided** exact permutation p-value is no smaller than `1/36 = 0.02778`; for diagram/geometry it is `1/28 = 0.03571`. The remaining classes are still less informative. Therefore no strict-class-conditioned text-description or explicit-geometry test in this sealed corpus can attain the preregistered significance threshold.

Pooling raw similarities across classes would violate strict class matching and would be confounded with geography because visual-class availability is itself strongly geography-dependent in the sealed corpus. Creating a new post-hoc cross-class standardisation after seeing the DINO confound result would change the inferential design.

## Decision

- `image/DINO`: **EXCLUDED** by Amendment 007 confound gate.
- `blind text-description`: **CONFIRMATORY NONRESOLVING / UNDERPOWERED** in the sealed visual corpus.
- `explicit geometry/morphology`: **CONFIRMATORY NONRESOLVING / UNDERPOWERED** in the sealed visual corpus.

No corridor-to-Voynich similarity is computed for either non-pixel visual family in the confirmatory run.

Their already-blind descriptions and deterministic geometry may still be retained for later exploratory/descriptive work, but cannot count toward H1/Tier-2 convergence in this run.

This stop preserves the selection firewall and prevents a class-imbalanced corpus from producing a geography result through unequal missingness.

## Consequence

Stage 5 now proceeds on the separately governed Amendment 006 primary route:

1. like-for-like codicology;
2. dated/localised palaeography where the writer hand is actually assigned;
3. documentary/prosopographic cross-node edges only (nodes without an edge score zero).
