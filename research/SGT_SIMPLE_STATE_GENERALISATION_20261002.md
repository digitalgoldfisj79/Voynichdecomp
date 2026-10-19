# Super-Grammar simple-state generalisation suite — 2026-10-02

## Scope

Tests whether the compact state-machine principle discovered in SGT13 generalises to other certified Super-Grammar domains. Five frozen targets were tested: within-token form, token junctions, recurrence, section/Currier modulation, and ED1–3 neighbourhoods.

## 1. Within-token grammar

Preserve the current glyph exactly. Cluster only the glyph one step further back into K routing states.

Across five EVA-family transliterations, four classes recover 92.3–94.3% of the unrestricted order-2 incremental held-out gain; six recover 94.9–97.4%.

ZLZI multi-start fit: K4 96.8%; K6 99.2%; K8 slightly exceeds unrestricted order-2 held-out performance.

Generative K6 replay initially leaked certified hard-zero bigrams at ~0.00022–0.00025. Adding the fixed 19-hard-zero mask eliminates every violation. Mean token length then closes in all five folds. Attested-token rate remains 5.7–13.5 simulation SD too low.

Interpretation: compact local form controller succeeds; repertoire/popularity remains missing.

## 2. Token junctions

Predict next-token opener from section/position controls plus compact previous-token state.

ZLZI heldout bits/opener:
- baseline 3.256885
- exact previous token 3.178461
- previous last-two glyphs + coarse length 3.122632

Compact morphology beats exact identity by 0.05718 bits/opener.

Observed compact-state gain +0.135601 bits/opener; feature-permutation null mean -0.074288, null SD 0.002791; observed-minus-null +0.209889 = 75.20 SD.

Compact state beats exact identity in all five alternate layers:
ZLZI +.05718; ZLZB +.05709; TTLI +.04191; VDRB +.04244; TTIA +.05004 bits/opener.

## 3. Recurrence

Exact-repeat opportunities at lags 1–64.

A 64-lag lookup slightly overfits held-out folios. Coarse lag bands perform better:
1 / 2–5 / 6–12 / 13–32 / 33–64.

Observed banded gain over constant +0.0000279034 bits/opportunity. Lag-label permutation null mean -0.0000021576, null SD 0.00000197394; observed-minus-null +0.0000300610 = 15.23 SD.

Interpretation: coarse reuse/refractory clock, not a detailed 64-step memory.

## 4. Section / Currier

Shared opener transition topology with ENTRY vs CONTINUATION, then shrink group-specific weights around shared kernel.

Section gain +0.048472 bits/opener. Folio-label permutation null mean -0.025966, null SD 0.006867; observed-minus-null +0.074438 = 10.84 SD.

Currier gain +0.033970 bits/opener. Null mean -0.020926, null SD 0.005554; observed-minus-null +0.054896 = 9.88 SD.

Interpretation: same machine, different routing weights; no new topology required by this test.

## 5. ED1–3

Adjacent within-line ED1–3 rate 26.5827%.
Within-line shuffle null 24.8282%, null SD 0.1785 pp.
Excess +1.7545 pp = 9.83 SD.

Naive edge-only edit share:
observed 87.6674%; null 88.3675%, null SD 0.2801 pp.
Effect -0.7001 pp = -2.50 SD.
Reject edge-only explanation.

Frozen ten-operator portable Currier/realisation library, tested as <=3-step graph reachability:
observed adjacent ED1–3 coverage 5.4617%; shuffle null 5.1484%, null SD 0.2054 pp.
Effect +0.3133 pp = +1.53 SD.
The metric does not resolve this.

Existing SG result remains: dedicated sequential ED1 mutation worsens the constructive generator by 0.133384 with null SD 0.015515 = 8.60 SD, 5/5 folds.

## Synthesis

Best current compact architecture:

- paragraph reset;
- d/q/y core wheel + anchor-specific exit spokes + off-wheel re-entry routing;
- separate paragraph-entry transition regime;
- token-junction state = previous tail2 + coarse length;
- within-token state = current glyph + ~4–6 backward-context classes;
- 19 deterministic hard-zero guards;
- q selector and i-run terminator microstates;
- recurrence clock = five coarse lag bands;
- section/Currier modify routing weights;
- repertoire/popularity/reuse remains unresolved.

The generalisation supported by the data is not “one wheel everywhere.” It is a stack of small local finite-state controllers.

Canonical Supabase handoff:
voynich_simple_state_generalisation_20261002_v01
