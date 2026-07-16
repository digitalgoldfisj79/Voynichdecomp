# v0.6 Family G — G2 initial implementation freeze

Date frozen: 2026-07-16

Status: **FROZEN BEFORE FULL G2 DEVELOPMENT RESULT**

## Authority

- Terminal protocol commit: `b751675e0ffdfa132579feacfe2f0d65f4884479`
- Frozen Family G protocol: `V060_PROTOCOL_FAMILY_G_CARRIER_STEGANOGRAPHY.md`
- G1 pass report commit: `10f9d7a6e0d581dd2f157876308e0cd557eb7035`
- Initial G2 implementation commit: `9daebe9e3f7cf7e373d05a175180bb6e70b6247b`
- Solver: `v060_family_g_stage_g2.py`

No test or Voynich data has been opened.

## Exact frozen carrier inventory

The solver evaluates **2,935** extraction rules on every cover:

| Carrier class | Frozen candidates |
|---|---:|
| Acrostic/telestic | 4 |
| Fixed token-position | 10 |
| Regular character/token periods | 154 |
| Repeated grille | 2,767 |
| **Total** | **2,935** |

Inventory SHA-256:

`00d3ae0d40c4cbf63c78c75eb779aae7c4cc15e524faa923d0d0d7a4b9584afe`

The grille inventory exhaustively enumerates widths 4–12 and every mask whose size is produced by the preregistered 10%, 20%, 30% or 40% density levels, capped at eight cells. Duplicate mask sizes are collapsed before enumeration.

## Initial blind scoring and selection

Every candidate is extracted to the frozen 96-character development payload length and receives:

1. train-only trigram/unigram language-model likelihood;
2. substitution-invariant recurrence-distance trigram likelihood;
3. entropy distance from untouched train-language chunks;
4. collision-rate distance from untouched train-language chunks;
5. zlib compression-ratio distance from untouched train-language chunks.

The fixed shortlist is the union of:

- the four highest identity-language scores;
- the twelve highest substitution-invariant scores.

Every shortlisted candidate receives a fresh-mono search of 50,000 iterations × 5 restarts. Identity and mono solutions are compared after the frozen mono key-description penalty. The selected candidate maximises the resulting evidence plus 0.15 × the invariant score.

For a selected mono arm on a payload cover only, recovery is refined with 700,000 iterations × 50 restarts. Null covers do not receive this recovery-only final refinement.

## Multiple-search calibration

The complete development run contains:

- 64 payload covers;
- 256 matched no-payload covers;
- four cover generators;
- all four carrier classes;
- four deterministic payload replicates per generator × carrier cell;
- four matched nulls per payload cell.

The family-level evidence is the maximum over all 2,935 candidates. The frozen operating threshold is the third-highest of the 256 null maxima, with detection defined strictly as evidence above threshold. This permits at most two empirical false positives, or 0.78125%, and therefore satisfies the predeclared ≤1% operating-point construction without seeing payload outcomes.

Abstention is evidence at or below this threshold. It counts as zero recovery on a payload cover.

## Frozen G2 gate

- AUROC ≥0.95;
- false-positive rate ≤1%;
- carrier-class accuracy ≥85% among detected payload covers;
- exact parameter accuracy ≥75%;
- mean recovered payload accuracy ≥80%, with abstentions scored zero;
- at least 54/64 payload covers recovered at 70% or better.

One development-only amendment remains available only if this initial full development run fails. The carrier inventory, cover generators, payloads, nulls, split and gates may not change.

## Execution preflight

CPU-basic job `6a58d4bf85d9643ce16d63e0` executed all 2,935 candidates on one payload cover and one matched null cover.

- exact carrier class: recovered;
- exact carrier parameters: recovered;
- payload recovery: 100%;
- null selected a different lower-evidence grille rule;
- output SHA-256: `0a79b4c410e2a44778a8cc6c5dce51ccae28579b2a6a8102a6a3a02cc63f53e2`.

The smoke is an execution check only and is not used in the G2 gate calculation.
