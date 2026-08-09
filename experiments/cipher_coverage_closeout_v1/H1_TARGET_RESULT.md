# Terminal Cipher Programme v1 — H1 target result

Date: 2026-08-09
Branch: `experiment/cipher-coverage-closeout-v1-20260809`
Protocol freeze: `394adef38ed7d724680c6ee722e2a5ad4d3b44ac`
Implementation: `ea5c5cccdf79d18db66097d181e4535ffcf03cda`
Qualification archive: `cb792a8b933803a5cc02950ef32c9bdc3ada93f0`
Qualification scientific SHA-256: `4614c17fc637ac4659d2cc15e362727f73cfa239f78144f8e349008b1b4f3886`
H1 Hugging Face job: `Digitalgoldfish79/6a78c2b0da2af92a634f067e`
H1 scientific SHA-256: `52794ec9f0d583d064f4a5860253bffe20f74ec70802807607da5abb19fea6a7`

## Formal verdict

**M0: CLOSED NEGATIVE / INCOMPATIBLE UNDER V1.**

**TQ: CLOSED NEGATIVE / INCOMPATIBLE UNDER V1.**

**NQ: CLOSED NEGATIVE / INCOMPATIBLE UNDER V1.**

No family reaches its preregistered absolute positive-control floor on H1. Q3 blind specificity is therefore not required and is not run. C1 remains sealed. No plaintext strings are emitted or inspected.

## Representation gate

Frozen target: Zandbergen Reference Transliteration reduced-EVA `RF1b-er.txt`.

- source SHA-256: `eb857a1f353b18983fbc25b954e1bbce227a26d99cefabfda9206ff9b57644d2`;
- parsed pages: 227;
- raw words: 37,848;
- retained words: 37,647;
- total alphabetic positions: 194,617;
- retained alphabetic positions: 193,776;
- retained-character coverage: **0.9956786919950467**;
- words excluded for rare surface letters: 113;
- words excluded for uncertain readings: 88.

Frozen minimum coverage was 0.995. The representation gate therefore passes.

Deterministic folio split:

- T1: 136 folios, 126,331 retained characters;
- H1: 45 folios, 34,965 retained characters;
- C1: 46 folios, 32,480 retained characters.

C1 was never scored.

## Qualification recap

All three families passed Q1 and Q2 before H1 was opened.

| family | Q1 | Q2 correct-language ranks | Q2 median recovery | Q2 minimum recovery | Q2 median language margin |
|---|---:|---:|---:|---:|---:|
| M0 | PASS | 24/24 | 1.0000 | 1.0000 | 0.3724 |
| TQ | PASS | 24/24 | 1.0000 | 0.9365 | 0.3717 |
| NQ | PASS | 24/24 | 1.0000 | 0.9768 | 0.3439 |

Thus the H1 negatives are not caused by a control instrument that failed to recover or recognize its own mechanisms.

## H1 family maxima

The frozen family decision uses the highest fixed-map H1 score among the complete family inventory. The winning candidate must converge and is then compared with that language/family's frozen Q2 5th-percentile positive-control floor.

| family | best H1 candidate | H1 score | Q2 floor | score − floor | A/B agreement | converged | verdict |
|---|---|---:|---:|---:|---:|:---:|---|
| M0 | `ID / German` | **-3.4110186680** | -3.0693364239 | **-0.3416822441** | 1.0000 | yes | CLOSED NEGATIVE |
| TQ | `TQ_REV / German` | **-3.3554979648** | -3.1420514370 | **-0.2134465278** | 1.0000 | yes | CLOSED NEGATIVE |
| NQ | `NQ_R1 / German` | **-3.5238679196** | -3.1593897518 | **-0.3644781678** | 1.0000 | yes | CLOSED NEGATIVE |

The runner-up raw-score candidates were:

- M0: `ID / French`, H1 -3.4350116698, floor -3.1876643794;
- TQ: `TQ_REV / French`, H1 -3.3718022968, floor -3.1951582270;
- NQ: `NQ_MID_CEIL / German`, H1 -3.5378907088, floor -3.1593897518.

All are also below their positive-control floors.

## Interpretation

### M0

Simple global monoalphabetic substitution is now no longer merely a recoverable reference family. Under the frozen 19-symbol RF representation, the best held-out Voynich fit is materially worse than known-message controls generated under the same global-bijection mechanism. This supplies the previously missing calibrated target bridge.

### TQ

The principal historical gap identified at Gate 1 — medieval within-word/local transposition — is now tested directly. The finite inventory includes reversal, last-to-first, end-swap and the two source-motivated outside-in permutations, each composed with one global substitution alphabet. The best target member is word reversal under the German language model, but it remains 0.21345 nats/character below the absolute positive-control floor.

This is a mechanism-level negative for the frozen historically grounded finite family, not merely a failure to beat shuffled text.

### NQ

The bounded deterministic inserted-null family also qualifies cleanly on controls but fails H1. Its best member is the German `NQ_R1` schedule, 0.36448 nats/character below its positive-control floor.

Arbitrary or content-dependent null placement remains non-identifiable and was never admitted to v1.

## Search-convergence note

Some non-winning rule/language fits reached the maximum optimizer budget without satisfying A/B convergence. They are retained as failed candidate fits and are not interpreted. Under the preregistered family decision rule, closure depends on the **best H1-scoring family member**; that winning member converged with A/B agreement 1.0 for all three families.

No optimizer budget, rule inventory, language panel, representation, threshold or nuisance parameter was changed after target scoring.

## One-way stopping decision

Because all three surviving families are below their absolute positive-control floors:

- do not run Q3;
- do not unlock C1;
- do not inspect recovered plaintext;
- do not rerun under alternative representations to seek significance;
- do not remove poorly performing rules or languages after the result;
- proceed directly to programme closeout.
