# VBM Joachim Exact v9 — Q1 fresh-fit specificity closeout

Date: 2026-09-01
Branch: `experiment/vbm-joachim-exact-v9-20260901`
Frozen protocol commit: `ca59a1127d0437087a19d10ed3d27ca5a30b82e5`
Primary successful execution job: `6a9731ef0718b0f6d890df77`
Summary execution job: `6a9733a321c5aa7c8364c4a0`

## Binding verdict

**NONSELECTIVE** under the frozen Q1 interpretation rule.

Decision: `FRESH_FIT_NO_EVIDENTIAL_WEIGHT_REQUIRE_GLOBAL_CODEBOOK`.

No H1 or C1 folio was opened.

## Published f115v.26 fixture

The source-faithful parser reproduces the proposed plaintext fit.

- tokens: 11
- bridges: 10
- unique bridge types: 9
- unique non-empty nucleus types: 8
- plaintext length: 30 characters
- fresh dictionary description cost: **122.9268086459 bits**
- fresh-key cost per plaintext character: **4.0975602882 bits/char**

The exact-fit fact therefore establishes feasibility only. It does not supply a reusable codebook and is expensive as a fresh line-specific description.

## Corpus fresh-fit audit

720 deterministic Voynich lines were sampled across 5–15 bridges.

German natural-orthography candidates:

- overall median fit fraction: 0
- mean fit fraction: 0.0210083
- fraction of lines with >=1 exact fit among <=5,000 candidates: 0.479167
- fraction with >=10 exact fits: 0.301389
- overall median smoothed surprisal: 12.2880 bits

The overall zero median is driven by longer lines and should not be read as universal selectivity. Exact-fit incidence by bridge count is:

| bridges | lines | fraction with >=1 DE fit | median DE fit fraction |
|---:|---:|---:|---:|
| 5 | 80 | 0.9250 | 0.0261 |
| 6 | 80 | 0.7625 | 0.0026 |
| 7 | 80 | 0.7375 | 0.0017 |
| 8 | 80 | 0.6375 | 0.0006 |
| 9 | 80 | 0.4750 | 0 |
| 10 | 80 | 0.3625 | 0 |
| 11 | 80 | 0.2625 | 0 |
| 12 | 80 | 0.1250 | 0 |
| 13 | 55 | 0.03636 | 0 |
| 14 | 15 | 0 | 0 |
| 15 | 10 | 0 | 0 |

Italian and English controls show the same qualitative length dependence. The shuffled-character German bank is less fit-friendly than natural German but still produces many exact fits on short lines.

## Structural-topology null

For 120 deterministic lines and 20 position-shuffle null replicates:

- real median DE fit surprisal: 11.2880008897 bits
- shuffle mean median surprisal: 11.8880008897 bits
- shuffle SD: 0.4472135955
- real-minus-shuffle z: **-1.3416407865**

The actual repeated-type placement is not more selective than the shuffled placement under this test. It is directionally less selective.

## MDL implication

For fitting candidate sentences, the maximum smoothed fit surprisal available with 5,000 sampled candidates is about 12.288 bits, while fresh dictionaries commonly cost tens to more than one hundred bits. The fixture itself costs 122.93 bits.

Thus a post-hoc exact sentence fit is not a compressed explanation of the line under the frozen fresh-key accounting. Its dictionary has substantially more descriptive capacity than the demonstrated fit supplies.

## Interpretation

Q1 does **not** show that VBM is false. It closes the evidential route based on isolated fresh-key readings.

Any further VBM claim must use one reusable codebook learned without the target plaintext and transferred unchanged to held-out material. Q2 is therefore restricted to synthetic qualification of such a global-codebook inference instrument before any Voynich language fit is permitted.
