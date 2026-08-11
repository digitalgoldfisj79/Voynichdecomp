# Expanded Amadi Babuini + VBM v1 — Closeout

Date: 2026-08-11
Branch: `experiment/amadi-expanded-vbm-v1-20260811`

## Executive result

Two requested programmes were run in sequence.

1. **Expanded Amadi 1,365-syllable Babuini:** structurally incompatible as a direct one-syllable/one-visible-sign model, including under Bavarian plaintext controls. No Voynich holdout decipherment was needed.
2. **Vowel-Bridge Model (VBM):** the preregistered C/V topology gate passed on a fresh six-folio H1 and selected Bavarian over German and Italian. The effect is higher-order rather than a trivial vowel-frequency match and Bavarian wins on all six H1 folios individually. However the subsequent typed substitution solver could not qualify on fresh Bavarian positive controls because independent map ensembles did not converge. Therefore no typed Voynich H1 decipherment was generated and the remaining six-folio VBM_C1 stays sealed.

Binding status:

**EXPANDED BABUINI DIRECT-SIGN MODEL: CLOSED STRUCTURALLY.**

**VBM v1: STRUCTURAL BAVARIAN SIGNAL AT S0; TYPED DECIPHERMENT INSTRUMENT NOT QUALIFIED; NO PLAINTEXT/LANGUAGE CLAIM.**

---

## A. Expanded 1,365-combination Babuini

### Source-fixed reconstruction

Sections 386–389 were treated as a finite syllabary of 1,365 distinct syllable identities. The source-derived grammar has one vowel nucleus per syllable (with listed diphthongs), no more than three consonants on either side, and the enumerated inventory includes V/VC/VCC/VCCC/CV/CCV/CCCV/CVC forms.

A deterministic minimal-unit syllabifier was frozen before the capacity tests.

### Italian source-native capacity

At Voynich-matched event scales, deterministic Italian controls activate about 1,304–1,334 distinct syllable identities. A literal one-syllable/one-sign realization therefore needs roughly the full 1,365-sign inventory, whereas the observed Voynich symbolic inventories at the comparable full-STA and connected-aaa levels are only about 158 and 129 types.

Result: direct expanded Babuini is surface-capacity incompatible.

### Bavarian rerun

After the user identified Bavarian as the preferred language hypothesis, the same frozen syllable grammar was applied to the Bavarian Wikipedia corpus.

Across eight deterministic matched windows:

- full-STA-scale windows (130,655 syllable events) activate **5,454–6,117** distinct syllables;
- aaa-scale windows (145,115 events) activate **5,715–6,411** distinct syllables;
- the sampled Bavarian stream contains 14,684 distinct deterministic syllable forms overall;
- 94.33% of sampled Bavarian words are segmentable under the broad Amadi one-vowel grammar.

Thus Bavarian is not an escape hatch; it makes the cardinality mismatch much larger.

### Whole-Voynich-word escape hatch

Treating an entire visible Voynich word as one expanded syllabary sign also fails:

- FIT-A visible word types: **7,517**;
- Amadi capacity: **1,365**;
- top 1,365 Voynich word types cover only **0.774654** of FIT-A token occurrences, far below the registered 0.995 representation threshold.

### Expanded-Babuini verdict

**CLOSED for the direct model.**

A future expanded-Babuini model would require an additional many-to-one/polygraphic surface layer that is not supplied by sections 386–389 themselves. Adding one after these results would be a different model.

---

## B. Vowel-Bridge Model v1

### Frozen inherited architecture

VBM was recovered from the June 2026 programme rather than invented after the Amadi run:

- visible EVA/RF spaces may be false boundaries cutting a linking/vowel bridge;
- approximate token architecture `VR | C1 | C2 | VL`;
- cross-space `VL.VR` is the bridge layer;
- line-start and line-end halves are operators, not ordinary payload;
- `e/ee/eee` may act as composite core units;
- previously observed couplings included `ed -> y`, `eed -> y`, and frequent `y.qo` boundaries.

Executable v1 froze:

- `VR=qo` when a retained token begins `qo`; otherwise VR is the first retained character;
- VL is the final retained character;
- internal `eed` and `ed` at the right edge are composite C2 units;
- remaining maximal `e+` runs are composite core units;
- no additional multigraphs are admitted.

FIT-A therefore yields:

- **21** core surface units;
- **123** bridge surface units at >=0.995 bridge occurrence coverage;
- bridge coverage **0.9950441**;
- most common bridge **`y.qo` = 2,866 occurrences**.

This reproduces the central older VBM boundary phenomenon in the new executable representation.

### Fresh target split

The 12 folios still sealed after Amadi Core Babuini v1 were prospectively split:

VBM_H1 (opened only after protocol freeze):
`f28v, f31v, f88r, f5r, f34r, f81v`

VBM_C1 (still sealed):
`f85r1, f53v, f33r, f10r, f23r, f111r`

FIT is the original 181-folio FIT-A; previously opened Amadi H2 material was not added.

### S0 — C/V topology qualification and H1

Under VBM, core events are constrained to consonants and bridge events to vowels. Before attempting substitution, an order-4 C/V model was fit independently for Bavarian, German and Italian. Sixteen held-out natural-text spans at the H1 event scale were used to establish each language's 5th-percentile absolute floor.

H1 contains:

- 2,526 VBM events;
- 1,898 core events;
- 628 bridge events;
- bridge/vowel-event fraction 0.2486144.

Binding H1 scores:

| language | H1 nats/event | natural-control p05 | gap | gate |
|---|---:|---:|---:|---|
| Bavarian | **-0.67439898** | -0.69444329 | **+0.02004431** | PASS |
| German | -0.71443016 | -0.61784736 | -0.09658280 | FAIL |
| Italian | -1.28015393 | -0.59790295 | -0.68225097 | FAIL |

Ranking: Bavarian > German > Italian.

Bavarian margin over German: **0.04003118 nats/event**, exceeding the frozen 0.02 candidate margin.

Therefore S0 verdict was:

**S0 PASS TO TYPED SUBSTITUTION — BAVARIAN TOPOLOGY CANDIDATE.**

### Nonbinding topology diagnostic

A post-S0 diagnostic was run only to understand the source of the preference; it does not modify any gate.

Language ranking by C/V model order:

| order | rank 1 | rank 2 | margin |
|---:|---|---|---:|
| 1 | German | Bavarian | 0.01770058 |
| 2 | German | Bavarian | 0.00923682 |
| 3 | **Bavarian** | German | 0.02303649 |
| 4 | **Bavarian** | German | **0.04003118** |

Thus the Bavarian preference is not produced by the marginal vowel fraction alone; German is actually the better unigram/bigram match. Bavarian wins only once higher-order C/V run context is included.

Per-folio order-4 diagnostic:

| folio | events | winner | Bavarian margin over runner-up |
|---|---:|---|---:|
| f28v | 221 | Bavarian | 0.03433459 |
| f31v | 375 | Bavarian | 0.05404165 |
| f88r | 447 | Bavarian | 0.04367013 |
| f5r | 187 | Bavarian | 0.04710952 |
| f34r | 474 | Bavarian | 0.03820109 |
| f81v | 822 | Bavarian | 0.03358091 |

Bavarian therefore wins **6/6 H1 folios individually** at order 4.

This remains a structural compatibility result, not evidence that the manuscript plaintext is Bavarian.

### S1 — typed substitution instrument

The conditional S1 solver has two typed homophonic surfaces:

- 21 core surface units -> consonant values only;
- 123 bridge surface units -> vowel values only.

It uses an unspaced 19-letter character trigram language model and independent A/B convergence-controlled annealing ensembles.

#### Engineering smoke

At a deliberately short 5k-character fitting scale:

- Bavarian ranked correctly, but recovery was 0.8235 and A/B agreement 0.7872;
- German ranked correctly, recovery 0.9751, agreement 0.7909;
- Italian ranked correctly, recovery 0.9673, agreement 1.0000.

This was nonbinding.

#### First full qualification

At 40k fit / 15k holdout characters, all four Bavarian controls again selected Bavarian with large language margins, but none reached the required A/B convergence gate. One of four German controls also failed convergence. The run was cancelled immediately after the positive controls, before the 50 structured negatives.

No typed Voynich H1 score had been generated.

#### Amendment 001 strong optimizer

Because the failure was purely instrument calibration and no target S1 score existed, a fresh namespace `VBMV1TYPEDQ2` was preregistered with:

- 160,000 proposals/restart;
- max 24 restarts/ensemble;
- 6-restart batches;
- unchanged representation and scientific thresholds;
- fresh hidden keys and control spans.

Fresh Q2 Bavarian controls:

| rep | selected | margin | recovery | A/B agreement | converged |
|---:|---|---:|---:|---:|---|
| 0 | Bavarian | 0.194972 | 0.858227 | 0.839867 | NO |
| 1 | Bavarian | 0.262244 | 0.959963 | 0.819992 | NO |
| 2 | Bavarian | 0.096994 | 0.858609 | 0.896318 | NO |
| 3 | Bavarian | 0.231508 | 1.000000 | 1.000000 | YES |

German Q2: 4/4 selected German; all converged; recoveries 0.9765–0.9857.

Italian Q2: 4/4 selected Italian; all converged; recovery 1.0000 throughout.

The Bavarian typed mapping therefore remains non-identifiable under the registered convergence criterion even though language ranking is perfect. Under Amendment 001 the qualification must stop here.

The paid job was cancelled immediately after all 12 positive-control rows were emitted; no 50-negative stage was run.

### VBM binding verdict

**VBM S0 STRUCTURAL RESULT: POSITIVE BAVARIAN COMPATIBILITY SIGNAL.**

**VBM S1 TYPED DECIPHERMENT: INSTRUMENT NOT QUALIFIED.**

Consequently:

- no typed Voynich H1 map was fit;
- no candidate plaintext was inspected;
- no Bavarian plaintext claim is made;
- VBM_C1 remains sealed;
- merely increasing annealing again is prohibited under v1.

A v2 may reopen S1 only with a materially different inference method (not another budget increase) qualified prospectively on fresh Bavarian controls before any typed use of VBM_H1 or VBM_C1.

---

## Compute closeout

Relevant bounded HF jobs:

- expanded source census `6a7b673227caad61c6eac290` — completed;
- expanded Italian capacity `6a7b679f27caad61c6eac298` — completed;
- expanded word-sign gate `6a7b67c8f6d0f3ee953a9f76` — completed;
- expanded Bavarian capacity `6a7b6846f6d0f3ee953a9f8c` — completed;
- VBM preflight `6a7b694827caad61c6eac2d1` — completed;
- VBM S0 `6a7b69def6d0f3ee953a9fc1` — completed;
- original slow typed smoke `6a7b6a8af6d0f3ee953a9fc7` — explicitly cancelled after no scientific output;
- fast typed smoke `6a7b6b68f6d0f3ee953a9fd9` — completed;
- first full typed qualification `6a7b6bbd27caad61c6eac318` — explicitly cancelled after positive controls, before negatives;
- strong Q2 typed qualification `6a7b6c4227caad61c6eac321` — explicitly cancelled after positive controls, before negatives;
- topology-order diagnostic `6a7b6d0927caad61c6eac332` — completed.

No target typed substitution job was launched.
