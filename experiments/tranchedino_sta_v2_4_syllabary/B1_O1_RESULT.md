# Tranchedino × STA v2.4 — B1-O1 result

Date: 2026-08-09
Protocol: `PROTOCOL_B1_O1_SYLLABARY.md`
Runner: `b1_o1_syllabary.py`
Status: **B1-O1 SYLLABARY COMPONENT NOT QUALIFIED**
Scientific payload SHA-256: `b4667314565983dd1a943e9f397216a019299894fa312746a095a72090b325cb`
Voynich target fit authorised: **NO**

## Source-power gate

The untouched Q1-O1 source was adequately occupied in every cell:

- minimum observed syllable identities: **61** (required >=45);
- minimum syllable occurrences: **491** (required >=400).

Therefore this is not reported as `SOURCE INSUFFICIENT`.

## Complete qualification table

`occ A/B` = occurrence-weighted true syllable recovery; `id A/B` = observed-identity mapping recovery; `edit A/B` = expanded-plaintext edit accuracy; `agree` = A/B occurrence-weighted semantic-map agreement.

| p_syll | p_null | ids | syll occ | occ A | occ B | id A | id B | edit A | edit B | agree | score Δ/event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.00 | 62 | 491 | 0.588595 | 0.714868 | 0.435484 | 0.564516 | 0.950755 | 0.962840 | 0.784114 | 0.004573 |
| 0.25 | 0.03 | 62 | 560 | 0.825000 | 0.776786 | 0.709677 | 0.645161 | 0.973108 | 0.962381 | 0.816071 | 0.003406 |
| 0.25 | 0.10 | 61 | 508 | 0.757874 | 0.783465 | 0.655738 | 0.655738 | 0.970539 | 0.973712 | 0.868110 | 0.000455 |
| 0.50 | 0.00 | 63 | 1046 | 0.957935 | 0.960803 | 0.857143 | 0.873016 | 0.987611 | 0.987763 | 0.970363 | 0.000559 |
| 0.50 | 0.03 | 64 | 1022 | 0.937378 | 0.963796 | 0.828125 | 0.859375 | 0.985194 | 0.991388 | 0.947162 | 0.001130 |
| 0.50 | 0.10 | 63 | 1027 | 0.937683 | 0.888997 | 0.825397 | 0.809524 | 0.982477 | 0.972373 | 0.895813 | 0.006805 |
| 0.75 | 0.00 | 63 | 1555 | 0.996141 | 0.936977 | 0.984127 | 0.888889 | 0.999094 | 0.971597 | 0.940836 | 0.017683 |
| 0.75 | 0.03 | 64 | 1527 | 0.969876 | 0.981663 | 0.906250 | 0.953125 | 0.985358 | 0.993201 | 0.971840 | 0.000307 |
| 0.75 | 0.10 | 64 | 1529 | 0.995422 | 0.995422 | 0.968750 | 0.968750 | 0.998942 | 0.998942 | 1.000000 | 0.000000 |
| 1.00 | 0.00 | 64 | 2061 | 0.995148 | 0.995148 | 0.968750 | 0.968750 | 0.998489 | 0.998489 | 1.000000 | 0.000000 |
| 1.00 | 0.03 | 64 | 2061 | 0.995148 | 0.995148 | 0.968750 | 0.968750 | 0.998489 | 0.998489 | 1.000000 | 0.000000 |
| 1.00 | 0.10 | 64 | 2061 | 0.995148 | 0.995148 | 0.968750 | 0.968750 | 0.998489 | 0.998489 | 1.000000 | 0.000000 |

## Registered gate summary

Metrics below pool the two independently seeded A/B ensembles where the protocol defines a recovery statistic, and use the 12 per-control agreements for the convergence statistic.

| gate | observed | threshold | result |
|---|---:|---:|---|
| median occurrence-weighted syllable recovery | **0.962300** | >=0.95 | PASS |
| minimum occurrence-weighted syllable recovery | **0.588595** | >=0.85 | **FAIL** |
| median observed-identity recovery | **0.880952** | >=0.90 | **FAIL** |
| minimum observed-identity recovery | **0.435484** | >=0.75 | **FAIL** |
| median expanded-plaintext edit accuracy | **0.987687** | >=0.97 | PASS |
| minimum expanded-plaintext edit accuracy | **0.950755** | >=0.93 | PASS |
| median A/B map agreement | **0.958763** | >=0.95 | PASS |
| minimum A/B map agreement | **0.784114** | >=0.85 | **FAIL** |

Four binding gates fail.

## Interpretation

The mechanism becomes highly recoverable when syllabic substitution is frequent. At `p_syll=0.75–1.00`, many runs are near exact. That does not rescue the registered family: the frozen nuisance range deliberately included sparse optional use because the historical key sheet does not state that syllabic signs were obligatory or dominant.

At `p_syll=0.25`, the solver can produce a superficially strong expanded plaintext score while recovering the actual syllable mapping poorly. The worst run has:

- expanded plaintext edit accuracy **0.9508**;
- but only **0.5886** occurrence-weighted syllable recovery;
- and **0.4355** observed-identity mapping recovery.

This is another direct instance of the programme's central distinction between **recovery of the hidden mechanism** and a high aggregate language/plaintext score produced with strong oracle assistance.

The failure cannot be blamed on lack of syllable exposure: the worst cell still contains 62 observed syllable identities and 491 syllable occurrences.

## Binding verdict

**B1-O1 SYLLABARY COMPONENT NOT QUALIFIED.**

Per the frozen advancement rule, no joint f.134v–135r solver is built and no Voynich target is scored under v2.4.

This closes the tested **one-visible-sign historical syllabary mechanism over the registered optional-use range** at the component-identifiability stage. It does not retroactively close a different historical key with a prospectively documented deterministic/near-obligatory syllable-use rule, nor a separate connected-aaa segmentation model. Those would require independent historical justification and a new freeze; they may not be recovered by simply dropping the failing 25% cells after seeing this result.
