# Paper 2 v15 — Number Audit Report
## Updated 2026-03-16 (post nomenclator-lock fix)

### Core Forward Cipher Scores (10-seed mean)
| Metric | Paper | Verified | Source |
|--------|-------|----------|--------|
| n/84 | 62.9 | 62.9 | v11 with P_STICKY=0.22, nomenclator lock |
| σ | 2.6 | 2.6 | 10-seed std |
| C15 | 12.6 | 12.6 | CORE-15 subset |
| BG42 | 33.4 | 33.4 | Bowern-Gaskell subset |

### Calibration Parameters
| Param | Paper | Verified |
|-------|-------|----------|
| COPY_ALPHA | 1.3 | 1.3 |
| P_STICKY | 0.22 | 0.22 |
| top50 | 0.411 | 0.411 (VMS: 0.406) |
| sfx_bi | 0.253 | 0.253 (VMS: 0.252) |

### Nomenclator
| Metric | Paper | Verified |
|--------|-------|----------|
| Training r | 0.96 | 0.9594 |
| CV r (CI) | 0.89 | 0.8933 |
| Null p | <0.0001 | 0/10000 |
| L≡N χ² | 5.6 | 5.64 |
| L≡N p | 0.47 | 0.4661 |

### Random Nomenclator Baseline
| Metric | Paper | Verified |
|--------|-------|----------|
| Mean | 62.0 | 62.0 |
| Range | 58-66 | 58-66 |

### Language Constraints
| Metric | Paper | Verified |
|--------|-------|----------|
| Sonorant % | >93% | 93.3% |
| CI χ² | 0.04 | 0.0385 |
| EC rate (full MS) | 52.7% | 52.7% |
| Word-length σ | 1.72 | 1.72 |

### Folio Enrichments (Bonferroni-surviving)
| Folio | CV | p-value | Verified |
|-------|-----|---------|----------|
| f2r | mi | 3.66e-07 | ✓ |
| f17v | fe | 3.17e-04 | ✓ |
| f23r | te | 4.27e-04 | ✓ |
| f23v | te | 5.43e-04 | ✓ |
| f20r | vi | 6.42e-04 | ✓ |
| f8v | bo | 8.80e-04 | ✓ |

### Cross-Section Scores (3-seed mean)
| Section | Tokens | n/84 | C15 | BG42 |
|---------|--------|------|-----|------|
| Herbal-A | 4,033 | 62.0 | 12.3 | 33.3 |
| Herbal-B | 5,783 | 56.0 | 9.3 | 30.0 |
| Pharmaceutical | 3,870 | 64.7 | 11.3 | 33.3 |
| Rosettes | 1,818 | 60.3 | 10.0 | 27.7 |
| Stars | 10,702 | 54.3 | 9.3 | 26.0 |
| Zodiac | 1,590 | 54.7 | 6.3 | 27.0 |
| Cosmological | 1,341 | 50.7 | 8.0 | 22.3 |
| Astronomical | 1,469 | 51.3 | 10.3 | 24.0 |
| Balneological | 6,859 | 45.7 | 8.0 | 25.0 |

### Boundary Innovation
| Metric | Paper | Verified |
|--------|-------|----------|
| Hapax rate line-initial | 24.4% | 24.4% |
| Hapax rate elsewhere | 12.2% | 12.2% |
| Ratio | 2.0× | 2.0× |

### Fixes Applied (this session)
1. Nomenclator family lock: EC words in NOMENCLATOR bypass rebalance_family and P_STICKY
2. P_STICKY recalibrated: 0.16 → 0.22
3. §5.8 worked example: folia/flores/aqua → radicem/pulverem/cortice (correct FC routing)
4. Boundary innovation: 2.8× → 2.0× (recomputed from enriched_records)
5. German overfitting warning added to §5.5
6. §5.1 corpus preparation: full provenance for CI (Wellcome MS 624, Transkribus) and Ald.211
7. §3.1 VMS data provenance: ZLZI → PGCS → enriched_records chain documented
8. Davis (2020a) solution criteria added to §11 conclusion
9. §10 rewritten: "key" decomposed into grid contents, house assignments, source text

### Verified on 3 platforms
- Claude compute (this session)
- Termux Run 1 (stages 2-5, 7, 8 — no scipy)
- Termux Run 2 (all stages including 6 — with scipy): 62.9/84 σ=2.6 C15=12.6 BG42=33.4
