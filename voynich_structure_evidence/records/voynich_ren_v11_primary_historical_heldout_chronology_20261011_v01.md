# ReN v1.1 historical chronology: eight-book true holdout, family failure, and positive cross-city charter calibration

- key: `voynich_ren_v11_primary_historical_heldout_chronology_20261011_v01`
- status: completed
- created: 2026-10-11T07:48:22.110165+00:00
- updated: 2026-10-11T07:48:22.110165+00:00

---

# RETRACTED FINDINGS / CORRECTIONS — READ FIRST
* CORRECTION: eight ReN manuscript TEI sources contain 140,794 usable diplomatic word tokens, NOT 140,494 as in an interim user-visible message.
* RETRACTION of prior access blocker: ReN tei_1.1.zip was supplied by the user and matches published MD5 bc63dbf7a83d108c60c5de92cd7a72e3.
* DO NOT PROMOTE Voynich writing chronology or D:25-32-first. No language-independent bridging instrument, validated direction anchor, or known-time single-manuscript production control. Providence direction-conditioned chronology remains retracted.
* The independent 2D MDS benchmark DOES NOT reproduce eo777's undisclosed exact 48-block 11-knot algorithm or his rho=.95. Our spread-word observed in-sample absolute Spearman rho=.857; excerpt selection differs.
# PRIMARY SOURCES
ReN-Team (2021) Reference Corpus Middle Low German/Low Rhenish v1.1 DOI https://doi.org/10.25592/uhhfdm.9195. ZIP SHA256 b4cc9664268f760517b822c5d3965050ad15d31d712ba7907742c87808b7841e. Eight concrete TEI witnesses with full sha256 and date provenance in local audit /mnt/data/ren_lobo20261011/01_manifest.json, all source bytes read to completion; line-aware word extraction verified exactly against source token order.
# EIGHT ORIGINAL MANUSCRIPTS — COMPLETE DOCUMENT HOLDS
6 x 1,000 contiguous non-overlapping words per witness, first or six spaced excerpts; context-weighted conditional JS of within-word and joined-line transitions (28x28, alpha .08), 2D classical MDS fitted on 42 training blocks, OOS 6 heldout blocks, 7-book ridge year calibration. 8! exact null dates permuted at manuscript level. Spread word: MAE 38.3753 yrs vs training-date-median 112.875 yrs; observed improvement 74.4997 yrs; null-excess +41.83 yrs, null SD 13.26 yrs, 3.15 SD, unrestricted p=.00362094. But family-aware baseline gain 48.125 yrs; null excess +31.63 yrs, null SD 19.92 yrs, 1.59 SD, p=.0405. Genre-matched date shuffle p=.01660; narrow family permutation p=.15385 (small permutation space). FAIL manuscript-family-independent temporal qualification. Per-book extremes Hamburg 1301 predicted ~1361 and 1497 predicted ~1436.
Word TF-IDF train-only 2D SVD: 73.47 MAE, permutation p=.2497. Full V/U+I/J/Y harmonization: 61.31 MAE, p=.1013. Within-five-legal-manuscript LOSO: word transition MAE 49.80 vs 84.00 baseline, null excess +32.04 years, null SD 15.28, 2.10 SD, exact p=.04132 (FAIL p<=.01).
# NEW MATCHED HISTORICAL DIACHRONIC CONTROL: CROSS-CITY CHARTERS
Seven ReN Urkunden charter series, each four dated 50-year collection intervals 1301-1350, 1351-1400, 1401-1450, 1451-1500. Six fully annotated cities Berl, Brs, Lüb, Mst, Oldb, Strals; seventh Hil is transcription-only. Every city-period sample 900 actual diplomatic tokens (minimum availability enforced). For each held-out city, fit MDS and year mapping on other cities only. Null: 40,000 independently sampled date-label permutations within each city, preserves genre, city and date multiset.
Six-city first-900 word transitions: heldout MAE 25.1068 yrs versus balanced bin-midpoint baseline 50.000 yrs; observed improvement 24.8932 yrs; matched-null excess +24.94 yrs, null SD 2.71 yrs, 9.19 SD; Monte Carlo p=.0000249994, below resolution. Every city improved. 100k city-cluster bootstrap 95% interval of advantage ~[22.65,27.72] yrs.
Six-city first-900 harmonized V/U I/J/Y: MAE 25.80, matched-null excess +24.25 yrs, null SD 2.72, 8.91 SD, p=.000025.
Seven-city first-900: 25.51 yrs MAE, matched-null excess +24.53 yrs, null SD 2.34, 10.47 SD, p=.000025. Spaced 3x300 and last-900 sampling reproduce significant results with both original and harmonized scripts. First-900 direct date formula / roman numeral token exclusion (117 tokens removed among 28 packets) leaves six-city MAE 25.10, z9.16, p=.000025.
Interpretation: recoverable between-document historical time for same-genre charter collections from unseen cities; NOT known exact date of tokens within broad 50y bins, not one-scribe temporal drift, NOT intra-Voynich chronology.
# AUDIT / QUALIFICATION
Self-projection numerical check 4.0e-16 maximum difference between in-sample and OOS training embedding. All source checksums, original paths, sample positions, predictions, 40k null results, atomic pickle files, exact code, 5 falsification Qs, and retractions are in user-visible file /mnt/data/ren_lobo20261011/FINAL_AUDIT.md and REPRODUCTION_PACKAGE.zip. Reproduction does not include redundant source ZIP; original user-supplied archive is required. Analyses were incrementally refined; all p values exploratory not prospective preregistered confirmatory tests.
# DECISION
Promote independently replicable conditional DIACHRONIC ORTHOGRAPHIC TIME SIGNAL for ReN same-genre charter collections, NOT publication as Voynich chronology. To resolve Voynich chronology, require independently dated writing-order calibration within physical manuscripts plus source-neutral generative bridge and physical codicology anchors.
