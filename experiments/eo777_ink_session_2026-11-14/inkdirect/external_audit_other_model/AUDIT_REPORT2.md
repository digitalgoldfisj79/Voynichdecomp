# RETRACTED FINDINGS — 14 November 2026

1. **RETRACT as evidential inference:** “Path 1 independently tracks historical inking sequence.” The original null omits text cluster, and the cluster-controlled result is inconclusive at the stated two-null-SD criterion.
2. **RETRACT:** “Controlling for section × hand eliminates textual confounding.” It leaves eo777’s own cluster assignment uncontrolled.
3. **RETRACT as a unique Path-1 finding:** other eo777 text-derived paths also show pre-control smoothness.
4. **RETRACT as established:** “The measured darkness is ink chemistry rather than photographic focus, parchment, or image processing.” These components are not identified separately by the available photographs.

These retractions concern inference from the measured statistic, not the numerical existence of the raw association.

# Independent audit: ink darkness and eo777 Path 1

## Independent primary-data rederivation

Read `words_y2014.csv.gz` directly via the connected GitHub API; decoded 963,637-byte GZIP to 2,582,921-byte CSV, 11,309 word records. Joined the independently obtained `golf-v21-assignments.csv` and `physical_map.csv`. 50 ranked sheets, 48 measurable; I:67-68 and J:69-70 lack 2014 word measurements. Used independent OLS regression of log darkness against glyph counts, then page mean and sheet mean (151 pages). Independently reconstructed sheet `D_net` differs from the repository `phase2d_sheets.csv` by average |error| 0.000558 and max 0.005425 log units. The original `analysis6.py` was inspected **only after** this clean-room reconstruction, and was never run. Original `phase5.pkl` was not obtained, so the remaining sheet-level discrepancy is not fully isolated. These limitations cannot be described as bit-for-bit verification.

Original reported: Path1 obs 0.0964, null mean 0.1337, SD 0.0099, z 3.79; binding 0.1417, null 0.1408, SD 0.0109, z −0.08. My fresh 30,000-permutation check (NumPy): Path1 obs 0.095981, null 0.133143, **effect +0.037162, null SD 0.010188, z 3.65**, p 0.00017; binding obs 0.141274, null 0.140354, **effect −0.000920, null SD 0.010565, z −0.09**, inconclusive at this power. Data: `independent_bounds.json`, script `independent_bounds.py`, checkpoint `02_independent_bounding_results.pkl`.

## Damaging failure 1: missing cluster in the null

38/47 Path1 sheet-to-sheet edges share the same source text-cluster. A fairer stratified permutation keeps *section, hand, and eo777 text cluster* fixed. Path1 gives **inconclusive at this power**: observed 0.095981 versus null mean 0.111880, **effect +0.015899, null SD 0.008735, z 1.82**, p 0.0363 nominal. Within Herbal Hand1 (24 sheets, all Currier A), cluster-matched result is **inconclusive at this power**: **effect +0.017541, null SD 0.015576, z 1.13**. The original null was not a clean test of sequencing beyond textual groups.

The remaining alternative is a cluster-group effect, *not* a sheet-order effect. The 1b–1a contrast among the 23 adequately-sized Herbal H1 sheets is 0.152314 log units, with signed label-shuffle null SD 0.049657; familywise correction for the three natural cluster-group pairs gives p 0.0591. Caveat: these are post-hoc clusters and tests; the most appropriate independent holdout remains absent.

## Damaging failure 2: photographic measurement non-identifiability

The `instrument.py` darkness variable is median skeleton-core contrast relative to estimated local background, not spectroscopy. Derived sheet darkness correlates with image focus reading sigma at **r = −0.612**. After independent sheet-level residualization on sigma, Path1 **inconclusive at this power**: **effect +0.003817, null SD 0.008507, z 0.45**. Residualizing focus within the eight section × hand strata also yields **inconclusive at this power**: **effect +0.003027, null SD 0.007845, z 0.39**. Matching clusters on the original word-level focus+width-adjusted measure: **effect +0.007009, null SD 0.007657, z 0.92**. Combining sheet-level focus adjustment with cluster matching: **effect −0.000631, null SD 0.008074, z −0.08**.

**Overcontrol warning:** These regressions do not prove photographic focus is the true causal confound. A test of 500 synthetic covariates forced to have the same correlation with darkness as actual focus showed that 67.6% also reduce the z-statistic below two. Actual focus adjustment fell at the lower 8.8th percentile of that simulated z distribution. Thus the alleged ink-specific signal is **not identified**, rather than demonstrably created by photography. See `focus_overcontrol_calibration.json` and checkpoint 03.

Page parchment brightness (`B`) alone does not carry this Path1 signal: **inconclusive at this power**, effect +0.4052 background-level units, null SD 0.3916, z 1.03. This negative test is insufficient to rule out focus, color correction, layout and exposure.

## Damaging failure 3: alternative orders and outcome selection

All four eo777 paths are smoother than the original section × hand null before source-cluster matching. Independently computed 5,000-permutation figures: Path1 **effect +0.03707, null SD 0.01001, z 3.70**; Path2 **effect +0.03309, null SD 0.01054, z 3.14**; Path3 **effect +0.03745, null SD 0.01063, z 3.52**; Path4 **effect +0.02561, null SD 0.01014, z 2.53**. After source-cluster matching the four ratios are respectively 1.83, 1.31, 1.66, and 0.54 (the latter three from the independent JS reanalysis; raw arrays were not stored in the packaged checkpoint). **Inconclusive at this power** for each matched path.

The earlier registered primary order outcome (`W_net`, stroke width) gave **inconclusive at this power:** observed 0.1293, null 0.1282, **effect −0.0011, null SD 0.0109, z −0.11** (this line comes from archived `phase2_results.json`, not an independently recovered word-level width test). A later, separately registered darkness protocol was written before its dataset was unsealed, but the whole session was already exploratory and involved several measures and several orders. The repository's running log claims 340 numerical z/p-bearing results, 44 Path1-related. An independent pass through its 32 JSON result artifacts counted **417 numerical fields named like z/p** (many are paired results, repeated or subsequent tests; not 417 independent hypotheses). No credible familywise adjustment for the actual analytic search was supplied.

## Influence and controls

The most influential single deletion is sheet H:58-65: dropping it leaves **effect +0.02544, null SD 0.00947, z 2.69**. This is **not** a one-sheet artefact. Its impact is substantial but not enough to remove the original weakly specified correlation. There are 14 quires represented in the 48 measured sheets; only 15/47 consecutive Path1 edges share a quire, versus 34/47 binding edges. Binding order has less than 2 null SD on this darkness statistic, which does **not** establish a historical ordering of ink application.

## Source and reproducibility limitations

- Independent primary words and order data were obtained through GitHub connector, not a local checked-out repository. Raw word bytes were not exported into this bundle; derived sheet summaries were. `rebuild_from_words.py` provides a clean-room NumPy/pandas reconstruction recipe requiring users to supply the three public primary data files and was not executed on the original GZIP in the local container; the *original* clean-room word-level computation was instead successfully executed in JavaScript against those bytes.
- Instrument validity is limited to photographic darkness in one campaign; the 2004 dataset's reported correlation with the 2014 dataset cannot serve as a separately calibrated chemical ink replication.
- The original source's `phase5.pkl` and unprocessed color-image measurements were not independently re-analysed. Physical inking dates/batches cannot be inferred from the present measurements.
- Source-linked paths: [word data](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/words_y2014.csv.gz), [instrument](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/instrument.py), [original test code](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/analysis6.py), [running log](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/RUNNING_RESULTS_inkdirect.md), [eo777 text assignments](https://github.com/eo777/voynich-clusters/blob/main/golf-v21-assignments.csv), [physical map](https://github.com/eo777/vms-spatial/blob/main/data/manuscript/physical_map.csv).

## Final inference and probability

Original numerical association: **replicated approximately under the originally specified section×hand null**; z ≈ 3.65 by my fresh 30,000 permutations. The substantive assertion that Path1 independently tracks historical ink sequencing: **unsupported**. Subjective probability this survives a future truly independent, focus-aware, cluster-matched image/ink analysis: **10%**. This is an expert judgement, not a posterior calculated from the permutation p-values. The narrow literal observation under the original inadequate null is much more likely to replicate; it is not a test of ink sequence.
