# RETRACTED / DOWNGRADED FINDINGS — placed first

1. **RETRACTED:** The description of all 49 facing pairs as belonging to different sheets. Four pairs are at the central fold of the *same* sheet: f20v/f21r, f36v/f37r, f44v/f45r, f79v/f80r. The correctly restricted denominator is 45.
2. **RETRACTED:** The description of the 37-sheet conjugate measurement as using single-panel pages only. Three included sheets have panel-labelled measurements (O:87-90, O:88-89, Q:94-95); excluding them and the resulting singleton quire leaves 33 matched sheets.
3. **RETRACTED:** Treating nonsignificance of the facing-page statistic as proof of absent facing-page similarity. The registered 25%-gap-reduction plant achieves only 3.7% power at z>=3.
4. **DOWNGRADED:** A whole-manuscript or ink-specific inference. The detectable association is concentrated in herbal quires, and the photographic measurement is confounded with focus/sheet-face effects; unmeasured material effects remain possible.

# Clean-room audit: bifolium conjugates and facing-page control

Date: 14 November 2026. Source: public branch `digitalgoldfisj79/Voynichdecomp`, `eo777-ink-session-2026-11-14`, with `eo777/vms-spatial/data/manuscript/physical_map.csv` and `eo777/voynich-clusters`. All numerical material below comes from actual 2014 word measurements or primary source-derived 2004 page measurements. **No primary photograph was re-opened or independently re-imaged in this run.**

## Decision rule (fixed before attacking)

A positive result must exceed 2 null SD on at least two defensible matched comparisons and demonstrate adequate planted-signal recovery. A facing-page non-result is *absence* only if a prespecified 25% gap reduction is detected >=80% of the time at z>=3. Cause attribution to ink requires defeating photographic and parchment alternatives. When ratio is below 2, the result is explicitly inconclusive. The plan was written to `PREREG_RULES.txt` before the attack analyses were executed and was also stated in chat before the reconstruction. **Deviations:** conditioning temperature 1 for the cross-wing image-matched null was chosen during implementation and is exploratory; additional M/T subgroup and same-face checks were run after reading the suspect's log, but all figures were independently recomputed. The screening influence search used 350 null permutations per candidate; it is not a confirmatory inferential p-value. The main and final tail checks use 15,000 and 500,000 permutations respectively.

## Original-source re-derivation

- `words_y2014.csv.gz` Git blob SHA `e009ac3bde66505a18eb8332232e23f8ff5f9b04`: 11,309 words. Text glyph-count OLS fitted to *log* darkness; page mean of retained words, minimum 20/page; 151 pages. Each leaf is the unweighted mean of its available page/panel means. Numeric values preserved in `independent_word_page_stats.csv`, derived from the source gzip using an independent JavaScript decompressor and OLS. `rebuild_word_page_input.py` provides an independent Python reimplementation for external verification from the gzip file.
- The program independently reconstructed the bifolium, quire, leaf, side and panel from the physical-map leaf ranges. **All 151 annotations were cross-checked with primary `physical_map.csv`, 0 mismatches.**
- Source and own computed statistics **agree**: 37 sheets, conjugate gap 0.0567435 vs null mean 0.1103383, **effect 0.0535948, null SD 0.0104206, z 5.14** (claimed z 5.18). Independent JS word-to-leaf reimplementation gave z 5.12. Fore-edge only: effect 0.0504898, null SD 0.0111579, z 4.53 (claimed z 4.60).
- Facing pages: 49 coded facing pairs, including four *same-sheet* central folds. The original-style 49-pair effect is -0.007023, null SD 0.0125634, z -0.56 (claimed -0.57). **Inconclusive at this power:** the corrected 45 different-sheet pairs have effect -0.0112524, null SD 0.0130189, z -0.86.

## Three most damaging analytical problems

### 1. Scope: detected mainly in the illustrated herbal quires

The 22 measurable sheets in quires A-G have effect **0.0740719, null SD 0.0139557, z 5.31** under separate within-quire nulls. **Inconclusive at this power:** M/T, 11 sheets, have effect **0.0045669, null SD 0.0124396, z 0.37**. The four late O/Q sheets likewise are inconclusive at this power: effect 0.0745067, null SD 0.0469022, z 1.59. The entire nonherbal remainder of 15 sheets has effect 0.0232737, null SD 0.0154448, z 1.51: **inconclusive at this power**. This is NOT an assertion of absence beyond herbal: a planted shared-sheet SD of 0.12 log units is detected at z>=3 in only 33.2% of M/T runs. It is a failure to establish generality.

### 2. Ink versus image formation and physical skin

- The *focus/edge-softness* reading `sigma_hat` itself pairs within a sheet: effect **0.0341741**, null SD **0.0107215**, z **3.19**. The leaf-level log-darkness/focus correlation is -0.619, and the within-quire correlation is -0.557. On 22 fully four-page sheets, the contrast in sigma similarity across the two **same physical faces** versus opposite faces is **0.0900353, null SD 0.0278588, z 3.23**; the analogous darkness difference is 0.0250548, null SD 0.0108978, z 2.30. Both shared skin sides and image geometry are live confounds; which mechanism produced them is not identified.
- Leaf-level regression on `sigma_hat` with quire fixed effects leaves effect **0.024044**, null SD **0.007466**, z **3.22**. Subtracting a sheet-level shared-focus prediction leaves effect **0.026685**, null SD **0.007934**, z **3.36**. **Important algebra:** a sheet-common subtraction cannot change the gap of the *true* paired leaves; it only changes random-pair gaps and is not an ink-specific identification test. Multi-level focus/background removal leaves effect 0.025569, null SD 0.008271, z 3.09.
- Within-quire **cross-wing** null: effect **0.046582**, null SD **0.011550**, z **4.03**. Cross-wing pairings *weighted* to match focus and background: effect **0.024840**, null SD **0.008017**, z **3.10**. **Inconclusive at this power:** apply leaf-level focus/background regression AND focus/B-weighted cross-wing pairing: effect **0.013803**, null SD **0.007069**, z **1.95**. The image-matched null is an *exploratory conditional sensitivity*, not an exact conditional permutation test. This 1.95 result is not proof of zero real shared ink.
- The measured local parchment brightness `logB` itself pairs within a sheet: effect **0.007833**, null SD **0.002122**, z **3.69**. Removing measured leaf `logB` barely changes the darkness result: effect **0.054087**, null SD **0.010434**, z **5.18**. Thus the recorded *background tone alone* does not explain it, but shared parchment surface, translucence, hair/flesh side, thickness, ink penetration or acquisition geometry remain unmeasured. They could produce a shared image contrast without requiring one batch of ink.
- **Known-answer plant:** a truly independent latent same-sheet Gaussian signal with SD 0.12 log units passes z>=3 in 75% of unadjusted random-repair trials and 75% after leaf-focus regression; with the image-matched null plus leaf-focus/background regression the power drops to 33% (at 0.21 log units, ~78%). A true sheet effect correlated 0.8 with focus is largely erased by focus adjustment: at SD 0.12 the full model has 13% power at z>=3. Therefore *focus confound versus over-control is not identifiable from this source alone.* All plants were calibrated on independently shuffled within-quire data, not on the actual outcome's reported p.

### 3. The facing-page "no" is not established

**Inconclusive at this power:** the genuine different-sheet facing-page comparison has effect **-0.011252**, null SD **0.013019**, ratio **-0.86**, based on 45 pairs. The original 49 included four same-sheet central spreads. A planted **25% contraction** of paired facing-page differences is detected at z>=3 only **3.7%** of the time (650 simulations). A **50% contraction** is detected 80.8% of the time. This procedure can exclude very large aligned effects, not modest similarity. These power figures apply to paired-difference contraction, not to an additive shared-effect generative model; mechanism-specific power may differ.

## Tests that failed to demolish the association

- **Single-panel only:** 33 sheets, effect 0.050799, null SD 0.010141, z **5.01**.
- **Within-quire cross-wing shuffled pairing:** effect 0.046582, null SD 0.011550, z **4.03**.
- **One-sheet deletion:** smallest recomputed z among 37 leave-one-sheet-out runs was **4.41**; it is not carried by a single outlier. **One-quire deletion:** smallest z **4.37**.
- **Hostile post-outcome selection**: greedy removal of six selected sheets (seven effectively excluded when a remaining quire becomes a singleton) reduces the test to **inconclusive at this power**, effect **0.014059**, null SD **0.007186**, z **1.96**. This selection exploits the observed outcome and cannot be treated as a valid p-value. Another deliberately unfair benchmark is optimal matching by *observed darkness* within quire: gap 0.04625 versus the real 0.05674; outcome-selected pairing is not an exchangeable null.
- **Second camera campaign sensitivity:** 2004 `phase2_pages.csv`, previously derived from `phase5.pkl`, re-paired independently by leaf and quire: 35 sheets, effect **0.015588**, null SD **0.003920**, z **3.98**. Sigma association in this 2004 set is **inconclusive at this power**, effect 0.006833, null SD 0.012505, z 0.55. Darkness failed the 2004 blur-calibration gate; this is *supportive sensitivity only*, not a validated replication of ink darkness. **I did not deserialize `phase5.pkl` directly:** the accessible derived 2004 page CSV was used. Original 2004 pixels were not remeasured.
- **Multiplicity:** a new 500,000-permutation within-quire tail test recorded 0 exceedances. Observed effect **0.0535500**, null SD **0.0103754**, z **5.16**, Monte Carlo `(hits+1)/(N+1)` **2.0e-6**; multiplying conservatively by the earlier ~340-tested-result count gives **0.00068**, or **0.00083** for the subsequently reported 417 z/p-named entries. These are Monte Carlo estimates/bounds and are not a trial-family preregistration; the raw association survives even a severe Bonferroni check. Correcting p does **not** solve mechanism confounding.

## 60-day subjective survival estimates

- **Literal measured-darkness conjugate association in the sampled sheets:** **90%**. Numerous independently recomputed nulls and sensitivity tests survive; the signal is mainly herbal.
- **Generality across the whole manuscript:** **35%**. Outside herbal, sample sizes and planted power are insufficient.
- **Claim that it specifically evidences shared ink, distinct from same parchment, surface and photography:** **25%**. No identifying experiment was available with these files.
- **Strong claim that facing different-sheet pages genuinely lack a comparable similarity of any moderate magnitude:** **30%**. The corrected test detects only quite large planted differences reliably.

These percentages are subjective calibrated judgments, **not** posterior probabilities computed from the p-values. A falsification would require real image calibration across controlled face, quire position and capture-session metadata, independent material/reflectance markers, or an acquisition-invariant ink estimator. **The conclusion is about measurement association, not inking sequence, provenance or shared ink stock.**

## Source links and limitations

- [2014 word CSV.gz](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/words_y2014.csv.gz) (original instrument data)
- [2004 word CSV.gz](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/words_y2004.csv.gz)
- [2004 source-derived page measures](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/phase2_pages.csv)
- [Physical map](https://github.com/eo777/vms-spatial/blob/main/data/manuscript/physical_map.csv)
- [Instrument](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/eo777-ink-session-2026-11-14/experiments/eo777_ink_session_2026-11-14/inkdirect/instrument.py)

The reconstructed source measurement has tiny numeric differences from the original (word canonicalisation and regression preprocessing were independently implemented). Key results agree to under ~0.5 null SD. The 2014 source word values were read in full, and page/leaf metadata were checked against the original mapping; no primary imaging pixel checks were possible in this run. The 2004 values were not read from the binary pickle directly. The photographer's session, white-balance, lighting, camera settings, page curvature and exact parchment side were not available, so a defensible *causal* explanation is unidentifiable. No invented observations or source-specific numbers are used.

## Reproducibility and atomic checkpoints

This package contains the independently reconstructed page table, source reconstruction recipe, independent pair recomputation, inference code, all JSON results, the running results with retractions at the top, and separate atomic pickle files `01`–`07`. Each pickle is written to a `.tmp` and atomically renamed after flushing. `rebuild_audit.py` and the follow-on scripts run in order from the saved page table without relying on original `analysis6.py` or `phase9.py`. The original large upstream word gzip is linked, not bundled, to preserve exact provenance. Effects are `null mean minus observed` for the absolute-gap tests, so positive means same-sheet or facing similarity.
