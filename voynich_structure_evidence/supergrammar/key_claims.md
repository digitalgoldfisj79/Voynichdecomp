# Super Grammar claims relevant to the transformation / ordering questions

Exported verbatim from the claims register (`claims_register_v03.csv`, all 147 rows). Status is the register's own
`certification_status`: only CERTIFIED / CERTIFIED_BOUNDED items are part of the certified v03 kernel; OPEN_UNCERTIFIED and
OPEN_HELD are provisional; TERMINAL_REJECTED means the claim was tested and rejected. Where a claim appears under more than one run, each row is shown.

## SG64_PORTABLE_REALIZATION_OPERATOR_LIBRARY  (OPEN_UNCERTIFIED; run supergrammar_v04_realization_library_20260924; layer surface_choice)

**Statement.** Beyond the Xy<->Xdy operator, ordinary Voynich running text contains a portable family-conditioned one-edit realization layer: multiple operator classes discovered in Herbal A/B preserve the same A/B polarity under independent Pharmaceutical-A versus Biological+Stars-B validation.

**Evidence.** 21 context-defined one-edit operator classes preserve discovery polarity and clear >=2 null SD in the independent domain. Large portable operators include pre-final d insertion before y (6.87 SD discovery; 4.83 SD validation), pre-final a/o (3.81; 5.31), internal e/o (3.64; 4.59), internal o insertion before d (3.96; 4.40), and pre-final d/e (5.34; 3.95).

**Falsifier / bound.** Kill or narrow if alternate transcription/atomic replay fails materially, clean-room implementation does not reproduce the 21-class frontier, or matched controls show the operator polarity is an artefact of one-edit support geometry.

**Provenance.** Exhaustive 5,076-type one-edit graph. Discovery restricted to Herbal A/B; validation restricted to non-Herbal Pharmaceutical-A versus Biological+Stars-B. Operator statistics in vms_realization_operator_outdomain_v01.

## SG65_REALIZATION_FAMILY_COMPRESSION_BOUND  (OPEN_UNCERTIFIED; run supergrammar_v04_realization_library_20260924; layer surface_choice)

**Statement.** A conservative portable realization library substantially reduces exact-EVA type and hapax counts, but the resulting components are formal surface families rather than demonstrated semantic equivalence classes.

**Evidence.** Top 8 externally portable operators reduce 5,076 exact types to 3,815 families (24.84% reduction), touch 65.92% of token occurrences, and reduce 3,375 raw hapax types to 2,594 family hapaxes (23.14%). Adding independently established k/t and p/f gallows relations yields 3,394 families (33.14% type reduction), touches 74.05% of token mass, and reduces family hapaxes to 2,328 (31.02%). All 21 externally validated operators give a broader 35.20% type-reduction bound before gallows.

**Falsifier / bound.** Do not interpret component membership as synonymy/plaintext identity. Kill the compression interpretation if operator replay fails; bound downward if graph chaining proves to merge distinct functional families.

**Provenance.** Derived graph-component bound from vms_realization_library_summary_v01; operators are externally validated SG64 classes plus separately tested gallows pairs where stated.

## SG66_LATENT_ROOT_OBFUSCATION_HYPOTHESIS  (OPEN_UNCERTIFIED; run supergrammar_v04_realization_library_20260924; layer latent_state)

**Statement.** A subset of the validated surface-realization operators is consistent with many-to-one latent-root obfuscation: collapsing those families removes substantially more Currier A/B lexical divergence than size-matched random collapses, and several operator pairs occupy unusually similar external contexts. This supports a latent-root mechanism hypothesis but does not yet establish semantic identity for every operator family.

**Evidence.** Raw A/B lexical JSD=0.46057. Top8 collapse lowers to 0.32460 (-29.52%); size-matched random-collapse mean reduction=0.07913, SD=0.00401; excess reduction=14.17 null SD. Top8+gallows lowers to 0.30624 (-33.51%); excess over size-matched random collapse=9.24 null SD. Across five strongest portable operators, external-context JSD=0.65227 versus matched one-edit null mean=0.79713, null SD=0.01533; similarity excess=9.45 null SD. Operator-specific context support: Xy/Xdy 5.59 SD; pre-final d/e 25.44 SD; pre-final a/o 2.67 SD; internal e/o 1.77 SD unresolved; internal o-before-d insertion 1.46 SD unresolved.

**Falsifier / bound.** Reject universal same-root semantics if variants consistently retain distinct external root-context distributions after macrostate/layout controls, if contextual similarity disappears under alternate transcription or stronger morphological matching, or if root collapse fails to improve held-out sequence prediction. Treat unresolved operators as functional alternations rather than root-preserving until independently supported.

**Provenance.** Post-SG64 hypothesis test on conservative top8 realization components plus separately established gallows; Currier A/B JSD collapse null and root-neighbour context JSD matched one-edit controls.

## SG47_NPCOMPL_MULTI_REGIME  (OPEN_HELD; run supergrammar_v03_20260920; layer register)

**Statement.** The proposed early-A / late-A-pharma / B-core / biological / ACZ page ecology is a useful descriptive partition, but it is not resolved as an independent held-out grammar layer under whole-physical-quire validation.

**Evidence.** REGIME5 reduces internal codelength by +3.24745 bits/token-equivalent versus the frozen parent, but physical-quire null SD is 2.46244 (z=1.33): the metric does not resolve this. REGIME5 versus Davis hand is likewise unresolved (gain +1.07743, SD 2.80476, z=0.38). BIO is single-quire; CELESTIAL is hand-4-confounded; B_STARS is effectively single-quire. Splitting B_HERBAL from B_STARS is not supported.

**Falsifier / bound.** Promotion requires independent/crossed support where each claimed regime occurs in multiple physical production units and is not deterministically tied to section/hand. A replicated whole-quire gain with z>=2 would reopen it.

**Provenance.** Voynich Ninja thread 6084 audit + Super-Grammar parent mediation 2026-09-18

## SG12_CURRIER_COORDINATE  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer modulation)

**Statement.** Currier A/B is a useful coarse graphotactic/register coordinate, but it is not independently identified as a grammar switch apart from Davis hand and section.

**Evidence.** Two-stage whole-quire audit. First, high-dimensional exact-token children for section-only, Currier, Davis hand and hand+Currier all selected the off-switch lambda=1e9 in 5/5 folds: no exact-vocabulary/register child is needed. Second, a lower-dimensional normalized character correction resolves versus section-only: Currier gain +0.051076 bits/token with quire-block null SD 0.021595 (z=2.36); Davis-hand gain +0.045818 with SD 0.019056 (z=2.41). Currier beats hand by only +0.005258 with SD 0.004419 (z=1.18): THE METRIC DOES NOT RESOLVE THIS. No section+Davis-hand cell contains both A and B, so Currier-after-hand is structurally non-identifiable. The residual signal is therefore best treated as a coarse graphotactic/register coordinate shared with hand/task ecology, not a separate A/B grammar.

**Falsifier / bound.** An independent Currier register can only be promoted with data containing A/B overlap within the same section+hand/task context, or an external/replicated corpus where Currier-specific held-out gain persists after those variables are controlled. Section-localized gains in single-Currier sections are not evidence for Currier because shared lambda tuning can alter shrinkage there.

**Provenance.** surface canonicalisation audit; Claude q72-style residualisation

## SGK17_CURRIER_NOT_INDEPENDENT  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer modulation)

**Statement.** Current manuscript support cannot identify Currier A/B as an effect independent of both section and Davis hand because no observed section-plus-hand stratum contains both A and B.

**Evidence.** Currier and hand both improve lower-dimensional character prediction versus section, but Currier-versus-hand is unresolved and Currier-after-hand is structurally non-identifiable.

**Falsifier / bound.** Can be reopened only by crossed A/B support within the same section+hand/task context or independent external replication.

**Provenance.** Currier support audit and whole-quire nested tests in SG1.0; vms_supergrammar_currier_support_v01.

## SGK17_CURRIER_NOT_INDEPENDENT  (OPEN_HELD; run supergrammar_v031_20260928; layer modulation)

**Statement.** Current manuscript support cannot identify Currier A/B as an effect independent of both section and Davis hand because no observed section-plus-hand stratum contains both A and B.

**Evidence.** Currier and hand both improve lower-dimensional character prediction versus section, but Currier-versus-hand is unresolved and Currier-after-hand is structurally non-identifiable.

**Falsifier / bound.** Can be reopened only by crossed A/B support within the same section+hand/task context or independent external replication.

**Provenance.** v03.1 correction candidate of supergrammar_v03_20260920; Currier support audit and whole-quire nested tests in SG1.0; vms_supergrammar_currier_support_v01.

## SG08_PAGE_LEAF_REGIME  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer regional_state)

**Statement.** Surface/subword regime is strongly localised at page and especially physical-leaf scale.

**Evidence.** Same-page and same-leaf paragraph similarity strongly exceed matched nulls; leaf-vs-other-leaf boundary contrast remains resolved in trigram and word representations.

**Falsifier / bound.** Must remain after exact section/hand/illustration matching and representation sensitivity.

**Provenance.** Sept 2026 boundary localisation/local-state carrier

## SG09_BIFOLIUM_NOT_INDEPENDENT_STATE  (TERMINAL_REJECTED; run supergrammar_v03_20260920; layer regional_state)

**Statement.** Do not model bifolium as an independently established linguistic state unit.

**Evidence.** Bifolium residual appears in trigram representation but collapses under word-set representation and strict equal-gap leaf-pair control.

**Falsifier / bound.** Reopen only if a representation-robust exact-distance bound resolves.

**Provenance.** Sept 2026 boundary localisation

## SG57_BIFOLIUM_REPERTOIRE_STATE  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer repertoire)

**Statement.** A low-dimensional repertoire-plan state has a reproducible physical-bifolium component, including a two-state mode within the clean Herbal hand-S1 subset that survives whole-quire holdout.

**Evidence.** Herbal q01-q08 leave-one-whole-quire-out: K2 low-rank uplift +.1440 Jaccard and dominant-component D2 +.1457, both positive 8/8 quires with exact p=.00778. In Herbal hand S1, same dominant K2 state occurs within physical bifolia at .810 vs .504 state-count null (z=7.06). Global D4 bifolium enrichment remains after section+hand matched permutation (z=3.04,p=.006), while adjacent manuscript-page state persistence is unresolved.

**Falsifier / bound.** Does not identify causal selection of the state. State may reflect bifolium task/exemplar/content/material ecology. Requires constructive forward replay and external/codicological discrimination.

**Provenance.** SG55 follow-ups C/D and four-state persistence localisation, 2026-09-18

## SG23_REMOTE_STATE_REUSE  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer regional_state)

**Statement.** Subword regimes recur on remote pages/quires after matching broad metadata, so local state is reusable rather than purely continuous page drift.

**Evidence.** Odd/even split-half remote recurrence resolves in both directions outside quire after Currier+hand+illustration matching.

**Falsifier / bound.** State carrier/meaning remains unknown; require representation sensitivity and no content gloss.

**Provenance.** Sept 2026 local-state carrier

## SGK12_REMOTE_STATE_REUSE  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer regional_state)

**Statement.** A subword-state match selected on one split of paragraphs predicts a remote matched page on the held-out split in both split directions.

**Evidence.** Even-heldout effect 0.0196012/null SD 0.00677852, z=2.892; odd-heldout 0.0273737/0.00733911, z=3.730.

**Falsifier / bound.** Fail if an independent split replay loses direction or falls below the frozen matched-null gate.

**Provenance.** local-state carrier programme; vms_local_state_carrier_results_v01.

## SGK10_SHORT_STATE_LAG1  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer regional_state)

**Statement.** Local subword-state similarity remains elevated at paragraph lag 1 under the registered matched null.

**Evidence.** Primary trigram lag1 effect 0.0121257 with null SD 0.00353763, z=3.428.

**Falsifier / bound.** Fail if independent matched-null replay does not resolve or if effect is explained by frozen physical/register controls.

**Provenance.** local-state carrier programme; vms_local_state_carrier_results_v01.

## SGK11_SHORT_STATE_LAG2  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer regional_state)

**Statement.** Local subword-state similarity remains elevated at paragraph lag 2 under the registered matched null.

**Evidence.** Primary trigram lag2 effect 0.0125631 with null SD 0.00219444, z=5.725.

**Falsifier / bound.** Fail if independent matched-null replay does not resolve or if effect is explained by frozen physical/register controls.

**Provenance.** local-state carrier programme; vms_local_state_carrier_results_v01.

## SG22_REPERTOIRE_RESERVOIR  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer repertoire)

**Statement.** Generation/selection operates from active local repertoires rather than a single manuscript-wide IID lexicon.

**Evidence.** Page/local established-vocabulary reservoirs plus controlled replay/innovation improve held-out prediction and diversity; page/leaf localisation independently supports local repertoire.

**Falsifier / bound.** Bifolium reservoir parameter should remain optional/held out because independent bifolium state is unresolved.

**Provenance.** Aug-Sep 2026 working-set and localisation programmes

## SG29_LINE_BOUNDARY_INNOVATION  (OPEN_HELD; run supergrammar_v03_20260920; layer innovation)

**Statement.** Novel/changed morphological material is enriched at physical line boundaries, especially line starts, but old PGCS-triple estimates are representation-bound.

**Evidence.** Old first/mid/last PGCS triple innovation rates showed strong first-line-position enrichment and section-stratified persistence. Later representation corrections demote PGCS-specific magnitudes.

**Falsifier / bound.** Rerun with frame-neutral atom/token-family innovations and page-block null before promotion.

**Provenance.** ChatGPT 2026-03; later line programmes

## SG31_DERIVATIVE_COPY_PARTIAL  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer near_form)

**Statement.** Local derivative/copy-mutate processes are sufficient for part of the near-form structure but are not a complete grammar.

**Evidence.** Faithful derivative mechanisms repair some ED1/local structure but fail joint surface closure and diversity/lag geometry.

**Falsifier / bound.** No historical copy-mutate claim without unique discriminant.

**Provenance.** faithful Timm and generator ablations; master consolidation

## SG25_NO_PARAGRAPH_CONTENT_STATE  (TERMINAL_REJECTED; run supergrammar_v03_20260920; layer semantic_state)

**Statement.** No reusable paragraph-level abstract/content state has been admitted by calibrated tests.

**Evidence.** Hard same-page adjacent-paragraph controls are all below the project resolution threshold; cross-bifolium state instrument failed synthetic power and target remained sealed.

**Falsifier / bound.** Reopen only with a calibrated representation recovering planted reusable state >=80% power <=5% FP.

**Provenance.** Sept 2026 latent-content closeout

## SG33_NO_TRANSFERABLE_CODE_UNIT  (TERMINAL_REJECTED; run supergrammar_v03_20260920; layer latent_state)

**Statement.** No stable transferable cross-scribe latent word/code unit has been recovered.

**Evidence.** Cross-scribe latent scale was unstable/nonreplicating; recurring payload channels did not survive correction.

**Falsifier / bound.** Does not exclude meaning carried in other representations; only blocks this grammar layer.

**Provenance.** July-Aug 2026 cross-scribe scale and payload programmes

## SG10_SECTION_REGISTER  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer modulation)

**Statement.** Section/register is a major conditioning variable on token families and surface probabilities.

**Evidence.** Section effects recur across morphology, repertoire and prediction models and are routinely required as a confound/control.

**Falsifier / bound.** Promoted rules should report section heterogeneity rather than treat section as nine independent grammars.

**Provenance.** v2.3; generation master; repeated held-out work

## SG11_HAND_MODULATOR  (OPEN_UNCERTIFIED; run supergrammar_v03_20260920; layer modulation)

**Statement.** Davis hand can modulate surface statistics but is not a standalone compact grammar.

**Evidence.** Hand labels are nonexchangeable descriptively, but hand-only grammar tests fail matched-null/complexity comparisons; local subword state can cross a secure hand boundary on f115r.

**Falsifier / bound.** Require within-section/within-task held-out improvement before adding hand-specific rule parameters.

**Provenance.** generation master; hand hostile tests; f115r natural experiment
