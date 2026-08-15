# Medieval Magic Formula × Voynich Mechanism Discriminator v0.2 FULL
Date: 2026-08-15
Seed: 20260815
Parent: v0.1 preflight classifier only.

## Aim
Run the full mechanism battery originally intended in v0.1. v0.1's omnibus A/B/C classifier is retained as one descriptive assay and is not a stop switch for unrelated mechanism tests.

## Integrity
1. All metric definitions, generator grids, nuisance controls and decision rules are frozen before this v0.2 run.
2. External development/qualification is completed and hashed before Voynich RF data are acquired/parsed by the runner.
3. No post-Voynich tuning or threshold changes.
4. Every planned test is run if its required external data exist. Unsupported tests are reported `NOT_TESTABLE_SOURCE_LIMITATION`, never silently synthesized.
5. Results are continuous effect sizes + bootstrap CIs + permutation p-values. BH-FDR q<=0.05 determines which external assays are inferentially qualified; no arbitrary omnibus AUC gate stops the programme.
6. f116v marginalia are not part of the main-text statistical test.

## External classes
A ordinary medieval Latin/German prose from the frozen seven-source control set.
B Lecouteux entries explicitly tagged corruption/hybrid (`class_label=B`).
C productive/opaque formula material (`class_label=C` or `C_single`), with productive-family subset separately analysed.
D real document-order mixed medical miscellany: only if an independently ordered source is present. Current frozen source pack does not contain one; expected status `NOT_TESTABLE_SOURCE_LIMITATION` unless discovered in the committed pack without web selection after run start.

## Splits
A split by source document; B/C split by deterministic hash of source_work + entry_id. No entry crosses splits. Development 60%, validation 20%, heldout 20%.
For block metrics, entries are concatenated only within split and class into deterministic 160–320-character corpus blocks; entry boundaries are retained as line boundaries. These are corpus blocks, not manuscript reconstructions.

## F1–F7 battery
F1: character/unit H0; conditional entropy orders 1–3; MI lags 1–20; excess-MI sum; zlib/bz2/lzma compression ratio.
F2: token length/TTR/hapax; nearest normalized edit distance; one-edit graph degree/component fraction; prefix/suffix sharing 1–3; shared-core coding ratio; branching entropy.
F3: exact repeat fraction/distance; adjacent similarity; near-copy rates lags 1–10; local-vs-random mutation advantage; template recurrence.
F4: line-initial/medial/final length and vocabulary divergence; source-structure proxy externally, genuine lines on Voynich.
F5: source/section conditioning: local-vs-global token cross-entropy gain and mean pairwise JSD; external source proxy, genuine Voynich sections.
F6: burstiness; exact/family recurrence lags 1–20; max half/sliding JSD change-point score and permutation-normalized excess.
F7: directional compression-transfer and symmetric NCD using zlib/bzip2/lzma on frequency-rank canonicalized character and token encodings. This removes alphabet-name identity as evidence.

## External qualification per metric
For each metric separately compare A-vs-B and A-vs-C on heldout blocks. Record standardized median difference, Cliff's delta, bootstrap 95% CI and 2000 source/block-label permutations. BH-FDR across all primary A/B and A/C metric tests at q<=0.05.
A metric is `QUALIFIED` for Voynich inference only if q<=0.05 and the bootstrap CI for the median difference excludes zero. Non-qualified metrics remain descriptive and cannot contribute to composite distances.

## Historical generators G0–G8
G0 ordinary character Markov baseline fitted to A development.
G1 scribal corruption of A using deletion/duplication/fusion/split plus attested confusion set rn<->m, ni<->m, st<->h, p<->f. Rate grid {0.01,0.03,0.06,0.10}; select on external development only.
G2 sound-class-preserving corruption (vowel↔vowel, consonant↔consonant) grid {0.02,0.05,0.10,0.15}; explicitly weak historical proxy.
G3 formula-family generator learned from within-formula near-neighbour transformations and productive-family C development rows.
G4 repetition/reduction generator learned from C development token/reduction statistics.
G5 prose/charm switching model only if D exists; otherwise NOT_TESTABLE.
G6 abbreviation/segmentation generator fitted only from abbreviation_evidence development rows; if support <20 rows, report low-support and do not use inferentially.
G7 simple source-attested controls: Caesar-like permutation, alphabet reversal, vowel masking/substitution. Structural metrics should be invariant or near-invariant; this is a null/control, not a cipher search.
G8 fixed combinations G1+G3, G2+G3, G1+G4, G3+G6; component parameters are frozen from external development.

Each generator produces 200 heldout-length replicates. Compare to heldout A/B/C in qualified metric space. Report nearest class, standardized distance, and fraction of qualified metrics within the class's empirical 95% interval. No generator is selected using Voynich.

## Adversarial N1–N6
N1 pseudo-Latin/German pronounceable-ish nonsense from A character transition model with A word-length distribution.
N2 matched order-2 Markov baseline.
N3 generic stem/affix slot grammar learned from A (tests whether C-like family geometry is generic morphology).
N4 corruption-only = G1.
N5 formula-only = G3.
N6 deterministic interleaving of unrelated A sources at matched block lengths.
All are scored identically to G0–G8.

## External freeze then Voynich
The runner writes and hashes `external_freeze.json` before downloading/parsing RF1b/bitrans/STA-aaa resources.

Voynich source is pinned RF1b reduced STA1 SHA-256 81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17.
STA-aaa.bit SHA-256 622621463ff2973ff456b02f0b46ba99fef8ad9103c464e44427762863e3cb64.
bitrans.c SHA-256 3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a.

Four analysis layers from the RF source:
R0 `RF_WORD`: long-word token strings in RF1b (word-level representation).
R1 `STA_FAMILY`: each STA member collapsed to family letter.
R2 `FULL_STA`: atomic RF STA member codes.
R3 `AAA_CONNECTED`: official STA→aaa conversion; colon-connected aaa pairs are one unit.
These are correlated views, not independent replications.

Voynich is evaluated by folio and as full corpus. F4 uses genuine line boundaries. F5 uses frozen `voynich_section_map.json` from repository if available; section-dependent metrics are otherwise omitted with explicit reason.

## Primary Voynich inference
For each qualified external metric, compute Voynich standardized location relative to heldout A, B and C distributions at each R0–R3 layer. Also compute nearest-centroid ridge-Euclidean distance in externally standardized qualified metric space.
Report per-layer closest class and Δ_magic = d(V,A)-min(d(V,B),d(V,C)).
No lexical matches enter the score.

A robust whole-manuscript affinity requires the same B/C class to beat A in at least 3/4 correlated layers and the sign to survive folio bootstrap. This is a robustness label, not a p-value.

## Generator-to-Voynich test
Apply frozen G/N generators to A development sources at Voynich-matched lengths. Score in the same qualified metric space. Report which historical mechanism, if any, is closer to each Voynich layer than G0/N1/N2/N3 controls and its metric-coverage fraction. No >=70% hard threshold is imposed post hoc; coverage is reported continuously with bootstrap intervals.

## Interpretation
- `NO_EXTERNAL_ASSAYS_QUALIFIED`: programme cannot compare mechanisms.
- `NO_ROBUST_MAGIC_AFFINITY`: qualified metrics exist but B/C do not beat A robustly across representations.
- `STRUCTURAL_MAGIC_AFFINITY_NONEXPLANATORY`: B/C are closer, but historical generators do not outperform adversarial controls.
- `HISTORICAL_MECHANISM_CANDIDATE`: B/C closer across >=3/4 layers and at least one frozen historical generator beats G0/N1/N2/N3 on the majority of qualified metric families with folio-bootstrap sign stability.
- D/G5 unsupported does not block other results.
