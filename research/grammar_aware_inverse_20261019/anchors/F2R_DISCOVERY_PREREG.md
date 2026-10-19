# F2R-CENTAURY discovery text assay — PRE-REGISTERED 2026-10-19

Status: descriptive hypothesis **already selected** by Stephen Bax (2014). This is NOT a blinded or independent decipherment experiment.

External anchor is BL Egerton MS 747 f21, a prior medieval illustrated herbal with manuscript-attested Centaurea minor/major; source https://searcharchives.bl.uk/catalog/032-001983805. This proves independent centaury terminology and depiction existed, NOT that the Voynich f2r image or text represents centaury.

Voynich image f2r (Herbal-A, Currier A) Yale IIIF https://collections.library.yale.edu/iiif/2/1006078; plant candidate database field Centaurea diffusa is unreviewed and derives from prior interpretation. No proven pictorial identity. f2r words line1 word0 and line8 word0 in pinned ZLZI corpus are kydainy and kydain. Line8 visual paragraph start not independently verified.

## Locked measurements

1. Fetch the public ZLZI corpus at GitHub fixed SHA 26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f. Count line tokens only if unit contains P (running text); never labels or circular.

2. Normalize nothing except lowercasing corpus words; preserve word boundaries. Compute Levenshtein integer and normalized edit distance for f2r line1/line8 first words.

3. Matched panel A: every OTHER Currier-A Herbal registered folio with line1 and line8 both running, both first tokens lower-case ASCII and each of length 5–10. Condition on first-word character lengths separately, within at most 1 character of f2r lengths (7 and 6). No other selection, no drawing/theme-based selection.

4. Matched panel B: within f2r line1 against first word of lines 2–13 excluding line8, same text type and eligible length within one of 6 chars. This checks folio-local near-repetition.

5. Descriptively report proportion of panel A with raw edit distance ≤1; also proportion of both panels with normalized distance ≤ target; exact control records. No one-sided significance claim because candidate and line8 were previously selected precisely for their similarity.

6. Primary failure of anchor is missing expert-reviewed Voynich plant identity and independently aligned text. Even positive rarity will NOT discriminate plant-name lexeme from production/FORM/LINE_ENTRY convention.

7. To promote: independently identify f2r without seeing selected transcription; verify exact text unit segmentation; use independent source-label exemplars on additional folios and heldout bifolia; test semantic vs production rival under both controls.

Data control register research/grammar_aware_inverse_20261019/anchors/herbal_a_register.json from research-agent public.vms_realization_folio_split_v01, N=89 Currier A/H. The stored DISCOVERY/VALIDATION split does not guarantee physical-bifolium independence and is not used to claim heldout validation.