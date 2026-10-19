# Anchor-1 real f2r pilot: completed build and first data gate
Date: 2026-10-19. This is the successor to the synthetic IDENT-GATE-1, **not** a Voynich translation.

## Actually delivered
- [Experiment protocol](ANCHOR1_F2R_PREREG_20261019.md)
- [Complete machine-readable f2r/Egerton witness manifest](anchor1_f2r_manifest_20261019.json)
- [Executable source-ecology audit](anchor1_real_loci_audit.py)
- [Corrected independently section-controlled audit](results/anchor1_real_loci_canonical_20261019.json)
- [A standalone, interactive manuscript comparison and label-rectangle reviewer](anchor1_f2r_review.html). Also delivered as a locally openable `anchor1_review_f2r.html` artifact with in-browser JSON export. Image URLs verified HTTP 200 with actual JPEG headers for Yale f2r and four British Library folios. Javascript passed `node --check`. Browser-render test was **not** completed (local Chromium could not start headless); do not claim full browser UX qualification.
- [Evidence adjudicator](anchor1_adjudicate.py) with [actual pre-review status result](results/anchor1_adjudication_pre_review_20261019.json).

## PRIMARY real corpus facts
Public TTLI/ZLZI corpus checksum `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f` verified on remote compute.
- In **ZLZI only**, f2r line **14** is explicitly `@Lp` plant label `ytoail` and line **15** is `@L0` `ios an on`. TTLI omits them, rather than agreeing on their exact spelling; do NOT pool layers.
- On f47r, the first running-text token `pchair` is present in TTLI but absent in ZLZI. Earlier cryptanalytic first-word assumptions using it must declare their layer.
- Very important cohort correction: numeric range f1–f66 does not equal Herbal-A. Read-only canonical Supabase `vms_catalogue.vms_folios` classifies f1r/f66r as text-only, f57v as cosmological. The eight ZLZI *preliminary* low-numbered label folios shrink to **five canonically Herbal-A folios with any labels**: f2r, f11v, f17r, f49v, f65r.
- More restrictive labels containing any >=3-letter EVA token appear at **four separate loci on only three Herbal-A folios**: f2r line14 `ytoail`; f2r line15 `ios an on`; f17r line13 `oteeeon oiil`; f65r line1 `otaim dam alam`. Many other `L` loci are single glyphs or ambiguous marks. This cohort has **three** folio-level candidate units, not 77 independent specimens.
- TTLI's only nontrivial canonical Herbal-A label locus in this audited group is f65r line1. This is a severe layer and population restriction.

## Independently grounded external botanical alternatives (not matches)
- British Library Egerton MS 747 dated 1280–1350 (institutional primary catalogue), text and illustrations in Latin with IIIF manifest: `https://bl.digirati.io/iiif/ark:/81055/vdc_100058663072.0x000001`.
- Exact verified primary IIIF canvases: **f21r** `...0x000031/canvas/c/49`, **f21v** `...0x000032/canvas/c/50`, **f72v** `...0x000098/canvas/c/152`. External descriptions identify historical `centaurea minor/maior` around f21 and `peonia` on f72v; neither has yet passed independent crop/reading verification. f20v is a matched adjacent manuscript-style control but has no certified species label.
- Voynich image official Yale IIIF `https://collections.library.yale.edu/iiif/2/1006078`; f2r wholeplant object from internal registry unreviewed, a proposed `Centaurea diffusa` identity inherited from a modern list is **not** independent botanical ground truth; sources also propose cornflower and peony.
- Taxonomy barrier: early manuscript `centaurea minor/maior` may name centaury species and must not automatically be mapped to modern `Centaurea cyanus` or `Centaurea diffusa`.

## Competing mechanisms / discrimination check
**Content model:** `ytoail` corresponds reproducibly to a independently authenticated species identity. For a test this predicts a specific botanical category at held-out *other same-register* manuscript labels, not merely similarity of text shape or illustration style.
**Production-only:** labels reflect drawing placement, register, folio/session conventions or arbitrary object indexing independent of species. Same folio co-location is expected under either model.

There are insufficient physically independent labelled Herbal-A folios and **no** independent species-level Voynich anchor. Therefore with current data, a semantic-vs-production **statistical test is not estimable**. Do not reuse running text, f66r text-only glyph rows or pharmaceutical labels as fake replicates. Do not run the old GPU dictionary search on `ytoail` and then select the best-fitting plant name: that would select the anchor from the answer.

The adjudicator correctly returned:
- `label_semantics: UNKNOWN`
- `status: NOT_QUALIFIED_MISSING_INDEPENDENT_ANCHORS`
- positive text locus physically identified, but target botanical identity, exact pixel rectangle, crop-verified source labels, independent blind expert reviews, matched production negatives, and independent heldout folios **all absent**.

## What the interactive instrument does
1. Opens actual Yale f2r image alongside Egerton MS747 f20v, f21r, f72v and f21v. All five image endpoints returned JPEG HTTP 200 in remote validation.
2. Records TWO clicks identifying the Voynich plant-label rectangle, normalized to the displayed image geometry. The page line number alone cannot supply this geometry.
3. Records morphology before supplying the text-label `ytoail` or explicit external botanical hypotheses.
4. Collects image similarity scores 0–3, production-alternative plausibility, and reviewer expertise/prior exposure; locks the observations and downloads a local JSON file.
5. Only after lock reveals the source key and ZLZI text. **Not a true blinded study** because full reference folios retain visible Latin botanical labels; independent cropped/masked botanical references and multiple independent reviewers remain a must.
6. [Adjudicator](anchor1_adjudicate.py) can read such JSONs and reveal only admissibility, descriptive similarity medians and missing prerequisites. It is deliberately incapable of outputting a translation of `ytoail`.

## Exit verdict
**Build done. Real corpus + manuscript source audit done. First candidate not qualified for semantic recovery, due to lack of independent botanical ground truth and tiny eligible label cohort.**

A stronger next empirical experiment would require an independent manuscript illustration/label collection with greater physically separate sample size. If switching to Zodiac circular labels or Herbal running text, preregister that ecology as a NEW experiment: the f2r label cannot be treated as ground truth for another register. Do not promote the preliminary f2r candidate after viewing the reviews.
