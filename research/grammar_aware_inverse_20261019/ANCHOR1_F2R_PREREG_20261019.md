# ANCHOR-1: One real manuscript anchor, f2r (2026-10-19)
Status: **BUILT FOR DISCOVERY; SEMANTIC CLAIM UNQUALIFIED**. It is a real Yale/BL comparison, not a decipherment, and the sample is far too small for an inferential decision.

## Independent inputs
- Yale Beinecke MS 408 **f2r**, Yale IIIF image service `https://collections.library.yale.edu/iiif/2/1006078`. The live original f2r image is essential; avoid thumbnails that mix other folios.
- Actual corpus frozen SHA256 `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`. **ZLZI locus line 14 `@Lp` = `ytoail`**; line 15 `@L0` = `ios an on`. Neither is in TTLI. Do not silently substitute TTLI first-word evidence for label evidence.
- Independent folio cataloguers ([voynich.nu q01](https://voynich.nu/q01/index.html), [VIB](https://vib.tamagothi.de/index.php?id=f2r&show=page)) agree a small isolated plant label exists but do not identify the plant with certainty.
- Our vms_catalogue.vms_objects f2r object `47d29a8b-1a67-45d8-abd0-db0ac4e1c187` is **unreviewed**; `Centaurea diffusa` is an inherited candidate, not ground truth. Rival proposals include `Cyanus/Centaurea` and `Paeonia`.
- British Library [Egerton MS 747](https://searcharchives.bl.uk/catalog/032-001983805) Latin illustrated Tractatus de herbis, c.1280–1350, digitised [IIIF](https://bl.digirati.io/iiif/ark:/81055/vdc_100058663072.0x000001). Registered historic botanical terms occur around f21r–v (`centaurea minor/maior`) and f72v (`peonia`). **Medieval centaury is not automatically modern cornflower**. No palaeographical crop/identity review has been conducted.
- A **four-reference viewer** uses Egerton f20v (adjacent-style control), f21r and f21v (centaury-sequence alternatives), and f72v (peony). Their labelled words are visible in whole-folio images, so stage one is **not fully blinded**. The comparator plant region and medieval caption must be masked and verified before confirmation.

## Separate the hypotheses
**SEMANTIC:** `ytoail` is linked to a source botanical identity. Predicts a reproducible relation among plant-verified matches, annotated labels, and source term categories on *previously unseen* herbal folios, not mere neighbouring physical text or similar stylised leaves.
**PRODUCTION:** `ytoail` is a line/object/page marker, arbitrary annotation, or context-conditioned token determined by drawing, placement, manuscript session or other structural constraints. Predicts contextual and shape matches without specific species identity.

With one target page and no independently verified botany, the two models need not differ observably. The first phase is **data qualification**. No lookup of `ytoail` against candidate plant name spellings and no posthoc key search is authorised. The f2r test is a herbal **label** ecology; never pool it with balneological running text or pharmaceutical labels.

## Registered sequence — no confirmation shortcut
1. **Freeze the real loci, both source layers and relative image objects.** Audit label types and count physically independent herbal folios. Reject inconsistent or conflated script layers.
2. **Image-only expert review** by multiple genuinely independent botanically trained assessors. Crop the target plant and corresponding 13th–14th-century witness plants. Redact Latin captions on comparison images before offering candidate names. Freeze morphological ratings and record negative similarities independently.
3. **Text-only palaeography.** Different reviewers verify line 14 `ytoail`, its pixel bounding rectangle and alternative glyph readings on the physical scan, independent of botanical guesses. Check source transcriptions. A line 14 location is NOT a pixel coordinate.
4. **Period nomenclature crosswalk.** Specialist distinguishes early *Centaurea* minor/major from modern *Centaurea cyanus/diffusa*, and names historic `Peonia`, without using the Voynich glyphs. Reject taxonomic/lexical homonymy.
5. **Matched hard negatives** are other same-hand, same-register, same-line-position *herbal labels* with independently known species, plus unrelated herbal folios. Preserve word length, FORM class, CONNECT context and illustrated-page production whenever possible. Do not interpret same-source adjacent folios f21r/v as independent negatives.
6. **Decode test only if eligible:** prereg a specific mapping from `ytoail` to an independently identified species and ask whether frozen FORM/SELECT mechanics predict a signal on held-out physically independent labelled folios above best-of-search matched production null. Selection of candidate among several species must be fully counted as multiple testing.
7. **Stop rule:** Without an independently verified Voynich plant identity, precise label coordinates, stable eligible paired controls and sufficient same-ecology heldout labels, the actual semantic inference is **VOID / NOT IDENTIFIABLE**, regardless of visual similarity or attractive token spelling. Do not manufacture 20 seed replicates of one page to pretend they are 20 independent manuscripts.

## Deliverables
- `anchor1_f2r_manifest_20261019.json`: source and image links, controls, status, hypothesis pair.
- `anchor1_real_loci_audit.py` and `results/anchor1_real_loci_audit_20261019.json`: machine-verified label ecology and transcription differences.
- `anchor1_adjudicate.py`: accepts returned reviewer JSON from a local discovery-stage image comparison and prints which *qualifying* conditions are missing; does not infer decipherment.
- Downloadable independent review HTML in the conversation: displays actual Yale f2r and British Library folio images, supports normalized label-box annotation, source-neutral image ratings, reviewer pre-exposure declaration, locking and local JSON export. Browser must load IIIF assets; library restrictions and CORS may impede remote display.
- Final decisions recorded separately. The current conclusion is deliberately **NOT QUALIFIED**, not a failure of the historical-language hypothesis.

## Independent outcome criteria
For an eventual confirmatory sample: at least 20 distinct physically independent target folios with independently verified **same label ecology** and adjudicated image identities, independent image/text reviewers, a locked SEMANTIC/PRODUCTION comparison and matched negatives. The exact power threshold and test statistic cannot be chosen from this single illustrated case. If the population has fewer than 20 such folios, the design must change, not recycle one folio 20 times.

Do not claim the external labelled plant images give a semantic gloss for `ytoail`. They only provide independently attested *historical alternatives* for an image review.
