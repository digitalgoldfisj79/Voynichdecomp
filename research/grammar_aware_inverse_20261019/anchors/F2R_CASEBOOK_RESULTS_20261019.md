# First historical text–image anchor: f2r / Egerton 747 (2026-10-19)

**Status: observational discovery complete; semantic correspondence unverified and NOT admissible for decipherment.** This builds a concrete evidence chain and a fixed actual-manuscript comparison rather than another synthetic cipher.

## Source separation
- External historical witness: British Library **Egerton MS 747**, folio 21r, _Tractatus de herbis_, c.1280–1350. Independent manuscript catalogue: https://searcharchives.bl.uk/catalog/032-001983805 ; official IIIF manifest https://bl.digirati.io/iiif/ark:/81055/vdc_100058663072.0x000001 and canvas index 48 labelled f.21r with image https://bl.digirati.io/images/ark:/81055/vdc_100058663074.0x000031/full/695,1024/0/default.jpg (200 and content-type image/jpeg verified). Image-identification catalogue/secondary caption: https://timelessmoon.getarchive.net/media/calamint-and-centaury-from-bl-eg-747-f-21-c0557e describes calamentum plus centaurea minor/major. **The historical name centaurea is not a modern single-species identification**; 'major' and 'minor' can denote different taxa in different traditions. The image and rubric still require expert review for manuscript-level exact reading.
- Voynich: Beinecke MS 408 **f2r**, Currier A Herbal, Yale IIIF https://collections.library.yale.edu/iiif/2/1006078. Source catalogue plant-object ID `47d29a8b-1a67-45d8-abd0-db0ac4e1c187` contains a Centaurea diffusa suggestion attributed to older Edith Sherwood material. `reviewed=false`; this is derivative, **not independent plant identification**.
- Prior literature explicitly selected **f2r line 1 `kydainy` and line 8 `kydain`** as potential centaury-name variants: Stephen Bax 2014 https://stephenbax.net/wp-content/uploads/2014/01/Voynich-a-provisional-partial-decoding-BAX.pdf . The f2r candidate and line8 were already known; this is a post-selection check. Alternate f2r botanical identifications exist.
- Additional candidate witness: BnF **Latin 6862** (9th century), f45r; external catalogue https://iiif.biblissima.fr/collections/manifest/a80ddf5b37d92bcff038b28023a41ff1a057a852. Research-agent Gemini 2.5 Flash machine-reading includes 'centauree marore', **UNVERIFIED OCR**. Excluded from any accepted semantic label until actual image and palaeographical validation.

## Operational artifacts
- [Candidate and provenance manifest](f2r_centaury_candidate.json) expressly records unmet semantic gates.
- [Frozen pre-run diagnostic](F2R_DISCOVERY_PREREG.md) specifies source SHA, Currier A/H folios, edits and matching rules.
- [Pinned 89-folio same-register population](herbal_a_register.json), from read-only research-agent `public.vms_realization_folio_split_v01`.
- [Executable f2r assay](f2r_opener_assay.py) fetches and verifies ZLZI corpus SHA256 `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`.
- [Raw outputs, all 28 matched controls](results/f2r_centaury_opener_assay_20261019.json), completed Hugging Face CPU job https://huggingface.co/jobs/Digitalgoldfish79/6ad5e6da632cfaf2180110d8 .
- An HTML **blind visual review sheet** with Yale and official BL IIIF images, comparison Herbal folios f3r/f4r, ratings locked before revealing textual interpretation, and exportable reviewer JSON was produced as a conversation artifact. Unlike the 28 text controls, the two visual comparisons are convenience decoys, **not** a confirmed statistically balanced selection.

## Actual manuscript outcome
| Registered diagnostic | Result | Inference |
|---|---|---|
| f2r ZLZI line1 word0 vs line8 word0 | `kydainy` vs `kydain`, Levenshtein 1, normalized 1/7 | Real distinctive surface recurrence already known |
| Other Currier-A Herbal folios with line1 & 8 running openers both ASCII and 5–10 chars, each opener length within 1 of target | **0/28** control pairs ED ≤1; median ED **6** | Uncommon in this narrowly defined matched sample; selected posthoc |
| f2r line1 opener vs other eligible within-f2r line starts 2–13 (except 8) | **0/5** ED ≤1; median ED **5** | Not a general f2r prefix repeated in every line |
| Source-vs-production discriminating test | **NOT CONDUCTED** | Both a plant-naming lexeme and a special paragraph-start/copying process predict this pattern |
| VMS species identity / spatial word-image alignment | **NOT VERIFIED** | No semantic decipherment claim allowed |

Important limits: matched line number ≠ matched physical paragraph boundary; `@P0` vs `+P0` in ZLZI does not itself verify that line8 begins a new physical paragraph. Source candidate and prominent repeated words were chosen before this test; no p-value or heldout independence is claimed. Bibliographic Egerton centaury illustrates historical vocabulary but is not an identified **Voynich-to-Egerton image match**.

## Next hard gate, not another unconstrained model
1. Independently judge f2r botanical imagery without reading the f2r text, including alternatives, and document exactly which historical witness drawings/labels were compared. A candidate class is insufficient.
2. Independently verify Egerton f21r rubric(s) and whether the matching depicted taxon is the correct 13th–15th century botanical sense of centaurea.
3. Hand-check actual physical paragraph separation at f2r line8 and coordinate the two text units in Yale scan.
4. Find a **new, untouched** Currier-A Herbal folio with independently documented same plant identity, or concede the single positive is insufficient for a prospective semantics test.
5. Match false positives not just by glyph lengths / line number but true paragraph starts, physical bifolium, manuscript session and an empirically fitted FORM/CONNECT/LINE_ENTRY generator. Source-vs-production must make different observable predictions; a recapitulation of paragraph opener similarity cannot adjudicate.

**Scientific conclusion:** We have built one concrete, reproducible medieval witness → suspected Voynich illustration → localized surface token **candidate**, and quantified the pre-existing repetition observation. The proposed semantic bridge is still missing its independent validation. This is the appropriate honest output of this build.