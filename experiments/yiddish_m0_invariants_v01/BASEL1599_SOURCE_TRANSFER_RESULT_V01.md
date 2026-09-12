# Basel 1599 anonymous word-equality source-transfer v0.1 — RESULT

Status: **`BASEL1599_ANON_WORD_SOURCE_TRANSFER_FAIL`**.

This is a source-transfer failure of the frozen Bovo-trained image segmenter. It is not evidence against Yiddish and it is not a Voynich result. Voynich was not loaded.

## Frozen protocol and provenance

Protocol frozen before any Basel page was downloaded/viewed: `BASEL1599_SOURCE_TRANSFER_PROTOCOL_V01.md`, commit `75dd46a042e1ac11e753dd865634126e352445e7`.

Fresh source: *Maʿaśe bait Daṿid bime paras*, Basel, Konrad Waldkirch, 1599, e-rara manifest 2611258, 40 canvases. This is a 1501–1600 transfer-panel witness only.

The first acquisition attempt failed before any image was loaded because the dynamic manifest endpoint returned non-JSON to the GitHub runner. Transport only was repaired using the exact 40 image IDs already frozen from the manifest and the repository's direct-IIIF image route. Successful acquisition run: GitHub Actions `34704498991`, artifact `basel1599-acquisition-v01`, artifact ID `10300742762`, artifact SHA-256 `a9031da282217307d6731a3a171f2a21f3ff44b2c792a0bf124824e7c38e8218`. All 40 source-image hashes were recorded by the run.

## Mechanical census

The unchanged Bovo v0.1 segmenter was applied after the prospectively fixed canonicalisation.

- 40 canvases processed.
- 35 met the automatic text-bearing rule (>=10 retained lines and >=80 retained word segments).
- Concatenated automatically qualifying quantity: **8,251 retained word segments**.
- Registered long quantity gate: >=4,128.
- Quantity gate: **PASS**.

Fixed audit pages and raw segmenter counts:

- [5]: 30 retained lines, 248 retained words.
- [11]: 32 lines, 276 words.
- [17]: 32 lines, 287 words.
- [23]: 31 lines, 254 words.
- [29]: 31 lines, 269 words.
- [35]: 20 lines, 142 words.

## Decisive visual-audit failure

The protocol required, on **all six** fixed audit canvases, zero missing substantive text lines, zero non-text substantive inclusions, and <=2 obvious word-boundary errors per page. No page substitution was allowed.

Image overlays show missing substantive text lines on every fixed audit canvas:

- **[5] FAIL** — opening heading/content above the first retained line is omitted; at least two full content lines are visibly outside the retained line set. A mid-page section heading is also not represented as ordinary retained text.
- **[11] FAIL** — the top substantive text line is omitted.
- **[17] FAIL** — the top substantive text line is omitted.
- **[23] FAIL** — two opening substantive lines are omitted.
- **[29] FAIL** — two opening substantive lines are omitted.
- **[35] FAIL** — opening heading/content is omitted and closing material below the last retained line is also omitted.

The principal failure mechanism is visible and non-linguistic: the v0.1 Bovo segmenter contains a frozen absolute vertical admissibility rule (`500 < line_start < 4600`). The Basel layout places genuine text outside that Bovo-specific band. This is exactly the kind of cross-source portability failure the transfer gate was intended to detect.

Because **any** visual-audit failure forces source-transfer failure, no predicted-equal audit, hard-negative audit, or 48-cell nuisance score can rescue v0.1. Full-source complete-link clustering was therefore not treated as evidence; an attempted naive whole-book clustering also exceeded the local compute window, but that engineering issue is scientifically moot once the prerequisite audit has failed.

## Decision / consumption

`BASEL1599_ANON_WORD_SOURCE_TRANSFER_FAIL`.

The Basel 1599 witness is **consumed for confirmation under detector v0.1**. It must not be reused as a fresh confirmation witness for a repaired detector. It may be used only as explicitly labelled DEVELOPMENT material in a later version.

No segmentation constant, `tau`, descriptor, page selection, or audit threshold is changed in response to this result.

## What this means

The narrower anonymous whole-token equality idea remains viable: v0.1 passed same-print Bovo holdout and nuisance qualification. What failed is the assumption that Bovo's absolute page geometry transfers unchanged to a different sixteenth-century print. A scientifically legitimate v0.2, if pursued, must solve page-layout portability using consumed BUILD/DEVELOPMENT sources only and then face a **new source-disjoint fresh witness**. It cannot rescue Basel 1599 retrospectively.

Current inference status remains: no historical-Yiddish population transfer certificate, no target admission, no Voynich inference.
