# Basel 1599 anonymous word-equality source-transfer protocol v0.1

Frozen before downloading or viewing any page image from the fresh source.

## Proposition

Test whether the already-qualified Bovo-1541 anonymous whole-token equality detector transfers unchanged to a different sixteenth-century Yiddish print/typeface. This is a **source-transfer** gate only. It does not establish language discrimination and does not authorize Voynich scoring.

## Fresh source

`Maʿaśe bait Daṿid bime paras`, Basel: Konrad Waldkirch, 1599, Universitätsbibliothek Basel, e-rara manifest ID `2611258`, 40 IIIF canvases, bibliographic description [18] leaves, language Yiddish. This source has not been used in BUILD or same-source HOLDOUT.

It is necessarily a **1501–1600 transfer-panel** source under the overarching Yiddish protocol, not core 1350–1500 evidence.

## Frozen detector

Use unchanged anonymous word-equality v0.1 from `ANON_WORD_EQUALITY_PROTOCOL_V01.md` and result `ANON_WORD_EQUALITY_RESULT_V01.md`:

- Bovo BUILD scaler only;
- complete-link cosine clustering;
- frozen `tau = 0.410`;
- no OCR, character identity, language model, dictionary, or lexical correction;
- segmentation constants unchanged after image canonicalisation below.

## Image acquisition and canonicalisation

Acquire all 40 canvases from the e-rara IIIF manifest in manifest order at maximum available IIIF resolution.

For each page:

1. convert to grayscale;
2. resize isotropically to width 3600 px;
3. if height >5400, centre-crop vertically to 5400; if height <5400, centre-pad white to 5400;
4. do not deskew, crop text regions, sharpen, denoise, or change contrast before primary extraction.

A page is automatically text-bearing iff the frozen segmenter returns >=10 retained text lines and >=80 retained word segments. All qualifying pages are concatenated in manifest order. No manual inclusion/exclusion after viewing is permitted.

## Prospectively fixed visual-audit pages

Irrespective of later equality scores, inspect manifest labels/canvases **[5], [11], [17], [23], [29], [35]** after canonicalisation.

For each page produce an overlay of detected line boxes and retained word boxes. Record:

- missing substantive text line: yes/no;
- included non-text line as substantive text: yes/no;
- obvious word-boundary split/merge errors visible in the overlay.

Audit-page pass: zero missing substantive lines, zero included non-text substantive lines, and <=2 obvious word-boundary errors per page. All six pages must pass. If a fixed audit canvas is genuinely blank/non-text, that canvas fails the fixed audit rather than being substituted.

## Quantity gate

The concatenated automatically qualifying pages must contain >=4,128 retained word segments. Report exact N. Failure => source cannot support the registered long transfer condition.

## Equality-pair audit

After applying the frozen detector, create two deterministic audit samples with RNG seed string `basel1599-anon-word-transfer-v01`:

### Predicted-equal sample

60 unordered pairs drawn from detector clusters of size >=2, preferring cross-page pairs and sampling at most 5 pairs from any one cluster. A human/image audit labels each pair `same visual word`, `different`, or `uncertain`. `uncertain` counts as failure.

Pass: >=57/60 same visual word.

### Hard-negative sample

For each pair of different detector clusters compute the minimum pairwise cosine distance. Sort ascending and take the first 60 distinct word pairs after excluding any word already used more than twice. Human/image audit labels as above.

Pass: >=57/60 different visual words; `uncertain` counts as failure.

No cluster threshold or preprocessing may be changed in response to these audits.

## Scan-nuisance robustness

On the six fixed audit canvases, apply the same eight perturbations used in Bovo qualification: contrast 0.90/1.10, Gaussian blur sigma 0.5/0.8, JPEG Q90/Q75, resize 0.99/1.01 then back to canonical dimensions.

Use the frozen nuisance positional matching rule. 48 cells total. A cell passes under the same criteria: token-count relative difference <=2%, matched-position fraction >=0.98, ARI >=0.90. Overall pass >=44/48.

## Source-transfer decision

`BASEL1599_ANON_WORD_SOURCE_TRANSFER_PASS` requires simultaneously:

1. all six visual audit pages pass;
2. N >=4,128;
3. predicted-equal audit >=57/60;
4. hard-negative audit >=57/60;
5. nuisance robustness >=44/48;
6. detector nondegeneracy on the concatenated source: TTR >=0.30 and maximum cluster fraction <=0.15.

Any failure => `BASEL1599_ANON_WORD_SOURCE_TRANSFER_FAIL`. Do not rescue by changing `tau`, segmentation, descriptor, page selection, or audit sampling. A failed source is consumed for this detector version.

## Interpretation if passed

A pass licenses the Basel-1599 image-derived whole-token equality partition as a primary-script **transfer-period Yiddish control** for later M0 necessary-condition work. It does not qualify core-period Yiddish, historical Hebrew/German discrimination, R/L/T, or Voynich. A separate population/comparator protocol is required before target admission.
