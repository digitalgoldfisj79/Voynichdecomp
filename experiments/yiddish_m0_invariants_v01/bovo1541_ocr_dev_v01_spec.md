# Bovo 1541 OCR development protocol v01

Status: FROZEN BEFORE OCR OUTCOMES
Purpose: qualify a diplomatic Hebrew-script transcription pipeline using an already-consumed Yiddish work family, so that the fresh 1599 Basel witness is not consumed during OCR tuning.

## Source
- Work: Elye Bokher, Bovo d'Antona / Bovo-bukh, first print Isny 1541.
- Development carrier: Judah A. Joffe 1949 facsimile edition, Internet Archive item `nybc207004`.
- This work family is already consumed in prior Yiddish qualification work; nothing in this development experiment can count as fresh confirmation evidence.
- The facsimile begins after Joffe's p.36; the eligible development leaf range is prospectively fixed as IA leaves n40..n120, avoiding front matter and keeping well inside the reproduced 1541 text.

## Deterministic audit sample
Seed material (literal UTF-8):
`BOVO1541_OCR_DEV_V01|nybc207004|facsimile_leaf_range_40_120|GOT_OCR2|FLORENCE2_OCR`
SHA-256: `bb5ddb855dce70777e4e1fd2008897431f007ea0e4947df538bdb9ac6c79181f`
PRNG: Python `random.Random(int(sha256[:16],16))`, sample 6 without replacement from inclusive integers 40..120, then sort.
Frozen leaves: **n51, n54, n73, n82, n86, n120**.

## OCR candidates
Candidate A: `fal-ai/got-ocr/v2`, plain OCR, one page at a time, no translation, no language-driven correction.
Candidate B: `fal-ai/florence-2-large/ocr`, one page at a time, no translation, no language-driven correction.
The e-rara/Internet Archive/Tesseract OCR is locator material only and is NOT gold and NOT an evaluator.

## Audit reference
For each frozen page, the image itself is ground truth. A page-level reference is produced independently of either OCR candidate by direct visual reading of the printed page. Preserve printed word divisions and line divisions. Unreadable atoms are marked `<?>`; they are not guessed from language context.

## Metrics
For each engine and page, after Unicode NFC only:
1. full-line omission count;
2. full-line hallucination count;
3. Hebrew-letter atom edit accuracy = 1 - Levenshtein edits/reference Hebrew-letter atoms;
4. word-boundary F1 after alignment;
5. exact-line rate.
Editorial punctuation is reported separately and cannot rescue/fail Hebrew atom accuracy.

## Development gates
An engine is individually viable only if across all six frozen pages:
- zero omitted or hallucinated printed text lines;
- Hebrew-letter atom accuracy >= 0.985;
- word-boundary F1 >= 0.985.

A two-pass consensus pipeline may be retained only if a prospectively mechanical reconciliation (exact agreement retained; unresolved disagreements marked `<?>`, never silently language-corrected) attains on the six-page audit:
- zero omitted/hallucinated printed lines;
- >= 0.995 Hebrew-letter atom accuracy on resolved atoms;
- >= 0.995 word-boundary F1;
- resolved-atom coverage >= 0.95.

If neither individual engine nor the mechanical consensus clears its gate, this OCR route is rejected. No third OCR model is introduced on this source.

## Model selection for later fresh source
If exactly one individual engine passes, it is the primary pass and the other remains an independent disagreement detector. If both pass, select higher pooled atom accuracy; tie-break by boundary F1, then exact-line rate, then Candidate A. The selection and reconciliation code are frozen before the fresh witness is processed.

## Target/source firewall
- Voynich is prohibited.
- The fresh 1599 Basel `Maʿaśe bait Daṿid bime paras` images/OCR are prohibited during this development stage.
- This experiment can only qualify transcription tooling; it cannot qualify R, L, T, or historical population transfer.