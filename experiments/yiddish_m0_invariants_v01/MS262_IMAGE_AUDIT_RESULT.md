# MS262 transcription-image audit result v0.1

Date: 2026-09-12
Frozen sample commitment: `MS262_IMAGE_AUDIT_SAMPLE.md`
Source commit: `cu-mkp/ms-262-data@1e7b9f78ae1b4d2bc3d6c2c593d0d14855bc664f`
Acquisition workflow run: 34695045178
Artifact: `ms262-image-audit-v01`, artifact id 10298765025, sha256 `5e8b0840c19f42a40d4d7e444342637704d270a6a1a921f6ed5af405410abdb7`.

## Result

**SOURCE-QUALITY GATE FAIL for treating the extracted p001r-p021v OWY stream as one continuous diplomatic historical-Yiddish text.**

The six score-independent sampled page sides were 002r, 003r, 004r, 013v, 014v, and 018r. The corresponding source images and frozen XML were inspected side-by-side after sample commitment.

Pages 002r, 003r, 004r, 013v, and 014v show no gross page-level omission of the transcribed outer `ab language="owy"` passages at inspection resolution. They contain expected deletions, additions, language switching, and marginal material represented in the XML; this statement is a page-level source audit, not a claim of a diplomatically perfect character-by-character edition.

Page **018r materially fails completeness**. The manuscript image contains two substantial blocks of text across the page. The frozen XML contains only a short `ab language="owy"` incipit (`ווען דוא וילט ... נשים מאכן`) corresponding to the beginning of the lower block and omits the substantial preceding block plus the continuation below the incipit. No omission/gap sentinel is inserted into the D0 stream.

Therefore the 4,154-word D0 extraction cannot be interpreted as a contiguous 4,154-word manuscript sequence. Concatenating the available XML creates artificial adjacency and recurrence distances across untranscribed source material. This directly affects registered sequence-sensitive invariants, especially recurrence-gap and order-dependent quantities, and prevents promotion of MS262 from implementation control to source-transfer control under v0.1.

The previous count `4,154 words; 26 words above 4,128` remains a count of *available extracted XML tokens*, not a validated long continuous historical-text cell.

## Consequence

The v0.1 instrument implementation result remains valid because planted M0 and destructive controls were tests over the extracted representation itself. The source-transfer state is downgraded/clarified to:

`M0_INVARIANT_INSTRUMENT_IMPLEMENTATION_PASS__MS262_CONTIGUITY_AUDIT_FAIL__SOURCE_TRANSFER_UNASSESSED`

No Voynich target use is authorised. Do not repair or silently fill MS262 after seeing this audit. A later source-transfer protocol requires either (a) a separately frozen gap-aware representation that never treats unknown gaps as adjacency, with adequate contiguous units established prospectively, or (b) a different complete audited historical-Yiddish source.
