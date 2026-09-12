# YIDTAK1711 primary-script image audit — exact page commitment

## RETRACTED PROCEDURAL FINDINGS

- **RETRACTED:** “the dedicated HUC probe never ran.”  **Correction:** it did run on commit `2df321423d2ba38edea1f1dafd41354d69d4974e` and failed because the obsolete `mss.huc.edu` host did not resolve.  This was a transport failure, not evidence that the source scan was unavailable.

## Status

**COMMITTED BEFORE SOURCE-IMAGE VIEWING**

No source JPEG in the 1711 scan stack, and no content of the corresponding `txt/` ground-truth files, was retrieved, rendered, or viewed before this commitment.  Only WebDAV metadata (path, filename, MIME type, byte size, ETag, resource type) was inspected.

Frozen upstream audit protocol: `YIDTAK1711_IMAGE_AUDIT_SAMPLE.md`, blob `510bb3edab2d18f25b2a0cf4849996834611cc0e`.

Frozen sides: `1v`, `3r`, `3v`, `6v`, `7r`, `8r`.

## Source stack and mechanical body-range rule

Published source-image dataset: Figshare DOI `10.6084/m9.figshare.25422844`, public SURFdrive directory:

`RBR_O_128_PDF תקנות 1711/`

Metadata-only inventory returned 31 JPEG files, `0001_p001.jpg` through `0031_p031.jpg`, plus a parallel `txt/` directory.

The parallel `txt/` metadata establishes the body range mechanically:

- `0001_p001.txt`–`0004_p004.txt`: 0 bytes.
- `0005_p005.txt`–`0028_p028.txt`: **24 consecutive non-zero files**.
- `0029_p029.txt`–`0031_p031.txt`: 0 bytes.

The frozen DOCX independently contains exactly 24 ordered side markers, `[1r]` through `[12v]`.  Therefore the deterministic pre-view mapping is ordinal across the unique 24-file non-empty run:

`[1r] -> p005`, `[1v] -> p006`, ..., `[12v] -> p028`.

This mapping is not allowed to be repaired for linguistic plausibility after image viewing.  A source-page/order mismatch discovered during audit is itself an audit failure to record, not a reason to silently choose replacement pages.

## Exact six committed source JPEGs

| Frozen side | Body ordinal (1-based) | Source JPEG |
|---|---:|---|
| `1v` | 2 | `0006_p006.jpg` |
| `3r` | 5 | `0009_p009.jpg` |
| `3v` | 6 | `0010_p010.jpg` |
| `6v` | 12 | `0016_p016.jpg` |
| `7r` | 13 | `0017_p017.jpg` |
| `8r` | 15 | `0019_p019.jpg` |

## Hard audit rule

Compare only these six precommitted JPEGs against the corresponding Penn/Yiddish transcription representation under the already-frozen audit protocol.  Record literal omissions, insertions, merges/splits, character substitutions/normalizations, damaged/uncertain readings, punctuation/abbreviation separately, and any page mismatch.  No linguistic plausibility repair is permitted.

If the transcription materially omits, inserts, merges, or normalizes distinctions relevant to M0, the Penn-to-primary-script bridge fails.  If the six-page audit is sufficiently faithful, quantify preservation and loss; this does **not** by itself establish a bijective Romanization.

Voynich/C7 remains sealed throughout.
