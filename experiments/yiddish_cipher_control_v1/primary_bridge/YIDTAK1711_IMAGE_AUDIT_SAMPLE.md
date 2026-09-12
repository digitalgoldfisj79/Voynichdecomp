# YidTakNL 1711 primary-script image audit — frozen sample

Status: **FROZEN BEFORE SOURCE-IMAGE VIEWING**  
Date: 2026-09-12

Source: `Takanot 1711 - YidTakNL.docx`, Zenodo record 10017358.  
DOCX SHA-256: `b949f2c089072a757fedeaacca5108f9c120981bd0f92b1479d73950531f0d46`.

The published edition contains 24 explicit manuscript-side markers, `[1r]` through `[12v]`.

Selection string:
`fc7fddb9d731585e3a483c8182a6e82095bdaf89c9b640e81cfbc3be499d9ddc|yidtak1711_primary_image_audit_v01`

The first eight bytes of SHA-256, interpreted big-endian, give RNG seed `3214507816572711821`. Python `random.Random(seed).sample(markers, 6)` yields the following six sides, sorted only for reporting:

- `1v`
- `3r`
- `3v`
- `6v`
- `7r`
- `8r`

## Audit rule

For each selected side, compare the complete published transcription between that side marker and the next side marker against the original 1711 page image. Record:

1. omitted substantive lines or inserted lines;
2. materially incorrect word segmentation;
3. literal character disagreements relevant to the proposed Hebrew-letter atom representation;
4. illegible/damaged spans and editorial uncertainty;
5. page/image mismatch;
6. punctuation/abbreviation phenomena separately from base Hebrew letters.

No linguistic plausibility repair is allowed. This audit is score-independent and occurs before any primary-script cipher scoring.

A material omission, page mismatch, or systematic transcription defect blocks promotion to a primary-script bridge. Small literal disagreements are reported and bounded; they are not silently corrected.

Voynich remains sealed.