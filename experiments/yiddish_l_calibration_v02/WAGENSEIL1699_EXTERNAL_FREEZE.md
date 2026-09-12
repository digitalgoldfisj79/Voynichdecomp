# Wagenseil 1699 Wieduwilt — external transfer freeze

## RETRACTIONS / CONTROLLING CORRECTIONS

1. An earlier three-image vision read labelled image `00395` as p.292. The full sequential pagination audit supersedes it: `00394=p.292`, `00395=p.293`.
2. The first Gemini sequential audit labelled `00396` as **p.294 HEBREW**. This is retracted. An independent Claude audit classified the scan as German type; a second four-scan sequential Claude audit read `00395=p.293`, `00396=p.294`, `00397=p.295`, `00398=p.296`, all German type. The isolated first Claude check read `00396` as UNNUMBERED German type, so the exact printed-number reading briefly disagreed, but **script eligibility does not**. The sequential neighbour-constrained pixel audit supports `00396=p.294 LATIN/GERMAN TYPE` and restores the bibliographic pp.293–302 Latin run.

## Status

`FROZEN_PAGE_RANGE_BEFORE_FULL_TEXT_EXTRACTION__P294_SCRIPT_ADJUDICATED_LATIN`

This source is **not** the primary v0.2 transfer panel. It is a hostile external historical-Yiddish transfer because its Latin-letter transliteration convention is source-specific and substantially more Germanizing than PPCHY's YIVO/YIVO-inspired Romanisation.

## Source

Johann Christoph Wagenseil, *Belehrung der Jüdisch-Teutschen Red- und Schreibart*, Königsberg: Rhode, 1699.

BSB digital object: `bsb10903876`.

Independent bibliography/editorial descriptions state that the *Wieduwilt* text is printed pp.158–292 with Hebrew/Yiddish script on even pages and Latin-letter text on odd pages 159–291, followed by Latin-type text pp.293–302.

## Mechanical scan mapping audit

Before body-text extraction, page-number/script checks established:

- image `00263` = printed p.161, LATIN;
- image `00267` = printed p.165, LATIN;
- adjudicated terminal run:
  - `00393` = p.291 LATIN
  - `00394` = p.292 HEBREW
  - `00395` = p.293 LATIN
  - `00396` = p.294 LATIN / GERMAN TYPE after independent adjudication
  - `00397` = p.295 LATIN
  - `00398` = p.296 LATIN
  - `00399` = p.297 LATIN
  - `00400` = p.298 LATIN
  - `00401` = p.299 LATIN
  - `00402` = p.300 LATIN
  - `00403` = p.301 LATIN
  - `00404` = p.302 LATIN

## Frozen primary page set

The bibliographic target is printed pp.159–302, restricted to Latin-type pages.

Before full transcription the frozen image set is:

1. Odd printed pages p.159,161,...,291 => image suffixes `00261,00263,...,00393` (67 pages).
2. Terminal printed pages p.293–302 => image suffixes `00395`–`00404` (10 pages).

Total frozen Latin-type candidate pages: **77**. No page may be dropped because OCR is difficult. A page may be excluded only if a later independent page/script audit proves it is not Latin-type Wieduwilt text; any such exclusion must be recorded as a protocol deviation before scoring.

## Required pre-use audit

Before any L score:

- transcribe all 77 frozen pages using one fixed OCR/VLM pipeline;
- precommit a deterministic six-page image-vs-transcription fidelity sample before looking at OCR discrepancies;
- report literal OCR error categories and quantity;
- perform exact 8-word overlap screening against every PPCHY L BUILD/DEVELOPMENT/transfer family; any material overlap disqualifies this as source-disjoint;
- reduce to literal lowercase a-z only after transcription; no YIVO repair, modernisation, dictionary correction or linguistic plausibility repair.

No result here can repair the failed YidTakNL primary-script bridge or issue L qualification by itself.