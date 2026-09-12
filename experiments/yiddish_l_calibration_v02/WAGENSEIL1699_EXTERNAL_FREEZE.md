# Wagenseil 1699 Wieduwilt — external transfer freeze

## Status

`FROZEN_PAGE_RANGE_BEFORE_FULL_TEXT_EXTRACTION`

This source is **not** the primary v0.2 transfer panel. It is a hostile external historical-Yiddish transfer because its Latin-letter transliteration convention is source-specific and substantially more Germanizing than PPCHY's YIVO/YIVO-inspired Romanisation.

## Source

Johann Christoph Wagenseil, *Belehrung der Jüdisch-Teutschen Red- und Schreibart*, Königsberg: Rhode, 1699.

BSB digital object: `bsb10903876`.

Independent bibliography/editorial descriptions state that the *Wieduwilt* text is printed pp.158–292 with Hebrew/Yiddish script on even pages and Latin-letter text on odd pages 159–291, followed by Latin-type text pp.293–302.

## Mechanical scan mapping audit

Before body-text extraction, page-number/script checks established:

- image `00263` = printed p.161, LATIN;
- image `00267` = printed p.165, LATIN;
- sequential terminal audit:
  - `00393` = p.291 LATIN
  - `00394` = p.292 HEBREW
  - `00395` = p.293 LATIN
  - `00396` = p.294 HEBREW was an erroneous model label in an earlier isolated check? NO: sequential audit output labels it HEBREW; therefore p.294 is excluded regardless of bibliography until independently adjudicated before corpus use.
  - `00397` = p.295 LATIN
  - `00398` = p.296 LATIN
  - `00399` = p.297 LATIN
  - `00400` = p.298 LATIN
  - `00401` = p.299 LATIN
  - `00402` = p.300 LATIN
  - `00403` = p.301 LATIN
  - `00404` = p.302 LATIN

### RETRACTION

An earlier three-image vision read labelled image `00395` as p.292. The full sequential pagination audit supersedes it: `00394=p.292`, `00395=p.293`.

## Frozen primary page set

The bibliographic target is printed pp.159–302, restricted to Latin-type pages.

Before full transcription the frozen image candidates are:

1. Odd printed pages p.159,161,...,291 => image suffixes `00261,00263,...,00393` (67 pages).
2. Terminal printed pages p.293–302 => nominal image suffixes `00395`–`00404` (10 pages), **subject only to script-type adjudication, not OCR quality selection**.

No page may be dropped because OCR is difficult. A page may be excluded only if independent script/page audit proves it is not Latin-type Wieduwilt text.

## Required pre-use audit

Before any L score:

- independently adjudicate terminal pp.293–302 script type, especially image `00396` / nominal p.294, because the sequential VLM audit conflicts with the bibliographic statement that pp.293–302 are all German type;
- transcribe all retained frozen pages using one fixed OCR/VLM pipeline;
- precommit a deterministic six-page image-vs-transcription fidelity sample before looking at OCR discrepancies;
- report literal OCR error categories and quantity;
- perform exact 8-word overlap screening against every PPCHY L BUILD/DEVELOPMENT family; any material overlap disqualifies this as source-disjoint;
- reduce to literal lowercase a-z only after transcription; no YIVO repair, modernisation, dictionary correction or linguistic plausibility repair.

No result here can repair the failed YidTakNL primary-script bridge or issue L qualification by itself.