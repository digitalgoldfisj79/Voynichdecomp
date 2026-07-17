# v1.6 external-corpus suitability audit

Date: 2026-07-17

Status: frozen before any external-corpus image inference.

## Purpose

This record applies the metadata-only corpus gate in Amendment 005. It does not report model performance and does not weaken any scientific gate.

## HisFrag20 provenance

- Official record: `https://zenodo.org/records/3893807`
- Training archive: `hisfrag20_train.zip`
  - bytes: 3,226,721,481
  - MD5: `a6c3a9a2c2f170605fcf153792dd78e8`
- Test archive: `hisfrag20_test.zip`
  - bytes: 1,428,688,298
  - MD5: `56ea22f6424cadb2208c1bfd171d8a8a`
- Filename convention successfully parsed for every image: `WID_PID_FID`.
- Metadata audit jobs:
  - `6a5a4a7bd216bd6f3a1fb900`
  - `6a5a4cd7bee6ee1cf4ecd935`

Only ZIP central-directory metadata and filenames were inspected. No HisFrag20 image was decoded or passed through any model.

## Exact archive inventory

### Test archive

- image fragments: 20,019
- non-image files: 0
- parsed filenames: 20,019
- malformed filenames: 0
- duplicate `WID_PID_FID` keys: 0
- writers: 1,152
- physical pages identified by `(WID, PID)`: 2,753
- writers with at least two pages: 557
- writers with at least three pages: 374
- pages per writer: minimum 1, median 1, maximum 5
- fragments per page: minimum 2, median 6, maximum 18
- page-count histogram:
  - 1 page: 595 writers
  - 2 pages: 183 writers
  - 3 pages: 35 writers
  - 4 pages: 8 writers
  - 5 pages: 331 writers

### Training archive

- image fragments: 101,707
- non-image files: 0
- parsed filenames: 101,707
- malformed filenames: 0
- duplicate `WID_PID_FID` keys: 0
- writers: 8,717
- physical pages identified by `(WID, PID)`: 17,222
- writers with at least two pages: 2,500
- writers with at least three pages: 2,388
- pages per writer: minimum 1, median 1, maximum 5
- fragments per page: minimum 2, median 2, maximum 77

### Published split overlap

- writer overlap between training and test: **331**
- physical-page overlap between training and test: **0**
- fragment-key overlap between training and test: **0**

The published train/test archives are therefore not writer-disjoint. They must not be used naively as a calibration/terminal split for the present study.

## HisFrag20 decision

**Conditionally accepted as the primary external cross-page pair benchmark, but not as the sole v1.6 confirmation corpus.**

It satisfies the following requirements:

- writer identity is encoded;
- physical page identity is encoded;
- 557 test writers provide at least two distinct pages;
- same-page fragments can be excluded exactly;
- cross-page same-writer positives and different-writer negatives can be constructed without ambiguity;
- page-level fragment aggregation and fragment-count calibration are feasible.

It does **not** independently satisfy the following requirements:

- the ZIP contains no institution, manuscript, digitisation-session, date, layout, transcription, glyph or grapheme metadata;
- the source/acquisition environment cannot be matched directly from the released filenames;
- cross-manuscript or cross-acquisition same-writer evaluation cannot be asserted from the archive metadata alone;
- content-conditioned testing cannot be performed without adding an independently validated transcription or grapheme annotation layer.

Image-derived nuisance variables such as dimensions, background statistics, ink density and layout proxies may be computed prospectively, but they do not substitute for known acquisition or manuscript identifiers.

## Frozen HisFrag20 split rule

Only writers in the **test archive** with at least two distinct PIDs are eligible for the primary v1.6 HisFrag20 analysis.

- eligible writer count before image QC: 557
- split seed: `20260720`
- compute `SHA256("20260720:" + WID)` for each eligible writer;
- sort ascending by that digest, with WID as a deterministic tie-break;
- first `floor(0.20 × N)` writers: calibration set;
- next `floor(0.20 × N)` writers: validation set;
- all remaining writers: sealed terminal set.

With the metadata-only inventory this yields 111 calibration writers, 111 validation writers and 335 terminal writers before image QC. Any writer failing objective image-integrity checks is excluded without replacement. Exclusion criteria must be fixed in code before decoding the archive.

The provided HisFrag20 training archive will not contribute writers, thresholds, calibration or residualization to the terminal analysis because 331 writer identities overlap the test archive.

## Frozen pair construction

- same-writer positives must use different PIDs;
- no query and gallery sample may share a PID;
- all FIDs from a PID remain in one page group;
- colour/binary derivatives or duplicate images, if discovered, remain in one group and cannot form positive pairs;
- adjacent or overlapping fragments from one page cannot appear on opposite sides of a pair;
- the primary fixed-evidence analysis uses 96 foreground patches per page, with the fragment-length ladder evaluated separately;
- sampling seeds and page manifests must be persisted exactly.

## Replication corpus

HisFrag20 alone cannot meet the final independent-replication and acquisition-heterogeneity requirements. The preferred replication corpus is **BullingerDB (2026)**, subject to exact-record and checksum verification before download.

Reasons for selection:

- 20,898 historical pages;
- 796 confirmed writers;
- six decades of temporal variation;
- writer identity and date metadata;
- layout information and transcriptions;
- multiple letters/pages for frequent writers;
- same writers can be tested across different letters and years;
- different writers share the same broad archival and digitisation environment.

Primary paper: `arXiv:2605.30235`, accepted for ICDAR 2026. The paper states that the full dataset is publicly available through Zenodo. The exact Zenodo record, archive checksums, licence and writer/page/date schema must be resolved and frozen in a separate metadata audit before any BullingerDB image inference.

AnyScript 2026 is retained as a reserve corpus because it offers 1,019 authors across 2,286 books and explicit extra-book retrieval, but its approximately 700 GB training set is unnecessary unless BullingerDB proves inaccessible or structurally unsuitable.

## Consequence

The accepted sequence is:

1. accept and persist the v1.5.1 reproduction;
2. export and verify the PCA, residual transforms and row-level provenance under Amendment 006;
3. decode and QC HisFrag20 under the frozen rules above;
4. run calibration and validation writers only;
5. preserve all analysis choices;
6. run the sealed 335-writer terminal split;
7. replicate on BullingerDB or another prospectively accepted cross-letter/cross-acquisition corpus;
8. open no Voynich material unless all Amendment 005 gates pass.

The Voynich seal remains intact.
