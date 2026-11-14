# eo777 crease / ordering / ink session (2026-11-13 to 2026-11-14)

Tests of the claims in voynich.ninja thread-6299 (npcompl33t = GitHub eo777), ending in direct ink measurement of Beinecke MS 408
on two independent photographic campaigns (2004 SID scans, 2014 Yale IIIF images).

**Status: provisional.** One Claude session. No fresh-chat demolition, no external test. Read the retraction blocks at the top of each running file first.

## Layout
- `eo777_crease/`  crease-set nulls (thread-6299). `RUNNING_RESULTS_crease.md`
- `order/`         folio / bifolium ordering against eo777's Golf paths. `RUNNING_RESULTS_order.md`
- `spatial/`       tests on eo777's vms-spatial data (layout, tilt, his ink measures). `RUNNING_RESULTS_spatial.md`
- `inkdirect/`     direct ink instrument and all ink phases. `RUNNING_RESULTS_inkdirect.md` is the full log: every pre-registration,
                   result, power check, retraction, in time order.
  - instrument: `instrument.py` (v3), `sharp.py` (focus gauge s3), `reg.py`, `reg2.py`; hashes in `hashes_phase2.txt`
  - calibration (known-answer): `phase1.py`, `phase1b.py`, `phase1c.py`, `phase1d.py`
  - measurement: `measure_pages.py` (2004), `measure_bein.py` (2014); per-page pickles in `pages/`, `pages_bein/`
  - word-level data: `words_y2004.csv.gz` (10,488 words), `words_y2014.csv.gz` (11,309 words)
  - analysis: `analysis2.py`, `bound2.py`, `phase3.py`, `phase4.py` (first run VOID: sort bug), `phase4b.py`, `phase4_bounding.py`,
    `analysis5.py`, `analysis6.py`, `db1_bounding.py`, `phase6.py` .. `phase9.py`
  - results: `phase*_results.json`, `*_results_table.csv`, `phase2d_pages.csv`, `phase7_pages.csv`, `tif_sha256.csv`

## Database
Findings ledger, runs, results, pre-registrations and retractions are in Supabase: tables `vms_ink_findings_v01`, `vms_ink_page_values_v01`,
`vms_crease_runs_v01`, `vms_crease_results_v01`, `vms_crease_prereg_v01`, `vms_crease_retractions_v01`, `vms_crease_files_v01`.

## To rerun
Needs eo777's `vms-spatial` (commit f0d04b7) and `voynich-clusters` checked out, and the paths at the top of `spatial/common.py` adjusted.
Scripts were run from the vms-spatial directory with a Python 3.12 environment holding his pinned requirements plus scikit-image.
Images: 2004 SID TIFFs (zodiackillerciphers.com/voynich-sid-tif), 2014 Yale IIIF (collections.library.yale.edu, oid 2002046). Images are not stored here.

## Credits
Word boxes, physical map, registrations and orders: eo777 (vms-spatial, voynich-clusters). Images: Beinecke Rare Book and Manuscript Library, Yale University.
