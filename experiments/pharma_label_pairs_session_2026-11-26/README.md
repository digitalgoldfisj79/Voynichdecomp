# Pharma labels ↔ drawings, repeated plants, T4 / T3 / T2b — session 2026-11-26

Session archive for the pharmaceutical-section label-pairing work (f88r–f102v1).
Read `RUNNING_RESULTS_pharma_pairs.md` first: retractions are at the top, steps 01–17 follow.
`BRIEF.txt` is the brief the session was run against.

## Verdicts (detail and numbers in the results file)
- Stage 1 object→label join: the geometric nearest-label join was wrong in 21/89 decidable sampled links; the final join uses eye-confirmed or eye-corrected links only.
- Stage 2 (existing similarity tables): no A grades; grades did not beat random pairs.
- Stage 2b (blind fragment pairs from crops): nominated A+B 12/48 vs random 0/29, Fisher p = 0.0019.
- T4 (names written differently each time): Inconclusive at this power (T4b ratio 0.99).
- T3 (label marks the part shown): Inconclusive at this power (planted power near zero).
- T2b (label appears in the next text block): FALSIFY as registered.

## Layout
| Folder | Contents |
|---|---|
| `scripts/` | All code. Run from this folder; paths are relative. `t2b.py` expects ZL3b-n.txt from quimqu/voynich-scribe @ 8f5c2c76 at `/home/claude/ext/voynich-scribe/` — edit `ZL` to your copy. |
| `data/` | Reference objects pulled by REST (`ref_objects.json`), concordance as uploaded to Supabase, SQL extracts. |
| `ckpt/` | Pickle/JSON checkpoints. `*_SEALED.json` are the blind keys (unsealed after grades were hashed; hashes in the results file). |
| `out/` | Result CSVs/JSON, graded pair tables, blind grades, `SHA256SUMS.txt`, `scripts_bundle.txt`. |
| `img/` | Yale IIIF page images at 2000 px wide (`<iiif_id>_2000.jpg`) and high-res regions (`reg_*`). Widths for scaling in `yale_dims.txt`. |
| `ov/` | Overlays used for eye-checking links: `ov_*` objects+labels, `link_*` link strips, `hr_*` high-res link crops. |
| `crops/` | Object crops used for Stage 2 grading (`<page_seq>_<obj>.png`). |
| `masks/` | Object masks fetched from the reference-object `mask_url`s (existing VLM masks; nothing re-masked). |
| `frag/` | Fragment crops `F<fragment_num>.png` used for Stage 2b. |
| `sheets/`, `sheets2b/` | Blind grading sheets and contact sheets as viewed. |

`SHA256SUMS_all.txt` covers every file in this folder.

## Supabase (research-agent, ymaqlcfjmdwncdbjprmw)
- `public.vms_pharma_object_fragment_join_v01` (372 rows)
- `public.vms_pharma_pair_grades_v01` (189 rows)
- `public.vms_pharma_label_tests_v01` (8 rows)
- `public.handoff_docs` keys `voynich_pharma_label_pairs_{results,brief,scripts}_20261126`, `voynich_slots_cold_review_running_results_20261126`, and `voynich_pharma_label_pairs_github_20261126` (pointer to this folder)

## Not here
The prior session's scripts (c1–c5, label_split.py, label_in_herbal.py) and `herbal_pages_labelled_from_pharma_v01.csv` were lost with that session's disk.

Page images are derived from Beinecke MS 408 (Yale Beinecke Rare Book & Manuscript Library, public-domain digital images).
