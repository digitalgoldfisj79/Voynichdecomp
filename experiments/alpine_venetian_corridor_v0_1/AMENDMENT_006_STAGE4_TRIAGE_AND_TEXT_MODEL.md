# Amendment 006 — Stage 4 blind triage and text representation

Date: 2026-08-08
Programme: Alpine–Venetian Corridor Illustration Programme v0.1
Run: `corridor_v01_20260808_run01`

## Status at freeze

No corridor-to-VMS similarity has been computed. The 12/12 VMS zodiac review has completed, but Stage 4 sees no VMS image or reference.

## IIIF-only triage sensitivity

The first Stage 4 pass is explicitly the preregistered IIIF/digitisation sensitivity subset, not the final primary corpus. It uses the sealed candidate identities only to resolve image sources. The VLM receives only page pixels plus deterministic opaque IDs.

Sampling is deterministic:

- ordinary multi-page witnesses: 10 evenly spaced non-binding canvases;
- Vat.lat.4082: 16 evenly spaced canvases;
- BSB astronomical witnesses and BnF Latin 7342: 12;
- Walsperger and Pizzigano 1424 single-sheet works: at most 2;
- Canon Misc. 554: catalogue-frozen ff.154–174, sampled to at most 10;
- obvious covers, spines, edges, colour targets and pastedowns are excluded mechanically.

No page is selected because it resembles the VMS.

## Frozen triage model and classes

Model: `Qwen/Qwen2.5-VL-3B-Instruct`, deterministic (`do_sample=false`).

Allowed classes only:

`plant`, `root`, `flower`, `zodiac`, `star_astronomy`, `bath_human`, `architecture_cartography`, `diagram_geometry`, `other_relevant`.

Maximum four tight non-overlapping objects per selected page. Invalid/out-of-range/degenerate boxes are discarded mechanically. Ordinary text, initials and generic ornament are excluded.

HF run: `6a77418bda2af92a634ef696`, one L4, hard timeout 40 minutes.

## Text arm

Freeze `sentence-transformers/all-MiniLM-L6-v2` for the blind structured-description representation. It is 384-dimensional and is not selected using any corridor outcome. Descriptions are neutral morphology/geometry descriptions produced during blind triage; manuscript/place/artist/source names are prohibited.

## Image arm

The image arm remains frozen to `facebook/dinov3-vit7b16-pretrain-lvd1689m`, no finetuning. Only tightly bounded, normalised crops may be embedded; whole-page RGB embeddings remain prohibited as inferential evidence.

## Geometry arm

Only explicit features emitted under the frozen class schema or deterministically computed from accepted boxes/crops are permitted. No manually added feature may be introduced after similarity inspection.
