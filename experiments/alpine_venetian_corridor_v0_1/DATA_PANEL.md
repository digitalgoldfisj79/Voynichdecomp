# DATA PANEL — v0.1 build baseline

Snapshot date: 2026-08-08

## Existing Manucomp registry

`public.manuscripts`: 1,583 rows at build inspection.

The corridor programme does not treat this registry as a neutral census. It is an existing Voynich comparanda registry and therefore every seed promoted from it is tagged `discovery_bias=legacy_voynich_context` until reproduced by neutral catalogue discovery.

## Metadata-only seed staged into `corridor_candidates`

The build seeded all 1350–1500 registry records whose **production/origin** field matched one of the frozen corridor/control geographies. No similarity score was consulted.

| Geography | B antecedent | A primary | C reception | D late reception | Total |
|---|---:|---:|---:|---:|---:|
| corridor_core | 2 | 3 | 1 | 4 | 10 |
| corridor_buffer | 1 | 2 | 0 | 0 | 3 |
| control_bavaria_swabia | 4 | 11 | 15 | 13 | 43 |
| control_east_alpine | 0 | 2 | 1 | 0 | 3 |
| control_lombardy | 3 | 2 | 1 | 0 | 6 |
| control_tuscany | 1 | 1 | 0 | 0 | 2 |

Total staged: **67**.

All are currently `needs_review`, not `included`.

## Immediate coverage issue

The existing registry contained only **5 corridor-core manuscripts in A+B combined**, plus **3 corridor-buffer manuscripts in A+B**. The preregistered primary gate requires >=12 verified illustrated corridor-core manuscripts across A+B. Therefore the present registry alone cannot establish the hypothesis; neutral external census work is mandatory.

At build inspection, only 5/10 corridor-core seed records had an IIIF manifest URL recorded. None of the initially queried corridor seeds had downstream `cat_herbal_folios` or `herbal_objects` rows under those registry IDs. Stage 0/1 must therefore resolve facsimiles and bind pages/objects before any similarity analysis.

## Named existing corridor seeds seen during build audit

Examples already present in `public.manuscripts` include:

- British Library Egerton MS 2020 (Erbario Carrarese) — Padua, 1390–1400.
- De Virga World Map — Italy/Venice, 1411–1415.
- Biblioteca Marciana Lat. VI.59 (2548), Roccabonella Herbal — Padua, 1430–1459.
- Bodleian MS. Bodl. 646, Basinio da Parma, *Astronomicon* — Padua, 1460.
- Padova Orto Botanico Ar.26, Pseudo-Apuleius herbal — later reception.
- Wellcome MS.208 — Padua/Santa Giustina, later reception.
- several Tyrolean/Austrian comparanda, treated as `corridor_buffer` unless a route-node production place is independently established.

These examples are leads, not results.

## Existing comparison infrastructure to reuse

- `public.manuscripts` — canonical comparanda registry.
- `public.cat_herbal_folios` — manuscript page inventory.
- `public.herbal_objects` — crops, descriptions and embeddings.
- `public.herbal_null_stats` — pair-level null calibration for botanical objects.
- `public.comparanda_illuminations` — Biblissima-derived illumination records.
- `public.zodiac_sign_ratings` / `cmp_zodiac_*` — existing zodiac comparanda.
- `public.cmp_manuscripts`, `cmp_images`, `cmp_depictions`, `cmp_embeddings` — general comparanda spine.
- `public.cmp_archive_*` — archive-mined manuscript leads; explicitly noisy and not a neutral census.

## Build-time warning carried forward

The `manuscripts` table documents a measured per-manuscript / scan-pipeline colour constant that makes raw image embeddings unsafe for cross-manuscript inference. The corridor protocol therefore makes the institution-classifier confound test a hard gate and prohibits raw-page RGB embeddings as primary evidence.
