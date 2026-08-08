# RUNNING RESULTS — Alpine–Venetian Corridor v0.1

## 2026-08-08 — BUILD

Status: **BUILT / NOT YET RUN**

### State

- Protocol frozen before corridor/control similarity analysis.
- Git branch created: `experiment/alpine-venetian-corridor-v0.1-20260808`.
- Additive Supabase schema migration applied successfully.
- Eight initial RLS-enabled tables created: `corridor_runs`, `corridor_candidates`, `corridor_pages`, `corridor_objects`, `corridor_embeddings`, `corridor_control_matches`, `corridor_scores`, `corridor_results`.
- Existing registry staged neutrally by date+production-place pattern only: 67 candidates, all `needs_review` and all tagged `legacy_voynich_context`.
- Existing corridor-core A+B seed count = 5, below primary coverage gate of 12.
- External neutral census is therefore required before the primary hypothesis can be evaluated.
- Existing corridor seed records currently have almost no downstream object coverage under their registry IDs; page/image binding is an explicit early stage.

### Amendment 001 — mandatory archive-first novelty gate

User requirement added after the inferential preregistration: proactively search the internal Voynich archive before interpreting external findings, so prior community discussion is not presented as new.

This amendment does not alter the frozen hypothesis, sample rules, controls, feature families, thresholds or statistical endpoint.

Schema additions applied:

- `corridor_archive_prior_art`
- `corridor_novelty_register`
- archive prior-art status/count fields on `corridor_candidates`

Initial archive scan results:

- `Venice`: 552 passages / 306 sources.
- `Padua`: 239 / 145.
- `Alps`: 210 / 136.
- `Alpine`: 152 / 104.
- `Tyrol`: 120 / 73.
- `Trento`: 48 / 30.
- `Verona`: 40 / 24.
- `Bolzano`: 11 / 7.
- `Brixen`: 12 / 8.
- `Brenner`: 4 / 4.

The broad corridor proposition is therefore not novel. Ten high-signal direct prior-art passages have been loaded into `corridor_archive_prior_art`, and the broad hypothesis has been explicitly registered in `corridor_novelty_register` as `already_discussed`.

High-signal prior art includes:

- Venice/Padua intellectual-crossroads discussion in 1997 (`voynich_nu_list_a1997h`, para 48).
- Veneto herbals / Carrara + Liber de Simplicibus comparison in 2001 (`voynich_nu_list_a2001k`, paras 42–43).
- Tyrol/Trento as a German–North-Italian synthesis in 2011 (`voynich_nu_list_a2011a`, para 46; `a2011b`, para 164).
- Trento–Bolzano visual transmission along Val d'Adige in 2021 (`voynich_ninja_thread-3339`, para 17).
- Brenner Pass explicitly framed as the main Italy–Northern Europe route in 2021 (`voynich_ninja_thread-3643`, para 81).
- Published southern-German/northern-Italian context with Padua comparanda in Brewer 2022 (`brewer_2022_emotions_encipherment`, para 6).
- A 2026 archive post explicitly combining Bozen/Bolzano, the Brenner axis and the intersection of Alpine and North-Italian architectural signals (`voynich_ninja_thread-3643`, para 503).

### Consequence for research priority

The programme will not spend effort proving that the corridor idea has been thought of before. Priority is now:

1. previously undiscussed eligible manuscripts;
2. previously unseen relevant folios/images in known manuscripts;
3. new ownership/copying/workshop/university/itinerary evidence connecting corridor nodes;
4. new negative evidence;
5. controlled measurements and matched-control tests that have not been performed previously.

Every future candidate is archive-scanned even if it remains statistically eligible; prior discussion affects novelty and research priority, not cohort inclusion.

### No hypothesis result yet

No VMS similarity score has been used to select or rank the cohort. No primary or family test has been run. No hypothesis verdict is permissible at this point.

### Next executable stage

`stage -1/0/1`: run candidate-by-candidate archive prior-art scans, verify the 67 staged metadata records, resolve facsimiles, and independently enumerate additional 1350–1500 illustrated manuscripts from neutral institutional/scholarly catalogues for the corridor and frozen controls. New external finds are checked against the archive before being escalated.
