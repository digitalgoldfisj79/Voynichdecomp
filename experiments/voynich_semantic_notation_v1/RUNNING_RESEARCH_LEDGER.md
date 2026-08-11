# VSN-v1 Running Research Ledger

## 2026-08-12 — Stage 1

### Repository
- Branch created: `experiment/voynich-semantic-notation-v1-20260812`.
- Parent: `experiment/vbm-prequential-mdl-v8-20260811`.
- New namespace: `VSN-v1`; no new Bavarian/German VBM variant.

### Preregistration
- `STRUCTURED_NOTATION_PROTOCOL.md` frozen before visual target scoring.
- Deterministic discovery/confirmation block split created before morphology×visual scoring.
- CONFIRMATION remains sealed.

### Data inventory
- Reused RF transliteration, catalogue metadata, existing herbal object crops/embeddings and `manucomp` database.
- Did not rerun vision extraction.
- Existing `voynich_dinov3.words` / `.folios` tables are empty schema stubs; no token x/y data available there.
- Davis/Currier hand catalogue fields are unpopulated for RF-linked rows.

### Data integrity incident
- First occurrence join inflated 36,680 raw RF token occurrences to 36,888 because canonical folio metadata duplicate foldout canvases.
- Rejected and rebuilt with cardinality-safe aggregated metadata join.
- Final snapshot = exactly 36,680 raw occurrences / 35,314 primary exact-letter occurrences.

### Morphology
- 7,893 RF types.
- 28,435 exact one-edit pairs.
- 361 support-qualified affix/component candidates.
- 401,615 matched component/core/type contrast rows.

### Visual assets
- 114 deterministic whole-plant folio targets.
- 114 deterministic root targets.
- Existing embedding dimension 3,072.

### Discovery
- Outcome-blind cross-family screen generated multiple BH-significant candidates.
- Leave-one-quire-out removed much of the screen.
- Broad `suffix:2:dy` / whole-plant is the clearest surviving discovery signal:
  - TEST-core mean cosine = 0.18544;
  - sign-flip BH q = 0.01795;
  - LOQO 7/7 quires positive, mean 0.09192;
  - block-bootstrap 95% interval ≈ [0.05125, 0.13813];
  - max weighted quire contribution ≈37.5%;
  - within-quire page reassignment observed 0.00574543 vs null p99 0.00306116; empirical p=1/257=0.003891.
- Nested `-edy/-hdy` signals are not independent until de-nested.

### Decisive blocker
- Sealed herbal confirmation arm has only 2 relevant blocks, 17 plant folios and 9 matched `-dy` residual cores.
- Preregistered 25%-shrunk planning power ≈43% at matched-core level and ≈68% even at folio level.
- Frozen >=80% power gate fails.
- CONFIRMATION NOT OPENED.

### Historical workstream
- Exact-anchor `manucomp` query returned no records for Vat. lat. 4082, Berlin lat. fol. 246, Canon. Misc. 554, Correr Cic. 3747 or Prosdocimo under those names.
- Grade B mechanism precedent: Prosdocimo/Paduan mensural notation (base sign + modifiers/context/proportion/ligature).
- Grade C direct Paduan/northern-Italian pedagogical precedents: Vat. lat. 4082; Berlin lat. fol. 246; Canon. Misc. 554; Correr Cic. 3747; calculatores/latitudines tradition pending tighter Paduan manuscript evidence.
- No grade-A precedent found in first pass.
- Actual target-folio inspection remains incomplete; no candidate promoted on unseen folios.

### Compute
- No HF jobs launched.
- HF status check: no running jobs.

### Database security note
- During Supabase inventory the platform reported RLS disabled on ten `public` tables: `handoff_docs`, `handoff_docs_deleted_2026_05_28`, `forum_url_backfill`, `forum_auth_probe`, `forum_ingest_config`, `forum_fetch_queue`, `forum_thread_paras`, `cmp_archive_url_pool`, `cmp_archive_watermark`, `cmp_archive_mentions`.
- No security setting was changed automatically because enabling RLS without complete policies can break legitimate access. This should be reviewed separately.

### Current state
- Workstream A: **promising discovery, formal confirmation blocked by frozen power gate**.
- Workstream B: **one grade-B mechanism precedent; several grade-C pedagogical precedents; no grade A**.
- Required closeout not yet written because the larger programme remains open; Stage 1 is a valid stopping/reporting point.
