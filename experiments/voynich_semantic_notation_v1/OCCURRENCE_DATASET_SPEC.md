# VSN-v1 Occurrence-Level Dataset Specification

Frozen 2026-08-12.

## Source text

Primary source is Supabase `public.transliteration_loci` with `transliteration_id='voynich_nu_data_RF'` (Reference combined ZL+GC, Basic EVA). Dot-delimited items are token occurrences. Raw strings are preserved.

Primary morphology inference uses only tokens matching `^[a-z]+$`. Ambiguous RF readings containing commas or other marks remain as rows with `primary_eligible=false`; they are never silently normalized.

## Snapshot tables

Schema: `voynich_semantic_notation_v1`.

### `rf_occurrences`

One row per RF token occurrence.

Identity/text:
- `occurrence_id`
- `locus_id`
- `folio`
- `locus`
- `line_no`
- `raw_token`
- `token_rf`
- `primary_eligible`
- `token_length`
- `token_frequency_global`

Codicological/context:
- `quire`
- `folio_seq`
- `section`
- `currier_hand`
- `davis_scribe`

Layout:
- `layout_marker` (raw IVTFF marker such as `@P0`, `+P0`, `@Lf`)
- `layout_family` (second marker character only; preserves class without expanding its semantics)
- `block_initial`
- `label_like` (`layout_family='L'`)
- `running_text_like` (`layout_family='P'`)
- `paragraph_id` (sequential start-marker segmentation for P-layout loci only)
- `word_position_in_line`
- `line_initial`
- `line_final`
- `tokens_in_line`

Observable page annotations retained but not automatically primary endpoints:
- `page_feature_annotation`
- `illustration_desc`
- `general_desc`

Cardinality rule: metadata joins must not duplicate RF loci. Duplicate canonical folio rows caused by Yale foldout canvases are aggregated to one metadata row before the join. The audited snapshot therefore has exactly the same raw token count as direct RF dot splitting.

### `visual_targets`

Existing herbal targets only; no new vision inference at Stage 1.

Fields:
- `folio`
- `folio_seq`
- `part` = `plant` or `root`
- `object_id`
- `obj_index`
- `bbox`
- `crop_path`
- `crop_qa`
- `description_text`
- `embedding`
- `embedding_dim`

Selection within folio/part is deterministic: prefer non-QA-flagged objects, then `obj_index=0`, then larger bounding-box area. `spurious_fragment` is excluded. Embeddings are copied by reference from existing data; no recomputation.

## Current audited coverage

At first snapshot audit:
- raw RF occurrences: 36,680;
- exact-letter primary RF occurrences: 35,314;
- RF folios represented: 227;
- quires represented: 18 plus records with missing quire metadata;
- stored visual targets after deterministic selection: 114 plant and 114 root folios;
- visual embedding dimension: 3,072.

## Explicit missingness / blockers

The present Supabase `voynich_dinov3.words` and `voynich_dinov3.folios` tables are empty. Therefore token-level x/y, word crops and local image-object distances are not available from that asset.

`vms_catalogue.vms_folios.currier_hand` and `davis_scribe` are currently unpopulated for RF-linked records. They remain null. No page-level surrogate is substituted under these column names.

Consequences:
- Stage 1 can validly test page-level visual embeddings, quire-block transfer, IVTFF layout, token family, frequency and line-position effects.
- It cannot yet claim survival of token-level spatial or hand controls.
- A future coordinate/hand enrichment is allowed only as a protocol amendment and must not revise already-opened confirmation outcomes.

## Derived morphology tables to create before visual discovery

1. `rf_token_types`: frequency and dispersion by folio/quire/layout/section.
2. `rf_edit1_pairs`: exhaustive distance-1 minimal pairs generated without visual outcomes.
3. `rf_affix_candidates`: prefix/suffix candidates lengths 1–4 satisfying frozen support thresholds.
4. `rf_component_contrasts`: candidate-bearing and matched core/alternative forms, with family/core identifiers.
5. deterministic `block_split`: discovery/confirmation assignment by frozen SHA-256 namespace.

## Independence rules

- Morphology tables are generated without joining `visual_targets`.
- Confirmation block assignments are computed before morphology × visual scoring.
- Free-text descriptions cannot be used to tune component discovery.
- Human-coded page features and image embeddings are separate evidence arms.
