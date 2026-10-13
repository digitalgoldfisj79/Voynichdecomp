# Velinska → citation ledger: running log

## 2026-10-13
- Scope (user decision): 142 Voynich-related posts; 14 non-Voynich + 14 image-only posts excluded; 12 double-posted comments dropped.
- Conventions copied from existing `dulov_` Blogger rows (source_id pattern, license, venue, sha256 = sha256(body)).
- Embeddings: Supabase/gte-small via Actions run (velinska-ledger-embed). Calibration vs stored production vectors:
  cos = 1.00000 on 3/3 passages (Russian + 2 English). Bounding test passed.
- crawl_sources: `velinska_` row inserted, is_enabled = false.
- sources: 142 rows inserted via SQL connector. Verification: count 142, sha256(body) matches 142/142,
  digest md5(sorted id:sha256:word_count:title) = 38700e8143e8c45769d5d536942966fe both locally and in DB.
- source_passages: PENDING (508 rows). Sources are not retrievable by search_sources until passages exist.
- Rollback: DELETE FROM sources WHERE source_id LIKE 'velinska\_%'; DELETE FROM crawl_sources WHERE source_id_prefix='velinska_';
