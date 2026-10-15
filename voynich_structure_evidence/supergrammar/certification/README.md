# Super Grammar v03 certification tables

These are the tables behind the frozen v03 release
([release note](../../../research/RELEASE_supergrammar_v03_20260920.md)). They hold the theorem → lemma → certificate →
obligation → evidence chain, so each certified statement can be traced back to the numbers that support it.
Exported from the research database on 15 October 2026. Every table is given as `.json`. Tables with no nested fields
are also given as `.csv`; `evidence_v03.csv` is the flat version of `evidence_v03.json` with the `source_snapshot` column dropped.

## How to trace a certified theorem

1. **`certified_theorems_v03`**: the 9 certified theorem wrappers (SGT01, 04–11). SGT07 is `CERTIFIED`, the rest are
   `CERTIFIED_BOUNDED`.
2. **`theorem_dependencies_v03`** / **`kernel_dependencies_v03`**: which lemmas (claim IDs, `SGK…`) each theorem needs,
   and whether each is `REQUIRED`, `NEGATIVE_CONSTRAINT` or `BOUND_ONLY`. Every dependency of the 9 certified theorems
   is `SATISFIED`. The only unsatisfied rows belong to the superseded wrappers SGT02 and SGT03, plus the optional
   order-3 strengthening of SGT01, which failed and is excluded.
3. **`certificates_v03`**: one certificate per claim per run. For run `supergrammar_v03_20260920`: 32 certified
   (2 `CERTIFIED`, 30 `CERTIFIED_BOUNDED`, including the 19 hard-zero glyph pairs `SGK03_HZ_*`), 2 `FAILED`
   (`SGK02_INTERNAL_ORDER3`, `SGK09_EXACT_IDENTITY_CLOSED`), 8 `REJECTED`, 5 `SUPERSEDED`, 1 `RETRACTED`, 55 `OPEN`.
   Each certificate records how many required obligations it had and how many passed; a certified claim has
   `open_required = 0` and `failed_required = 0`.
4. **`claim_obligations_v03`**: every obligation per claim (2,537 rows), with its status. **`obligation_types_v03`**
   defines the 18 obligation codes and the order they are audited in.
5. **`obligation_evidence_v03`** → **`evidence_v03`**: the evidence rows that discharge each obligation, with effect,
   null SD, effect/null SD and the source relation and selector each number was read from.

## Supporting tables

- **`release_v03`**: the frozen release record (9 theorems, 32 certified claims, 2 failed, 5 excluded open kernel
  candidates, 9 retractions), the full release manifest, and its SHA256 (`c4bbace8…027e`, same as the release note).
- **`validation_v03`**: the fail-closed release checks. All 10 pass: 6 for v03, 4 for the v03.1 correction run.
- **`verifier_runs_v03`**: every independent replay job, including the ones that failed before producing an outcome
  (`FAILED_PREOUTCOME`) and one that was invalidated. Failed runs are kept and not counted as evidence.
- **`retractions_v03`**: the retraction and correction ledger (16 rows across runs).
- **`control_registry_v03`**: the external control corpora, with what each may and may not be used for.
- **`canonical_sources_v03`**, **`runs_v03`**: source documents and run manifests that define the authority chain.

## Status vocabulary

`certificate_status` here is authoritative for certification. The `certification_status` column in
[`../claims_register_v03.csv`](../claims_register_v03.csv) is the claim ledger's own label, set before certificates
were issued and not updated afterwards, so certified lemmas still show `OPEN_UNCERTIFIED` there. The register now
carries `certificate_status` alongside it; read that column.

No later run has a certified certificate yet. Open required obligations, from `claim_obligations_v03`:

| Run | Claims | Still open |
|---|---|---|
| `supergrammar_v031_20260928` (correction run) | 33 | 32 claims open only on `EXTERNAL_ADVERSARIAL_REVIEW`; 1 failed (order 3) |
| `supergrammar_v032_20261002` (new candidates) | 5 | `EXTERNAL_ADVERSARIAL_REVIEW` on all 5; `IMPLEMENTATION_REPLAY` on 4 |
| `supergrammar_v04_realization_library_20260924` (SG64–66) | 3 | No certificate. Circularity, confounds, control fairness, implementation replay, representation dependence and others still open |
| `supergrammar_v03_extension_gallows_20260924` | 3 | No certificate. Most obligations open, including implementation replay and matched null |
