# Yiddish cipher control — canonical state

Date: 2026-09-12

## Controlling status

**`V02_SMOKE_PASS__FULL_BLINDED_CONTROL_READY_NOT_RUN__C6_NOT_RUN__VOYNICH_SEALED`**

This file is the restart point for future sessions. Read it together with Supabase handoff `voynich_cipher_instrument_recovery_standard_20260912_v01` before doing new cipher work.

## What happened

1. The first non-qualifying smoke harness (`run_m0_control.py`) executed end-to-end in GitHub Actions run `34708340714`.
2. C0/C1/C2 executed successfully, official ReF 1.0.2 diplomatic German material was acquired and parsed, and all target-access flags remained false.
3. The smoke metamorphic check MR1 **failed**: arbitrary global renaming of ciphertext symbols changed recovery from atom 0.7422 / non-pass to atom 1.000 / pass. This exposed a real search implementation dependency on raw cipher-symbol IDs (frequency-tie ordering and RNG index swaps).
4. No full qualification and no Voynich target outcome had been exposed. The failure was therefore legitimate DEVELOPMENT evidence.
5. Candidate v0.2 (`run_m0_control_v02.py`) adds one repair only: before search, ciphertext symbols are canonicalised by order of first occurrence in the fit ciphertext. The frozen S1 objective/search is then run on that canonical representation and the returned mapping is conjugated back to the external labels.
6. GitHub Actions run `34708602061` reran the smoke control under v0.2. It completed successfully and C5 reported **zero violations**. MR1 therefore passed under the repaired representation.

## v0.2 smoke evidence

- C0 provenance boundary: PASS (smoke semantics only).
- C1 secondary Penn representation path: PASS.
- C2 known-answer encoder/decoder checks: PASS.
- C3 smoke positive trials: non-qualifying by design; Bovo 512 0%=1/2, 1%=2/2; Cracow 512 0%=2/2, 1%=2/2 under reduced smoke search budget.
- C4a smoke structural negatives: 0 false calls in each tiny n=2 family, explicitly UNBOUNDED / non-qualifying.
- C4b German comparator: ReF F016 yielded 8,385 diplomatic normalized words; 0/1 recovery under reduced smoke budget; diagnostic only.
- C5 smoke metamorphic checks: 7 checks, **0 violations**.
- C6 independent historical/source transfer: NOT RUN.
- C7 Voynich: SEALED.

Artifact v0.2 smoke digest: `sha256:337e2832c442be663f753503e6811cc3ef0de6160b052e0f87f000df9f4dbb55`.

## Statistical correction frozen before full run

`STATISTICAL_BOUND_ADDENDUM_V01.md` controls full-profile false-positive bounding:
- family-wise Bonferroni alpha = 0.05/6;
- at least 94 negative trials per gated family; 0/94 gives one-sided upper bound ≈0.0496555;
- stochastic metamorphic relations use at least 32 planted instances;
- smoke evidence can never issue a scientific PASS.

## Full blinded architecture now built

Directory `full/` contains:
- `prepare_full.py` — generates a public ciphertext/training bundle and a separate private truth/root bundle; commits the plant-root hash before solver outcomes.
- `solve_full.py` — shardable label-equivariant frozen S1E solver; sees only public ciphertext and BUILD training words.
- `score_full.py` — joins private truth only after solver mappings are committed; reports key/plaintext recovery, oracle-vs-returned objective, search misses vs objective misalignment, exact bounds, and metamorphic paired results.

Workflow `.github/workflows/yiddish_cipher_control_full.yml` is **manual dispatch only** and uses 12 parallel blind solver shards. Solver jobs never download the private truth artifact. The score job runs only after all shards complete.

The full case generator covers:
- C3 positive power surface: 128/256/512/1024/2048 words where source quantity permits; 0/1/3/5% erasure; 32 keys/cell across consumed development works.
- C4a non-global substitution negatives: per-word and registered key-drift regimes, 94 trials/family.
- C4b language/nuisance diagnostics: within-word shuffled Yiddish, unigram-matched nulls, and official ReF historical German under genuine global M0.
- C5: MR1 global ciphertext relabelling, MR2 consistent plaintext+LM relabelling, exact duplication/chunk identities, independent decoder checks, and boundary-destructive controls.

## Important claim boundary

A future C0-C5 PASS would qualify only the **M0 mechanism instrument** in the declared secondary normalized representation and operating envelope. It would still not establish Yiddish identity because historical language discrimination L remains unresolved, and it would not establish primary-script transfer because C6 remains outstanding.

No Voynich target may be loaded until the relevant instrument, language, representation, and transfer certificates are all prospectively qualified.