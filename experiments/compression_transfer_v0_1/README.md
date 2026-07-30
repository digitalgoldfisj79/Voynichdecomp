# Compression-Transfer Distance Programme v0.1

This directory implements the preregistered compression cross-entropy and normalized compression-distance programme inspired by *Language Trees and Zipping*.

## Current status

- Research branch created from CoReMA closeout commit `f86759d6651fdde135c427682c79a07ef8df38f9`.
- Voynich analysis is sealed.
- Metric core, representation registry, corpus manifest validation, row-level runner, NCD tree builder, null controls, consensus evaluator and independent validator are implemented.
- Deterministic smoke fixture passes.
- Formal corpus manifests are not yet populated; no scientific calibration result exists.

## Directory layout

- `PROTOCOL.md` — scientific design, gates and stop rules.
- `DATA_PANEL.md` — required corpus inventory and acquisition rules.
- `CORPUS_MANIFEST_TEMPLATE.csv` — document-level provenance schema.
- `configs/` — smoke and formal-stage templates.
- `code/compression_metrics.py` — directional conditional cost, excess cost and NCD.
- `code/representations.py` — frozen text encodings.
- `code/run_benchmark.py` — row-level benchmark runner.
- `code/evaluate_consensus.py` — compressor/representation consensus.
- `code/make_null_controls.py` — deterministic shuffle controls.
- `code/build_tree.py` — deterministic UPGMA/Newick output.
- `code/validate_results.py` — independent output arithmetic and hash checks.
- `tests/` — unit tests.
- `fixtures/smoke/` — deterministic synthetic fixture.

## Smoke run

```bash
python -m unittest discover -s tests -v
python code/make_smoke_fixture.py
python code/run_benchmark.py configs/smoke.json
python code/validate_results.py results/smoke
python code/evaluate_consensus.py \
  results/smoke/directional_observations.csv \
  --required-votes 2 \
  --output results/smoke/consensus.json
python code/build_tree.py \
  results/smoke/ncd_pairs.csv \
  --representation codepoint_u32_ws \
  --compressor zlib9 \
  --output results/smoke/tree.newick
```

The smoke fixture is an engineering test only. Its four synthetic sources have deliberately different alphabets and transition systems.

## Formal execution sequence

1. Populate and hash the Stage 1 known-source corpus manifest.
2. Freeze acquisition, duplicate screening, split assignment and config.
3. Run development diagnostics; modify only before locking.
4. Execute the untouched Stage 1 test once.
5. Freeze Stage 1 decision.
6. Build and freeze Stage 2 fresh-key cipher/generator panel.
7. Execute Stage 2 once.
8. Open the sealed Voynich config only if the protocol's opening rule is satisfied.

## Dependencies

The mandatory implementation uses only Python 3 standard-library compressors. `zstandard` and `pyppmd` are optional preregistered compressors with pinned versions in `requirements.txt`.

## Scientific boundary

The strongest permissible result is bounded source-family compatibility. Compression proximity is not decipherment and does not prove that a message exists.
