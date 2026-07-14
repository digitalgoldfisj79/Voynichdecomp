# Frozen MDL Codec v0.1

Executable accounting layer for bounded cipher-versus-production comparisons.

This branch contains no Voynich manuscript result. It provides:

- canonical, prefix-free model serialization;
- full and surface-inventory-conditional historical costs;
- enumerative structural costs;
- KT universal categorical costs;
- exact finite-state latent-path marginalization;
- cost-envelope verdict logic;
- deterministic conformance and fuzz tests;
- a leave-one-selection-policy-out synthetic calibration gate.

Start with `SPEC.md`.

## Local checks

```bash
python -m unittest -v test_codec.py
python run_conformance.py
python fuzz_codec.py
python synthetic_gate.py --quick
```

`registry_fixture.json` is not a production registry. No manuscript run is admissible until the verified external artefacts are entered into a separately frozen registry.
