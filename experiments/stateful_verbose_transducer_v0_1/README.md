# Stateful Verbose Transducer v0.1

This experiment tests the remaining intersection left by Terminal Cipher v0.6: a **stateful + variable-length + hidden-segmentation** cipher.

The primary head mechanism is order-free: state-specific substitution alphabets are inferred without assuming an Alberti-style circular ordering of visible glyphs. The surface mechanism expands each hidden head to 1–3 same-alphabet glyphs with locally conditioned continuation structure.

The programme is deliberately staged. `svt_v01.py` contains the synthetic generator, boundary lattice, order-free stateful solver and hostile controls. `run_synthetic.py` is the only entry point allowed before the locked gate. `run_voynich.py` validates the frozen gate hash before opening target bytes.

Binding protocol: `FROZEN_PROTOCOL_v0_1.md`.

Example development command:

```bash
python experiments/stateful_verbose_transducer_v0_1/run_synthetic.py \
  --repo . --output results/svt_v0_1_dev.json --split dev --stage joint \
  --iso de --length 384 --replicates 4
```

The GitHub Actions workflow `stateful-verbose-transducer-v0.1.yml` exposes the frozen synthetic stages. Target application is intentionally not part of the default workflow and remains blocked unless the untouched locked gate passes.
