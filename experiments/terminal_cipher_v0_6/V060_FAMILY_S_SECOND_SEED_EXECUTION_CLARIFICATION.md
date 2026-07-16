# v0.6 Family S3 — second-seed execution clarification

Date: 2026-07-16

Status: **FROZEN BEFORE ANY S3 DEVELOPMENT EVALUATION.**

The registered final S3 amendment requires two independent Transformer models with identical architecture and different frozen seeds. The implementation and persistence records named seed `1731`, but the amendment report did not record the companion seed. This is an execution omission, not a scientific result.

To complete the already-registered two-model ensemble without inspecting any development recovery output, the companion seed is fixed as:

- model 1: `1731`;
- model 2: `1732`.

The rule is the adjacent integer to the already-recorded first seed. It was selected before loading either trained checkpoint into an evaluator and without reference to plaintext recovery, boundary F1, development examples, test data, Voynich data or Davis labels.

This clarification changes none of the following:

- architecture;
- synthetic generator;
- train split;
- update count;
- batch size;
- optimiser or learning-rate schedule;
- checkpoint selection;
- beam width;
- lattice search;
- scoring rule;
- gate thresholds.

It is not an additional development amendment. It supplies the omitted deterministic identifier required to execute the already-frozen two-model ensemble. No S3 evaluation is permitted until both 30,000-update checkpoints have completed exact-byte persistence verification.