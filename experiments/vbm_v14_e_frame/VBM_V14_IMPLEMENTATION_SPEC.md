# VBM v14 — implementation specification

Date: 2026-09-02
Status: **FROZEN BEFORE BINDING OUTPUT**

Operational details:

- Jensen-Shannon vectors concatenate separately normalised left/right categorical distributions and are then renormalised to unit sum before distance calculation.
- Full-TRAIN half vocabularies are reused unchanged in TRAIN-A and TRAIN-B.
- Frequency deciles and all matching pools are defined from full TRAIN only.
- Branch-B frame overlap is computed from the split being evaluated; candidate residual distances are precomputed but matching rules are unchanged.
- Branch-C outcome vocabulary is the full-TRAIN external-half vocabulary plus OTHER and EDGE.
- For Branch C, skeleton backoff distribution uses Dirichlet 0.5 per outcome. Both `P(outcome|frame,skeleton)` and `P(outcome|frame,skeleton,m)` receive prior mass 1.0 distributed according to that skeleton backoff distribution. No interpolation weight is fitted.
- All null RNG seeds are SHA256-derived from namespace `VBMV14EFRAME20260902` and branch/split/null index.
- Exact ties in matched pools are resolved by lexical nucleus order only after deterministic random sampling indices are generated.
- The binding run is one `cpu-upgrade` job containing A, B and C. No GPU is permitted.