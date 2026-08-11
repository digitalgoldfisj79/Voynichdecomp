# VBM v1 — Amendment 001: Strong Optimizer

Date: 2026-08-11

The first full typed-substitution qualification generated no Voynich typed-target score. All four Bavarian controls ranked Bavarian first, but A/B map convergence failed; one German control also failed convergence. The run was manually cancelled immediately after the positive-control table and before structured negatives.

This is therefore an instrument-calibration amendment, not target rescue.

Frozen representation, language panel, VBM_H1/VBM_C1 split, bridge vocabulary rule, absolute-score logic, language-margin gate, recovery gates and negative-control gate are unchanged.

A fresh qualification namespace `VBMV1TYPEDQ2` is used. No Q1 hidden key, fitted map or optimizer state is reused.

Optimizer only:
- proposals per restart: 160,000;
- maximum restarts per ensemble: 24;
- batches: 6 restarts per ensemble;
- same annealing kernel, surjectivity constraints and deterministic coordinate polish;
- same convergence criterion: score gap <=1e-7 nats/event and occurrence-weighted A/B map agreement >=0.95.

If fresh Q2 positive controls do not qualify, typed VBM v1 stops as `INSTRUMENT NOT QUALIFIED` and no typed Voynich H1 score is generated.
