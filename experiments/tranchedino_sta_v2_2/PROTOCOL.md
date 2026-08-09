# Tranchedino × STA v2.2 — Final T20 optimizer qualification

Date: 2026-08-09
Parent: `b2735191578960d7c59829400313903088c90ad5`
Namespace: `TRANCHSTA22`

This is the final allowed search-budget escalation for the direct f069v 36-sign alphabetic model. It uses T20 only and cannot inspect H21/C21.

Run six fresh independent ensembles K, L, M, N, O, P. Each receives 96 restarts, the unchanged f069v 36→19 multiplicity profile, unchanged Paduan quadgram, frequency initialization, incumbent perturbations/full random starts, and exact pair-block polish.

The T20 fit gate passes only if the highest objective among the six is attained by at least two ensembles within 1e-7 nats/retained symbol and their occurrence-weighted maps agree >=0.90. The canonical map is the lexicographically smallest among agreeing top maps.

If this gate fails, the direct fixed 36-sign Tranchedino alphabetic model closes as `TARGET MAP NOT ROBUSTLY IDENTIFIABLE UNDER CALIBRATED SEARCH`; no further restart escalation is permitted in this programme.

If it passes, the untouched v2.1 H21/C21 split and all v2.1 held-out thresholds/null rules are inherited verbatim. H21 is then unlocked; C21 remains sealed pending H21 plus cross-transliteration replication.

No diagnostic map from v2.0/v2.1 is inserted or used as an initializer.
