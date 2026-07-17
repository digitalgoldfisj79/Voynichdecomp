# Corrigendum 001 — Amendment 005 KMeans statement

Date: 2026-07-17

Amendment 005 line 72 states that the v1.5.1 launcher left KMeans `n_init` at a version-sensitive default.

Inspection of the hash-verified assembled scientific source (`fd8f93893a488b59d41eba4395de82e5690ebb491bc8bbe6c1de581a2884cdd8`) shows that this statement is incorrect. Both `MiniBatchKMeans` calls explicitly set:

- `random_state=SEED`;
- `n_init=3`;
- `max_iter=300`;
- fixed `batch_size=4096`.

This corrigendum changes no scientific gate, endpoint, split, metric or v1.6 decision rule. The reproduction environment should still pin `scikit-learn` because library implementation differences may affect exact numerical reproducibility, but missing `n_init` is not an identified defect in the v1.5.1 source.
