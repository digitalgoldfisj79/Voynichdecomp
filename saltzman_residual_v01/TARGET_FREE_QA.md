# Target-free QA

Before workflow launch:

- protocol SHA-256 fixed to `b600d0e80beecc609823e2b24f303eec45c7e8862fae7b14ca6f725ff7b67dd0`;
- authoritative section map SHA-256 fixed to `a5d2a9e7aec3d3511ff00de828a17abd2d2255d065c70940ba72ed8abc753cb3`;
- runner compiles under Python 3.12;
- toy ED1 tests distinguish substitution, insertion, deletion, first/final position and reject exact/non-ED1 pairs;
- toy higher-order fingerprint test detects an ABA motif;
- RF parser structural preflight retained 37,193 clean words / 6,515 clean segments and all nine primary sections exceeded support thresholds;
- no numerical residual metric, model distance, mechanism gain, shared-residual result, or section-interaction result was inspected before freeze;
- `enriched_records.pkl` is prohibited by both protocol and workflow grep guard.

Workflow is committed last.
