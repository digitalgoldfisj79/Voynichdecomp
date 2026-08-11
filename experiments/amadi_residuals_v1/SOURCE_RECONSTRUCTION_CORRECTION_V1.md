# Source Reconstruction Correction — section 024 `g`

Date: 2026-08-11
Status: **FROZEN BEFORE Q1 / VOYNICH H2**

This note supersedes the `g -> i` line in `SOURCE_RECONSTRUCTION_V1.md`.

The source transcription says that `g` is avoided and gives the examples:

- `gioue -> ioue`
- `giocho -> iocho`
- `gienere -> ienere`

The long reduced-text example also contains `negotii -> neotii` and `manegiare -> maneiare`.

These outputs consistently remove `g`; they do not insert a second `i`. The source-derived deterministic R12 formalisation therefore uses:

`g -> deletion`

The earlier shorthand `g -> i` came from reading the prose description literally while failing to test it against the worked outputs. The smoke Q0 gate exposed the inconsistency before qualification and before any Amadi-residual Voynich score.

Final `R12_V1_024` deterministic formalisation used in v1:

- `b -> u`
- `d -> t`
- `f -> deletion`
- `g -> deletion`
- `h -> deletion`
- `p -> deletion`
- `q -> c`
- modern consonantal `v -> o` as the operational counterpart of historical consonantal u/v
- inherited post-reduction normalisations `j->i`, `w->u`, `y->i`, `x/z->s`
- unsupported modern foreign `k` causes whole-word exclusion from R12 controls rather than an invented historical mapping.

No target-derived information entered this correction.