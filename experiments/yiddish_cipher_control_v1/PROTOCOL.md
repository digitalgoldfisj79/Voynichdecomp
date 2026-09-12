# Yiddish Cipher Instrument Qualification Control v1.0

Status: **FROZEN DESIGN — TARGET SEALED**  
Date: 2026-09-12  
Branch: `gpt56/yiddish-control-v1-20260912`

## 0. Purpose

This is not a Voynich decipherment experiment. It is a verification/validation benchmark for any computational instrument that claims to detect, recover, reject, rank, or exclude a cipher mechanism when the underlying plaintext language is historical Yiddish.

The benchmark is deliberately verbose because previous Voynich cipher work sometimes conflated solver failure with mechanism absence. This control reverses that logic: **no negative target result is admissible until the instrument has demonstrated its operating characteristics on blinded positives, matched negatives, nuisance transformations, and independent transfer data.**

The first implemented family is `M0`: a globally fixed bijection on orthographic atoms with preserved word boundaries and order. The harness is designed so later families can be registered under the same gate structure without changing the validation logic.

Voynich data MUST NOT be loaded by any script in this directory.

## 1. External methodological basis

The design borrows from established verification and validation practice rather than a bespoke one-off protocol:

1. NIST CAVP/ACVP: implementation-under-test, known-answer tests, independent vectors, fixed request/response semantics, prerequisite validation of dependent algorithms.
2. Simulation-based calibration / synthetic-ground-truth validation: repeated draws from a known generative mechanism to measure whether the inference/search procedure is calibrated across a parameter envelope rather than on one planted example.
3. Positive + negative control benchmarking: empirical sensitivity and false-positive behaviour must both be measured on controls representative of the application domain.
4. Metamorphic testing: transformations whose expected effect is mathematically known are used as test oracles where the exact output is otherwise unavailable.
5. Independent validation / transfer: BUILD/DEVELOPMENT material is separated from CONFIRMATION material and retuning after confirmation exposure is prohibited.
6. Sealed target admission: the application target may only be examined after all prerequisite certificates pass.

## 2. Claim boundary

A passing M0 control licenses only the following claim:

> Within the prospectively tested representation, length, damage, alphabet-size, source and search-budget envelope, this implementation can recover or detect a globally fixed monoalphabetic bijection on historical-Yiddish-like material at the measured sensitivity while maintaining the measured false-positive rate on registered matched negatives.

A pass does **not** prove that Voynich is or is not Yiddish, does not validate other cipher families, does not validate arbitrary tokenisation, and does not permit extrapolation outside the tested envelope.

## 3. Certificates

The control issues seven independent certificates. All must be PASS before target admission for the corresponding family.

### C0 — provenance / executable freeze

Required:
- source commit and hashes recorded;
- plaintext source IDs and hashes recorded;
- encoder and decoder are separate modules or independently hashed code paths;
- RNG roots committed before outcomes and revealed only after final rows are durable;
- BUILD / DEVELOPMENT / CONFIRMATION assignments frozen;
- no Voynich loader/import reachable from the benchmark entry point.

Failure state: `C0_PROVENANCE_FAIL`.

### C1 — representation verification

For the first executable M0 suite the representation is explicitly **secondary Penn historical-Yiddish Romanisation reduced to a-z atoms**. It is acceptable for algorithm verification only and cannot by itself certify primary-script transfer.

Checks:
- deterministic extraction under repeated execution;
- explicit atom inventory and boundary convention;
- exact source hashes;
- no silent normalization beyond the declared rule;
- word counts and alphabet cardinality recorded;
- alternative representation sensitivity must be separately labelled, never silently substituted.

Primary Hebrew-script / diplomatic transfer is a later independent certificate.

Failure state: `C1_REPRESENTATION_FAIL`.

### C2 — known-answer tests (KAT)

Before stochastic search, test exact encoder/decoder correctness on fixed fixtures:
- identity permutation;
- cyclic permutation;
- reverse permutation;
- deterministic random permutation;
- shortest legal words;
- repeated letters;
- full alphabet coverage;
- word-boundary preservation;
- 0% and 1% erasure semantics.

Every non-erased atom must round-trip exactly. Any mismatch invalidates the implementation.

Failure state: `C2_KAT_FAIL`.

### C3 — blinded positive recovery / power surface

Generate blinded M0 ciphertexts from historical Yiddish plaintext controls. Truth keys are hidden from the solver and stored behind a pre-outcome commitment.

Primary grid:
- plaintext windows: 128, 256, 512, 1024, 2048 words where source quantity permits;
- erasure/damage: 0%, 1%, 3%, 5%;
- 32 independent planted keys per cell;
- at least 3 source families across DEVELOPMENT when quantity permits;
- same frozen solver objective and hyperparameters throughout a registered candidate.

Per-trial outputs:
- exact eligible-atom recovery;
- exact eligible-word recovery;
- key accuracy;
- objective value returned;
- oracle objective value after solver commitment;
- runtime;
- search-miss witness (`oracle_objective > returned_objective`) vs objective-misalignment witness.

Default qualification thresholds for the primary declared operating envelope (512+ words, <=1% erasure):
- atom recovery >=0.90;
- whole-word recovery >=0.80;
- >=29/32 trials pass per work × length × damage cell.

The full power surface is reported even if only a subset is designated as the qualified envelope. No failed cell may be omitted.

Failure state: `C3_POWER_FAIL`.

### C4 — matched negative / false-positive control

The instrument must also encounter inputs where M0-Yiddish recovery is not the generating mechanism.

Registered negative families:
1. historical German plaintext encrypted under its own M0 key but scored against the Yiddish recovery objective;
2. historical Hebrew plaintext under M0 where a comparable source representation is available;
3. Yiddish with word-internal atom permutation independently redrawn per word (destroys global key while approximately preserving local inventory/length);
4. Yiddish with per-line independently redrawn substitution keys;
5. Yiddish with key drift at registered change points;
6. word-order shuffled Yiddish (for tests claiming sequential-language evidence; should not matter to purely equality-invariant tests and therefore has family-specific expected behaviour);
7. Markov / frequency-matched synthetic nulls preserving unigram or bigram nuisance statistics.

For a recovery solver, a `positive call` requires the same frozen success rule used on planted positives. Empirical false-positive rate and exact binomial confidence bounds are reported. Default admissibility gate: upper one-sided 95% bound <=0.05 on the designated negative panel. If sample size is insufficient to bound this, status is `C4_UNBOUNDED`, not PASS.

Failure state: `C4_NEGATIVE_CONTROL_FAIL`.

### C5 — metamorphic relations

M0-specific mathematical relations are prospectively registered:

MR1 Global ciphertext-symbol relabelling: applying any bijection to ciphertext atoms and composing the recovered key accordingly must leave recovery statistics unchanged up to exact arithmetic / deterministic search-seed mapping.

MR2 Plaintext/cipher alphabet renaming: arbitrary one-to-one relabelling of plaintext atoms before planting must not change recoverability when the language model is transformed consistently.

MR3 Dataset duplication: duplicating the fit corpus exactly must not alter the normalized objective ranking of keys. It may alter runtime, not the optimum.

MR4 Order-preserving chunk concatenation: splitting and recombining a fixed fit sequence without changing atom/boundary order must give the same sufficient statistics and objective.

MR5 Independent decoder oracle: after solver commitment, independently inverting the planted key must decode all non-erased atoms exactly.

MR6 Boundary perturbation destructive control: deliberately splitting/merging words must change boundary-dependent statistics in the expected direction; if the instrument is invariant to a transformation it claims to use, the claimed evidence channel is invalid.

At least 32 planted instances per applicable MR. Any unexplained violation invalidates the implementation before transfer.

Failure state: `C5_METAMORPHIC_FAIL`.

### C6 — independent transfer validation

Only after C0–C5 pass on consumed BUILD/DEVELOPMENT material may an untouched source family be opened.

Transfer source must be independent by work/family and preferably by edition/printer/scribe/representation. No threshold, preprocessing, search budget, alphabet rule or acceptance criterion may change after source exposure.

For the current Yiddish programme:
- secondary Penn confirmation is only a **secondary-normalized transfer certificate**;
- primary-script / image-derived transfer requires its own source-quality audit;
- Basel 1599 is consumed DEVELOPMENT for image-layout work and cannot be reused as fresh confirmation;
- Shmuel-bukh 1544 is reserved as a potential fresh transfer-period witness for the image-level programme, subject to a separately frozen acquisition/audit protocol.

Failure state: `C6_TRANSFER_FAIL`.

### C7 — sealed target admission

Voynich can only be loaded if the exact family/representation has C0–C6 PASS and a target protocol has been committed first.

Target protocol must predeclare:
- exact transcription/version;
- exact unit and ordering;
- target windows;
- uncertainty handling;
- scores and thresholds;
- null construction;
- multiplicity correction;
- interpretation language;
- no-rescue rule.

A target negative is admissible only inside the certified operating envelope. If target properties fall outside that envelope, verdict is `OUT_OF_DOMAIN`, not cipher rejection.

## 4. Dataset governance

Roles are source-family disjoint where possible:
- `BUILD_SOLVER`: language model / objective fitting only.
- `BUILD_EVALUATOR`: if an evaluator is used, its training corpus is distinct from solver build.
- `DEVELOPMENT`: algorithm/hyperparameter decisions.
- `CONFIRMATION`: untouched until executable freeze.
- `TARGET`: sealed Voynich, absent from this repository path's runtime dependencies.

Exact 8-word overlap screens are mandatory across BUILD/DEVELOPMENT vs CONFIRMATION for normalized corpora; exact 5-word overlaps are reported diagnostically.

Repeated pages/chunks from one work are not independent works and are not counted as population replication.

## 5. Candidate / tuning policy

For a family version:
- at most two versioned search/evaluator candidates may be selected on DEVELOPMENT;
- after the second candidate fails, status is `INSTRUMENT_UNRESOLVED`; do not tune indefinitely;
- a fresh confirmation source is consumed immediately upon first outcome exposure;
- a failed confirmation source may become DEVELOPMENT only for a later version.

## 6. Statistical reporting

Every headline rate must include an exact binomial interval/bound or be marked `AWAITING_BOUNDING`.

Report separately:
- sensitivity / recovery probability by cell;
- false-positive probability by negative family;
- abstention probability if applicable;
- key-recovery vs plaintext-recovery;
- search failure vs objective misalignment;
- work-family counts vs trial counts.

Do not use hundreds of planted keys as if they were hundreds of independent historical works. Simulation trials estimate algorithmic operating characteristics conditional on the chosen source distribution; historical transfer remains finite-panel unless a justified population model exists.

## 7. Failure semantics

The harness is fail-closed:
- `PASS` means all required evidence exists and gates are met.
- `FAIL` means a registered gate failed.
- `UNRESOLVED` means the instrument did not establish adequate operating characteristics.
- `UNBOUNDED` means evidence quantity cannot support the requested error-rate bound.
- `OUT_OF_DOMAIN` means the eventual target lies outside the qualified envelope.

None of FAIL / UNRESOLVED / UNBOUNDED / OUT_OF_DOMAIN is evidence that the target lacks the cipher mechanism.

## 8. Historical-result mapping

Old Voynich cipher experiments must be audited into exactly one status before they contribute to any aggregate conclusion:
- `MATHEMATICALLY_VALID`
- `INSTRUMENT_QUALIFIED`
- `EXPLORATORY_UNQUALIFIED`
- `FAILED_CALIBRATION`

The previously quoted aggregate `P(cipher narrow) ≈ 0.12` is not a valid calibrated posterior under this standard and is withdrawn pending the historical audit.

## 9. First executable implementation

`run_m0_control.py` implements C0–C5 on the existing Penn normalized historical-Yiddish benchmark and emits a machine-readable certificate bundle. It intentionally cannot import or load Voynich data. C6 primary-script transfer and C7 target admission are separate later programs.

The first executable goal is not to prove anything about Yiddish/Voynich. It is to demonstrate that the M0 solver itself deserves to be treated as a scientific measuring instrument.
