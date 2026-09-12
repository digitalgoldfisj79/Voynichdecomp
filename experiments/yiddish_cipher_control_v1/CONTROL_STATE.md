# Yiddish cipher control — canonical state

Date: 2026-09-12

## Controlling status

**`C0_C5_M0_MECHANISM_PASS__LANGUAGE_L_REQUIRED__C6_NOT_RUN__C7_SEALED`**

This file is the restart point for future sessions. Read it together with Supabase handoff `voynich_cipher_instrument_recovery_standard_20260912_v01` before doing new cipher work. Do not redesign or rerun C0–C5 unless the instrument or representation changes.

## Development history

1. Initial smoke harness run `34708340714` exposed a real MR1 failure: arbitrary global ciphertext-symbol relabelling changed recovery because the search path depended on raw symbol IDs.
2. v0.2 repaired only that defect by canonicalising ciphertext symbols by first occurrence before search and conjugating the returned mapping back. Objective, search budget and thresholds were unchanged.
3. Smoke rerun `34708602061` passed C5 with zero metamorphic violations.
4. The full control was then executed under a blinded architecture: `prepare_full.py` emitted separate public ciphertext/training and private truth/root bundles; 12 solver shards received only the public bundle; truth was joined only after all solver outputs were committed.

## Full blinded qualification result

GitHub Actions run: `34709034207`  
Final artifact: `yiddish-m0-full-final-certificate`  
Artifact digest: `sha256:0f2cb389fcec7f02a62072facee10ee182babe0b5cb2ad5413cbafdccf7165e8`  
Cases/results: `3796 / 3796`  
Plant-root commitment: verified.

Final status:

**`C0_C5_M0_MECHANISM_PASS__LANGUAGE_L_REQUIRED__C6_NOT_RUN__C7_SEALED`**

### C0 — provenance / executable freeze
PASS.

### C1 — representation verification
PASS for the declared **secondary Penn historical-Yiddish Romanisation reduced to literal a–z atoms**. This is not a primary Hebrew-script certificate.

### C2 — known-answer tests
PASS.

### C3 — blinded positive recovery / power surface
PASS in the preregistered primary operating envelope.

Primary envelope = windows `>=512` words and erasure `<=1%`.
- 22 work × length × damage cells were eligible.
- **22/22 cells passed.**
- Every primary cell was **32/32 successful**.
- Cell-wise one-sided 95% lower bound for 32/32 = `0.9106318010`.

Outside the primary envelope the instrument is not universal and must not be extrapolated:
- 14 shorter/harder cells failed.
- Cracow 1588 at 128 and 256 words failed through **objective misalignment** (oracle objective did not support the true key strongly enough), not search failure.
- Several 128-word Kine/Sam-Hayyim cells also failed, mostly objective misalignment.
- Bovo 128/3% had 27/32 through search misses.
- At 512 words, all six available work families passed at all registered 0/1/3/5% erasure levels, although the formal qualified envelope remains the prospectively declared <=1% damage range.

Interpretation: a negative target result below 512 words is `OUT_OF_DOMAIN`, not cipher rejection.

### C4a — matched non-global substitution negatives
PASS.

Each family produced 0/94 false-positive calls:
- per-word key: 0/94
- key drift every 4 words: 0/94
- key drift every 16 words: 0/94
- key drift every 64 words: 0/94

Bonferroni one-sided upper bound per family = `0.04965553295561137` under the frozen alpha `0.05/6` rule.

### C4b — language/nuisance diagnostics
Diagnostic only; this is **not** a language certificate.

- ReF historical German F016, F018, F034, F037, F148: each 0/32 Yiddish-solver recovery successes.
- unigram-matched null: 0/94.
- within-word-shuffled Yiddish: 0/94.

These diagnostics are encouraging but cannot overturn the earlier L result: the registered Yiddish-vs-German language evaluator remained unresolved because nuisance arms performed equally well. M0 mechanism recovery and Yiddish language identification remain separate claims.

### C5 — metamorphic relations
PASS.

- MR1 global ciphertext-symbol relabelling: 32/32 paired passes.
- MR2 consistent plaintext+language-model relabelling: 32/32 paired passes.
- MR3 duplicate normalized counts: exact pass.
- MR4 chunk recombination: exact pass.
- MR5 independent decoder: 32/32.
- MR6 boundary-destructive controls: 32/32 changed the expected boundary-dependent statistics.
- Pair failures: 0.

### C6 — independent transfer validation
**NOT RUN.**

This is now the next scientific step. It must use an untouched historical/source representation and no retuning. The certificate above does not establish primary Hebrew-script transfer.

### C7 — Voynich target admission
**SEALED.**

No Voynich inference is permitted yet.

## Claim boundary

What is now qualified:

> The current v0.2 instrument can recover a globally fixed monoalphabetic substitution on the declared secondary normalized historical-Yiddish representation in the tested primary envelope (>=512 words, <=1% erasure), with the measured sensitivity and bounded false-positive behavior above.

What is not qualified:
- Yiddish language identity (`L` unresolved).
- primary Hebrew-script / diplomatic representation transfer (`C6` not run).
- arbitrary short-text behavior below 512 words.
- any other cipher family.
- any Voynich conclusion.

## Next action

Proceed to **C6 independent historical transfer** without changing the qualified M0 instrument. In parallel, any renewed Yiddish-language (`L`) work must be a new registered language-identification instrument rather than reusing the previously failed nuisance-dominated evaluator. Voynich stays sealed until both relevant language and representation/transfer certificates are qualified.