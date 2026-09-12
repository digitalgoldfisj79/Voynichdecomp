# Yiddish cipher control — canonical state

Date: 2026-09-12

## Controlling status

**`C0_C5_M0_PASS__C6_LATE_NORMALIZED_PASS__EARLY_PRIMARY_TRANSFER_OPEN__LANGUAGE_L_OPEN__C7_SEALED`**

This is the restart point for future sessions. Read it with Supabase handoff `voynich_cipher_instrument_recovery_standard_20260912_v01`. Do not redesign or rerun C0–C5 unless the instrument or representation changes.

## Qualified M0 mechanism instrument

Development history:
- smoke run `34708340714` exposed an MR1 failure: arbitrary ciphertext-symbol names affected the search path;
- v0.2 repaired only that defect by canonicalising cipher symbols by first occurrence before the frozen search and conjugating the mapping back;
- smoke rerun `34708602061` passed metamorphic checks;
- full blinded run `34709034207` used separate public ciphertext/training and private truth artifacts, 12 solver shards with no truth access, and truth-side adjudication only after mapping commitments.

Full final artifact: `yiddish-m0-full-final-certificate`  
Digest: `sha256:0f2cb389fcec7f02a62072facee10ee182babe0b5cb2ad5413cbafdccf7165e8`

Full status: **`C0_C5_M0_MECHANISM_PASS`**.

Evidence:
- C0 provenance/freeze PASS.
- C1 PASS for the **secondary Penn historical-Yiddish Romanisation reduced to literal a-z atoms** only.
- C2 KAT PASS.
- C3 primary envelope `>=512 words`, `<=1% erasure`: 22/22 work×length×damage cells passed, every cell 32/32. One-sided 95% lower bound for each 32/32 cell = `0.9106318010137353`.
- Below 512 words the instrument is not universally reliable; short failures were mostly objective misalignment. Short target results are `OUT_OF_DOMAIN`, not rejection.
- C4a mechanism negatives: per-word key, key-drift-4, key-drift-16, key-drift-64 each 0/94 false calls. Bonferroni one-sided family upper bound = `0.04965553295561137`.
- C4b diagnostic only: ReF German F016/F018/F034/F037/F148 each 0/32 Yiddish-recovery successes; unigram-null and within-word-shuffle each 0/94. This does **not** constitute a Yiddish language certificate.
- C5 metamorphic PASS: MR1 32/32, MR2 32/32, MR3/MR4 exact, MR5 32/32, MR6 32/32; zero paired failures.

## C6a — fresh late historical source-family transfer

Frozen protocol: `c6/C6_LATE_NORMALIZED_PROTOCOL.md`.

Fresh witness: `1783e-ukraine-1.psd`.

Freshness/preflight before outcome:
- no occurrence in canonical Supabase Yiddish/cipher handoff;
- no occurrence in Voynichdecomp code search;
- source not used in prior registered control source lists;
- extracted words = `1702`, sufficient for 512 + 32 + 512;
- exact BUILD/source shared 8-word types = `0`;
- exact BUILD/source shared 5-word types = `0`.

GitHub Actions run: `34709502376`  
Final artifact: `yiddish-c6-late-final-certificate`  
Digest: `sha256:6f3a9f995e792e9dd76f8cb6758afd4fa92f89e64c8f790f4d7f39ea3413d2f0`

Verdict: **`C6_LATE_NORMALIZED_PASS`**.

Registered cells:
- 512 words, 0% erasure: **32/32**, mean atom recovery 1.000, mean word recovery 1.000, no search misses, no objective-misalignment failures;
- 512 words, 1% erasure: **32/32**, mean atom recovery 1.000, mean word recovery 1.000, no search misses, no objective-misalignment failures.

Root commitment verified; all results complete.

Claim boundary: this proves independent source-family transfer to one previously untouched **1783** Yiddish work in the **same Penn-normalized representation**. It does not establish 15th/16th-century transfer and does not establish primary Hebrew-script/diplomatic transfer.

## What remains open

### Early / primary-script transfer
**OPEN.** There is no untouched pre-1601 Penn work long enough for the qualified 512+32+512 design. Goetz 1518 (~626 words) and Anshel 1534 (~857) are individually too short; concatenating unrelated works is prohibited. A primary Hebrew-script route therefore needs its own representation qualification and cannot silently inherit C1.

### Yiddish language certificate L
**OPEN / UNRESOLVED.** Earlier language-evaluator candidates were stopped after two registered candidates because unigram/length/within-word-shuffle nuisance arms performed as well as the primary classifier. M0 recovery and language identity are separate claims. Any renewed L programme must be a newly registered instrument, not Candidate 3 of the failed evaluator.

### Voynich C7
**SEALED.** No target inference is permitted yet.

## Next scientific action

Build a new language-identification control whose observable is **blinded cipher recovery under competing, equally budgeted historical language models**, rather than the previously nuisance-dominated direct corpus classifier. It must distinguish Yiddish from historical German and, if an adequate comparable Hebrew source can be constructed, Hebrew, with BUILD/DEVELOPMENT/CONFIRMATION separation and nuisance/metamorphic controls. In parallel, primary-script transfer remains a separate representation problem.