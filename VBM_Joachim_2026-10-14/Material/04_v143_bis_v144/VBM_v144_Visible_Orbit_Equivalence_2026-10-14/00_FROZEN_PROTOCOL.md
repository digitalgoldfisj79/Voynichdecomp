# v144 synthetic orbit-and-bridge preflight, frozen before results
2026-10-14. This study is NOT a historical Voynich test.

Motivation: v143's carried variable equals the final vowel in the preceding decoded source word. In v142b and v143, that vowel is given by the final *visible bridge pair*, which is invariant under cell permutations. The earlier arbitrary final-three-character proxy is retracted as a discriminating instrument.

Input: frozen v143 manifest and three held-out Hartlieb ciphertext arms: unchanged v142b, v143 carry, and index-hash statefree. The analyst never reads plaintext in feature extraction or prediction. Manifest provides ciphertext lexing and nine bridge-pair meanings; it does NOT provide event ledger or actual decoder state sequence. Same original source, ciphertext word count and per-word length in all arms.

Deterministic features for each ciphertext word: (a) parse emitted bridge pairs and prefix-free cells (as specified in existing, not learned, frozen manifest), (b) surface skeleton = ordered bridge pair symbols + code-cell initial key and length, (c) outcome = full first code cell, (d) previous feature X = final source-vowel class as read from the previous ciphertext word's last bridge pair; this X must be exactly identical across all three arms at each index. All targets/skeletons/covariates are ciphertext-derived, no raw source words.

Held-out test: five fixed folds floor(position/1000) % 5. Smoothed baseline P(outcome | current skeleton) versus P(outcome | skeleton, predecessor final bridge class), using additive 3 / 12 priors. Admit same target indices across arms only if each arm's training shell has >=12 examples and >=2 outcomes. Held-out log2 likelihood gain per admitted event. Matched null: 80 seeded permutations of X within (skeleton, corpus block=floor(position/2000)), with same index permutation across arms, keeping same eligible positions; if singleton groups result, report effective permutability. Use sample SD of 80 null replicates, paired differences for contrasts.

Primary claim requires carry-minus-baseline AND carry-minus-statefree effects >= 2 paired-null SD, and positive in >=4 of 5 folds. Controls: permuted cross-word labels should null association; statefree and unchanged v142 must be contrasted, not ignored. Source corpus chronological dependence remains a confound; comparison is about this synthetic grammar only.

Admission to Voynich requires (independently) an unambiguous, externally justified lexer from manuscript FORM, demonstrated parser coverage in *both* Currier A and B without v142 fitted inventory, and a matched autonomous selector with comparable model capacity. No Voynich inference will be made if synthetic discriminability or lexer coverage fails.

Stop on either synthetic gate; preserve negative outcomes. Audit order: circularity; leakage; confounds; matched nulls; fairness; degeneracy; representation; fragility; audit completeness.
