# Frozen observational-equivalence audit (before execution)
Competing mechanisms:
H_source: v143 encoder obtains carry = last vowel of the previously decoded/known normalised German plaintext word, and permutes cell codes accordingly.
H_surface: a deterministic ciphertext-to-ciphertext transducer obtains carry = value of the final bridge pair in the preceding *v142b ciphertext word*, never accessing German, then performs the same allowed cell-code permutations.

Theorems/decisions:
If final-bridge and final-source-vowel coincide for every source word by construction, the two must produce identical emitted streams, byte for byte, for any v142b-valid input. This is a direct constructive counterexample to using the bridge-orbit observable as a unique historical indicator of plaintext-driven state.

Implementation audit: transducer may read ONLY frozen v143 manifest and v142b base ciphertext; forbidden from importing v143 encoder, independent decoder, or source text during generation. Compare exact bytes and SHA-256 with existing v143 carry outputs for BOTH development and held-out Hartlieb. Then independently verify source-vowel/bridge equivalence as an audit only (not input to transduction). Failures must be preserved without modifying initial mechanism.

Precommitted outcome: if bytewise identical, classify bridge-orbit as a detectable *surface-state mechanism*, not evidence of an underlying plaintext carry; no Currier A/B test is justified as proof of plaintext. If nonidentical, audit errors and do not claim equivalence.
