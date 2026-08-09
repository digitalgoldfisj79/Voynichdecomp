# BnF 7342 marked single-value ambiguous-code programme v0.5 — result

Date: 2026-08-09
Branch: `experiment/bnf-marked-single-value-v0.5-20260809`
Protocol freeze: `17feffb4d4afc3fa75944649b6af4a7d6d91f3f8`
Runner: `d879f63e1f5065db9422412f14f344ab6e3dc608`
HF job: `6a781725da2af92a634efe28` — cancelled prospectively after decisive control failure.

## Verdict

**INSTRUMENT NOT QUALIFIED. NO VOYNICH INFERENCE.**

v0.5 tested a global injective mapping from each surface cipher glyph to one of the 57 exact marked `(table,value)` codes defined by the five BnF lat. 7342 letter-value tables. A marked code could decode contextually to any plaintext letter sharing that value in that table.

The frozen qualification gate required the correct language to rank first in 8/8 known-plaintext controls, with median true-letter compatibility accuracy >=0.95, minimum >=0.85, and median target-language permutation z >=10.

The first control alone made qualification mathematically impossible:

- target plaintext: Latin, replicate 0;
- generated control used a valid 25-code M57 subset covering all 23 plaintext letters;
- correct Latin language model ranked **8th of 8**;
- Latin held-out mapping-permutation z = **2.0275**;
- best-ranked model was Hebrew, z = 4.0749;
- fitted Latin marked-code true-letter compatibility accuracy = **0.1939**.

Because Q1 required 8/8 correct controls, one failed control is sufficient to fail the whole instrument. The job was therefore cancelled immediately to avoid unnecessary compute. It never entered the Voynich stage and produced no Voynich language score.

## Interpretation

This is a solver/scorer qualification failure, not evidence against the historical mechanism class. The simple averaged bigram-compatibility objective does not recover the underlying marked-code assignment even from 45k training + 39k held-out known plaintext letters.

No attempt is made to repair the scorer after observing Voynich, because Voynich was never scored. A future M57 attempt would require a materially stronger latent-state decoder/EM or exact likelihood model and fresh preregistration.
