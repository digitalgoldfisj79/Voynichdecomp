# v0.6 Family T — final permitted development amendment

Date frozen: 2026-07-16

Status: **FROZEN BEFORE AMENDED DEVELOPMENT RESULT**

## Trigger

Family T component oracles passed decisively in job `6a589123b1669a49bf077088`:

- T1 mean recovery 99.9674%; mode accuracy 100%; width accuracy 100%; exact permutation accuracy 93.75%;
- T2 mean recovery 99.5117%; every trial at least 98.1771%.

The initial blind T3 solver failed in job `6a5893c485d9643ce16d5f68`:

- mean recovery 65.8529%;
- median 99.3490%;
- 9/16 trials at least 80%;
- mode accuracy 100%;
- width accuracy 87.5%;
- permutation accuracy 50%.

The result is strongly bimodal. Code audit identified a specific search defect: the refinement stage passed forward the provisional substitution key but `coordinate_candidate` reset the provisional permutation to identity. Thus the best joint screen state was not actually refined as a joint state.

## Final amendment

The final Family T development solver will:

1. preserve both substitution key and column permutation between all coordinate stages;
2. use deterministic independent starts for every permitted `(mode, width)` structure;
3. evaluate every development structure rather than hard-pruning by the first weak screen;
4. alternate monoalphabetic and seeded-permutation refinement monotonically from the retained state;
5. use the validated full mono solver only after the structural shortlist is formed;
6. perform a final seeded permutation refinement and, if it changes, one final mono refinement;
7. select solely by the frozen train-only language-model plus MDL score.

The corpus, ciphertexts, modes, widths, language model, candidate family and development gates are unchanged. No test data or Voynich data may be opened.

## Frozen development gates

Across the same 16 English development trials:

- mean plaintext recovery at least 80%;
- median at least 90%;
- at least 14/16 trials at or above 80%;
- mode accuracy at least 14/16;
- width accuracy at least 13/16;
- no recovery below 40%.

Failure closes Family T without locked testing. Passing authorises one untouched locked test with no further modification.