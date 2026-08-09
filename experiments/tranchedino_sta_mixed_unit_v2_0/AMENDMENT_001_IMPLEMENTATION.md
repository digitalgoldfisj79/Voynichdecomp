# Amendment 001 — solver implementation details

Date: 2026-08-09
Status: prospective; frozen before Stage A0 control execution.

The protocol's smoothed 19-letter Paduan quadgram is fixed at additive `alpha=0.5`, matching the earlier Paduan programme's smoothing convention.

Scoring is line-reset: a quadgram contributes only when all four symbols fall within the same retained line/segment. Scores are mean natural-log probability per scored plaintext event.

The f069v multiplicity vector is fixed in alphabet order `abcdefghilmnopqrstu` (19 columns):

`[1,2,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]`.

Frequency initialisation is deterministic: ciphertext surface symbols are ranked by observed frequency and matched to the 36 expected homophone slots ranked by Paduan training unigram probability divided by that plaintext letter's fixed multiplicity. Ties are resolved by numeric surface ID / alphabet order.

Each independent optimizer ensemble uses the protocol's deterministic namespace-derived seeds. Restart 0 uses the frequency initialisation; later restarts alternate perturbations of the incumbent and full random permutations of the fixed plaintext-slot multiset. Every restart receives exhaustive legal pair-block polish over all plaintext-letter pairs. No multiplicity may change.

No threshold or scientific gate is changed by this amendment.
