# Amendment 003 — train-unseen homophones in synthetic controls

Date: 2026-08-09

The K-QUIRE run completed Latin and Italian positive controls, then stopped during the German positive-control calculation with a mapping-length mismatch. No K-QUIRE Voynich language score had been generated.

Cause: in a small synthetic quire group, one generated T2 homophone occurred in the synthetic held-out segment but not in that group's synthetic training segment. The base optimizer correctly creates a key only for train-observed symbols, while the control evaluator had retained the full 25-symbol generated alphabet.

Prospective repair, matching Amendment 001:

1. For every synthetic control group, the fitted cipher alphabet is restricted to homophone IDs observed in that group's synthetic training segment.
2. A synthetic held-out homophone unseen in training becomes a hard break for scoring and is excluded from character-accuracy denominator, exactly as a train-unseen Voynich glyph is handled.
3. The runner reports whole-rung synthetic held-out mapped-symbol coverage for each target-language control.
4. A positive-control rung can PASS only if every target-language control has >=99% mapped-symbol coverage.
5. Existing P1–P5 thresholds remain unchanged.

This amendment cannot rescue a weak control by ignoring substantial held-out material: the new >=99% coverage condition is additional to, not a replacement for, the frozen qualification criteria.

No K-QUIRE Voynich score existed at amendment time.