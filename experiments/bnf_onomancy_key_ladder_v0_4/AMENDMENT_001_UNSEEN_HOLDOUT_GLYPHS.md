# Amendment 001 — train-unseen held-out glyph labels

Date: 2026-08-09

The first K-CURRIER run completed its positive-control gate (PASS) and then stopped with a shape/broadcast error before emitting any Voynich language score. Cause: `build_actual_items` defined a group alphabet from train+hold labels, while the v0.3 optimizer sizes its mapping from labels actually present in the training sequence. In one Currier group, the highest-index union label occurred only in held-out folios.

This is an implementation defect, not a result.

Prospective repair before any v0.4 Voynich score is observed:

1. A group key is defined only on literal glyph labels observed in that group's sampled training folios.
2. A held-out glyph label unseen in training is treated as a hard break for 4-gram scoring, not silently deleted and not assigned using held-out information.
3. The runner reports held-out mapped-position coverage for each rung.
4. A rung's primary Voynich result is admissible only if >=99% of held-out glyph positions are covered by train-observed labels. Otherwise verdict is `UNDERPOWERED: UNSEEN HOLDOUT GLYPHS`.
5. Positive controls are unchanged; their generated cipher labels are expected to be observed in training at these sample sizes.

No language score from Voynich existed at the time of this amendment.