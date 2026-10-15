# Visible-state proxy preflight (pre-analysis)

## Retractions / preserved failures
- v143's known-plaintext carry is not a pre-identified function of preceding visible glyphs. Neither the v143 synthetic carry success nor this experiment identifies the historical mechanism.
- Earlier v142/v143 B-only surface NO-GO, V10 qualification failure, and v138 bridge null persist.

## Observable & qualification
- Observable X: last three characters of the previous *ciphertext* token on the same line / in the same source stream.
- Observable Y: middle character of current ciphertext word, conditional on its shell = word length + first two characters + final character (middle character excluded).
- Baseline: frozen shell multinomial probability, additive smoothing to corpus-wide Y prior (alpha 3).
- Carry proxy: shell-by-X multinomial, smoothed to the baseline shell probabilities (beta 15).
- Validation: five consecutive-block folds of 1,000 tokens in a round-robin split; admit shells with >= 10 training observations and at least two training outcome glyphs; word length >=5.
- Matched null: permute X among test events within shell and corpus block (when block size too sparse, entire section-shell). Keep Y, shell, lengths, vocabulary and corpus unchanged. Null controls whether the observed association transfers to withheld material, but not whether a plaintext encoder caused it.
- Positive: synthetic v143 carry.
- Hard negative: matched statefree v143 arm, plus v142b baseline.
- Precondition: only if carry distinguishes itself from both controls by >= 2 matched-null SD can this specific observable advance to real A/B tests. Otherwise STOP.
- Representation dependence: raw ciphertext glyphs. Any alternative atomization requires a fresh qualification run, not retrofitting.
- No historical claim from a synthetic pass. A/B subsequent tests still need section/hand/folio controls and autonomous-generator matching.
