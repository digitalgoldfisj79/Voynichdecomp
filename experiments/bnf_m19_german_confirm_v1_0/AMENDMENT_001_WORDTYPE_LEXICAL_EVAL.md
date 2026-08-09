# Amendment 001 — exact word-type-counted lexical evaluation

Date: 2026-08-09

The first v1.0 launch reached only the frozen fresh-panel census (`C10 = 122` with zero overlap) and had produced **no language, lexical, bucket, or decoded-word result** when cancelled.

Reason for cancellation: the inherited lexical routine Viterbi-decodes every token separately for each of 256 mapping permutations. Because Voynich contains many repeated tokens, this repeats identical deterministic calculations and risks wasting CPU / hitting the job timeout.

Prospective implementation-only repair:

- Replace token-by-token lexical evaluation with an exactly equivalent word-type-counted calculation.
- For a given transcription surface and mapping, count word-token multiplicities with `Counter`.
- Viterbi-decode each distinct word type exactly once under that mapping and German LM.
- Multiply its dictionary-hit indicator by its token count.
- The denominator is likewise the sum of token counts for decodable word types.
- Use the same 256 mapping permutations and the same frozen seed namespace as the original runner.

This changes no data, mapping, decoder, dictionary, thresholds, null distribution, or random permutations. It is algebraically identical to the token-by-token statistic because Viterbi output is deterministic for a fixed `(word type, mapping, LM)`.

No confirmatory score was observed before this amendment.