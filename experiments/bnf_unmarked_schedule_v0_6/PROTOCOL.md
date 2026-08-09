# BnF 7342 unmarked numerical-value schedule programme v0.6 — preregistration

Date: 2026-08-09
Status at freeze: no v0.6 Voynich schedule-feasibility result observed.

## Question

Could the five BnF lat. 7342 letter→number tables be used with **table identity supplied by position**, so that the visible cipher alphabet need encode only the numerical value and not the table?

This is structurally different from v0.3–v0.5. A surface Voynich glyph has one global numerical value; the plaintext letters compatible with that value depend on whichever BnF table is active at that occurrence.

## Frozen tables and values

The exact F/M/G/L/H tables remain as frozen previously. Across all five tables the union of unmarked values is:

`0,1,2,3,4,5,6,7,8,9,10,12,16,20,22,23,24,28,30`

Exactly 19 values.

For table `T`, let `V(T)` be the values actually present in that BnF table. A surface glyph globally assigned value `v` may occur under active table `T` only if `v ∈ V(T)`.

## Surface alphabet

Primary transcription: ZLZI from `main/voynich_transcriptions_slim.json`.

- Alphabetic transcription characters are lowercased.
- The one capital `I` is therefore merged with `i`.
- The whole manuscript is used for the structural test.
- No glyphs are deleted because they are rare.

The observed surface alphabet is expected to contain 25 lowercased labels; this is verified by the runner rather than assumed.

## Fixed schedule family

Five table schedules are evaluated, fixed prospectively:

1. `CHAR_CONTINUOUS`: F,M,G,L,H cycle over all alphabetic glyph positions continuously in manuscript folio/line/token order; spaces do not advance phase.
2. `CHAR_WORD_RESET`: F,M,G,L,H cycle over character positions inside every word, resetting to F at each word initial.
3. `WORD_CONTINUOUS`: every word uses one table for all its letters; tables cycle F,M,G,L,H over words continuously across the manuscript.
4. `WORD_LINE_RESET`: every word uses one table; the word-table cycle resets to F at the start of every transcribed line.
5. `LINE_CONTINUOUS`: every transcribed line uses one table for all its letters; lines cycle F,M,G,L,H continuously over manuscript order.

No offset is fitted after seeing results. For each schedule all five cyclic rotations are reported as a predeclared nuisance panel. A schedule family survives only if **at least one rotation** passes the structural gate; later language tests must pay a five-way multiplicity penalty or freeze the surviving rotation independently.

## Structural legality

For a schedule+rotation, collect for each Voynich surface glyph `g` the set of active BnF tables under which it occurs, `P(g)`.

Its legal unmarked values are:

`L(g) = intersection_{T in P(g)} V(T)`.

If `L(g)` is empty, that schedule+rotation is impossible.

The literal unmarked-number model further requires that the manuscript-scale ciphertext can express the numerical repertoire expected from ordinary plaintext. This is checked using positive controls before binding Voynich interpretation.

## Positive-control repertoire calibration

Before the Voynich gate is interpreted, construct long plaintext controls for Latin, Italian, German and Hebrew using the same disjoint held-out corpora as v0.3/v0.5. For every schedule family and all five rotations:

- apply the exact BnF table schedule directly to 150,000 normalized plaintext letters (or all available if fewer);
- record which of the 19 numerical values occur;
- record frequency of every value.

A value is `EXPECTED` for a schedule family if it appears in **all 20 controls** (4 languages × 5 rotations), unless a target corpus lacks 150,000 letters, in which case use all available and still require appearance in all controls.

The expected-value set is determined before inspecting the Voynich assignment result for that schedule family.

## Surjective assignment gate

For each Voynich schedule+rotation, solve an exact bipartite coverage problem:

- every one of the 25 observed glyph labels must be assigned exactly one legal numerical value from `L(g)`;
- every EXPECTED numerical value must be assigned to at least one glyph;
- no numerical value may have more than **3** surface-glyph homophones.

The max-3 bound is frozen as a generous allowance: with 25 surface labels and 19 numerical values, six duplications are already required if all 19 are expressed; max 3 permits additional concentration without allowing the entire alphabet to collapse onto the six universally available low values.

Feasibility is solved exactly by backtracking / integer search, not heuristically.

A schedule+rotation PASS requires such an assignment. If none exists, it is `STRUCTURALLY REJECTED` before language modeling.

## Additional frequency plausibility diagnostic

For each feasible assignment, find the assignment minimizing squared distance between normalized Voynich glyph frequencies aggregated by numerical value and the median positive-control numerical-value frequency vector for that schedule family. This diagnostic is reported but does not establish or reject the model in v0.6.

## Stop rule

- If all 25 schedule+rotation combinations are structurally rejected, v0.6 ends. No language decoder is built for this mechanism.
- If one or more survive, archive the exact feasible rotation(s) and assignments. Build a **new preregistered v0.7 language-decoding programme** only for the surviving schedule family/rotations. Do not choose among feasible rotations by language fit in v0.6.

## Interpretation

A structural PASS is not evidence of encryption or language. It only establishes that an unmarked-number scheduled use of the BnF tables is combinatorially compatible with the Voynich glyph/position pattern.

A structural rejection is stronger: under the frozen schedule and global glyph→number assumption, some required numerical values cannot be represented while respecting the tables at every observed glyph occurrence.
