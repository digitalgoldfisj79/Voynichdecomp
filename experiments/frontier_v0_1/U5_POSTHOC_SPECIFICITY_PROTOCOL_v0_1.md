# U5-D — post-hoc specificity challenge v0.1

Date frozen: 2026-08-14
Status: **POST-HOC INTERPRETATION CHALLENGE, FROZEN BEFORE THESE CONTROLS ARE SCORED**

## Why this exists

The confirmatory U5-C target has already returned `PASS_COMPATIBLE_FROZEN_VERBOSE`: 13/13 canonical Voynich blocks exceeded the frozen U5-B threshold and 0/13 blocks were positive in each target-derived destructive null. The five transliteration sensitivities were also uniformly positive.

That confirmatory verdict is not retroactively changed by this post-hoc panel. The new question is narrower and adversarial:

> Is the U5 recogniser specific to the frozen fresh-codebook verbose architecture, or does it also fire on ordinary or generated reusable morphology/compositionality that has no encoded message?

This protocol is written before scoring any control below.

## Frozen recogniser

No classifier, feature, threshold or preprocessing is changed. Threshold remains `0.9997460219719421`; the U5-B locked qualification remains recall 0.96, precision 1.00, 0/400 false positives on its preregistered nulls.

Primary unit remains a 2,731-token block.

## Challenge A — direct natural-language word surfaces

Use the *raw plaintext word tokens*, not Naibbe outputs, of the two authors that were held out from U5-B fitting:

- Pliny, *Naturalis Historia* XVI;
- Dante, *Divina Commedia*.

Tokenise on alphabetic word runs, then apply Naibbe's already-frozen Latin-script normalization **inside each word** (diacritics removed, W→UU, J→I, K→C), dropping empty tokens. Partition into consecutive non-overlapping 2,731-word blocks. Do not reorder, abbreviate, encipher or respace.

Caesar and Collodi—the U5-B development authors—are scored separately as descriptive source-reuse sensitivities and do not determine the primary natural-language challenge.

## Challenge B — independent non-message ROOT+AFFIX generator

Generate 100 independent 2,731-token blocks over the artificial alphabet `abcdefghijklmnopqrst`. There is no plaintext/message stream.

For each block independently:

- 16 unique prefixes, lengths sampled uniformly from {1,2,3};
- 128 unique roots, lengths uniformly from {2,3,4,5};
- 24 unique suffixes, lengths uniformly from {1,2,3};
- root ranks follow a fixed Zipf weight `1/rank^1.1`;
- every root is assigned one of four prefix classes and one of four suffix classes;
- prefix appears with probability 0.70 and, when present, is selected only from the four prefixes assigned to the root's prefix class;
- suffix appears with probability 0.80 and, when present, is selected only from the six suffixes assigned to the root's suffix class;
- the root is always present;
- no token is copied from Voynich, Naibbe or a natural-language source.

This deliberately creates reusable prefix/root/suffix composition, dependency, Zipfian lexical ecology and combinatorial rectangle closure without carrying a message.

Seeds are `SHA256('u5d-root-affix|' + block_index)`.

## Challenge C — target-derived PGCS component-recombination null

This challenge asks whether the positive U5-C result is explained by the already-known PGCS-like component inventories alone.

For each of the same 13 canonical 2,731-record Voynich blocks:

1. take the frozen `prefix`, `gallows`, `core`, and `suffix` fields;
2. independently permute each of the four component columns within that block using seeds `SHA256('u5d-pgcs|' + block_index + '|' + component)`;
3. reconstruct a token by concatenating the permuted components in frozen field order, treating `None`, empty string, `∅`, `EMPTY` and `-` as empty;
4. if reconstruction is empty, use a single artificial atom `x` rather than borrowing the original token;
5. score the resulting 2,731-token block with the untouched U5 recogniser.

This preserves the block's component marginals and reusable component vocabulary but destroys original component conjunctions, token identity, local syntax and any putative encoded payload.

It is explicitly target-derived and post-hoc; it is an explanatory ablation, not an independent null for the original U5-C p-value/decision.

## Frozen specificity adjudication

Let `F_nat` be the positive fraction across all complete held-out Pliny+Dante direct-word blocks; `F_morph` across the 100 ROOT+AFFIX blocks; `V_pgcs` the number of positive PGCS-recombined blocks out of 13.

- `SPECIFICITY_SURVIVES` iff `F_nat <= 0.05`, `F_morph <= 0.05`, and `V_pgcs <= 2/13`.
- `GENERIC_COMPOSITION_CONFOUND` iff **any** of: `F_nat >= 0.20`, `F_morph >= 0.20`, or `V_pgcs >= 11/13`.
- otherwise `SPECIFICITY_AMBIGUOUS`.

These bands are deliberately separated. The 20% confound line requires a large failure of specificity rather than one or two accidental positives.

## Interpretation

If `GENERIC_COMPOSITION_CONFOUND` fires, the U5-C confirmatory result remains factually true but its scientific interpretation is demoted to:

> Voynich strongly exhibits reusable compositional token structure captured by the U5 feature family; the experiment does not distinguish a verbose cipher from non-message morphology/slot generation.

If `SPECIFICITY_SURVIVES`, the frozen fresh-codebook architecture remains a genuinely discriminating live cipher-family compatibility result and merits a new independent negative family in a future preregistered version.

No decipherment, source language, key, provenance, date or historical attribution is tested here.