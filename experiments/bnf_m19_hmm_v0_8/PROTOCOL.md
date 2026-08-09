# BnF 7342 free-switch M19 + forward-HMM programme v0.8 — confirmatory preregistration

Date: 2026-08-09
Status at freeze: no v0.8 qualification-control or Voynich score observed.

## Development lock

v0.7 showed that the M19 numerical key can be recovered but its pairwise permutation-z statistic is not a valid cross-language classifier. Control-only development on a separate branch then showed exact hidden-letter forward likelihood correctly identified fresh diagnostic Latin, Italian and German M19 ciphertexts; target numerical mappings were recovered at 100% weighted accuracy, with correct-language forward margins 0.091–0.160 nats/plaintext letter.

Those development controls are not reused for v0.8 qualification.

## Cipher model

M19 is unchanged from v0.7:

- exact 19 unmarked numerical values from the five BnF tables;
- each plaintext letter may choose uniformly among its distinct BnF values;
- every numerical value has one surface glyph and exactly six values have a second surface homophone, giving 25 surface glyphs;
- surface→number map is global, surjective onto all 19 values, and multiplicity is 1 or 2 only;
- word boundaries are preserved;
- no changing keys, table markers, nulls, transposition, syllabic units or positional schedules.

The pairwise induced numerical model from v0.7 is retained **only to fit the surface→number mapping**. Language identification is replaced prospectively by exact word-level forward likelihood under the hidden plaintext-letter Markov model and the frozen BnF emission channel.

## Exact forward score

For a mapped numerical word `v1...vn`, hidden state is the 23-letter plaintext alphabet. The frozen language model supplies word-initial probabilities and character transition probabilities; the BnF channel supplies `P(v|letter)=1/|V(letter)|` for legal values and zero otherwise.

Forward recursion marginalizes over all compatible plaintext-letter sequences. A word-final letter factor from the frozen language model is applied at the end. Scores are summed and reported as nats per mapped ciphertext letter. All candidate languages are scored on the same held-out words and mapping coverage.

## Fresh corpus split

For **all eight languages**, sentence indices are partitioned by `i mod 10`:

- LM training: residues `{2,3,4,7,8,9}` (60%);
- v0.8 qualification pool: residues `{1,6}` (20%);
- residues `{0,5}` are excluded from v0.8 because they overlap the earlier development holdout pattern.

Thus no v0.8 qualification plaintext sentence is used for LM fitting or earlier HMM development.

Frozen language panel remains:
Latin, Italian, German, French, Ancient Greek, Hebrew, Arabic, Spanish/Castilian.

## Qualification languages

The exact 25-surface/all-19-value M19 generator requires the plaintext normalization to exercise all singleton anchor values `0(y),22(o),23(n),28(f),30(s)`.

The fresh normalized corpora support the full repertoire for:

**Latin, Italian, German, French, Arabic, Spanish/Castilian.**

These six languages form the qualification panel. Greek and Hebrew remain candidate languages in Voynich ranking, but are not forced into a synthetic all-19-value control they cannot generate under the frozen `Unidecode` romanization. This limitation must be stated in any result concerning Greek/Hebrew.

## Fresh positive controls

One independent 84,000-letter control per qualification language:
- first 45,000 letters for mapping fit;
- next 39,000 for held-out evaluation.

Span selection uses seed namespace `20260809|v08qual|language` and the fresh qualification pool.

Generation is exact M19. If all 25 surface forms do not occur in the training segment, redraw value/surface choices under deterministic attempt seeds while keeping the plaintext span fixed. Record attempt count.

For each control:
1. fit an M19 mapping independently under each of the eight candidate language models using training ciphertext only;
2. freeze each fitted map;
3. compute exact forward held-out score under that same language;
4. rank the eight languages by held-out forward nats/letter;
5. for the target language report occurrence-weighted true numerical mapping accuracy;
6. fit the target language a second time under an independent optimizer seed namespace and report occurrence-weighted mapping agreement.

### Qualification gate

All must pass:
- Q1 correct language ranks first in **6/6** controls;
- Q2 smallest correct-language margin over runner-up >= **0.05 nats/letter**;
- Q3 median target weighted mapping accuracy >=0.95;
- Q4 minimum target weighted mapping accuracy >=0.85;
- Q5 minimum independent-fit mapping agreement >=0.90.

Failure of any item stops the programme before Voynich.

## Voynich split

Primary transcription: ZLZI in `main/voynich_transcriptions_slim.json`.

Whole-folio split by SHA-256 namespace `20260809|M19HMM|folio`:
- 20% held-out folios;
- remaining folios are training candidates.

Training folios are taken in deterministic hash order until both:
- >=45,000 non-space alphabetic glyph positions are accumulated;
- all 25 lowercased ZLZI surface glyph labels present in the training partition have been observed.

Held-out evaluation uses all held-out folios. Train-unseen held-out glyphs are hard breaks. Mapped held-out glyph coverage must be >=99%.

## Voynich fit and ranking

For each of the eight languages:
- fit surface→number map on training sample using the pairwise numerical objective;
- refit once under an independent seed namespace;
- report occurrence-weighted agreement between the two maps;
- freeze the better training-objective map;
- compute exact forward likelihood on held-out ZLZI words.

Primary candidate requires all:
- top forward score exceeds runner-up by >= **0.05 nats/letter**;
- top-language independent-fit mapping agreement >=0.90;
- exact M19 surjection/multiplicity constraints pass;
- held-out mapped-glyph coverage >=99%.

There is deliberately no permutation-z threshold in v0.8: v0.7 proved that statistic is not a valid cross-language discriminator for M19. The held-out forward margin is qualified directly on fresh synthetic controls.

## Candidate decoding gate

Only if a primary candidate exists:

1. Decode each held-out word by Viterbi over hidden plaintext letters using the candidate language model and frozen BnF emission law.
2. Compute candidate-language dictionary hit fraction.
3. Compare against 128 seeded permutations of the fitted surface→number map; lexical z must be >=5.
4. Report a sample of decoded held-out words only after this gate; do not inspect or hand-select words before it.

## Cross-transcriber confirmation

Transfer the literal ZLZI glyph→number map unchanged to TTLI and VDRB on the same held-out folios.

For each:
- shared-glyph coverage >=90%;
- candidate language ranks first by exact forward likelihood;
- candidate forward margin over runner-up >=0.03 nats/letter;
- Viterbi lexical enrichment z >=3.

Only then verdict = `CONFIRMED M19-HMM SIGNAL`.

Otherwise:
- primary ZLZI candidate failing transfer = `TRANSCRIPTION-DEPENDENT / NOT CONFIRMED`;
- no primary candidate = `NO M19-HMM SIGNAL`.

## Scope

Even a confirmed result would establish statistical compatibility with the free-switch unmarked-number M19 mechanism, not historical derivation from BnF lat. 7342. Greek/Hebrew rankings remain less strongly instrument-qualified because their frozen romanizations cannot generate the full 19-value control repertoire.