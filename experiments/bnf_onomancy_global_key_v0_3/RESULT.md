# BnF 7342 numerical-alphabet global-key programme v0.3 — Result

Date: 2026-08-08
Branch: `experiment/bnf-onomancy-global-key-v0.3-20260808`
Protocol freeze: `dc693e82f7c7da2b3866d7652a53a12ff6cc64a8`
Runner: `ac8fa90f3959fb5346d9837159d194ddb4ab9bd7`
HF job: `6a77ae2bda2af92a634efbbb` — COMPLETED, 495 s running

## Verdict

**GLOBAL_FIXED_KEY REJECTED** for the preregistered T2 model and eight-language panel.

This is a resolving negative for the narrow tested model because the global-key instrument passed its positive-control gate decisively before Voynich optimization was permitted.

It rejects the hypothesis that one fixed mapping from each Voynich transliteration glyph to one plaintext letter, with preserved word spaces and at most four BnF-derived homophones per plaintext letter, globally encodes Latin, Italian, German, French, Greek, Hebrew, Arabic or Castilian/Spanish under the frozen normalizations.

It does **not** reject changing keys, polyalphabetic/context-conditioned switching, hidden table-pair schedules, non-space word boundaries, syllabic/logographic or compositional codes, transposition, nulls, abbreviation/stenographic systems, untested languages, or non-linguistic generation.

## Corpus-scale observation count

The user's point that the manuscript provides far more evidence than one folio is decisive for instrument power.

From `main/voynich_transcriptions_slim.json`:

- ZLZI: 226 pages, 37,465 word tokens, 7,598 word types, **186,727 non-space glyph positions**.
- After lowercasing the single capital `I`, ZLZI has only **25 literal cipher-symbol labels**.
- TTLI: 34,351 tokens / 170,585 glyph positions.
- VDRB: 34,038 tokens / 170,017 glyph positions.
- 131 pages contain all seven fixed transcription families.

A direct global `one Voynich word-type = one BnF code symbol` model is not available: ZLZI has 7,598 word types whereas the exact marked BnF constructions contain only 92 T2 or 199 T3 code symbols. The global test therefore used glyph-level substitution.

## Positive-control qualification

Synthetic controls used one fixed 25-symbol global homophonic key satisfying the conservative T2 capacity of at most four cipher symbols per plaintext letter. Spaces were preserved. The true key was retained only for scoring.

### Recoverability frontier at 11,264 plaintext letters

All **8/8** controls identified the correct language. Held-out plaintext recovery ranged from **0.9299 to 1.0000**.

| language | rep 0: z / accuracy | rep 1: z / accuracy |
|---|---:|---:|
| Latin | 27.945 / 1.0000 | 28.124 / 0.9996 |
| Italian | 27.431 / 1.0000 | 13.892 / 0.9898 |
| German | 14.994 / 0.9969 | 14.080 / 0.9299 |
| Hebrew | 15.587 / 0.9822 | 17.904 / 0.9805 |

Thus a global fixed key becomes strongly recoverable well below the full Voynich corpus size.

### Binding gate at 45,056 plaintext letters

| Gate | Frozen requirement | Result | Status |
|---|---|---:|---|
| P0.1 correct language | 8/8 | **8/8** | PASS |
| P0.2 median held-out character accuracy | >=0.90 | **0.99756** | PASS |
| P0.3 minimum held-out character accuracy | >=0.75 | **0.98258** | PASS |
| P0.4 median mapping-permutation z | >=10 | **15.6200** | PASS |

Individual 45,056-character controls:

| language | rep 0: z / accuracy | rep 1: z / accuracy |
|---|---:|---:|
| Latin | 28.009 / 1.0000 | 31.376 / 1.0000 |
| Italian | 14.382 / 0.9988 | 15.658 / 0.9984 |
| German | 16.342 / 0.9955 | 14.422 / 0.9967 |
| Hebrew | 15.473 / 0.9916 | 15.582 / 0.9826 |

The instrument therefore entered the Voynich stage legitimately.

## Voynich split

Primary transcription: ZLZI.

- training folios: 183
- held-out folios: 43
- representative optimization sample: 60 training folios, 45,122 non-space glyph positions
- held-out evaluation: 38,980 non-space glyph positions
- cipher-symbol labels: 25

The mapping was optimized on training folios only and evaluated unchanged on the 43 held-out folios.

## T2 — conservative BnF model, primary result

T2 permits at most four homophones per plaintext letter.

| language model | held-out mapping-permutation z | frozen signal threshold |
|---|---:|---:|
| Spanish/Castilian | **7.8523** | 10 |
| Latin | **7.2689** | 10 |
| Greek | 6.2910 | 10 |
| Arabic | 6.1787 | 10 |
| German | 6.1240 | 10 |
| Italian | 6.0736 | 10 |
| French | 5.8811 | 10 |
| Hebrew | 5.4232 | 10 |

The top language is Spanish only in the mechanical ranking. It is **not a candidate plaintext**. It fails the absolute threshold, and its margin over second-ranked Latin is only **0.5834**, versus the frozen required margin of 5.

All fitted T2 mappings respected the four-homophone capacity.

Because no T2 candidate satisfied z >=10 plus margin >=5, the preregistered TTLI/VDRB cross-transcription transfer stage was not triggered.

## T3 — aggressive flexibility diagnostic

| language model | held-out mapping-permutation z |
|---|---:|
| Spanish/Castilian | 7.2042 |
| Arabic | 6.3382 |
| Latin | 5.8859 |
| French | 5.6946 |
| Italian | 5.5925 |
| Hebrew | 5.3730 |
| Greek | 5.3658 |
| German | 5.1729 |

T3 does not rescue the hypothesis. By protocol, T3 alone could never establish the primary signal in any case.

## Interpretation

The contrast with the positive controls is the central result. Known T2 global homophonic ciphers are recovered nearly perfectly at 45k letters and already very strongly at 11k letters. The Voynich held-out corpus instead produces a broad, nonspecific cluster of optimized mapping z-scores around 5–8 across mutually incompatible languages, with no language reaching the preregistered z=10 threshold or separating from the rest.

Therefore the correct interpretation is not `Spanish`, `Latin`, or another tentative language. The tested global fixed-key BnF-style homophonic substitution model is rejected for this language panel.

The earlier one-folio v0.1 result remains nonresolving because its instrument was underpowered. v0.3 is materially stronger: its positive controls demonstrate that a true global fixed key of the tested form would be recoverable at the available corpus scale.
