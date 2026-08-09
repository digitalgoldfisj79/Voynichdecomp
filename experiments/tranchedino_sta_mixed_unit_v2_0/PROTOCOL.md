# Tranchedino × STA Historical Cipher Programme v2.0 — Frozen Protocol

Date: 2026-08-09
Namespace: `TRANCHSTA20`
Parent: BnF STA/aaa v1.9 terminal result `a533fc38f27021eedaeef376ac1ce80074856f96`.

## Purpose

Revisit the historical Tranchedino/Paduan cipher line under Zandbergen STA rather than EVA-era unitisation. The old July programme tested conditional same-line, same-family variant ordering and explicitly did **not** test alternative glyph/ligature unitisation or a direct historical homophonic alphabet over such units. v2.0 addresses that specific gap.

No BnF/M19 constraints are used.

## Frozen sources

### Historical source

Recovered Library archive: `tranchedino_paduan_payload_program_complete.zip`.

- archive SHA-256: `ddae949a2d4ff13714204f3751feaf9e836333ef57a45def77c803cd87fc7b61`
- `paduan_cipher_letters.txt`: 227,702 letters; SHA-256 `9d21818c13a425639a68ae2c6fb400f35d3f81a49a77bb1f9d610162012f39fe`
- `paduan_lines.csv`: SHA-256 `c5eba63cbe8055d3506d099043f5df23fd427df709546df6de70e084fedd3cf6`
- `tranchedino_homophone_cells.csv`: SHA-256 `d0b96b4c5311e7d1f620a0e63742fc949ec3c7b382f0430478f53f782db97053`
- old Paduan split retained verbatim: first 72% of text-bearing pages for language-model training; remaining 28% held out for payload/control material.

### Voynich symbolic source

Primary: René Zandbergen RF1b STA1 IVTFF file, `https://www.voynich.nu/data/sta/RF1b.txt`.

- expected bytes: 463,638
- SHA-256: `81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`

Replication streams, invoked only after primary H20 passes:

- IT2a STA: SHA-256 `215f2d05690828c00bd4ae00d6201df31050adcd81601343b142ae91b9dfeee4`
- ZL3b STA: SHA-256 `8438ba1c45f47fe1d06b5262cbcdf60ce69158a0edbd4dd802612896f3217e2a`
- GC2a level-1 STA: SHA-256 `0c0d1eea4b5ab87f8a65fb7f4346864cd90758ad993812b4f2122b3899d4ac88`

## Historical geometry gate

The 41 verified Tranchedino alphabet grids have strict inventory median 43, IQR 38–50, range 23–70.

Primary historical key: **f069v / SYS-007**, because it is the first high-occupancy exact match to the already-frozen full-STA K=36 vocabulary:

- 36 strict alphabetic signs / 38 available grid cells = 94.74% occupancy;
- exactly 19 plaintext letter columns;
- multiplicities: `a=1,c=1`; each of `b,d,e,f,g,h,i,l,m,n,o,p,q,r,s,t,u=2`;
- total = 36.

This key was selected by historical inventory geometry, before any v2.0 Voynich language score.

Context-only representation arms:

- STA family K=22: historically below every strict Tranchedino alphabet inventory; cannot become the primary arm.
- connected-aaa K=26: within only the sparse lower tail (4/41 pages at or below); diagnostic only.

## STA tokenisation

RF full-STA IVTFF is parsed at the STA character level. A full STA symbol is one family code plus member code (e.g. `A1`, `Aa`, `Ch`, `Lp`). Certain and uncertain word spaces are not cipher symbols. Drawing intrusions and locus/line boundaries are hard scoring breaks. Uncertain bracketed readings are excluded from the primary stream.

The full-STA surface vocabulary is the same already-frozen K=36 rule used in the v1.9 hierarchy: smallest RF full-STA member vocabulary covering at least 99.5% of occurrences. All other STA members are breaks, never post-hoc merged.

## Folio split

Among RF folios with retained primary symbols, sort by SHA-256 of `TRANCHSTA20split::<folio>` and partition 50/20/30:

- T20: first 50% — map fitting only;
- H20: next 20% — first manuscript adjudication;
- C20: final 30% — sealed confirmation.

The split algorithm is frozen before any Paduan-language score. Surface-only coverage/inventory census is allowed before scoring. C20 is not scored unless H20 passes all gates.

## Plaintext alphabet and language model

Use the historical 19-column f069v alphabet:

`a b c d e f g h i l m n o p q r s t u`

Apply the old Paduan normalisation rules unchanged: `j→i`, `v/w→u`, `y→i`, `x/z→s`; non-alphabetic material removed for the core letter model.

Fit a line-reset smoothed quadgram model on the old 72% Paduan LM partition only. No modern Italian corpus and no Voynich-derived language statistics enter Stage A.

## Stage A0 — solver qualification

Before any Voynich score, generate 12 fresh controls from the old held-out Paduan payload partition. Each control:

- uses a deterministic 12,000-letter sample window selected by `SHA256(TRANCHSTA20control::<replicate>)`;
- uses a fresh opaque 36-symbol key drawn uniformly from permutations respecting the exact f069v multiplicity profile;
- preserves held-out Paduan line breaks but not spaces for the primary cipher score;
- is solved blind by two independent optimizer ensembles A/B.

The solver is the recovered fixed-inventory pair-block homophonic search family from `recoverability_frontier_v0_5`, adapted only to the exact f069v counts and Paduan quadgram model. Search is adaptive: 6 restarts per ensemble minimum, then +6 batches to maximum 36 if A/B convergence is not reached. Each restart receives exhaustive pair-block local polish.

Control convergence requires objective difference <= `1e-7` nats/event and occurrence-weighted A/B map agreement >=0.90.

Binding qualification requires all of:

- 12/12 controls converge;
- median plaintext recovery >=0.95;
- minimum plaintext recovery >=0.85;
- median occurrence-weighted true-map recovery >=0.95;
- minimum occurrence-weighted true-map recovery >=0.85;
- minimum A/B map agreement >=0.90;
- best recovered objective is no worse than the known true-map objective by >`1e-5` nats/event on every control.

If this fails: `TRANCHEDINO-STA INSTRUMENT NOT QUALIFIED`; no Voynich language score is admissible.

## Stage A1 — RF T20 fit / H20 adjudication

Fit the fixed 36→19 mapping on T20 only with two independent adaptive ensembles. The same f069v multiplicity profile is mandatory.

T20 fit must converge under the same A/B criteria; otherwise STOP.

Freeze the best T20 map verbatim and evaluate H20 without refitting.

Primary H20 gates:

1. retained-symbol coverage >=0.97;
2. T20 A/B occurrence-weighted map agreement >=0.90;
3. H20 decoded Paduan quadgram score is no worse than the 5th percentile of the 12 qualified synthetic-control fixed-map held-out scores;
4. H20 observed decoded score exceeds the 99th percentile of 200 deterministic within-line order-shuffle nulls evaluated with the same fixed map;
5. all four deterministic H20 folio buckets have observed-minus-median-null score >0.

If any H20 gate fails: `NO TRANCHEDINO-STA ALPHABETIC SIGNAL`; C20 and replication streams remain sealed.

## Stage A2 — cross-transliteration replication

Only after RF H20 passes, apply the **same fixed RF T20 map**, with no refitting, to matching H20 loci in IT, ZL and GC STA streams.

At least 2/3 independent streams must have:

- retained coverage >=0.90 on shared loci;
- observed score above its 95th-percentile within-line shuffle null;
- observed score within 0.05 nats/retained symbol of RF H20.

Failure gives `RF-ONLY / TRANSCRIPTION-SENSITIVE`; C20 remains sealed.

## Stage A3 — C20 confirmation

Only after A1+A2 pass, score RF C20 with the frozen T20 map. No refit.

Confirmation requires:

- coverage >=0.97;
- score no worse than the 5th percentile of qualified synthetic fixed-map scores;
- score >99th-percentile 200-shuffle null;
- observed-minus-median-null >0 in all four C20 folio buckets;
- same-map replication on at least 2/3 IT/ZL/GC streams at p<=0.05 against order-shuffle nulls.

Formal positive label: `CONFIRMED TRANCHEDINO-STA ALPHABETIC SIGNAL`.

This label is a statistical mechanism signal, not a plaintext claim. Decoded text/word readings remain sealed until confirmation.

## Stage B — mixed-unit/nomenclator extension

Stage B is **not automatically unlocked by a Stage-A negative**. It is a separate programme because the existing mixed-unit calibration had limited sensitivity and because adding nulls/syllables/words materially changes model capacity.

Stage B may be designed only after the Stage-A result is archived. It must reuse the independently reproduced mixed-unit homophonic calibration framework, use empirical Tranchedino distributions where measurable, and qualify on genuine/synthetic historical controls before Voynich scoring.

## Verdict vocabulary

- `TRANCHEDINO-STA INSTRUMENT NOT QUALIFIED`
- `NO TRANCHEDINO-STA ALPHABETIC SIGNAL`
- `RF-ONLY / TRANSCRIPTION-SENSITIVE`
- `H20 TRANCHEDINO-STA CANDIDATE / C20 FAILED`
- `CONFIRMED TRANCHEDINO-STA ALPHABETIC SIGNAL`

No thresholds may be relaxed after control or Voynich results are seen.
