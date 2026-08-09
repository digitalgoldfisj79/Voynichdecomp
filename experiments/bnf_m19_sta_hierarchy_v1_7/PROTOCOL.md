# BnF M19 STA/aaa Transliteration Hierarchy v1.7 — Frozen Protocol

Date: 2026-08-09
Seed namespace: `M19STAv17`
Parent diagnosis: `experiments/bnf_m19_why_german_v1_1/RESULT.md`

## Question

Does the BnF Latin 7342 M19 unmarked numerical channel produce a language-specific signal when Voynich text is represented in René Zandbergen's common STA / Reference Transliteration framework rather than literal EVA ASCII?

This experiment is explicitly designed to remove the EVA `ch`/`sh` character-splitting confound diagnosed in v1.1 without attempting new image OCR.

## Frozen external sources

All source files are acquired from `voynich.nu` with browser-equivalent HTTP headers. Their bytes are SHA-256 checked before execution.

- RF1b reduced STA1: `https://voynich.nu/data/sta/RF1b.txt`, SHA-256 `81c331b7d8e76761e27d350c3b37ccfbe192848e6c8a227bcb5d40fb29259b17`.
- IT2a STA1: `https://voynich.nu/data/sta/IT2a.txt`, SHA-256 `215f2d05690828c00bd4ae00d6201df31050adcd81601343b142ae91b9dfeee4`.
- ZL3b STA1: `https://voynich.nu/data/sta/ZL3b.txt`, SHA-256 `8438ba1c45f47fe1d06b5262cbcdf60ce69158a0edbd4dd802612896f3217e2a`.
- GC2a STA level 1: `https://voynich.nu/data/sta/GC2a_1.txt`, SHA-256 `0c0d1eea4b5ab87f8a65fb7f4346864cd90758ad993812b4f2122b3899d4ac88`.
- bitrans C source: `https://www.voynich.nu/software/bitrans/bitrans.c`, SHA-256 `3ffc7e6c74078f9b395179aaf5daaae3c8dfbbfc2896d21162c8ff0354108e9a`.
- reduced-STA to aaa rule: `https://www.voynich.nu/software/bitrans/STAR-aaa.bit`, SHA-256 `d5c78527ed417e11140bb491a3f36836771cf1c075273bb7088659c2b115af96`.

RF is a reference constructed from aligned ZL and GC material. Therefore ZL/GC transfer is a representation/transcriber robustness check, **not independent confirmation**. IT/Takahashi is the independent transliteration replication stream.

## Source parsing

- IVTFF locus metadata is retained only for page/locus identity; metadata is never treated as text.
- Full-line comments and inline `<...>` comments are ignored; drawing-interruption markers become sequence breaks.
- Any bracketed alternative reading is removed as a break rather than selecting a preferred alternative.
- `Z1` / aaa `z*` illegible/unknown readings are breaks and are never assigned an M19 value.
- Primary word policy is the IVTFF **long-word** convention: `.` is a certain word boundary and `,` is an uncertain space that is not counted as a boundary.
- A predeclared short-word robustness rerun treats both `.` and `,` as boundaries.
- Words containing an out-of-vocabulary surface unit are excluded whole; retained character and word coverage are reported.

## Frozen representations

Three RF representations are tested independently.

### R1 — STA family

Each recognised STA code is collapsed to its uppercase family (`A1 -> A`, `K2 -> K`, etc.). `Z` is excluded as unknown. Surface vocabulary is all non-Z families observed in RF T17. The resulting K must satisfy 19 <= K <= 38; otherwise the arm is underpowered.

### R2 — full STA member

Each two-byte STA code is an atomic surface form. Using **RF T17 only**, choose the smallest K in [19,38] whose most frequent codes cover >=99.5% of recognised T17 STA occurrences. Freeze that vocabulary before any language score. If no K <=38 reaches 99.5%, the arm is underpowered.

The data-only whole-RF preflight, performed before protocol freeze and before any language score, found that whole-corpus RF needs K=36 for 99.5326% coverage. This is diagnostic only; the binding K is recomputed on T17 under the rule above.

### R3 — aaa connected unit

Compile the pinned bitrans source and convert each STA file with the pinned `STAR-aaa.bit` in direction 1. Parse aaa pairs `[a-z][0-9]`; consecutive pairs joined by `:` form one atomic connected surface unit. Adjacent pairs without `:` are separate surface units. `z*` is excluded as unknown.

Using RF T17 only, choose the smallest K in [19,38] covering >=99.5% of recognised T17 connected-unit occurrences, then freeze it. The data-only whole-RF preflight found K=26 at 99.5525%; binding K is T17-derived.

## Folio split

The RF folio/page inventory is sorted by SHA-256 of `M19STAv17split::<folio>` and split 60/20/20:

- T17: first 60% — all surface-vocabulary selection and key fitting;
- H17: next 20% — first language test;
- C17: final 20% — sealed confirmation.

No H17/C17 language score may influence representation vocabulary, optimization budget, thresholds or source parsing.

## BnF M19 channel

Use the exact five BnF tables and 19 unmarked numerical values frozen in v0.7–v1.1. A surface unit maps to one numerical value. For a K-unit surface vocabulary, legal maps are surjective onto all 19 values, each numerical value has one or two surface homophones, and exactly K-19 values have two forms. Thus 19 <= K <= 38.

The plaintext emission law is unchanged: a plaintext letter selects uniformly among its distinct values across the five BnF tables.

The exact forward likelihood is the 23-letter language HMM -> frozen BnF letter/value mixture used in v0.9.

## Language panel and models

Frozen ranking panel:

1. Latin
2. Italian
3. German
4. French
5. Ancient Greek
6. Hebrew
7. Arabic
8. Spanish/Castilian

Use the same UD language-model sources and normalization as v0.9 for direct comparability. The v0.9 LM training sentence residues `{3,4,8,9}` are retained.

`German` is a model label, not a dialect localization. Bavarian/Austro-Bavarian diagnostics are permitted only after a fully confirmed German signal.

## Fresh synthetic qualification

The generalized K-surface optimizer must qualify separately for each representation K before RF language scoring.

Control plaintext for Latin, Italian, German, French, Arabic and Spanish comes only from the UD **dev+test** files of the same treebank; these files were not used to train the v0.9 language models or the prior v0.8/v0.9 qualification mappings.

For each control language:

- generate 45,000 training and 39,000 held-out plaintext letters under the exact M19 law;
- construct K opaque surface forms by assigning one form to every value and duplicating the K-19 most frequent training values;
- require every surface form to occur in training, using deterministic rejection/resampling if necessary;
- fit the correct-language key twice independently;
- rank all eight languages on held-out ciphertext using the better correct-language fixed key.

Qualification gate for each K:

- 6/6 controls rank the true language first;
- minimum true-language margin >=0.05 nats/letter;
- median exact numerical-map recovery >=0.95;
- minimum recovery >=0.85;
- minimum occurrence-weighted independent-fit agreement >=0.90.

Failure locks the corresponding RF arm. Thresholds may not be relaxed after seeing controls. If failure is clearly optimizer convergence rather than information, any optimizer development must occur on controls only and requires a new fresh qualification namespace before Voynich scoring.

## RF T17 -> H17 test

For every qualified representation:

1. build T17 aggregate symbol transition/start/end/frequency statistics from fully covered words;
2. fit a separate legal surface->M19 key under each of the eight language models;
3. use two independent fits per language; report occurrence-weighted agreement;
4. freeze the better key for each language;
5. rank fixed keys on H17 with exact forward likelihood.

An arm produces an H17 candidate only if:

- H17 retained-character coverage >=0.97;
- candidate ranks first;
- candidate margin over runner-up >=0.05 nats/mapped unit;
- candidate independent-fit agreement >=0.90.

## Hierarchy gate

C17 remains sealed unless **all three R1/R2/R3 arms qualify and produce the same H17 candidate language**, with each satisfying its H17 gate.

This is intentionally strict: a language effect that exists only under one arbitrary transliteration granularity is treated as representation-sensitive, as in v1.1.

## C17 confirmation

If the hierarchy gate passes, freeze the three RF T17 mappings verbatim and score C17 without refitting.

RF C17 confirmation requires for all three representations:

- same candidate ranks first;
- margin >=0.05 nats/mapped unit;
- retained-character coverage >=0.97;
- candidate margin >0 in each of four deterministic C17 folio buckets `sha256(M19STAv17bucket::<folio>)[0] mod 4`.

## Cross-transliteration transfer

Only after RF C17 confirmation, apply each fixed RF map by common STA/aaa surface name to the same C17 folios in IT, ZL and GC1. No key crosswalk or refitting is allowed.

For **IT** (independent replication), all three representations must have:

- candidate rank 1;
- margin >=0.03;
- mapped/retained character coverage >=0.95.

For ZL and GC1 (component-source robustness), candidate must rank 1 with positive margin and coverage >=0.95 in all three representations.

The short-word boundary policy must leave the candidate rank 1 on RF and IT C17 for all three representations; this is a robustness requirement, with no fixed margin threshold.

## Post-confirmation diagnostics only

Only after all above gates pass may the programme:

- inspect Viterbi/decoded strings;
- compute dictionary-hit statistics;
- compare full-STA mappings with family-level mappings;
- run historical German/ReF and Bavarian-vs-non-Bavarian diagnostics if German is the candidate;
- audit disputed STA/aaa units against image/DINO data.

These diagnostics cannot rescue a failed primary verdict.

## Verdict vocabulary

- `STA/AAA INSTRUMENT NOT QUALIFIED`
- `NO STA/AAA M19 SIGNAL`
- `REPRESENTATION-SENSITIVE / NO HIERARCHY SIGNAL`
- `H17 STA/AAA CANDIDATE / C17 FAILED`
- `RF CONFIRMED / IT FAILED`
- `CONFIRMED STA/AAA M19 SIGNAL <language>`

A confirmed statistical signal is not itself a plaintext solution; readable independently constrained recovery remains required for any decryption claim.
