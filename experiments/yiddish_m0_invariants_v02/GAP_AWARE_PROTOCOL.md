# Yiddish M0 necessary-condition instrument v0.2 — gap-aware source-transfer protocol

**Frozen 2026-09-12 before any historical-source compatibility or Voynich target score.**

Parent: `experiments/yiddish_m0_invariants_v01/PROTOCOL.md`.
Parent implementation result: exact planted-bijection/nondegeneracy PASS.
Reason for v0.2: the score-independent MS262 image audit showed that treating incompletely transcribed historical material as a continuous stream creates false adjacency/recurrence. v0.2 changes missing-data handling only; it does not authorise target scoring.

## 1. Scope and state

This protocol tests a **necessary condition** for M0 only: globally fixed bijection on declared orthographic atoms, with manuscript/print word boundaries and order preserved within known continuous source spans.

A PASS can only retain M0-Yiddish as compatible with the registered structural envelope. It cannot decipher the target or establish Yiddish. A robust FAIL after instrument/source-transfer qualification can reject M0-Yiddish for the tested representation/scope.

**Voynich remains sealed until the source-transfer gate in section 9 is passed and a separate target-admission commitment is written.**

## 2. Source authority

For a historical source, the image/facsimile is authoritative. A scholarly transcription may provide alignment/scaffolding, but:

- supplied/restored text is never silently treated as observed;
- illegible/uncertain material is never guessed from language context;
- transliteration collisions are never resolved from lexical plausibility;
- unreviewed OCR/vision output is DEVELOPMENT only and cannot be confirmation gold.

Every source representation must retain folio/page and line provenance sufficiently to locate each accepted word on an image.

## 3. Atom declaration

For primary Hebrew-script Yiddish, an atom is one observed source grapheme unit under the frozen source-specific grapheme inventory. Unicode encoding is an implementation representation, not the definition of a palaeographic atom.

Editorial expansions, normalized spellings, reconstructed missing letters, and letters inferred solely from a Roman transcription are not observed atoms.

If a published Roman transcription maps injectively to exactly one declared source grapheme at a position, it may be mechanically back-mapped. If two or more source graphemes map to the same Roman symbol at that position, the atom is **AMBIGUOUS** until resolved against the image.

## 4. Hard-gap rule

A word is `CERTAIN` only if:

1. all of its atoms are observed or image-resolved with no unresolved collision;
2. its complete word boundary is visible/attested;
3. it contains no supplied/restored/illegible span affecting its surface;
4. it is in the declared Yiddish source layer rather than explicitly marked non-Yiddish quoted material when such marking exists.

Any word failing one of these conditions is `UNCERTAIN`. An UNCERTAIN word creates a **hard segment break before and after itself**. Two CERTAIN words separated by even one uncertain, omitted, unread, or untranscribed word are never considered adjacent.

No statistic may bridge a hard gap.

## 5. Segmented packets

The basic measurement object is a `SEGMENTED_PACKET`: an ordered list of physically continuous runs of CERTAIN words.

Registered nominal packet sizes remain 512 and 2048 CERTAIN words. A packet can contain multiple runs. It is admissible only if:

- every run contains >= 8 CERTAIN words;
- >= 80% of packet words lie in runs of >= 32 words;
- the packet contains at least 4 runs unless the source is independently established as fully continuous for the whole packet;
- no run is created or selected using invariant outcomes or target data.

For calibration/comparison, a counterpart packet from another corpus must be deterministically partitioned to **the identical ordered vector of run lengths**. This prevents missingness/fragmentation itself from becoming a discriminator.

If a corpus cannot supply the same geometry, that comparison cell is `CORPUS_LIMITED`, not imputed.

## 6. Registered per-run invariants

All registered quantities must be exactly invariant under a global bijection on the declared atoms. They are computed per run; cross-run relations are forbidden.

Exact structures:

1. word-length sequence;
2. within-word equality pattern for every word (first-occurrence canonicalisation inside word);
3. token-equality partition within the run;
4. canonical full symbol stream of the run after first-occurrence renaming, with word separators preserved;
5. repeated-token gap multiset within the run only;
6. within-word symbol-adjacency multigraph canonicalised up to atom relabelling.

Packet summaries are deterministic aggregates of per-run quantities, weighted by registered word/atom denominators. No feature is allowed to encode source ID, folio label, run count, absolute page position, transcription convention, editor, or missingness beyond the matched run geometry.

## 7. Implementation gates before source transfer

### 7.1 Planted M0 identity
For each candidate source representation and each packet geometry: 32 random bijections per packet. Every registered exact structure and every summary component must be exactly unchanged (integer/rational quantities) or agree to <=1e-12 for floating summaries. Required 32/32 in every cell.

### 7.2 Gap isolation
Insert synthetic gaps at deterministic positions into a complete control. Recomputing the packet after splitting runs must equal direct independent per-run computation. Required 32/32.

### 7.3 Cross-gap contamination sentinel
Permuting the order of words **between** runs while leaving each run unchanged must not change any registered packet feature. Required 32/32. If it changes, some statistic is illegally bridging a gap.

### 7.4 Nondegeneracy
At least three destructive within-run transformations (within-word shuffle, within-run word-order shuffle, non-bijective atom merge) must change the registered vector in >=29/32 trials per cell. Boundary perturbation is applied only inside runs and must also meet >=29/32 where mechanically feasible.

## 8. Representation sensitivity

Before source-transfer scoring, run at least two prospectively declared defensible representations if the source admits them (for example, strict observed-only vs observed plus explicitly certain abbreviation expansion). Headline inference requires either:

- the same source-transfer decision under all registered representations, or
- an explicit `REPRESENTATION_DEPENDENT` result.

No representation may be selected because it moves Yiddish closer to or farther from a target.

## 9. Source-transfer qualification

The v0.2 instrument is not qualified for target use until it demonstrates that the registered invariant family is interpretable across source families.

Minimum DEVELOPMENT/qualification design:

- historical Yiddish: at least two independent source families, including at least one core-period (1350–1500) primary-source representation if available;
- contemporary historical German: ReF diplomatic sources, source-family independent and date/genre matched as closely as feasible;
- contemporary historical Hebrew: independent sources/registers matched as closely as feasible;
- structural nulls/generators that preserve selected marginals but destroy linguistic organisation.

Packets are the unit of measurement but **source family is the inferential unit**. Multiple packets/keys from one work are dependent technical replicates and never increase n of historical works.

Before outcomes, freeze a distance/compatibility rule using DEVELOPMENT only. Candidate source-transfer rule must show nuisance resistance: source/editor/genre/fragmentation-only baselines must not match the primary discrimination. If shallow baselines explain the same separation, result is `SOURCE_TRANSFER_NOT_RESOLVED`.

No population claim from one Yiddish manuscript. With fewer than the protocol's planning counts, any successful result is labelled `FINITE_PANEL_ONLY`.

## 10. Target admission

Only after sections 7–9 pass may a separate commitment name:

- exact Voynich transcription and version/hash;
- glyph atomisation/canonicalisation;
- treatment of uncertain glyphs, labels, line/paragraph/folio boundaries;
- fixed target packet geometries matched to historical-source run lengths;
- frozen compatibility threshold and nulls;
- one-shot target decision language.

Until then: **VOYNICH TARGET SEALED.**

## 11. Stop rules

- Do not repair a failed source after seeing compatibility outcomes and reuse it as fresh confirmation.
- Do not add Candidate 3 to the failed character-language evaluator family.
- Do not convert a many-to-one scholarly transliteration back to source script by lexical guessing.
- If source transfer cannot be distinguished from nuisance controls, stop: the invariant instrument is non-resolving for Yiddish identity even though its M0 algebra is valid.
