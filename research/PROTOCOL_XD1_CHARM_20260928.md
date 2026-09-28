# XD1-CHARM: externally defined medieval charm control

**Protocol ID:** XD1-CHARM-20260928  
**Frozen:** 2026-09-28, before any charm lag-1/lag-2 recurrence outcome is computed.  
**Parent:** XD1-CLOSEOUT-20260928 / SGT12.  
**Purpose:** adversarial downgrade test of SGT12 specificity.

## Scope

This follow-up cannot strengthen the original five SGT12 promotion gates. It can only:
1. leave SGT12 unchanged because the charm evidence is underpowered or inadmissible; or
2. downgrade SGT12 if an independently selected charm cohort reproduces its joint recurrence geometry.

The old claimed 13-Aug charm result remains RETRACTED/UNSUPPORTED. No historical result is resurrected.

## Selection before outcome

Primary source selection is independent of Voynich recurrence outcomes.

### Primary Cambridge cohort

P1. Cambridge, University Library, MS Add. 9308, late 14th/early 15th c., England.
Independently catalogued as c.265 medical recipes and charms, principally one hand, single column, 16 ruled physical lines/page.
Frozen charm loci known before outcome:
- childbirth / peperit charm, ff.49r-50r;
- Middle English verse charm, ff.68r-68v.
Additional loci may enter the primary manifest only if they are explicitly labelled as charms in Cambridge/Curious Cures structured metadata and are discovered in the source-only preflight before any recurrence output is opened.

P2. Cambridge, University Library, MS Dd.5.53, 15th c., England.
Independently catalogued charm loci:
- childbirth, ff.107r-107v;
- staunching blood, f.114v;
- spasms, f.117v;
- fever, f.122v.
The two rabid-dog charms on f.144r are secondary because they are later/additional hands.

P3. Cambridge, University Library, MS Dd.3.52.
Admissible only if the source-only preflight recovers independently catalogued exact charm loci and physical lineation. Otherwise exclude without replacement.

### German sensitivity

Voynich Ninja thread 6047 ("German medicinal charm recipe metastudy") is NOT primary because its witness selection was explicitly Voynich-aware. It may be used only after primary adjudication as a preregistered geographical/language sensitivity arm.

## Source-only preflight

The preflight may inspect:
- metadata, folio/page identifiers, hands, dates, languages;
- diplomatic/hyperdiplomatic transcription availability;
- XML/TEI/PAGE structure;
- explicit charm labels and boundaries;
- physical line-break markers and reading order.

It MUST NOT calculate token recurrence, adjacent equality, lag-2 equality, or any statistic from which the SGT12 outcome can be inferred.

Primary admissibility requires genuine manuscript physical lines supplied by the source transcription (e.g. TEI <lb>, PAGE TextLine, or an equivalent explicit line object). Browser wrapping, normalized prose, catalogue line counts, or lines reconstructed by OCR are inadmissible.

A parser correction after seeing recurrence outcomes requires a new protocol version. Parser corrections during source-only preflight are permitted and logged.

## Tokenization

Identical to frozen XD1:
- Unicode NFC;
- lower-case;
- retain Unicode letters, combining marks and digits inside tokens;
- strip punctuation/markup;
- require at least one Unicode letter.

Editorial expansions/normalisations are not silently substituted for diplomatic forms. Where the source exposes both, diplomatic is primary.

## Statistic

For every admissible physical line and d in {1,2}:

observed = fraction of eligible token pairs where token[i] == token[i-d].

Null: independently permute the exact token multiset within each physical line.

Primary Monte Carlo: 2,000 deterministic permutations.
Analytic exact-multiset expectation is a required cross-check.

Report:
- physical lines;
- tokens;
- eligible opportunities;
- observed rate;
- null mean;
- effect;
- null SD;
- signed effect/null SD;
- observed/null.

If |effect|/null SD < 2, interpretation begins exactly:
**the metric does not resolve this.**

## Within-manuscript control

For each primary manuscript, build a non-charm medical/recipe control from the same transcription and hand/context where feasible.

Controls are selected without recurrence outcomes:
- exclude all catalogued charm loci;
- prefer the same codicological unit and main hand;
- use all remaining independently catalogued medical/recipe text if clean boundaries are available;
- otherwise use symmetric neighbouring non-charm folios around each charm locus, with the window frozen during source-only preflight.

This is the preferred comparison because it controls manuscript, scribal environment, transcription pipeline and much chronology.

## Primary pooled adjudication

Pool primary charm lines only after preserving manuscript/charm identifiers.

A charm cohort reproduces the SGT12 joint geometry strongly enough to force a specificity downgrade if ALL hold:
1. lag1 observed/null >= 0.90;
2. lag1 signed effect/null SD > -2.0 (immediate repetition is not resolved as suppressed);
3. lag2 observed/null >= 1.10;
4. lag2 signed effect/null SD >= +2.0;
5. the conclusion survives leave-one-charm-unit-out analysis wherever >=4 independent charm units are available.

If fewer than 4 independent charm units or <100 eligible lag2 opportunities survive source QC, the pooled result is explicitly underpowered for the downgrade gate, though descriptive metrics are reported.

## Sampling uncertainty

When >=4 independent charm units survive, bootstrap charm units nested within manuscript (2,000 deterministic replicates).
Do not bootstrap individual lines as though they were independent charm witnesses.

Report charm-vs-same-manuscript-recipe differences for lag1 and lag2 with pooled bootstrap SD. No arbitrary p-value threshold replaces the frozen SGT12 geometry rule.

## Length sensitivity

Fixed token-count strata:
ALL, 2-5, 6-10, 11-20, 21+.
A stratum is formal only with >=50 physical lines pooled across primary charms.

## Stop rule

After primary Cambridge adjudication:
- if the source lineation is inadmissible, stop and report BLOCKED_PHYSICAL_LINEATION;
- if underpowered, report UNDERPOWERED_CHARM_CONTROL;
- if the downgrade gate fires, downgrade SGT12 before any German sensitivity;
- if it does not fire, SGT12 remains bounded; do not upgrade it on this basis.

The German arm and any later charm sources are sensitivity only unless a fresh protocol is frozen before their outcomes.
