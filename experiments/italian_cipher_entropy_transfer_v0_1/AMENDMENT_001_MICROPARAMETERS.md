# Amendment 001 — uncalibrated microparameters and hostile preflight

Date: 2026-08-15
Status: **pre-run, external-only amendment**. No Voynich data have been loaded or scored.

The historical sources establish mechanism classes but do not calibrate several entropy-relevant implementation details. Hiding those choices behind a single arbitrary setting would make the transfer map circularly fragile later. Therefore the v0.1 external run adds the following *diagnostic uncertainty variants* without changing the primary mechanism battery:

- homophone emission: primary uniform choice; diagnostic 70% dominant-choice skew (`HOM34_SKEW70`);
- null alphabet: primary 4 null symbols; diagnostics 1 and 8 null symbols at the fixed 2.5% insertion rate;
- two-glyph rendering: primary hash-derived ordered pairs from a shared 16-component pool; diagnostic unique-component pairs as an upper-bound dependence case;
- the same microparameter diagnostics are applied to `COMBINED_MID`.

These diagnostic settings are sensitivity probes, not historical claims and not target-fit parameters. A later target comparison must use the frozen primary settings and report whether any inference is sensitive to these diagnostics.

Implementation clarifications frozen before the formal run:

1. Evaluation uses exact non-overlapping 3,000 alphabetic-character windows. Partial words at window edges are ineligible for nomenclator replacement.
2. Nomenclator word lists and bigram inventories are learned only from a deterministic 50% training split within each external source family; entropy is evaluated on the disjoint 50% split.
3. Bigrams are coded only within words, never across word boundaries.
4. In combined mechanisms, whole-word nomenclator replacement is applied first, then bigram units, then letter-level substitution, then null insertion.
5. The primary null alphabet contains 4 distinct null symbols.
6. `BIGRAM20+DIGLYPH50` counts as producing a measurable atomic-vs-glyph H1 difference only when the family median absolute difference exceeds **0.05 bits/symbol**. Gate 4 requires this in at least four source families.
7. Diglyph-rendering robustness is reported explicitly; if primary and unique-component renderings disagree in the sign of H0 or H1 in more than two source families, any future diglyph-based target inference must be labelled rendering-sensitive.

The amendment introduces no target values, target thresholds, or target-selected mechanisms.
