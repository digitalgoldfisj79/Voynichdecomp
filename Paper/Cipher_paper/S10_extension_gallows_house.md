## S10.8 Gallows × house analysis

This section reports three related tests of the house structure in EC tokens. The first asks whether gallows identity recovers suffix-family assignments. The second asks whether gallows class constrains house choice within each suffix family. The third asks whether scribe identity predicts house preference.

### S10.8.1 Gallows → suffix-family mapping (negative)

If each gallows value predominantly encoded one suffix family, gallows identity would directly recover function-word assignments. Table S10.8a tests this.

**Table S10.8a.** Suffix-family distribution per gallows value (EC tokens only, n = 19,758).

| Gallows | n | Y | N | L | R | BARE | M |
|---|---|---|---|---|---|---|---|
| ∅ | 10,791 | 29.4% | 19.8% | 21.3% | 17.8% | 9.0% | 2.2% |
| k | 4,707 | 49.6% | 24.0% | 13.4% | 10.5% | 0.9% | 1.5% |
| t | 2,797 | 48.4% | 18.1% | 15.6% | 13.6% | 1.6% | 2.7% |
| p | 210 | 30.5% | 16.7% | 21.4% | 21.9% | 7.1% | 2.4% |
| f | 53 | 20.8% | 15.1% | 17.0% | 22.6% | 24.5% | 0.0% |
| ckh | 515 | 85.4% | 1.6% | 8.5% | 3.1% | 0.6% | 0.8% |
| cth | 556 | 70.0% | 3.6% | 13.3% | 11.5% | 0.9% | 0.7% |
| cph | 98 | 65.3% | 7.1% | 16.3% | 10.2% | 0.0% | 1.0% |
| cfh | 31 | 58.1% | 6.5% | 19.4% | 12.9% | 0.0% | 3.2% |

Every gallows value is Y-dominant. The ornate gallows (ckh, cth, cph, cfh) show 58–85% Y-family concentration versus 20–50% for plain gallows. No gallows value maps predominantly to N, L, or R. Gallows identity therefore does not recover per-family assignments. This approach is **killed**.

### S10.8.2 Family × gallows-class → house interaction (positive)

The flipped question asks: within each suffix family, does gallows class (bare, plain, ornate) constrain which house the prefix selects? If the cipher designer assigned gallows variants non-independently of houses, this interaction should be detectable.

**Table S10.8b.** House distribution per gallows class within each suffix family.

| Family | Gallows class | n | H1 (∅/qo) | H2 (o/d) | H3 (ch/sh) | H4 (y/s) |
|---|---|---|---|---|---|---|
| Y | bare | 3,169 | 2.5% | 12.8% | 83.0% | 1.7% |
| Y | plain | 3,766 | 54.5% | 34.2% | 2.9% | 8.3% |
| Y | ornate | 911 | 44.1% | 6.5% | 48.6% | 0.8% |
| N | bare | 2,141 | 32.2% | 54.0% | 4.6% | 9.2% |
| N | plain | 1,679 | 52.5% | 37.7% | 2.8% | 7.0% |
| N | ornate | 37 | 78.4% | 2.7% | 18.9% | 0.0% |
| L | bare | 2,297 | 25.8% | 39.7% | 29.2% | 5.3% |
| L | plain | 1,122 | 50.8% | 40.7% | 3.3% | 5.2% |
| L | ornate | 140 | 84.3% | 3.6% | 11.4% | 0.7% |
| R | bare | 1,920 | 30.4% | 40.2% | 22.3% | 7.0% |
| R | plain | 930 | 48.2% | 39.8% | 2.8% | 9.2% |
| R | ornate | 94 | 86.2% | 4.3% | 8.5% | 1.1% |

Gallows-class × house interaction is significant within every family: Y (χ² = 4886.1, df = 6, p ≈ 0), N (χ² = 212.4, df = 6, p = 4.4 × 10⁻⁴³), L (χ² = 532.5, df = 6, p = 8.5 × 10⁻¹¹²), R (χ² = 307.3, df = 6, p = 2.3 × 10⁻⁶³).

Three systematic patterns emerge:

1. **Bare tokens prefer H3 in Y-family** (83.0%) but H2 in N/L/R (40–54%). Bare-Y tokens are overwhelmingly ch/sh-prefixed, the most common tokens in the manuscript (chy, shy, chaiin, etc.).

2. **Plain tokens prefer H1** across all families (48–55%), with secondary H2 (34–50%). Plain gallows are rarely H3 (1.6–3.3% in non-Y families).

3. **Ornate tokens are overwhelmingly H1** in non-Y families (78–86%). In Y-family, ornate tokens split between H1 (44.1%) and H3 (48.6%). H2 and H4 are almost excluded (combined <8% in every family).

### S10.8.3 Y-family pairwise house tests

The three gallows classes within Y-family show maximally divergent house distributions despite sharing positional behaviour (§S10.8.4).

**Table S10.8c.** Pairwise χ² tests of house distribution within Y-family EC.

| Comparison | n₁ | n₂ | χ² | df | p |
|---|---|---|---|---|---|
| bare vs plain | 3,169 | 3,766 | 4773.7 | 3 | < 10⁻¹⁰⁰ |
| bare vs ornate | 3,169 | 911 | 1184.3 | 3 | < 10⁻¹⁰⁰ |
| plain vs ornate | 3,766 | 911 | 1564.5 | 3 | < 10⁻¹⁰⁰ |

All three classes use radically different house distributions to encode Y-family tokens. This is consistent with homophonic substitution: the same function word appears under different house prefixes depending on the gallows variant selected.

### S10.8.4 Positional identity across gallows classes

Despite divergent house usage, bare-Y and ornate-Y show indistinguishable positional behaviour, while plain-Y is the outlier.

**Table S10.8d.** Line-initial and line-final rates per gallows class × suffix family (EC, n ≥ 20).

| Family | Gallows class | n | Line-initial | Line-final |
|---|---|---|---|---|
| Y | bare | 3,169 | 3.1% | 10.3% |
| Y | plain | 3,766 | 9.4% | 8.7% |
| Y | ornate | 911 | 2.7% | 12.3% |
| N | bare | 2,141 | 16.2% | 12.3% |
| N | plain | 1,679 | 9.4% | 7.9% |
| N | ornate | 37 | 2.7% | 18.9% |
| L | bare | 2,297 | 8.1% | 8.4% |
| L | plain | 1,122 | 12.6% | 8.3% |
| L | ornate | 140 | 6.4% | 9.3% |
| R | bare | 1,920 | 12.3% | 8.3% |
| R | plain | 930 | 13.3% | 5.7% |
| R | ornate | 94 | 5.3% | 10.6% |

Within Y-family: bare-Y (3.1% LI) and ornate-Y (2.7% LI) are statistically indistinguishable (χ² = 0.1, p = 0.70). Plain-Y (9.4% LI) differs from bare-Y at χ² = 112.0, p = 3.6 × 10⁻²⁶. The positional identity of bare-Y and ornate-Y, combined with their divergent house distributions (§S10.8.3), is consistent with the same function word encoded under different house variants occupying the same syntactic position.

### S10.8.5 Scribe-level homophonic preferences

If the prefix slot functions as a homophonic variant selector, different scribes should exhibit distinctive prefix preferences, just as documented encipherers in Sforza correspondence show individual preferences among homophonic alternatives. Using the Davis (2020b) folio-level scribe assignments, each of the five identified scribes does exhibit a distinctive prefix distribution:

**Table S10.8e.** Prefix distribution per scribe (% of EC tokens).

| Scribe | Dialect | ∅ | qo | o | d | ch | sh | y | s | n |
|--------|---------|------|------|------|------|------|------|------|------|------|
| 1 | A | 20.1 | 8.7 | 16.2 | 15.5 | 19.9 | 8.8 | 6.2 | 4.6 | 9,013 |
| 2 | B | 18.1 | 20.0 | 22.1 | 8.7 | 13.1 | 10.2 | 3.7 | 4.1 | 9,133 |
| 3 | B | 26.5 | 16.6 | 22.8 | 4.8 | 16.2 | 7.3 | 3.8 | 2.0 | 11,653 |
| 4 | B | 16.5 | 2.0 | 39.2 | 8.8 | 16.0 | 4.8 | 9.5 | 3.2 | 650 |
| 5 | B | 26.4 | 9.0 | 18.1 | 12.3 | 13.0 | 9.8 | 8.5 | 2.8 | 1,054 |

Scribe-level variation accounts for most of the section-level house variation (house max/min ratio: 2.86× at section level, 2.22× at scribe level). If prefixes encoded cipher-layer content (e.g. keyword-driven alphabet rotation), scribe identity should not predict prefix choice. It does (scribe-level variation 2.22×, compared with gallows variation 1.14× which is content-determined and therefore scribe-independent).

### S10.8.6 Within-house pair compensation

If two prefixes within the same house are genuinely substitutable homophones, their folio-level frequencies should show negative correlation: when the scribe uses more of one, they use less of the other.

**Table S10.8f.** Folio-level Pearson r between prefix pairs within each house.

| House | Prefixes | Folio-level r | Compensatory? |
|-------|----------|---------------|---------------|
| H1 | ∅ / qo | +0.28 | No |
| H2 | o / d | −0.46 | Yes |
| H3 | ch / sh | −0.29 | Yes |
| H4 | y / s | +0.09 | Weak |

H2 (o/d) and H3 (ch/sh) show negative within-pair correlation at folio level, confirming substitutable homophones. H1 (∅/qo) shows positive correlation, which may reflect co-variation with section or folio length rather than compensation. H4 (y/s) shows no compensatory behaviour, and the two prefixes differ structurally: y carries gallows 63% of the time versus s at 1.5%, and their suffix-family distributions diverge (χ² = 257, p = 10⁻⁵³). The H4 grouping is the weakest of the four and may represent two distinct functional roles rather than a single homophonic pair.

### S10.8.7 Currier A/B as house preference

The Currier A/B dialect distinction, first identified in the 1970s and widely treated as evidence for two different languages or encoding systems, partially maps to scribe-specific house preference:

| Dialect | Scribes | H1 (∅+qo) | H2 (o+d) | H3 (ch+sh) | H4 (y+s) |
|---------|---------|-----------|----------|------------|----------|
| A | Scribe 1 | 28.8% | 31.7% | 28.7% | 10.9% |
| B | Scribes 2–5 | 40.0% | 25.0% | 23.3% | 7.1% |

Dialect A (Scribe 1) uses more H3 and H4; Dialect B scribes use more H1. The cipher architecture (two tables, suffix-family assignments, grid structure) does not differ between A and B; the scribes' selection among homophonic alternatives does.

### S10.8.8 Interpretation

The gallows axis does not independently recover function-word assignments (§S10.8.1). It does constrain house selection in a systematic way (§S10.8.2): ornate gallows in non-Y families select H1 prefix at 78–86%, reducing the effective house space for these tokens to a single value.

Scribe identity independently predicts house preference (§S10.8.5), and within-house prefix pairs show compensatory behaviour consistent with substitutable homophones for H2 and H3 (§S10.8.6). This is the expected behaviour of any homophonic system with multiple encipherers: the Sforza cipher keys in S12 provide multiple symbols per plaintext unit, and different secretaries using the same key develop individual selection habits. The VMS scribes did the same. The Currier A/B distinction, long treated as evidence for two encoding systems, is at least partly explained by scribe-specific homophonic preferences within a single cipher architecture (§S10.8.7).

Two alternative explanations for the ornate/H1 concentration were tested and rejected. First, a production rule based on glyph complexity (ornate gallows avoided at line-initial position for mechanical reasons) was falsified: token length correlates positively with line-initial rate across the manuscript (Spearman ρ = 0.89, p = 0.0001), and ornate tokens are enriched rather than depleted at line-final position. Second, the suffix-composition confound (ornate EC is Y-heavy, and Y is non-initial) was controlled by within-family testing: the positional divergence persists within every suffix family (Table S10.8d).

**Source:** enriched_records.pkl (37,465 tokens, 19,758 EC); Davis (2020b) folio-level scribe assignments. Reproduction code: `jojo_s_analysis.py` (deposited with supplementary materials at DOI 10.5281/zenodo.19056347).
