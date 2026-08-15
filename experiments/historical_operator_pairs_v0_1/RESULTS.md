# Historical Operator Pairs v0.1 — results

Run date: 2026-08-15
GitHub Actions run: 31909970418
Runner commit: `61ff6142a68ec9c4d6817ebf8e1dd15f0fe54da9`
External canonical freeze SHA: `726b10647f763e625c723141bf11825b7fc36c7b1705104e146f2299195e0f4e`
Artifact digest: `sha256:36f85c0e02f9f899e5f2ff7bf6024d86db78e87e56f60550442e5a1e7ef63b94`

## Primary verdict

**HISTORICAL_ABBREVIATION_NOT_SUPPORTED**

Both independent historical corpora fail the preregistered requirement that real abbreviation lower first-order conditional character entropy. The Voynich target was therefore never opened by the runner.

### Nuremberg Letterbooks

QC: 47,784 paired diplomatic lines; 40,012 abbreviation-bearing lines; 7,772 no-abbreviation lines; no-abbreviation identity rate 1.000. 3,159 abbreviation-bearing documents were metric-eligible.

Operator convention is ABBREVIATED minus EXPANDED.

- H0 shift: median **+0.08332 bits**, 95% bootstrap CI **[+0.08177,+0.08512]**.
- H1 shift: median **+0.03137 bits**, 95% CI **[+0.02847,+0.03358]**.
- H0-H1 shift: median **+0.04876 bits**, 95% CI **[+0.04647,+0.05081]**.
- Length-matched H1 shift: median **+0.10510 bits**, 95% CI **[+0.10222,+0.10777]**.

Thus abbreviation raises, rather than lowers, H1. Pooled H1 rises from 3.61296 expanded to 3.79358 abbreviated.

### ORIFLAMMS Dated and Datable Manuscripts

QC: 101 paired manuscripts; 2,261 perfectly line-ID-aligned lines; 1,795 abbreviation-bearing lines; median and minimum line-ID overlap 1.000. All 101 manuscripts were metric-eligible.

- H0 shift: median **+0.31587 bits**, 95% bootstrap CI **[+0.27286,+0.35895]**.
- H1 shift: median **+0.01491 bits**, 95% CI **[-0.00004,+0.02831]**.
- H0-H1 shift: median **+0.30616 bits**, 95% CI **[+0.27099,+0.33606]**.
- Length-matched H1 shift: median **+0.04452 bits**, 95% CI **[+0.02259,+0.06378]**.

Again, abbreviation does not lower H1. Pooled H1 rises from 3.50845 expanded to 3.89489 abbreviated.

## Prespecified secondary result: a higher-order inversion

The H2 diagnostic (conditional entropy given the preceding two characters) was prespecified, although it was not part of the primary qualification gate. It behaves in the opposite direction in both corpora:

- Nuremberg H2 shift: median **-0.12329 bits**, 95% CI **[-0.12546,-0.12129]**; 99.49% of abbreviation-bearing documents have negative H2 shift.
- ORIFLAMMS H2 shift: median **-0.23534 bits**, 95% CI **[-0.27222,-0.20919]**; 100% of manuscripts have negative H2 shift.

This is not evidence for the preregistered Voynich-abbreviation hypothesis because the H1 gate failed and the target remained sealed. It is, however, a replicated empirical fingerprint of real medieval abbreviation: abbreviation increases character inventory/unigram entropy and does not reduce one-character-context entropy, while it strongly increases predictability once two preceding characters are known.

The H2 effect also strengthens with abbreviation density in both corpora (exploratory Spearman correlations approximately -0.44 Nuremberg and -0.73 ORIFLAMMS). This density analysis is post-hoc and should be treated as descriptive.

## Interpretation

The result rejects the simple mechanism previously under consideration: ordinary medieval scribal abbreviation is not an empirical route to low first-order conditional entropy. Synthetic abbreviation simulations therefore should not be used as evidence for that mechanism unless they reproduce this real historical operator fingerprint.

A distinct higher-order abbreviation signature exists and is worth a separately labelled follow-up if the programme chooses to test conditional-entropy spectra; such a follow-up must not be described as confirmation of v0.1.

## Caveat

The Nuremberg XML can encode an expansion by deleting the editorial `<ex>` letters without always preserving a separate physical abbreviation mark in the abbreviated rendering. ORIFLAMMS `alto_abbr/without-norm` is the stronger graphical witness because it retains manuscript abbreviation characters and combining marks. The fact that the primary H1 direction is non-negative in both representations makes the primary negative result robust to this difference.
