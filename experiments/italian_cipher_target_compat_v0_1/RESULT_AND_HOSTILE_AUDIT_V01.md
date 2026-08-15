# Northern Italian Cipher Entropy Compatibility v0.1 — result and hostile audit

Date: 2026-08-15
Successful target run: `31907936454`
Target-run head: `febbf035ba0a2a2db97c175b49eac5d64ed31df5`
Artifact: `9252867055`
Artifact SHA-256: `5460f2854be9de3352bf3e10e3e5cf6cd8afd08f0e277f356f8d4f57c7cbb891`

## Frozen result

`ENTROPY_ADVANTAGE_BUT_OUT_OF_DISTRIBUTION`

External target-stage calibration reproduced the preregistered PASS:

- primary positive-control groups: HOM PASS, NOM PASS, BIGRAM PASS;
- ACTIVE mechanisms: BIGRAM10/20/40; HOM2; HOM34; HOM34+null 1/2.5/5%; NOM25/50/100;
- global identity/plaintext null threshold: **1.6165962624929362**.

The repaired STA_CODE representation passed all integrity gates:

- 5,385 common RF/STA loci;
- zero word-count mismatches;
- zero unparsed STA residues;
- synthetic `P2A3K1 -> [P2,A3,K1]` PASS;
- 37,848 STA words across 227 folios.

Whole-manuscript target:

- 157,254 STA atomic symbols;
- 131 non-overlapping 1,200-symbol windows;
- median `Delta_any = 2.6715915359204283`, which exceeds the frozen global plaintext-null threshold;
- **no ACTIVE mechanism passes its externally calibrated compatibility threshold**;
- `NOM100` is the least-distant ACTIVE mechanism in all 131 target windows.

Therefore the frozen positive-candidate rule does not fire.

## Hostile result audit

The superficially interesting feature is that the target gains a large standardized-distance advantage over the identity/plaintext model and always chooses NOM100. This must **not** be interpreted as a nomenclator identification.

### 1. The target is not actually close to NOM100

Median target distances versus external LOSO compatibility thresholds:

| Mechanism | Target median distance | Compatibility threshold |
|---|---:|---:|
| ID | 8.371 | 3.024 |
| NOM100 | 5.688 | 3.984 |
| NOM50 | 8.251 | 1.729 |
| NOM25 | 10.489 | 1.544 |
| BIGRAM10 | 7.002 | 2.446 |
| HOM2 | 12.334 | 2.360 |
| HOM34 | 14.166 | 4.644 |

Only 2/131 target windows fall within the NOM100 threshold; none fall within ID, NOM25, NOM50, BIGRAM10, HOM2, or HOM34 thresholds. The whole-manuscript median is outside every ACTIVE mechanism.

### 2. Conditional entropy is the decisive mismatch

Target window medians:

- `H0 = 4.0196 bits/symbol`
- `H1 = 2.4659 bits/symbol`
- `MI1 = 1.5849 bits/symbol`
- median alphabet size `K = 35`.

The unenciphered external ID model has centre:

- `H0 = 4.0934`
- `H1 = 3.2313`.

Thus target H0 remains approximately ordinary-text-like, while target H1 is lower by about **0.77 bits/symbol**.

This is not reproduced by the historically anchored primary mechanisms:

- all 131 target windows have H1 below the minimum H1 observed among 105 external ID windows (`3.0088`);
- 129/131 target windows have H1 below the minimum external NOM100 H1 (`2.7977`);
- all target windows have H1 below the external minima for NOM25, NOM50, BIGRAM10, BIGRAM20, HOM2, and HOM34;
- 128/131 are below the minimum BIGRAM40 H1 (`2.7389`).

At the same time, nomenclator and bigram mechanisms raise H0 substantially. NOM100's external centre is `H0=4.8878, H1=3.1882`; BIGRAM10's is `H0=4.6505, H1=3.4358`. The target is therefore not sitting near those mechanism clouds.

### 3. Why NOM100 nevertheless wins every window

NOM100 is the least-wrong standardized model, not a compatible model. Its external H0/H1 dispersion is broader than the ID model, so an extreme low-H1 target receives a smaller robust-z penalty even though its raw H0/H1 point is farther from the NOM100 centre than from the ID centre.

For the target median H0/H1 point, approximate component z-scores are:

- ID: H0 `-0.56`, H1 `-11.82`;
- NOM100: H0 `-4.25`, H1 `-6.77`.

Raw Euclidean H0/H1 distance of the target median is actually smaller to ID (~0.769) than to NOM100 (~1.129). The global plaintext-null correctly controls ordinary-source false improvements, but it does not convert an out-of-distribution target into positive mechanism evidence.

Therefore `Delta_any > null` is only a directional diagnostic once compatibility fails.

### 4. Historical mechanism direction

The external-only transfer experiment had already established the causal directions:

- dense homophony strongly raises both H0 and H1;
- nulls raise them further;
- nomenclators raise H0 and alter H1 only modestly;
- atomic bigram codes raise H0 and generally raise H1;
- diglyph rendering can lower glyph-level H1 somewhat, but its H1 sign was rendering-sensitive and its H0 remains much higher than the target;
- combined mechanisms create very high H0 (roughly 6.1–7.0 in the target-stage 1200-symbol controls), inconsistent with target H0 near 4.02.

The target residual that a successful mechanism would need to explain is therefore quite specific:

> **substantially lower conditional entropy / stronger local dependence while leaving marginal symbol entropy close to ordinary plaintext levels.**

The 1435–1448 Northern Italian diplomatic ingredients tested here do not do that.

## Section diagnostics

Section `Delta_any` medians are heterogeneous, but these are directional diagnostics, not compatibility results, and cannot upgrade the whole-manuscript failure:

- Balneological 4.381
- Stars 3.220
- text-only 2.936
- Rosettes 2.690
- Zodiac 2.523
- Pharmaceutical 2.341
- Herbal-A 1.988
- Herbal-B 1.893
- Cosmological 1.528
- Astronomical 0.917

The high Balneological/Stars values merit later residual analysis, but no section-specific historical-cipher claim is licensed by this run.

## Scientific conclusion

The result is a **negative for the tested pre-1450 Northern Italian diplomatic-cipher entropy family**, not a positive nomenclator result.

A fair formulation is:

> Relative to ordinary external plaintext, the Voynich entropy vector moves in the broad direction for which a large nomenclator is the least-bad tested comparator, but the Voynich remains decisively outside every externally calibrated historical mechanism distribution. Its defining entropy residual is much lower H1 at roughly ordinary H0, which the tested homophonic, null, nomenclator and atomic-bigram mechanisms do not reproduce.

This result should constrain future mechanism searches rather than trigger parameter tuning. Any new proposed mechanism selected because it can lower H1 must be treated as exploratory and independently calibrated on historical/external controls before another target test.
