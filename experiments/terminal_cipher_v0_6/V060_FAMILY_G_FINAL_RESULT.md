# v0.6 Family G — final result

Date: 2026-07-16

Status: **FAILED AT DEVELOPMENT — FAMILY CLOSED**

## Provenance

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/terminal-cipher-programme-v0.6-20260716`
- Terminal protocol commit: `b751675e0ffdfa132579feacfe2f0d65f4884479`
- Frozen Family G protocol: `V060_PROTOCOL_FAMILY_G_CARRIER_STEGANOGRAPHY.md`
- G1 implementation commit: `9e5fb7b97f9078748687033441f1c5e350cb3e1f`
- G1 combined report commit: `10f9d7a6e0d581dd2f157876308e0cd557eb7035`
- Initial G2 implementation commit: `9daebe9e3f7cf7e373d05a175180bb6e70b6247b`
- Initial G2 freeze commit: `044e75d8f6cd9b51c47d29ed0233d3ddea7c7c93`
- Initial G2 job: `6a58d530b1669a49bf077861`
- Initial result SHA-256: `6d658fe149fde328319aec5a06309c51cfdb75e2c34afd34bb40bbe2728ed76e`
- Sole amendment commit: `06ec6c3db3c1ef7410fe07806c3e9ca5f85416ca`
- Final solver commit: `e631167b45026b1b6710c64033aedb06f8a7af87`
- Final encrypted smoke job: `6a58d724b1669a49bf0778b3`
- Final smoke SHA-256: `3bd45405e7e92450d1a6adc6d76074ec53b56549e5d3818dbd39aaff298b77b4`
- Final full job: `6a58d7b985d9643ce16d6405`
- Final scientific result SHA-256: `c5d4bd663f19a14101fa0d2c3f31b6cb91c0411488cb125371cc8939087c8bc5`
- Machine-readable closure result: `v060_family_g_g2_final_result.json`
- Closure payload SHA-256: `43e99e18164a780fd4a82a6c09bf92cac13b7b0af25b443ca0bf19ac5e5554a2`

No locked test or Voynich data was opened.

## Development sequence

### G1 oracle-carrier stage

G1 passed strongly over 64 development covers:

- mean recovery: 98.1934%;
- minimum recovery: 81.2500%;
- at least 85%: 63/64;
- encrypted-payload mean: 96.3867%.

This established that the bounded carriers and downstream plaintext/mono recovery were viable when the true extraction rule was supplied.

### Initial blind G2

The initial 2,935-rule blind search failed:

- AUROC: 0.675903;
- detected payloads: 32/64;
- carrier class and parameters among detected: 100%;
- mean recovery with abstentions scored zero: 50.0000%;
- recovered at least 70%: 32/64;
- false-positive rate: 0.78125%.

It recovered the plaintext half perfectly and systematically suppressed the fresh-mono half.

### Sole permitted amendment

The amendment corrected the mono arm's double penalty, expanded the substitution-invariant shortlist from 12 to 128 candidates, increased its screening budget from 50,000 × 5 to 100,000 × 8, and retained the unchanged 256-null operating-point calibration.

The mandatory encrypted smoke executed correctly but still selected a spurious grille:

- true carrier: line telestic;
- selected carrier: width-12 grille;
- recovery: 18.75%.

No further amendment was permitted.

## Final amended G2 result

| Frozen requirement | Observed | Pass |
|---|---:|:---:|
| AUROC ≥0.95 | 0.785522 | no |
| False-positive rate ≤1% | 0.78125% | yes |
| Carrier-class accuracy ≥85% among detected | 94.7368% | yes |
| Exact parameter accuracy ≥75% among detected | 92.1053% | yes |
| Mean recovery ≥80%, abstention = 0 | 57.6497% | no |
| At least 54/64 recovered ≥70% | 38/64 | no |

Additional results:

- detected payload covers: 38/64;
- minimum recovery among detected: 88.5417%;
- matched null covers: 256;
- frozen threshold: `-1.0508602310135067`.

**Global final G2 gate: FAIL.**

## Interpretation constrained by the protocol

The final solver remains reliable after it has selected a payload carrier: among detected covers, carrier class and parameters are highly accurate and minimum recovery is 88.54%. The unresolved problem is blind family-wide separation and ranking under fresh mono substitution. Too many encrypted payload carriers either lose to incidental extraction maxima or fall below the matched-null threshold.

This does not establish that bounded steganographic carriers are impossible in general. It establishes that the preregistered Family G candidate inventory and permitted solver portfolio failed the frozen synthetic recoverability standard required before Voynich use.

## Decision

Family G is closed at development. No locked synthetic test is permitted. No bounded steganographic extraction from this family may be scored on Voynich data. The identical previously closed fixed-position Polygraphia-style global mechanism remains closed and is not reopened.
