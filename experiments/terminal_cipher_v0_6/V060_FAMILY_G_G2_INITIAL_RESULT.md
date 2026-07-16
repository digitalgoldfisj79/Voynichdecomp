# v0.6 Family G — initial G2 development result

Date: 2026-07-16

Status: **FAILED — ONE PERMITTED DEVELOPMENT AMENDMENT TRIGGERED**

## Provenance

- Repository: `digitalgoldfisj79/Voynichdecomp`
- Branch: `experiment/terminal-cipher-programme-v0.6-20260716`
- Terminal protocol commit: `b751675e0ffdfa132579feacfe2f0d65f4884479`
- Initial implementation freeze: `044e75d8f6cd9b51c47d29ed0233d3ddea7c7c93`
- Full development job: `6a58d530b1669a49bf077861`
- Inventory: 2,935 frozen extraction candidates
- Inventory SHA-256: `00d3ae0d40c4cbf63c78c75eb779aae7c4cc15e524faa923d0d0d7a4b9584afe`
- Result SHA-256: `6d658fe149fde328319aec5a06309c51cfdb75e2c34afd34bb40bbe2728ed76e`

No locked test or Voynich data was opened.

## Frozen-gate result

| Requirement | Observed | Pass |
|---|---:|:---:|
| AUROC ≥0.95 | 0.675903 | no |
| False-positive rate ≤1% | 0.78125% | yes |
| Carrier-class accuracy ≥85% among detected | 100% | yes |
| Exact parameter accuracy ≥75% among detected | 100% | yes |
| Mean recovery ≥80%, abstention = 0 | 50.0000% | no |
| At least 54/64 recovered ≥70% | 32/64 | no |

Additional result:

- payload covers: 64;
- matched null covers: 256;
- detected payload covers: 32;
- minimum recovery among detected covers: 100%;
- empirical threshold: `-2.531994074619792`.

**Global initial G2 gate: FAIL.**

## Failure structure

The failure is not diffuse carrier confusion. The 32 detected payload covers were recovered perfectly:

- carrier class: 32/32 exact;
- carrier parameters: 32/32 exact;
- payload recovery: 32/32 at 100%.

The programme contains exactly 32 plaintext payload covers and 32 fresh-mono payload covers. The detected count, perfect detected recovery and trial logs establish that the initial operating score detected the plaintext half and systematically abstained on the monoencrypted half.

G1 had already shown that the same 32 encrypted payloads are recoverable when the true carrier is supplied, with encrypted mean recovery 96.3867%. The G2 defect is therefore in blind encrypted-carrier ranking and arm calibration, not in carrier capacity or the downstream mono solver.

## Decision

The initial G2 solver fails development. The one protocol-permitted G2 amendment may now address the concrete encrypted-arm ranking/calibration defect. The carrier inventory, generators, payload and null sets, operating-point construction, gates, split and test/Voynich seals remain unchanged.
