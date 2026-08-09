# Tranchedino × STA v2.3 — B1 D1 development result

Date: 2026-08-09
Protocol: `PROTOCOL_B1_CALIBRATION.md`
Data: D1 source panel only; no Q1 control and no Voynich target was scored.

## Development discipline

Three bounded solver revisions were explored, exhausting the protocol's allowed D1 development budget. No further D1-driven architecture/threshold changes are permitted.

### Revision 1 — unrestricted variable-output character-score polish

A frequency-based K92 initial map was polished while all semantic classes could exchange freely. The variable-output objective pushed decoded expansion toward the upper admissible boundary and damaged semantic recovery. Approximate four-control plaintext recoveries were 0.392, 0.614, 0.775 and 0.325. **Rejected on D1.**

### Revision 2 — event-normalised search with fixed inferred class partition

Search used total Paduan character log-likelihood per observed cipher event, while preserving the frequency-inferred alpha/gem/null/word class partition. Plaintext recovery rose to approximately 0.844, 0.793, 0.873 and 0.856; median ≈0.850. This passed the D1 >=0.80 plaintext gate but left the historically important residual classes poorly identified.

### Revision 3 — final architecture

The final architecture preserves only the frequency-inferred **alphabet-vs-residual** partition. Alphabetic signs can exchange alphabet semantics; the 56 residual signs can exchange geminate/null/nomenclator roles and nomenclator candidate words. This gives the residual historical classes a chance to be inferred rather than frozen from frequency alone.

The final bounded deterministic A search over all four null-rate proposal strata produced:

| D1 control | planted p_null | plaintext recovery | occurrence-weighted semantic recovery | alpha occurrence recovery | geminate occurrence recovery | nomenclator occurrence-word recovery | null occurrence F1 | selected p proposal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | .01 | 0.8477 | 0.9073 | 0.9649 | 0.2047 | 0.0000 | 0.1322 | .06 |
| D1 | .03 | 0.8412 | 0.8988 | 0.9553 | 0.2422 | 0.0510 | 0.3230 | .06 |
| D2 | .06 | 0.9082 | 0.9524 | 0.9848 | 0.1145 | 0.0745 | 1.0000 | .06 |
| D3 | .10 | 0.8327 | 0.8776 | 0.9361 | 0.0769 | 0.0000 | 0.7005 | .10 |

Median plaintext recovery: **0.8444**.

## D1 gate

Frozen D1 requirement: median plaintext recovery >=0.80.

Result: **PASS**.

This pass is deliberately narrow. It does **not** imply that the full mixed key is recoverable. The D1 truth audit shows that most plaintext recovery is carried by the alphabetic component; exact geminate and nomenclator recovery remains weak. That is precisely why Q1 has binding component-level minimum gates.

No Q1 threshold has been changed in response to these D1 diagnostics.

## Consequence

D1 development is now closed. The revision-3 implementation must be frozen before the first Q1 result. If any Q1 control violates a binding minimum gate, later Q1 controls cannot repair the minimum and compute must stop.
