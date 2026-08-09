# BnF M19 STA/aaa v1.7 — Binding Q3 Qualification Result

Date: 2026-08-09
Namespace: `M19STAv17Q3`
Protocol: `PROTOCOL.md` plus Amendments 001–005.

## Execution

Three fresh qualification jobs were run independently for the binding surface-vocabulary sizes K=22, K=26 and K=36. Each job reran all six control languages (Latin, Italian, German, French, Arabic, Spanish) from untouched UD dev+test control pools. The fitting half of every accepted control span was required to support all 19 BnF numerical values. No RF/Voynich H17 or C17 language score was generated.

HF jobs:
- K=22: `6a7861ebda2af92a634f013d`
- K=26: `6a7862093e1f34a7e32c03b7`
- K=36: `6a786220da2af92a634f0143`

All three passed the frozen score self-test before qualification.

## Binding gates

| K | 6/6 language rank | min margin | median map recovery | min map recovery | min fit agreement | Gate |
|---|---:|---:|---:|---:|---:|---|
| 22 | 6/6 | 0.089999 | 1.000000 | **0.763205** | 1.000000 | **FAIL** |
| 26 | 6/6 | 0.086044 | 1.000000 | 1.000000 | 1.000000 | **PASS** |
| 36 | 6/6 | 0.088197 | 1.000000 | 0.965154 | 1.000000 | **PASS** |

Frozen minimum map-recovery threshold: 0.85.

### K=22 failure

The only failing metric is Arabic exact numerical-map recovery: **0.7632051282**. Arabic is nevertheless ranked correctly with margin **0.2238711532**, and the two independent fits agree 1.0 occurrence-weighted. The other five K=22 controls recover the exact map at 1.0.

This is therefore an identifiability/recovery failure under the frozen control gate, not a language-discrimination failure.

### K=26

All six languages rank correctly. Every exact numerical map is recovered at 1.0 and all independent-fit agreements are 1.0. Minimum language margin is German at **0.0860436832**.

### K=36

All six languages rank correctly. Minimum map recovery is Arabic at **0.9651538462**; German is 0.9994102564; all others are 1.0. All independent-fit agreements are 1.0. Minimum language margin is German at **0.0881969186**.

## Formal verdict

**STA/AAA INSTRUMENT NOT QUALIFIED** under the frozen v1.7 hierarchy protocol.

The protocol requires all three representation scales to qualify before RF H17 is language-scored. Because K=22 fails the frozen 0.85 minimum exact-map-recovery gate, the hierarchy is locked. H17 and C17 remain sealed and no Voynich language result is admissible from v1.7 Q3.

This result does not reject the BnF M19 hypothesis or the STA/aaa representations. It establishes that the current generalized K=22 control instrument is insufficiently identifiable under the exact preregistered recovery criterion, despite perfect control-language classification.
