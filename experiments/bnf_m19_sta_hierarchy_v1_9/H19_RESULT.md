# BnF M19 STA/aaa v1.9 — RF H19 Result

Date: 2026-08-09
Protocol freeze: `67ff4e032395030d18b84a630c216e55854ff3a3`
Qualification result: `55e67834351bd8bafd7281ea786dd1ae39520c73`
H19 runner: `5b0688f1f71730c5376dc64ff2fbe79894cce587`

## Qualification prerequisite

All three synthetic-control scales passed the binding v1.9 qualification under the stronger convergence-controlled optimizer:

- K=22 STA family: PASS
- K=26 connected aaa: PASS
- K=36 full STA: PASS

RF H19 was therefore legitimately unlocked.

## H19 execution

Three RF H19 jobs were launched independently:

- STA family K=22: `6a786a7bda2af92a634f01a2`
- connected aaa K=26: `6a786a94da2af92a634f01a6`
- full STA K=36: `6a786aadda2af92a634f01a8`

C19 remained sealed throughout.

### Binding STA-family result

The K=22 family arm completed all eight language fits with successful A/B convergence for every language. H19 coverage was 0.9983818770 (word coverage 0.9980126692).

Held-out H19 ranking:

1. Spanish: **-2.5711088593**
2. French: **-2.5934305936**
3. Greek: -2.6259873716
4. German: -2.6319475217
5. Latin: -2.6527180797
6. Hebrew: -2.6835728413
7. Italian: -2.7021448099
8. Arabic: -2.7801834577

Top-language margin Spanish over French: **0.02232173435 nats/retained unit**.

Frozen H19 requirement: margin >= **0.05**.

Other binding family-arm conditions passed:

- coverage >=0.97: PASS
- top-language A/B map agreement >=0.90: PASS (1.0)
- top-language optimizer convergence: PASS
- all eight language fits converged: PASS

But the margin gate fails decisively: **0.02232 < 0.05**.

### Remaining representation jobs

Under the frozen hierarchy, all three representations must independently pass H19 and select the same top language before C19 can be unlocked. Once the family arm failed its binding margin gate, neither the connected-aaa nor full-STA result could restore the hierarchy. Their still-running jobs were therefore cancelled prospectively for compute discipline rather than allowed to consume unnecessary resources.

At cancellation, many language-specific fits in both jobs had already converged, but neither job had emitted a final held-out language ranking. These incomplete fits are diagnostic only and are not used for any inference.

## Formal verdict

**NO STA/AAA M19 SIGNAL** under the frozen v1.9 hierarchy.

The stronger optimizer successfully repairs the v1.7 control-identifiability failure, so this is no longer an instrument failure. It is a genuine held-out failure of the first binding Voynich representation: the RF STA-family stream does not separate its best-fitting language from the runner-up by the preregistered 0.05 margin.

Spanish is the numerical top model at this representation, but the margin is only 0.02232 and therefore does **not** constitute a candidate-language signal. No C19 score, decoded text, or independent IT/ZL/GC transfer was generated.
