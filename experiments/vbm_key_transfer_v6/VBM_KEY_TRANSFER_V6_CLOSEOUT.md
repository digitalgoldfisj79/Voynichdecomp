# VBM v6 — key-transfer closeout

Date: 2026-08-11
Branch: `experiment/vbm-key-transfer-v6-20260811`

## Binding conclusion

**VBM v6 KEY-TRANSFER INSTRUMENT: CALIBRATION SEPARABLE, UNTOUCHED POSITIVE VALIDATION FAILS 7/9; CLOSED BEFORE VOYNICH FIT.**

No v6 Voynich FIT scoring was run. H1 was not reused. `VBM_C1 = f85r1 f53v f33r f10r f23r f111r` remains sealed. No target plaintext or decode mapping was inspected.

## Scientific target

v6 asked whether the VBM stream carries reusable cross-folio symbol-identity information consistent with a stable cipher transducer over and above the already-established C/V topology.

The primary statistic was identity-transfer excess:

`ITE = score(observed held-out folios | frozen train transducer) - median score(topology-preserving per-folio label permutations | same transducer)`.

A secondary map-stability statistic compared surface→latent assignments from two disjoint train-folio halves.

Per Amendment 001 the transducer was the typed moment-factorisation emission matrix, without post-moment Baum–Welch.

## Smoke

HF job `6a7b900b27caad61c6eac6a2` completed.

Median smoke ITE:
- Bavarian global key: +0.7902
- German global key: +1.3390
- Bavarian global key + sparse local same-type swaps: +0.7230
- Bavarian fresh key per pseudo-folio: -0.0320
- German fresh key per pseudo-folio: -0.0112
- stable non-language Markov surface process: +0.5577

The smoke therefore showed the desired qualitative separation, while correctly identifying the stable Markov process as the hard adversary.

## Q0 CAL

HF job `6a7b9044f6d0f3ee953aa1c2`.

### Same-key positives

BAV_GLOBAL:
- rep0 ITE 0.9008186; STAB 0.9046914
- rep1 ITE 0.9047399; STAB 0.8000000
- rep2 ITE 1.2304065; STAB 0.8850926

GER_GLOBAL:
- rep0 ITE 1.4517683; STAB 0.5661728
- rep1 ITE 1.4908237; STAB 0.5984568
- rep2 ITE 1.4106912; STAB 0.5470679

BAV_GLOBAL_SWAP:
- rep0 ITE 0.8032301; STAB 0.7728704
- rep1 ITE 0.9790919; STAB 0.5162346
- rep2 ITE 0.8447670; STAB 0.7234259

### Fresh-key negatives

BAV_FRESH:
- rep0 ITE +0.0061547; STAB 0.1215432
- rep1 ITE -0.0050924; STAB 0.1206790
- rep2 ITE +0.0067752; STAB 0.1474074

GER_FRESH:
- rep0 ITE +0.0220233; STAB 0.1634259
- rep1 ITE -0.0020011; STAB 0.1108025
- rep2 ITE -0.0257181; STAB 0.1489506

### Stable non-language Markov adversary

STABLE_MARKOV:
- rep0 ITE 0.6457984; STAB 0.5924691
- rep1 ITE 0.6087056; STAB 0.4127160
- rep2 ITE 0.5474241; STAB 0.7681790

Calibration was fully separable under the preregistered combined gates.

Frozen thresholds from CAL:

- `TAU_ITE = 0.7245142406459757`
- `TAU_STAB = 0.33983024691358025`

The ITE threshold lies midway between weakest same-key positive (0.8032301) and strongest negative/adversarial ITE (0.6457984).
The stability threshold lies midway between weakest same-key positive stability (0.5162346) and strongest fresh-key stability (0.1634259).

## Untouched VAL — decisive failure

The preregistered positive gate required >=8/9 same-key positive replicates to clear **both** frozen thresholds.

Completed positive VAL rows before the stop:

BAV_GLOBAL:
- rep0 ITE 0.9003803; STAB 0.5915432 — PASS
- rep1 ITE 0.9114255; STAB 0.2673457 — **FAIL STABILITY**
- rep2 ITE 0.8630194; STAB 0.8764506 — PASS

GER_GLOBAL:
- rep0 ITE 1.4387456; STAB 0.7762654 — PASS
- rep1 ITE 1.4545131; STAB 0.5297222 — PASS
- rep2 ITE 1.4280571; STAB 0.9539815 — PASS

BAV_GLOBAL_SWAP:
- rep0 ITE 0.8576067; STAB 0.4727160 — PASS
- rep1 ITE 0.8186903; STAB 0.6859568 — PASS
- rep2 ITE 0.9377897; STAB 0.2041358 — **FAIL STABILITY**

Thus the maximum attainable positive pass count became **7/9**, below the frozen >=8/9 requirement. The validation gate was mathematically unrecoverable before the VAL negative families needed to complete.

The job was cancelled immediately under the compute stop rule.

## Interpretation

The primary ITE statistic itself is strikingly robust in the completed positive validation rows: all nine same-key validation replicates exceed `TAU_ITE`, including both rows that failed overall. The failure is entirely in the secondary map-stability endpoint.

This means v6 has discovered an important distinction:

1. **Cross-folio symbol identity is recoverably predictive under a reusable synthetic key**, and the topology-preserving permutation intervention cleanly destroys that advantage.
2. **Fresh-key genuine language ciphertext collapses ITE to approximately zero**, so ITE genuinely distinguishes reusable-key from per-folio-key language ciphertext.
3. A stable non-language Markov surface grammar can also generate substantial ITE (up to 0.6458 in CAL), so ITE alone is not a sufficient cipher diagnostic.
4. The proposed surface→latent map-stability statistic is too variable even for true same-key positives: two untouched same-key validation replicates fell below the frozen stability threshold.

Therefore the combined v6 instrument is **not qualified** and no inference about Voynich key reuse is permitted.

The failure is informative rather than a null result on the cipher question. It isolates the unresolved methodological problem: we need a stability measure invariant to homophone/latent-map non-identifiability but still capable of rejecting stable non-language surface grammars.

A legitimate successor should keep the successful topology-preserving ITE intervention, replace the fragile argmax latent-map agreement with a permutation/label-invariant representation (for example emission-subspace, predicted-pair matrix, or canonical transport similarity), and requalify from scratch before Voynich FIT or C1 access.

## Compute ledger

- smoke `6a7b900b27caad61c6eac6a2` — completed.
- full Q0 `6a7b9044f6d0f3ee953aa1c2` — cancelled at irrecoverable 7/9 positive VAL gate.
- Voynich FIT — **not launched**.
- C1 — **not launched**.
