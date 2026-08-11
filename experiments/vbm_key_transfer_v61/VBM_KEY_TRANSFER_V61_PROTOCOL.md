# VBM v6.1 — label-invariant key-transfer protocol

Date: 2026-08-11
Status: prospective protocol; no v6.1 synthetic or Voynich scores generated before this file.

## Scientific question

Does the VBM surface stream carry reusable cross-folio symbol-identity structure consistent with a stable latent language/cipher transducer, over and above its C/V topology, when stability is measured without relying on brittle argmax plaintext labels?

## Inherited primary intervention

Keep v6 identity-transfer excess (ITE) unchanged:

`ITE = score(observed held-out pseudo-folios | frozen train transducer) - median score(topology-preserving per-folio within-type label permutations | same transducer)`.

The permutation preserves the entire C/V sequence, segmentation, boundaries and core/bridge positions, while destroying reusable cross-folio surface-symbol identity.

## v6.1 replacement stability endpoint

Replace v6 argmax map agreement with **emission-kernel stability (EKS)**.

For a fitted typed emission matrix `E` and language stationary distribution `pi`, define the surface-side latent co-membership kernel

`K(E) = E.T @ diag(pi) @ E`.

Normalize `K` to unit Frobenius norm. Fit the same candidate language independently on two disjoint train-pseudo-folio halves, yielding `K1` and `K2`. The binding stability statistic is their Frobenius cosine:

`EKS = <K1,K2>_F / (||K1||_F ||K2||_F)`.

This compares the geometry induced among surface symbols rather than exact latent-letter labels. It is insensitive to homophone label switching and other equivalent latent assignments that preserve surface co-membership geometry.

Diagnostic only: predicted surface-pair-matrix cosine between the two halves. It is not a binding endpoint because a stable non-language Markov process may reproduce it directly.

## Instrument

- typed moment-factorisation transducer only; no post-moment Baum–Welch;
- candidate latent languages: Bavarian and German;
- source-grounded `FREQ_PROP` core allocation and Amadi `CYCLE` bridge scheduling for language cipher controls;
- 12 pseudo-folios per replicate;
- 4 cross-folio folds;
- 24 topology-preserving held-out permutations per fold;
- 500 moment steps for formal CAL/VAL;
- fresh deterministic namespace `VBMKEYTRANSFERV61`.

## Synthetic families

Positive reusable-key families:
1. `BAV_GLOBAL`
2. `GER_GLOBAL`
3. `BAV_GLOBAL_SWAP` — one global Bavarian key plus sparse local same-type swaps

Negative/adversarial families:
4. `BAV_FRESH` — genuine Bavarian, fresh key per pseudo-folio
5. `GER_FRESH` — genuine German, fresh key per pseudo-folio
6. `STABLE_MARKOV` — one stable non-language surface first-order Markov grammar generated from the same VBM-like alphabet/geometry

These directly distinguish reusable-key language ciphertext from real language with no reusable key and from stable non-language symbol grammar.

## Q0 smoke

One fresh replicate per family, reduced steps/permutations. Diagnostic only. No thresholds frozen from smoke.

## Q1 formal calibration

Fresh CAL namespace, 4 replicates per family = 12 positives and 12 negatives.

CAL must satisfy both:
- minimum positive median ITE > maximum negative median ITE;
- minimum positive median EKS > maximum fresh-key (`BAV_FRESH`,`GER_FRESH`) median EKS.

If separable, freeze:
- `TAU_ITE` = midpoint between weakest positive and strongest negative CAL ITE;
- `TAU_EKS` = midpoint between weakest positive and strongest fresh-key CAL EKS.

A replicate passes iff both median ITE >= `TAU_ITE` and median EKS >= `TAU_EKS`.

## Q2 untouched synthetic validation

Fresh VAL namespace, again 4 replicates per family.

Binding gates:
- >=11/12 reusable-key positives pass both frozen thresholds;
- each positive family >=3/4;
- 0/12 negative/adversarial replicates pass both thresholds;
- `STABLE_MARKOV` exactly 0/4;
- `BAV_FRESH` exactly 0/4;
- `GER_FRESH` exactly 0/4.

Stop immediately when a gate becomes mathematically irrecoverable.

## Q3 Voynich FIT

Only if Q2 passes.

Do not reuse H1. Do not open C1.

Use the inherited frozen VBM FIT set, excluding H1 and C1 by construction. Six deterministic leave-folio-out folds.

Binding FIT gates:
- median ITE >= `TAU_ITE`;
- >=5/6 folds ITE > 0;
- median EKS >= `TAU_EKS`;
- >=5/6 folds EKS >= `TAU_EKS`;
- one latent language selected in >=4/6 folds.

No plaintext or per-symbol argmax map may be inspected.

## Q4 C1

Only if Q3 passes.

Train one final transducer on all FIT, freeze it, and score sealed `VBM_C1 = f85r1 f53v f33r f10r f23r f111r` once against 64 topology-preserving per-folio permutations.

C1 passes iff:
- ITE >= `TAU_ITE`;
- observed C1 score > all 64 permutations;
- selected latent language matches FIT plurality.

EKS is a train-set reproducibility endpoint and is therefore not recomputed on C1.

## Interpretation

A positive result supports reusable cross-folio symbol identity compatible with a stable cipher transducer, not plaintext recovery. Bavarian specificity requires Bavarian plurality on FIT and C1. Failure before target leaves all Voynich inference unchanged.
