# VMS Residual Recovery Ladder v1.0
Date: 2026-10-08
Status: MASTER PROGRAMME — prospective ladder
Scope: running paragraph text only unless a rung explicitly says otherwise.

## 0. Current evidence base

The ladder begins from the completed 2026-10-08 programme, not from a blank slate.

### Established
- VMS-ECOLOGY-1: the compact SELECT + line-position architecture leaves a strong 15-D ordered residual in Herbal-A, Herbal-B, Balneological/Q13, and Recipes.
- ABA is not a universal mechanism. It is a recurring observable fingerprint, especially robust in Balneological and Recipes, but transcription-sensitive in Herbal.
- RESIDREC1: one universal linear predictive-state basis does not explain all ecologies.
- RESIDREC2: a variable-order observable K12 context correction improves held-out prediction in all four ecologies; the tested line-reset GLM-HMM is uniformly worse.
- RESIDREC3: observable context fully absorbs the registered broad residual only in Herbal-B; residual structure remains in Herbal-A, Balneological, and Recipes.
- RESIDREC4/4B: in Recipes, a small K12-history GRU produces a null-qualified held-out gain beyond variable-order context (+0.0077477 bits/event, Z_G~7.52) but fails residual absorption catastrophically (15-D Z_D2~17.42).
- The largest remaining Recipes deviations after q3 are Pearson/calibration energy, lag-2 repeat residual, lag-2 surprise autocorrelation, and excess surprise.

### Therefore
The next programme is not "try a larger neural network".
The remaining hypothesis space is:

H1. Missing observable channel: K12 opener history is too lossy; FORM/final-piece/token-shape history carries predictive state.
H2. Missing observed slow context: line/entry/paragraph/page/hand/Currier/physical packet variables drive selection.
H3. Missing latent slow state: a line/entry/page-scale latent regime with persistence/dwell time is required.
H4. Distributed/factorial state: more than one process jointly drives the observations (e.g. SELECT/repertoire and FORM/shape).
H5. Wrong state representation: the process is predictive/unifilar or infinite-order rather than a conventional finite hidden Markov chain.
H6. Nonstationary mixture: the apparent state is heterogeneity across hands, packets, folios, Currier regimes or local production episodes.
H7. Exogenous/source innovations: even the best autonomous observable-history process fails free generation and requires unresolved innovations.
H8. Mere capacity: a flexible model can predict better without recovering the structural anomaly. This is a null/negative hypothesis and must be rejected at every rung.

## 1. Academic design principles

### Predictive-state representations / observable operator models
Singh, James & Rudary (2004), "Predictive State Representations: A New Theory for Modeling Dynamical Systems", DOI 10.48550/arxiv.1207.4167.
Gordon & Boots (2018), "Spectral Approaches to Learning Predictive Representations", DOI 10.1184/r1/6723110.v1.

Use: define state through predictions of observable futures rather than semantic or hidden-state labels; use spectral/subspace identification only after the observable representation is adequate.

### Computational mechanics / causal states
Shalizi & Crutchfield, "Computational Mechanics: Pattern and Prediction, Structure and Simplicity", DOI 10.1023/A:1010388907793.
Brodu & Crutchfield, "Discovering Causal Structure with Reproducing-Kernel Hilbert Space epsilon-Machines", DOI 10.1063/5.0062829.

Use: histories should be grouped only when they imply the same future distribution. Kernel causal-state recovery is a late rung, after simpler observed-channel and timescale explanations fail.

### Predictive information bottleneck / predictive rate-distortion
Still (2014), DOI 10.3390/e16020968.
Marzen & Crutchfield (2016), DOI 10.1007/s10955-016-1520-1.
Hahn & Futrell (2019), DOI 10.3390/e21070640.

Use: complexity is governed by retained predictive information, not by arbitrary state count. Finite-history clustering can fail for long-memory and hidden-state processes; use causal-state/variational bounds where necessary.

### Multiscale and structured latent-state models
Sidrow et al. (2021), "Modelling multi-scale, state-switching functional data with hidden Markov models", DOI 10.1002/cjs.11673.
Michelot, "hmmTMB: Hidden Markov Models with Flexible Covariate Effects", DOI 10.18637/jss.v114.i05.

Use: distinguish fast token-level dynamics from slow line/entry/page state and test observed covariates before inventing latent state.

### Factorial/distributed state
Ghahramani & Jordan, "Factorial Hidden Markov Models", DOI 10.1023/A:1007425814087.

Use: if SELECT and FORM carry separable predictive factors, a single latent state can be the wrong model class; factorized state is licensed only after single-channel and multiscale rungs show complementary gains.

## 2. Global programme rules

### G1. Ecology firewall
Primary linear ladder populations:
- Herbal-A strict +P0
- Herbal-B strict +P0
- Balneological/Q13 strict +P0
- Recipes strict +P0

No Pharmaceutical labels, Zodiac circular text, Cosmological radial/circular text or Rosettes spatial inscriptions enter this ladder.

### G2. Physical-fold firewall
Five physical-bifolium outer folds remain canonical.
All model selection uses TRAIN / VALIDATION / untouched TEST separation.
No event is evaluated by a model trained on its own physical fold.

### G3. Outcome hierarchy
No model is "recovered" from held-out log-loss gain alone.
Every promoted model must satisfy:
1. positive held-out predictive gain;
2. complexity-matched dynamic-null qualification;
3. reduction or absorption of the frozen residual panel;
4. robustness to the relevant negative controls;
5. replication at the next promotion level.

### G4. Frozen residual panels
Primary legacy panel: VMS-ECOLOGY-1 15-D panel.
For Recipes additionally retain the RESIDREC4B q3 residual vector as the current target.
No adaptive feature additions after real outcomes.
Any new panel requires a new preregistration and synthetic qualification first.

### G5. Model capacity controls
Every flexible model gets:
- architecture-frozen dynamic q0/q1 nulls;
- wrong-channel/permuted-channel controls when new observables are introduced;
- boundary-shuffled controls when slower state is introduced;
- capacity-matched ablations.

### G6. Promotion levels
P0 — Recipes discovery/qualification.
P1 — Balneological replication.
P2 — Herbal-A and Herbal-B stress test.
P3 — TTLI cross-transcription replication where the representation is supportable.
P4 — free-generative closure.

Failure at P1 prevents manuscript-wide promotion but does not invalidate a Recipes-specific mechanism.

### G7. Interpretation firewall
No state gets semantic labels.
No language/plaintext/source-vocabulary claim.
No historical wheel/controller claim.
Operator/state interpretation is permitted only after P3 and generative closure.

---

# THE LADDER

## RUNG 0 — Instrument and recoverability qualification

Question:
Can each candidate architecture recover its own planted mechanism through the frozen SELECT/FORM renderer at Voynich-sized sample counts?

Synthetic panel:
- planted extra FORM channel only;
- planted slow line state only;
- planted semi-Markov entry state;
- planted two-factor process;
- planted predictive/unifilar non-HMM process;
- pure q0/q1 null;
- wrong-family controls.

Required before every later family:
- >=90% detection at target effect size;
- <=5% false positive under q0/q1;
- correct family discrimination >=80% where family identity is a claim;
- parameter/state recovery is NOT required unless explicitly claimed.

If an instrument cannot recover its planted family, that rung cannot interpret a Voynich failure.

Decision:
INSTRUMENT_QUALIFIED / INSTRUMENT_UNDERPOWERED.

---

## RUNG 1 — Observable-channel sufficiency: does FORM history carry the missing state?

Hypothesis H1.

Rationale:
The current q3 sees only K12 opener-class history. The strongest remaining error is calibration/Pearson structure, consistent with omitted predictive information in another observable channel.

Frozen channel ladder, added one at a time:
C0 K12 opener only (current q3 reference).
C1 previous final FORM piece / final K12-family.
C2 piece-count / token-length bucket.
C3 coarse token family = opener x final x length bucket.
C4 recent FORM-family occupancy/repertoire over last 2/3/5 events.
C5 multichannel low-rank interaction between opener and final/shape history.

No raw whole-token identity initially; it is too high-capacity and is reserved as a sensitivity.

Models:
- q1 multichannel variable-order context.
- q3-sized GRU with identical hidden dimension/epochs but multichannel input.
- low-rank bilinear context model as an interpretable competitor.

Critical negative controls:
- independently permute added FORM channel within line-position x section x opener-class strata;
- lag-shift FORM channel by a random within-line offset;
- preserve marginal FORM frequencies.

Primary tests:
A. incremental TEST gain over K12-only q3;
B. reduction in real 15-D D2;
C. specific reduction in Pearson-energy and lag-2 surprise/repeat coordinates;
D. channel-shuffle null.

Promotion gate:
MULTICHANNEL_OBSERVABLE_STATE if:
- gain null-qualified;
- D2 falls by >=50% relative to q3 OR enters calibrated residual envelope;
- real improvement exceeds all channel-shuffle controls;
- >=4/5 folds positive.

Branch:
- If C1/C2 suffice -> H1 supported; go to Rung 6 complexity governor.
- If only C4/C5 suffice -> distributed/context interaction likely; go to Rung 3.
- If none suffice -> Rung 2.

---

## RUNG 2 — Timescale ladder: is the missing variable slower than token-local history?

Hypotheses H2/H3/H6.

Fast-to-slow state hierarchy:
T0 current line-reset q3.
T1 carry state across physical lines within the same paragraph/star entry; reset at entry/paragraph boundary.
T2 line-level summary state derived only from previous lines in same entry/paragraph.
T3 entry/paragraph-level state carried across lines.
T4 folio-local slow state.
T5 physical-bifolium / packet state.

Observed covariates tested before latent state:
- line index within entry/paragraph;
- previous-line K12/FORM composition;
- entry length;
- folio/quire/bifolium;
- Currier A/B;
- hand where externally available;
- Recipes exact star-entry coordinate;
- known mixed-hand f115r flag only as an observed covariate, never inferred from outcome.

Models:
A. observed-covariate recurrent model;
B. hierarchical recurrent model with token state + line state;
C. hidden semi-Markov slow state with explicit dwell distribution;
D. switching regression/HMM with covariate-dependent transitions.

Semi-Markov is preferred over ordinary HMM if dwell-time structure is detected.

Negative controls:
- shuffle whole lines within the same folio/entry-length stratum;
- shuffle entry order within folio;
- preserve within-line sequences while destroying slow order.

Promotion gate:
SLOW_STATE_REQUIRED if:
- cross-boundary state improves held-out prediction beyond Rung1-best;
- slow-order shuffling destroys that gain;
- D2 reduction >=50% or calibrated absorption;
- effect survives excluding f115r in Recipes.

If observed covariates absorb the signal, stop: no latent slow state is licensed.
If latent state adds beyond observed covariates, proceed to Rung 3 or 4 depending on channel results.

---

## RUNG 3 — Factorial/distributed state: do two processes jointly generate the surface?

Hypothesis H4.

Licensed only if:
- Rung1 finds FORM-channel gain and Rung2 finds independent slow-state gain, OR
- neither single extension absorbs residual but their residual signatures are complementary.

Candidate factorization:
Factor A: fast SELECT/repertoire process over K12 history.
Factor B: FORM/shape process over final-piece/family history.
Optional Factor C: slow line/entry regime.

Models:
- two-chain factorial HMM with q0 offset;
- coupled state-space model with additive low-rank logits;
- hierarchical factor model where slow state modulates fast transition operator.

Compare against:
- single-chain HMM with matched total state count;
- matched-parameter GRU;
- additive no-interaction factor model.

Primary question:
Does factorization outperform an equally large monolithic state representation out of fold and absorb the residual?

Promotion:
FACTORIAL_STATE_SUPPORTED only if:
- beats monolithic matched-capacity alternative;
- both factors have nonzero unique conditional predictive information;
- ablating either factor restores a distinct part of the residual;
- dynamic-null qualified.

Otherwise reject factorial interpretation.

---

## RUNG 4 — Predictive causal-state recovery: is conventional latent state the wrong representation?

Hypothesis H5.

Only open after the best observed-channel/timescale representation is frozen.

Methods:
1. regularized past-future CCA / spectral subspace identification;
2. OOM/PSR operators in the qualified observable space;
3. kernel causal-state reconstruction / kernel epsilon-machine;
4. finite unifilar causal-state approximation as an interpretable compression.

Past features:
qualified multichannel observables and boundary state only.

Future:
1-5 step future distributions plus registered residual observables.

Key criterion:
histories are merged by predictive equivalence, not by superficial similarity.

Tests:
- rank/dimension stability across physical folds;
- future prediction;
- residual absorption;
- state/operator stability under bootstrap;
- leave-one-ecology-out operator transfer.

Promotion:
PREDICTIVE_STATE_RECOVERED if:
- predictive state transfers or has a stable ecology-conditioned shared subspace;
- residual absorbed;
- state dimension stable over resampling;
- kernel/linear versions agree on the dominant predictive directions.

Do not infer a hidden state count from centered singular values alone.

---

## RUNG 5 — Nonstationarity and mixture audit

Hypothesis H6.

This is a guard against mistaking manuscript heterogeneity for dynamics.

Fit the Rung4-best representation with:
- shared architecture + hand-specific parameters;
- Currier-specific parameters;
- folio/bifolium random effects;
- ecology-specific transition operators;
- mixture-of-regimes model.

Use hierarchical shrinkage: shared global operator + local deviation.

Questions:
- Is there one common architecture with ecology-specific parameters?
- Are Recipes/Balneo genuinely a shared regime?
- Does Herbal differ because of transcription/FORM representation rather than mechanism?

Gate:
NONSTATIONARY_SHARED_ARCHITECTURE if hierarchical shared model predicts held-out ecologies/folios better than both:
(a) one universal model;
(b) completely separate ecology models.

If completely separate models win, stop manuscript-wide mechanistic claims.

---

## RUNG 6 — Predictive information / complexity governor

Applies to the best candidate from Rungs1-5.

Methods:
- predictive information bottleneck;
- predictive rate-distortion;
- variational NPRD if finite-history methods are unstable.

Measure:
- predictive information retained at 50/80/90/95%;
- bits of state required;
- marginal gain per state bit;
- horizon-specific information.

Purpose:
find the minimum sufficient predictive representation.

Hard rule:
the chosen mechanism is the smallest representation that:
- retains >=90% of qualified predictive gain;
- remains inside residual calibration envelope;
- survives P1/P2 replication.

This prevents state proliferation.

---

## RUNG 7 — Autonomous-generator versus unresolved innovation test

Hypothesis H7.

At this point the surface mechanism should be fixed.

Two tests:

### 7A conditional closure
Does the model whiten the one-step innovations?
Frozen diagnostics:
- residual autocorrelation;
- conditional calibration/Pearson energy;
- recurrence;
- transition residual;
- surprise coupling;
- horizon 1-10 predictive residuals.

### 7B free-generative closure
Generate complete section text from the model without teacher forcing.

Must pass:
- frozen 15-D residual panel;
- relevant 84-metric generative harness;
- line/entry length distribution;
- FORM conformity;
- recurrence spectrum;
- opener/final structure;
- section-specific repertoire;
- novel-form behavior where FORM generation is in scope.

If one-step prediction passes but free generation fails:
GENERATOR_INCOMPLETE.

Only if both pass may the autonomous surface generator be considered sufficient at the tested resolution.

If no autonomous model passes despite qualified state recovery, retain:
UNRESOLVED_INNOVATION_REQUIRED.
This does not prove an external semantic source; it only says the tested autonomous observable-history generator is insufficient.

---

## RUNG 8 — Cross-ecology and cross-transcription promotion

Every surviving mechanism must pass:
P1 Balneological.
P2 Herbal-A/B.
P3 TTLI where representation mapping is well-defined.

Three allowed outcomes:
A. universal architecture + shared parameters;
B. universal architecture + ecology-specific parameters;
C. ecology-specific mechanisms.

Only A/B justify manuscript-wide running-text claims.

Cross-transcription failures trigger representation audit before mechanism rejection if the disputed feature is known to depend on transcription segmentation.

---

## RUNG 9 — Mechanistic perturbation and interpretation

Only after Rung7 generative closure and Rung8 replication.

Perturb recovered state/operator and measure:
- which K12 families change;
- which FORM families change;
- recurrence/surprise consequences;
- boundary behavior;
- section/ecology effects.

State labels remain operational:
"high-return / low-surprise state", etc.

No semantic interpretation until a state has:
- cross-fold stability;
- cross-transcription stability;
- generative consequences;
- physical/codicological correlate or independently interpretable observable signature.

---

# 3. Execution order from current state

CURRENT NODE:
Recipes q3 is predictive but leaves Z_D2 ~17.42.

Therefore execution is:

L1 = Rung1 multichannel observable sufficiency in Recipes.
If L1 absorbs -> replicate Balneo -> Herbal -> complexity governor -> generator.
If L1 helps but does not absorb -> Rung2 slow-timescale ladder.
If L1 does nothing -> Rung2 immediately.

After Rung2:
- channel + slow both independently useful -> Rung3 factorial.
- slow only -> Rung4 using slow-qualified observables.
- neither -> Rung4 directly with current observables.

Rung5 is mandatory before any manuscript-wide interpretation.
Rung6 is mandatory before choosing a state complexity.
Rung7 is mandatory before calling anything a generator.
Rung8 is mandatory before calling anything manuscript-wide.
Rung9 is interpretation only.

# 4. Standard success table

Each rung must report:

1. held-out bits/event vs parent;
2. fold-wise sign;
3. dynamic-null Z and add-one p;
4. residual D2 and calibrated Z_D2;
5. percent D2 reduction vs parent;
6. coordinate-wise residual localization, descriptive unless separately preregistered;
7. negative-control result;
8. model complexity;
9. replication level achieved;
10. exact licensed conclusion.

# 5. Stop rules

STOP A:
Predictive gain without residual reduction -> useful predictor, not mechanism.

STOP B:
Residual reduction without null-qualified predictive gain -> likely descriptive overfit.

STOP C:
Effect disappears under channel shuffle/boundary shuffle -> channel/timescale artifact; record and do not escalate model class.

STOP D:
A simpler rung absorbs >=90% of residual anomaly -> do not open more complex rungs.

STOP E:
A model fails synthetic family recovery -> no Voynich interpretation.

STOP F:
Generative closure fails -> do not infer source semantics or decoding architecture.

# 6. Why this ladder is academically preferable

It orders hypothesis classes by identifiability:
observed variables before latent variables,
observed timescale before hidden timescale,
single process before factorial process,
finite structured state before kernel/nonparametric causal state,
prediction before generative explanation,
and complexity only after predictive sufficiency.

This directly reflects the lessons of PSR/OOM theory, computational mechanics, predictive rate-distortion, multiscale HMMs and factorial state models.

It also addresses the failure mode already demonstrated by RESIDREC4B:
a model can be strongly, genuinely predictive and still be a very poor explanation of the process.

