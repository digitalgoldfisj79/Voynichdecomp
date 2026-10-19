# Grammar-aware inverse: running results — 2026-10-19

## Status and controlling documents
- Programme: [PROGRAMME_PREREG.md](PROGRAMME_PREREG.md).
- Frozen prior constraints: `research/SELECT_FORM_INTERFACE_SPEC_20261004.md`, `research/INVERSE_RECOVERY_CONTRACT_20261004.md`, `research/Q13_INFORMED_SELECT_SOURCE_SPEC_20261004.md`, and `research/VMS_RESIDUAL_RECOVERY_LADDER_20261008.md`.
- Prior brute-force stage: [f75r word-first 63-billion assignment study](https://github.com/digitalgoldfisj79/Voynichdecomp/blob/experiment/f75r-progressive-cipher-ladder-20261018/research/f75r_cipher_ladder_20261018/RUNNING_RESULTS.md) produced many dictionary matches without linguistic coherence; structure-destroying controls could not validate meaning.
- Archived source: [q0_frozen_form_oracle.py](q0_frozen_form_oracle.py). Original strict FORM renderer pinned at `fa4321e75365d83c7ea9d07b0d6c24ee77c01b2f`, SHA256 `16050050c11c41d222902668e84fa6e185a589c03be9a78b19529258f4b653de`. Underlying corpus SHA256 `26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f`.
- Population at this Q0 stage is **synthetic only**; no real Voynich inversion, source-language guess or decoder output.
- The K0/strict-FORM socket has 57 exact pieces, 12 FORM classes, 8 collapsed continuation groups, 70 legal continuation edges, empirical FORM mean length ~2.482 pieces. Source may bias only legal ENTRY with strength 5 and a weak token-constant ROUTE preference strength 0.5; piece realization and exact STOP likelihood frozen. CONNECT and LINE_ENTRY are **not simulated in Q0** and remain to be wired in before any real running-text test.

## Q0 Stage: can structured content survive the renderer under oracle decoding?
Input design: 8 controller states; 32 or 64 possible **latent source item identities**; each is mapped into one of the 8 controllers. Source identities within a controller are indistinguishable from a single token, except to the extent sequence context separates them. All source families share the same FORM and SELECT renderer. LANG = random sparse reusable-item Markov source with frequency skew (a **proxy**, not natural language). NOTATION = near-deterministic cyclic source with strong syntax (intentionally easy). TABLE = independent item choices with Zipf-like frequencies. NONE = same latent dynamics but rendering fully source-independent.

Inference is *generously oracle-assisted*: supplied the TRUE source transition matrix, its item-to-SELECT-controller assignments and its controller emission parameters. Hidden source truth is used for scoring only; no fitting or restart selection. Log likelihood includes exact FORM normalized emission with exact END/CONTINUE decisions. Ordered-vs-within-block-shuffled evidence and source NMI are reported. The metric is over the last quarter of 800 synthetic tokens. Thus a Q0 success gives only a recoverability ceiling; blind success is **not** implied.

### Preflight and discovery
- [16-type one-seed smoke](results/q0_preflight_16types.json): LANG NMI .76575, TABLE .59045, NONE 0. This single case was a plumbing check, not a recovery claim.
- [32-type four-seed discovery](results/q0_oracle_32_4seeds.json): LANG median NMI .58156; NOTATION .90222; TABLE .55688; NONE 0.
- [64-type four-seed discovery](results/q0_oracle_64_4seeds.json): LANG median NMI .66828; NOTATION .89889; TABLE .49929; NONE 0.

### Frozen twenty-seed calibrated oracle panel
Fresh seeds 20264000–20264019, 800 tokens each, source identity recovery on heldout last 200 tokens per replicate; no architecture/hyperparameter tuning between cases. Exact numerical data and job URL:
- [32 latent source items](results/q0_oracle_32_20seeds.json)
- [64 latent source items](results/q0_oracle_64_20seeds.json)

| Source shape | 32 types: median NMI | 32 types: 10th%-ish NMI | 32 types: fraction NMI≥.6 | 64 types: median NMI | 64 types: 10th%-ish NMI | 64 types: fraction NMI≥.6 |
|---|---:|---:|---:|---:|---:|---:|
| LANG proxy | **0.62614** | ~0.561 | 13/20 | **0.67117** | ~0.631 | 19/20 |
| NOTATION rigid cycle | **0.88144** | ~0.840 | 20/20 | **0.93079** | ~0.881 | 20/20 |
| TABLE iid | 0.53528 | ~0.452 | 1/20 | 0.48827 | ~0.394 | 0/20 |
| NONE source-independent | 0 | 0 | 0/20 | 0 | 0 | 0/20 |

**Q0 oracle gate evaluation** — prereg ceiling criteria: median NMI ≥.70, 10th percentile ≥.60, and ≥90% NMI≥.60. LANG **fails** for 32 (all three) and 64 types (median alone). TABLE **fails**; NONE **correctly gives no source recovery**. NOTATION **passes Q0 oracle-only** at both inventory sizes, but its deterministic-like cycle is an intentionally easy source class, not a historically appropriate hypothesis.

The source–observation order carries oracle sequential information beyond shuffled order in structured types:
- LANG order effect ~**+0.1954 bits/token (32)**, ~**+0.1602 bits/token (64)**.
- NOTATION ~+1.1547 / +0.9364 bits/token.
- TABLE and NONE essentially zero.
These use the TRUE A and SELECT emissions and should not be extrapolated to real-text heldout bits, nor interpreted as calibrated source-family classification.

### Disposition and what changes
1. **Transfer of source *identity* through eight collapsed SELECT controllers is bottlenecked** for the tested weakly predictable language-like and table-like synthetic sources, even under oracle inference. This is a **family/instrument limitation**, not evidence Voynich lacks linguistic content.
2. A structured code with extraordinarily strong transitions can transmit individual source identities under the same renderer, but this oracle result says nothing about whether Voynich is such a code; the next model needs **matched-complexity controls**.
3. Recoverable units may be **source equivalence classes**, source recurrence relations, or conditional likelihoods, not recoverable alphabet letters or each underlying word. The proposal is to test identity-vs-class-vs-relation separately, without pretending the full source sequence is identifiable.
4. **DO NOT launch blind real-text inversion** of LANG/TABLE at these inventory/difficulty settings. Pseudo-language SOURCE-A and notation SOURCE-N are not authenticated historical corpora.
5. Before real Voynich work, instrument qualification needs larger/informative realistic vocabularies, fixed structural nuisances including CONNECT and LINE_ENTRY, strict bifolium split, 20+ fresh blinded encoder instances, >=90% planted-family sensitivity and <=5% false positives under grammar-only controls.
6. Treat this Q0 experiment as **first phase instrument calibration**, not a source-family adjudication on the actual manuscript. The NMI pass of NOTATION is not a permission to name or translate its states.

### Next controlled experiment
A Q1 blind synthetic **class/relation** recovery ladder:
(A) first estimate 8 SELECT-equivalence classes at the source socket;
(B) decode source-type equality/inequality and next-source-category predictive information;
(C) only if these hold, increase to recover individual source types;
(D) distinguish LANG vs NOTATION vs TABLE after **equalizing entropy rate, lexical occupancy, average FORM length, and source controller separation**, instead of giving NOTATION an obviously easier deterministic cycle;
(E) use exact forward likelihood for final ranking, not post-hoc dictionary hits.

Gate failure stops the relevant branch; keep the full record and don't respond by adding unregistered cipher flexibility. No GPU-heavy extension is justified before blind synthetic qualification.

## Q1 — first blind SELECT-category recovery instrument (2026-10-19)
Design, algorithm, validation-only selection and nulls are frozen in [Q1_BLIND_CATEGORY_PREREG.md](Q1_BLIND_CATEGORY_PREREG.md) and [q1_blind_category.py](q1_blind_category.py). Uses exact FORM log probability incl. END, 8 latent SELECT groups, separate 1200/400/400 TRAIN/VAL/TEST synthetic chains, IID vs HMM family and L2/restart selection on VAL only. Source item and controller truth sealed until TEST interpretation; K=32 source item types, each belongs to one of 8 controllers.

**Discovery two seeds per family; NOT confirmation**:
- LANG: NMI .45625, same-controller AUC .76697; sequential heldout gain +.02518 bits/token.
- NOTATION: NMI .58725, AUC .86073; sequential gain +.09401.
- TABLE: NMI .64765, AUC .89511; sequential gain −.02206.
- NONE: NMI .04003, AUC .48751; sequential gain −.00530.

Evidence files: [LANG](results/q1_blind32_pilot_lang.json), [NOTATION](results/q1_blind32_pilot_notation.json), [TABLE](results/q1_blind32_pilot_table.json), [NONE](results/q1_blind32_pilot_none.json).

Verdict: **instrument fails NMI identification** for the structured families, and uses HMM states partly to fit sequence context rather than stable controllers. Cannot promote to Voynich. Do not count this as proof that source identity is absent. A prospective factorized alternative was then registered rather than posthoc tuning Q1.

## Q1B — emission-only categories, then freeze emissions and fit dynamics (fresh 20-seed panel)
Protocol: [Q1B_FACTORISED_BLIND_PREREG.md](Q1B_FACTORISED_BLIND_PREREG.md); code [q1b_factorised_blind.py](q1b_factorised_blind.py) commit `bc2adbe4504ba5a03d9350c7aa815f60ce32192f`. First 2 discovery seeds per family are archived separately as `results/q1b_blind32_pilot_*.json` and excluded. Confirmation seeds 20269000–20269019 per family (+10k family offset); 32 source types, 8 controller classes, 1200/400/400 TRAIN/VAL/TEST, fixed `l2` and restart bank, 8→8 transition fit on *unlabeled* TRAIN and model selection through VAL exact likelihood only.

**Frozen-panel confirmatory outcomes, all 20 seeds per family:**

| Synthetic family | Blind class NMI median | NMI ≥ .60 | Same-SELECT pair AUC median | Pair AUC ≥ .80 | Within-same-SELECT source-item AUC median | HMM−IID test bits/token median | Test gain ≥ .02 bits |
|---|---:|---:|---:|---:|---:|---:|---:|
| LANG abstract structured sequence | **.532318** | 4/20 | **.844946** | 17/20 | .500741 | +.045966 | 16/20 |
| NOTATION rigid motif sequence | **.570858** | 7/20 | **.847809** | 16/20 | .503210 | +.098526 | 19/20 |
| TABLE IID repertoire | **.590681** | 7/20 | **.870456** | 17/20 | .485065 | −.014602 | 0/20 |
| NONE source-independent grammar | **.035801** | 0/20 | **.497868** | 0/20 | .502753 | −.012817 | 0/20 |

Primary artifacts with **each** case, including source seed and standalone HF job link:
- [LANG 20 seeds](results/q1b_blind32_20seeds_lang.json)
- [NOTATION 20 seeds](results/q1b_blind32_20seeds_notation.json)
- [TABLE 20 seeds](results/q1b_blind32_20seeds_table.json)
- [NONE 20 seeds](results/q1b_blind32_20seeds_none.json)

Note on logs: HF truncated the large aggregate `GAI_Q1B_FINAL` line at ~16 KB, so the saved JSONs were reconstructed exclusively from the **20 intact** `GAI_Q1B_CASE` records per job and validated against the exact expected source seeds; there was no case selection or rerun.

### What passed and failed
1. **Blind stable controller identity FAIL**: none reaches registered NMI median .70 or 90% test replications NMI ≥.60. Even oracle recovered underlying item types only in some stylized source conditions. The instrument is not licensed to invert real Voynich into source classes or words.
2. **Same-class relation detectability, PARTIAL**: same-controller pair AUC medians .845/.848/.870 vs grammar-only null .498. Positive families have only 16–17 of 20 cases above .80 (80–85% power under that threshold), **below a 90% reliability standard**. Pure IID TABLE has excellent AUC despite zero sequence signal; AUC does **not** establish natural language or encoded lexemes. Finer false-positive thresholds and context-matched negatives still needed for real text.
3. **Source identity after conditioning on true controller, FAIL/UNIDENTIFIABLE IN THIS EMISSION MODEL**: within-class same-source-item AUC is essentially random ~.50 in all arms. That is mathematically expected because multiple source items are **designed** to share exactly the same controller-dependent FORM emission. This is a planted identifiability limitation, **not** an empirical claim that Voynich words have no meaning. The superficially high *unconditional* same-source-item AUC ~.81 arises via shared controller and must not be interpreted as lexical recovery.
4. **Dynamic distinction QUALIFIED ONLY FOR THIS ARTIFICIAL NOTATION PROXY**: HMM vs IID test gain ≥.02 in NOTATION 19/20 and zero in TABLE/NONE (40 matched no-sequence negatives). LANG only 16/20, below prereg 90% sensitivity gate. Moreover synthetic NOTATION has near-deterministic transitions and is NOT entropy-matched to LANG; no cross-family historical inference.
5. **Renderer scope incomplete**: current Q1B freezes FORM incl. 70 legal class transitions and END, but CONNECT and LINE_ENTRY are not installed. Negative families do not yet match true historical sections, physical bifolia, transcription errors or position-conditioned graphotactics.
6. **No real Voynich inversion permitted**. This is a nontrivial instrument diagnostic, not a decipherment or a mechanism rejection. Stop this original blind NMI rung; design a separate label-invariant relation-specific instrument if needed, preregister new positive and hard matched nulls before use.

### Recommended scientific next step (not yet completed)
Target a directly measurable *relation* (posterior probability of shared SELECT constraints) rather than 32 individual source words. Synthetic calibration must:
- demand ≥90% power and ≤5% FPR on challenging controls preserving FORM/CONNECT/LINE_ENTRY and observable source frequencies;
- evaluate relation precision at frozen low FPR, not ROC AUC alone;
- confirm transfer across independent folio/physical bifolium boundaries and transcription layers;
- compare entropy-matched LANG/NOTATION/TABLE alternatives, not rigid notation against loose language;
- use sectioned running text only, never contaminate with Pharma labels, Zodiac circular or Rosettes spatial inscriptions.

Proceed to actual Voynich only after those qualification gates; otherwise retain the uncertainty and do not escalate arbitrary search complexity.

## 2026-10-19 — Step-back identifiability audit and revised build decision

After reviewing the frozen Q0/Q1/Q1B, Q13 informed-source, existing VMS residual recovery ladder and horseshoe retractions, the proposed **next relationship decoder is demoted**: another unsupervised SELECT-category similarity model would not discriminate source semantics from production/copyist structure. Q1B's within-controller identity AUC ~.50 is a formal result of the deliberately shared emission channel and cannot be overcome through brute-force search.

**Implemented:** [IDENT_GATE_PREREG_20261019.md](IDENT_GATE_PREREG_20261019.md), [ident_gate.py](ident_gate.py), [results/ident_gate_20seeds_20261019.json](results/ident_gate_20seeds_20261019.json), [IDENT_GATE_RESULTS_20261019.md](IDENT_GATE_RESULTS_20261019.md). Frozen FORM exact stochastic mechanism, synthetic-only causal-source versus production twins, plus matched anchor confounding stress test. Twenty independent seeds:
- Exact observable source/production twins **all 20/20 identical surface hashes**, full marginal log-likelihood differences exactly 0; intentional observational equivalence.
- Plant synthetic semantics-anchor score median +.008006 bits/line, classifier AUC .5720, just 6/20 runs above prereg +.02 threshold. Matched **production anchor** numerically identical by construction.
- Independent anchor control median −.006701 bits/line, AUC .5484, 2/20 score above .02. **Instrument power gate FAIL (6/20 < 18/20); empirical negative gate FAIL (2/20 > 1/20)**. Semantic interpretation additionally **NONIDENTIFIABLE** against exact production twin.
- Real CONNECT and LINE_ENTRY empirical rules are NOT actually installed in this synthetic feasibility test; the contextual variables are shared nuisance proxies. Do not mislabel it a complete grammar simulation or claim real manuscript inference.
- This is a negative methodological result, not evidence against encoded content.

**Independent anchor inventory** (read-only research-agent SQL in [anchor_audit_readonly.sql](anchor_audit_readonly.sql)): Balneological image objects 44 across 3 folios, zero reviewed; 92 Balneological masks over four folios geometric only; Herbal-A 272 objects across 105 folios, just 25 reviewed as "root"/"plant" classes (not semantic plant IDs); 416 manual masks across 150 folios geometric. De Balneis comparator has 37 text/image sections and 407 witness-section concordance rows, but there is NO independently certified alignment from that material to specific Voynich words.

**Deliverable for the correct next step:** [anchor_candidate_gate.py](anchor_candidate_gate.py), a read-only machine-checkable noncircularity gate for independent historical citations, a specific TEXT locus, reviewed picture/witness identity, competing semantic-vs-production predictions, same-register matched negatives and independent physical folds. Self-test PASS rejecting a generic "plant" class. This is an *input validation tool*, not proof of any identity.

**Decision:** Do not promote Q1B categories to lexemes. Do not rerun an upgraded unsupervised similarity solver under a new name. Register and validate one independently grounded text–meaning alignment that a production-only mechanism cannot readily mimic, then define a falsifiable heldout test. Without such an anchor, treat source-content-versus-production as nonidentifiable; predictive residual modeling remains descriptive.
