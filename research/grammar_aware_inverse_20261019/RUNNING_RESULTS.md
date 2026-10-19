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
