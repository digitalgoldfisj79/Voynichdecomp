# VBM v11 — Structural Constraint Programme closeout

Date: 2026-09-02
Job: `6a97c0510718b0f6d890fcfc`
Hardware: HF `cpu-upgrade`
Status: **COMPLETED**
Protocol: `VBM_V11_STRUCTURAL_CONSTRAINTS_PROTOCOL.md`
Implementation specification: `VBM_V11_IMPLEMENTATION_SPEC.md`

## Frozen programme decision

`V11_MULTIPLE_STRUCTURAL_CONSTRAINTS_JUSTIFY_V12_SYNTHETIC_MODEL`

Exactly two of five preregistered primary structural branches passed: **B** and **C**. A, D and E failed their frozen gates. No Voynich plaintext search was opened and no GPU was used.

This decision permits only a new, explicitly specified and more restrictive **synthetic generative model** incorporating independently supported constraints. It does not validate VBM, identify a language, or permit a plaintext fit.

## Corpus

Unchanged Joachim-exact Q0b parser, H1/C1 firewalls excluded:

- total valid segments: 4,930
- TRAIN segments: 3,887
- INTERNAL_HOLDOUT segments: 1,043
- nonempty transcription lines: 4,929

## A — contextual five-class bridge homophones

Verdict: `A_NO_FIVE_CLASS_EVIDENCE`

The Voynich bridge contexts themselves were highly structured:

- eligible bridge types: 89
- k=5 silhouette: `0.148947`
- context-permutation null p: `0.00990099`
- TRAIN-A / TRAIN-B k=5 partition ARI: `0.950640`
- split-stability null p: `0.00990099`

However the method failed its mandatory known-answer synthetic calibration. Six true five-vowel-class VBM synthetic replicates yielded ARIs:

`0.7286, 1.0000, 0.2540, 0.2991, 0.1213, 0.2131`

Median ARI = `0.276526`, below the frozen `0.70` requirement, and only 2/6 exceeded 0.50 rather than the required >=4/6.

Also, k=5 was not close to the best observed cluster count under the frozen tolerance: k=2 silhouette was `0.191723`, versus k=5 `0.148947`.

Therefore the striking Voynich split stability cannot be interpreted as recovery of five latent vowel classes. It demonstrates stable bridge-context organisation only.

## B — compositional nucleus morphology / e-family ladders

Verdict: `B_E_LADDER_COMPOSITIONALITY_SUPPORTED`

148 nucleus types met the >=20 TRAIN-occurrence threshold. There were 22 eligible e-ladder pairs: nuclei with the same skeleton after collapsing maximal `e+` runs to `E`, but different total numbers of `e` glyphs.

Full TRAIN:

- observed median Jensen-Shannon context distance: `0.245251`
- matched-null mean: `0.377407`
- null SD: `0.028642`
- z = `4.61413`
- empirical p = `0.00009999`

TRAIN-A:

- observed median JS: `0.250676`
- null mean: `0.348543`
- z = `5.03104`
- p = `0.00009999`

TRAIN-B:

- observed median JS: `0.244810`
- null mean: `0.343194`
- z = `5.51397`
- p = `0.00009999`

The effect therefore survives the two preregistered independent TRAIN halves with substantial margin.

Narrow interpretation: nucleus strings that differ principally in e-run multiplicity occupy unusually similar bridge contexts compared with matched unrelated nuclei. This is evidence that the `e` family participates in a compositional/morphological system rather than every whole nucleus behaving as an unrelated atomic type.

It does **not** establish Joachim's proposed consonant values, nor that e-count encodes consonant-cluster length. Branch D tested simple length rules separately and failed.

## C — factorisation of bridges into visible halves

Verdict: `C_BRIDGE_HALVES_FACTORISE`

The additive-half model predicts INTERNAL_HOLDOUT adjacent-nucleus context from the two visible halves `R` and `L` separately. It was compared with a pooled baseline and an unrestricted full-pair bridge model.

Left-nucleus context:

- pooled M0: `-2.577528`
- full-pair MPAIR: `-2.058229`
- half-factorised MADD: `-2.017214`
- pair gain: `0.519298`
- additive gain: `0.560314`
- factorisation ratio: `1.07898`
- half-permutation null p: `0.00990099`
- null 99th-percentile additive gain: `0.519166`

Right-nucleus context:

- pooled M0: `-2.564289`
- full-pair MPAIR: `-2.098676`
- half-factorised MADD: `-2.068932`
- pair gain: `0.465612`
- additive gain: `0.495357`
- factorisation ratio: `1.06388`
- half-permutation null p: `0.00990099`
- null 99th-percentile additive gain: `0.236279`

Both frozen side gates passed.

Narrow interpretation: a bridge's visible right and left halves carry substantial reusable information about the adjacent nucleus contexts; treating each `R|L` bridge as an arbitrary indivisible symbol discards real structure. The constrained half model even slightly outperformed the full-pair model on HOLDOUT, consistent with useful regularisation.

Important limitation: because the bridge halves are themselves token-boundary glyph material, this factorisation can arise from Voynich's native positional/morphological grammar without implying that the pair encodes a vowel. It supports a compositional boundary mechanism, not five vowel values.

## D — simple morphology to consonant-run length

Verdict: `D_NO_SIMPLE_MORPHOLOGY_LENGTH_RULE`

The same rule won on TRAIN and HOLDOUT (`D2`: `1 + min(4, longest e-run)`), but it did not beat the familywise matched permutation null.

HOLDOUT D2 scores:

- German run-length model: `-1.412933`, familywise p = `0.49151`
- Italian run-length model: `-1.400378`, familywise p = `0.66134`

Thus the e-family compositionality from Branch B cannot be promoted to the specific claim that visible e-run length directly determines plaintext consonant-run length under any of the five frozen simple rules.

## E — line-edge closure

Verdict: `E_NO_SEQUENTIAL_CLOSURE_EVIDENCE`

TRAIN sequential closure:

- n = 3,718
- internal-bridge support rate = `0.980097`
- support null p = `0.00059994`
- mean internal-bridge log probability = `-4.997028`
- log-probability null p = `0.063094`

The support statistic was exceptional, but the mean probability statistic failed the joint gate. INTERNAL_HOLDOUT also failed the required replication direction:

- HOLDOUT support = `0.978937` vs null median `0.976931`
- HOLDOUT mean log probability = `-5.014523` vs null median `-5.002246`

The latter is worse than null, so sequential closure is not supported.

Cyclic closure was also negative (`support p=0.7311`, `logp p=0.0890`).

## Overall interpretation

V11 changes the picture in one specific way. V10 showed that an unconstrained dictionary of arbitrary whole nuclei and arbitrary bridge pairs is non-identifying. V11 finds evidence that **both of those objects contain internal structure**:

1. `e`-family-related nuclei have reproducibly similar boundary contexts (B);
2. bridge behaviour is substantially factorisable into the two visible token-edge halves (C).

Those two facts point away from Joachim's current arbitrary dictionary formulation and toward a smaller **compositional transducer** in which nucleus morphology and boundary halves are reusable components.

The negative branches matter equally:

- there is no validated recovery of exactly five bridge-context classes (A);
- no tested simple e-count/e-run rule determines consonant-run length (D);
- open line edges cannot presently be replaced by sequential or cyclic bridge closure (E).

Accordingly V11 does not rescue the published VBM. It identifies two constraints that a new model would have to exploit and three tempting simplifications it is not allowed to assume.

## Permissible next step

Per preregistration, the only justified continuation is **V12 synthetic model construction**. V12 must be specified before any new Voynich plaintext output and should incorporate only the supported structural ideas:

- compositional nucleus representation with an explicit e-family operator;
- bridge/boundary value generated from visible half-components rather than independent pair lookup.

It must not assume five vowels, a direct e-count-to-consonant-length map, or cross-line closure unless independently re-established.

A V12 model must first prove synthetic identifiability and key/plaintext recovery under its own generating mechanism. Only after that could a separate Voynich TRAIN/HOLDOUT test be considered.
