# Palaeographic Instrument Gate v0.1 — results

Date: 2026-08-08
Branch: `experiment/palaeo-instrument-gate-v0.1-20260808`

Binding documents:

- `PROTOCOL.md`
- `AMENDMENT_001_G0C_BLOCKING.md`
- `AMENDMENT_002_G0B_BACKBONE_MATCH.md`

No Voynich-to-comparator affinity score was computed.

## G0A — minimum isolated-shape competence

Job: `6a7784bd3e1f34a7e32bf491`

The job completed G0A and printed its result before later terminating on a JSON serialization error in the original G0C output. The G0A values are unaffected by that later serialization defect.

Dataset: sklearn handwritten digits, 1,797 shapes; held-out test n=540; fixed 72×64 preprocessor images.

| Representation | Balanced accuracy |
|---|---:|
| frozen naive 8-feature morphology | 0.8123365244 |
| DINOv2-small CLS | 0.9759156148 |

Difference: `+0.1635790904`.

Paired-bootstrap accuracy-difference 95% CI (2,000 replicates):

`[+0.1296296296, +0.1944444444]`

Frozen rule: PASS when lower CI endpoint >=0.

**G0A = PASS.**

Interpretation: DINOv2-small is not generically incapable of isolated handwritten-shape discrimination, and the unrecovered historical claim that an earlier 72×64 test was 15.3 null-SD below the naive descriptor cannot be treated as a universal property of the backbone. This floor test is not a medieval-palaeography validation and does not reproduce the missing 2026-08-01 artifact.

## G0B — DINOv2-matched manuscript/background confound

Job: `6a7786ecda2af92a634efaeb` — completed.

Exact Stage-5 confound design, same manifest and preprocessing, with DINOv2-small substituted prospectively under Amendment 002.

Acquisition:

- requested crops: 59;
- eligible crops: 58;
- eligible manuscripts: 10;
- one transient acquisition error: Wikimedia returned HTTP 429 for the Carrara Herbal violet crop;
- all ten manuscripts retained >=2 independent source pages;
- acquisition coverage remained above the inherited 80% minimum.

| Variant | Page-held-out manuscript macro OVR AUC | Top-1 | Frozen decision |
|---|---:|---:|---|
| `rgb_norm_v1` | 0.7761776693 | 0.5862068966 | FAIL |
| `gray_bgdiv_v1` | 0.7527084820 | 0.5172413793 | FAIL |
| `inkmask_v1` | **0.7267968228** | 0.4827586207 | **FAIL** |

Frozen thresholds: PASS <=0.65; CAUTION >0.65 and <=0.70; FAIL >0.70.

**G0B = FAIL.**

The primary masked result is 0.0268 AUC above the failure boundary. It is therefore a formal failure under the frozen rule but should be described as a modest-margin failure, not an enormous effect.

For comparison only, the earlier sealed DINOv3 Stage-5 `inkmask_v1` AUC was 0.7905697030. The backbone change reduces the confound but does not clear the gate.

## G0C — digitisation-source leakage stress test

Initial job: `6a7784bd3e1f34a7e32bf491`. Its crop-level permutation was prospectively invalidated before any G0C output was observed because digitisation source is manuscript-level.

Corrected diagnostic job: `6a77863c3e1f34a7e32bf4a7` — completed.

Panel:

- 23 masked crops;
- 5 independent manuscripts;
- BSB: 17 crops;
- DigiVat: 6 crops;
- corridor-core domain: 16 crops;
- Bavarian-control domain: 7 crops;
- zero acquisition failures.

Training/testing swaps geography so a source classifier cannot succeed merely because source and geography are synonymous:

1. train BSB-v-DigiVat on corridor manuscripts, test on Bavarian manuscripts;
2. train on Bavarian manuscripts, test on corridor manuscripts.

### Directional source AUC

| Representation | corridor→Bavaria | Bavaria→corridor | Mean | Pooled AUC | Status |
|---|---:|---:|---:|---:|---|
| DINOv2-small CLS | 0.4000 | 0.3125 | 0.35625 | 0.42157 | INDETERMINATE |
| naive morphology-8 | 0.3000 | 0.45833 | 0.37917 | 0.45098 | INDETERMINATE |
| masked 32×32 pixels | 0.2000 | 0.16667 | 0.18333 | 0.21569 | INDETERMINATE |

No representation meets the Amendment-001 diagnostic-leakage criterion (mean directional AUC >=0.70 and each direction >=0.65).

**G0C = INDETERMINATE.**

The observed mapping reverses rather than transferring across geography. This small panel therefore does not support the specific claim that a stable BSB-v-DigiVat acquisition signature is straightforwardly recoverable from masked ink. But only five independent manuscripts are present and glyph identity is not fixed, so absence of transfer cannot certify source invariance.

The scientifically required follow-up remains a larger manuscript-blocked fixed-glyph/homologous-form source-crossed test with an equivalence criterion around chance.

## G0D — design identifiability

See `G0D_IDENTIFIABILITY_AUDIT.md`.

**G0D = REPAIRABLE / NOT YET PASS.**

The historical comparison is not intrinsically source-nested. At least two useful same-platform crossings can be assembled for Padua/Veneto versus German material (BSB/MDZ and DigiVat), and a British-Library crossing is plausible for Veneto/Paduan-associated versus Lombard material. Actual hand-level matching, balancing and metadata verification remain prerequisites.

## Programme verdict

| Gate | Result |
|---|---|
| G0A shape floor | **PASS** |
| G0B masked manuscript-confound | **FAIL** |
| G0C source leakage | **INDETERMINATE** |
| G0D identifiability | **REPAIRABLE / NOT PASS** |

# FINAL: NOT UNLOCKED

The current DINOv2-small representation is **not licensed for cross-manuscript Voynich provenance inference**.

The failure mode is more specific than the pre-run suspicion:

- the model plainly can encode isolated handwritten shape (G0A passes strongly);
- simple cross-geography BSB-v-DigiVat source prediction does not transfer in this small masked panel (G0C does not show diagnostic source leakage);
- nevertheless the same masked embeddings retain manuscript identity above the frozen nuisance threshold (G0B fails).

Thus the present evidence does **not** justify saying that the instrument merely reads scanner source. It shows that it reads some combination of manuscript-specific visual information strongly enough to contaminate palaeographic attribution. Possible contributors include writing content/form distribution, page/crop construction, ink/threshold morphology, acquisition interactions, or genuine manuscript-specific hand/style signal; the current experiment does not identify which.

No historical inference about Padua, Germany, Lombardy/Pavia, the Alpine corridor, or Beinecke MS 408 follows from this instrument result.

## Next admissible experiment

Do not collect a broad provenance specimen panel yet. Build a small source-crossed homologous-form calibration in which identical Latin glyph/form classes are sampled across multiple manuscripts on at least BSB/MDZ and DigiVat. Test:

1. form identity competence against the naive descriptor;
2. manuscript identity conditional on form;
3. digitisation-source identity conditional on form;
4. writer/style retrieval only after (2) and (3) are within preregistered nuisance bounds.

Only a representation that passes all four can be used on Candus, Fita, Clm 78/Holkham or VMS for localisation.
