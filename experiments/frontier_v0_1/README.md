# Voynich Frontier Programme v0.1

**Branch:** `experiment/voynich-frontier-programme-v0.1-20260814`  
**Base:** `main` at `f5a2ac824c9f720738f6706360477273f467e563`  
**Status:** BUILT — NOT YET SCIENTIFICALLY FROZEN  
**Primary exam question:** GENERATED vs ENCODED, only where the relevant instrument has passed its own calibration gate.

## Purpose

This programme operationalises the surviving research gaps after a combined review of the ChatGPT and Claude research histories. It is deliberately not another generic generator search, language-ranking exercise, or historical-cipher resemblance tournament.

The programme has six modules:

1. **U1 — Transliteration uncertainty propagation**
2. **U2 — D'Imperio 1978 anomaly replication**
3. **U3 — Physical-unit latent-regime adjudication**
4. **U4 — Surface closure and residual payload capacity**
5. **U5 — Blind verbose-cipher recovery / recognition instrument**
6. **U6 — VTPS v0.2 external visual-instrument qualification**

U1–U4 form the core sequential programme. U5 and U6 are independent instrument-development branches and may proceed in parallel, but neither may be applied to the Voynich target before calibration.

## Scientific posture

The programme distinguishes:

- **measurement robustness**: does a result survive uncertainty in what was read?
- **mechanism adequacy**: can a compact prospective surface model predict and generate unseen physical units?
- **identifiability**: can competing mechanism classes be distinguished under fresh controls?
- **payload capacity**: after surface closure, what structured recurrent payload rates would be detectable?
- **physical level**: does the known textual persistence state have a handwriting/allographic correlate?

These are not interchangeable questions. A failure at an earlier gate blocks downstream interpretation.

## Frozen inheritance

The programme must reuse, not reconstruct ad hoc:

- `enriched_records.pkl` / `enriched_records.json` as the canonical enriched record layer;
- `Paper/p70c_full_spec_v1.json` and `Paper/p70c_full_layer.pkl` where the existing surface pipeline requires them;
- `voynich_transcriptions_slim.json` as the aligned multi-transliterator container;
- the inherited five physical-bifolium outer folds from the validated surface-closure programme;
- existing component-retention event definitions;
- the existing surface-closure/payload-capacity architecture unless a pre-target amendment is separately frozen.

Any source-hash mismatch is a Gate-0 failure.

## Run order

### Gate 0 — provenance and implementation hardening

Run:

```bash
python -m src.run_programme gate0 --repo-root /path/to/Voynichdecomp \
  --fold-manifest /path/to/inherited_physical_folds.json
```

Gate 0 checks hashes, required fields, section names, counts, multi-transliterator structure, fold integrity and forbidden leakage conditions. It writes an immutable freeze candidate and SHA-256 manifest.

### U1 — transliteration uncertainty

Build an empirical uncertainty lattice with one vote per independent transliterator family, not one vote per duplicated file version.

```bash
python -m src.run_programme u1-build \
  --slim /path/to/voynich_transcriptions_slim.json \
  --out results/u1
```

U1 does not select whichever transliteration gives the strongest Voynich result. It samples or bounds over the frozen uncertainty model and reports posterior/sign robustness and the observed-reading envelope.

Primary load-bearing effects for later adapters:

- lag-1 same-family midfix persistence;
- lag-1 same-family suffix persistence;
- RED1 and RED2;
- ED1 neighbourhood / contextual-equivalence statistics;
- within-line order measures;
- line-boundary reset measures;
- Currier separation as a descriptive sensitivity, not a target criterion.

A result becomes **measurement-robust** only if its preregistered sign/effect criterion survives the posterior draws, observed-reading envelope and physical-fold test.

### U2 — D'Imperio replication

The 1978 anomaly test is already preregistered. Its thresholds are inherited unchanged:

- 4–5 of the five historical anomalies replicated → CONFIRM;
- 0–1 → FALSIFY;
- 2–3 → AMBIGUOUS.

The page-number mapping is a separate prerequisite gate. No anomaly result may be interpreted while the mapping is unresolved.

### U3 — latent-regime adjudication

The unit is the physical bifolium. The target labels Currier, Davis hand and section are not inputs to the unsupervised regime model.

The model uses equal-weight feature families, grouped cross-validation, stability tests and synthetic one-state/multi-state calibration. It asks whether the manuscript's independently measured anomaly families are explained by:

- one shared regime;
- a small set of common latent regimes;
- feature-family-specific regimes;
- or no stable discrete regime structure.

Only after model selection is frozen are Currier, hand, section and codicology opened for descriptive association.

### U4 — surface closure and payload capacity

U4 inherits the existing causal design:

1. adjudicate exchangeable page composition versus order-sensitive production;
2. build the strongest compact surface realiser justified by admitted mechanisms;
3. require predictive and forward-generative closure;
4. calibrate payload detectors on matched hybrid surfaces;
5. open the Voynich residual only if both surface closure and payload calibration pass.

Payload-bearing and payload-free models must share the exact same surface renderer. The only difference is the explicitly charged information channel.

### U5 — blind verbose-cipher instrument

Target remains sealed until the solver can recover and recognise fresh, hidden-key examples from a preregistered verbose/nomenclator family inventory under source-disjoint testing.

### U6 — VTPS v0.2

VTPS v0.1 failed calibration and did not open the Voynich visual target. v0.2 must therefore introduce an independently calibrated handwriting/stroke instrument, preferably on known-writer medieval material with manuscript-level holdout. No target label may be used to tune the representation.

## Global decision rules

Every formal module must return exactly one of:

- `PASS`
- `FAIL`
- `ABSTAIN_UNRESOLVED`
- `CONTAMINATED_SUPERSEDED`

A failed calibration never counts as negative evidence about the manuscript.

One bounded repair is permitted only when it is explicitly preregistered, triggered by a declared diagnostic and completed before any target outcome is opened.

## Mandatory outputs

Every module must write:

- protocol or protocol reference;
- freeze/hash record;
- input manifest;
- deterministic seeds;
- machine-readable results;
- fold/bifolium/regime diagnostics;
- QA report;
- formal verdict separate from interpretation;
- retraction/amendment entry where required;
- `P(survives 60d)` field for every newly promoted analytical finding;
- reproducibility manifest.

## What the programme cannot establish

Even a complete pass cannot rule out a perfectly distribution-matched, one-time, externally keyed or cryptographically random payload. A positive residual does not identify a language. A generative closure result establishes sufficiency under the tested observables, not historical proof of meaningless production.
