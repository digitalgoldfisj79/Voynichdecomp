# Fractionation Signature Gate v0.1 — Result and Closure

## Status

**Decision: `STOP_NON_IDENTIFIABLE`.**

The preregistered synthetic gate completed successfully on the frozen six-language v0.5 corpus infrastructure. No Voynich text was used.

An earlier workflow attempt is discarded as an implementation failure: dynamic import of the v0.5 module failed under Python 3.12, and `tee` initially masked the Python exit code. The import and workflow error propagation were fixed without changing the hypothesis, generators, features, classifier, block ranges, controls, or decision thresholds. Only the corrected run is scientific output.

## Locked results

| Test | Locked AUC | Effect above chance | Permutation-null SD | Effect / null SD | Result |
|---|---:|---:|---:|---:|---|
| Fractionation vs easy adjacent-bigraph control | 1.000000 | 0.500000 | 0.028947 | 17.273 | resolves |
| Fractionation vs observational twin | 0.523112 | 0.023112 | 0.028505 | 0.811 | **does not resolve** |

The easy test demonstrates that the chosen features can recognize the deliberately planted two-stream block-regrouping geometry. The hard-twin comparison fails the preregistered attribution gate. Because the effect is only 0.81 null SD, **the metric does not resolve this comparison**.

## Block-width recovery

Locked test widths were unseen during training: `{5,7,9,10,11,12}`.

- Clean positive samples: exact recovery 1.000 (192/192); within-one 1.000; mean best JSD 1.000.
- Noisy positive samples: exact recovery 0.5833 (112/192); within-one 0.6771; mean best JSD 0.04992.

Thus the planted block geometry is perfectly recoverable in the clean construction but degrades sharply under the bounded overlap/homophony/null model used in v0.1.

## What the hard-twin result means

This is **not negative evidence against fractionation existing in the Voynich Manuscript**. It is an identifiability result.

The positive generator assigns plaintext units to keyed coordinate pairs and then emits the two coordinate streams in bounded blocks. The hard negative is an arbitrary keyed bigraphic code over the same two symbol roles with the same regrouping and noise law. Once the coordinate square is arbitrarily keyed, those constructions can induce the same surface distribution. A detector using only the observed symbol stream therefore cannot infer whether the latent pair had specifically Polybius/coordinate semantics rather than generic bigraphic semantics.

The easy AUC=1.0 result prevents the hard-null failure being blamed on a generally powerless detector: it sees the two-stream geometry when a comparator lacks it. What it cannot do is attribute that geometry uniquely to coordinate fractionation.

## Verification questions

1. **Did the detector have enough power to find the planted signature?** Yes. Easy-control AUC was 1.0, 17.27 null SD above chance.
2. **Did it generalise to unseen block widths?** Yes in the clean construction: 192/192 exact width recovery. Under the noisy construction exact recovery fell to 58.3%.
3. **Can the surface signature distinguish coordinate fractionation from a generic bigraphic structural twin?** No. AUC 0.523, only 0.81 null SD above chance.
4. **Could the hard-null failure be interpreted as evidence that Voynich is not fractionated?** No. Voynich was never run, and the failure concerns attribution/identifiability, not compatibility.
5. **Does v0.1 permit a Voynich application?** No. The preregistered rule requires both easy and hard controls to resolve; the hard control failed.

## Consequence for the cipher programme

The Claude-style information-theoretic loophole remains formally possible, but this experiment narrows what can be learned from surface statistics:

- a two-stream / split-and-regroup production geometry is detectable in favorable synthetic conditions;
- that geometry is not diagnostic of Polybius/coordinate fractionation once an observationally equivalent polygraphic code is admitted;
- realistic bounded noise substantially weakens block-parameter recovery;
- therefore a positive Voynich match to these features would establish at most **compatibility with a broader paired/polygraphic production family**, not a coordinate cipher or plaintext-bearing fractionation specifically.

Under v0.1, the branch is closed without touching Voynich.

## Files

- `PROTOCOL_V010.md` — preregistered design and thresholds.
- `fractionation_signature_gate_v01.py` — deterministic runner.
- `results_v010.json` — full locked result rows and summaries.
- `run_v010.log` — concise locked headline output.
