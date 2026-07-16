# v0.7 Stage A0 oracle source-transfer precheck

Date frozen: 2026-07-16

Status: **FROZEN BEFORE ORACLE RESULTS**

This precheck completes the Stage A protocol before development. It implements the hostile-review stop rule that the source-transfer criterion must work with the true latent mapping before mapping inference is evaluated.

## Inventory

- 16 source-message positives: four independent cipher renderers × two Greek corpora × two deterministic replicates;
- 16 ordered non-message controls: four control generators × four deterministic replicates;
- 12 documents and 180 tokens per document;
- the same complete-document train/test split, source registry, production registry and accounting as Stage A.

For every trial, the renderer's true surface-to-latent mapping, partition and null structure are supplied to the scorer. They are still charged in the two-part codelength. The source model, order, emission policy and selector are selected using training documents only.

Controls receive their true renderer mapping as well. Thus success cannot arise merely because positives are easier to decode.

## Oracle decision

The source-message arm is selected only if:

- total two-part advantage is at least 0.05 bits per token;
- held-out predictive advantage is at least 0.02 bits per held-out token;
- both full and conditional accounting prefer the source-message arm;
- positive trials select the appropriate leave-target-corpus-out source profile.

No mapping-accuracy or fold-stability condition is needed because the mapping is supplied.

## Gate

All conditions must pass:

- positive sensitivity at least 75%;
- every positive mechanism detected at least once in two trials;
- each source corpus sensitivity at least 62.5%;
- overall ordered-control false-positive rate at most 15%;
- no control family accepted in more than one of four trials;
- median positive held-out advantage greater than zero;
- median control held-out advantage less than or equal to zero.

Failure closes v0.7 before inferred-mapping development. No feature, source model, threshold or accounting amendment is permitted after the oracle result.
