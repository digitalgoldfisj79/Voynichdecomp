# Changelog v0.6.2

Version 0.6.2 closes the principal C2ST artefact gap and responds to the final adversarial review.

Scientific changes:

- withdraws the statement that five contextual axes accounted for 28.9% of PGCS entropy after a permutation audit showed severe high-cardinality plug-in bias;
- adds explicit abstract and Table 6 language that adverse results are implementation-bounded and that the framework, not the numerical error rates, is intended to transfer;
- corrects the C2ST full feature count from ten to thirteen;
- corrects the ablation description from removing two features to removing five;
- records exact clean-environment reproduction of all five full-feature C2ST means and standard deviations;
- records an eight-feature labelled reconstruction uniquely identified by exhaustive search of all 1,287 subsets and reproducing all published ablation cells within 0.002;
- replaces the former principal artefact gap with precise residual qualifications: the original `corpus.pkl` builder and original ablation edit were not textually recovered.

No cipher-recovery, recognition, compression, source-transfer, or CoReMA result changed.
