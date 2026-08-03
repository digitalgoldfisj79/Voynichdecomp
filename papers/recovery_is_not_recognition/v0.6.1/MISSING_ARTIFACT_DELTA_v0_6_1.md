# Missing-artifact delta for v0.6.1

This record distinguishes absence from the submission snapshot, absence from the Git repository, and absence from the surviving research record.

## Principal unresolved gap

### Classifier two-sample test feature-construction implementation

The aggregate C2ST record is preserved and specifies the observation unit, classifier, cross-validation design, reference partition, headline AUC values, and an eight-feature ablation. The exact original code constructing all ten per-chunk features was not located in any `Voynichdecomp` ref or in the H2 archive. This is the principal genuine reproducibility gap.

A replacement implementation may be reconstructed from the documented design, but it must be labelled a reconstruction. It cannot be represented as the lost original harness unless its source is recovered.

## Present outside GitHub, not scientifically absent

### Full H2 archive

`vms_h2_archive_2026-07-12.zip` is available in the working archive. Its SHA-256 is:

`8e7f6205154990db88b19fe3c378fcabb43a55a5f056d1e2897370aff2062a39`

The external checksum and all internal `SHA256SUMS.txt` entries pass. The archive includes code, derived data, result objects, the authoritative ledger, provenance records, and research copies. It was omitted from the reviewer ZIP because it includes copyrighted or otherwise non-redistributable material. The corrective action is a portable filtered snapshot, not recovery from another model.

## Historical row-level artefacts not committed

The following experiments preserve code, inputs or deterministic generators, protocols, aggregate reports, job identifiers, and scientific hashes, but do not preserve every original row-level output as a committed file:

- generic multilingual decoder;
- monoalphabetic six-language locked test;
- fresh-key homophonic final test;
- joint substitution-transposition development test;
- periodically shifting-alphabet locked recovery;
- carrier steganography final run retains an aggregate machine JSON rather than every final row;
- true-map MDL retains a documented clean recomputation but not a normalized standalone scientific JSON.

These are persistence defects, not evidence that the experiments were never run. They can be closed by deterministic clean-clone reruns and by committing normalized machine outputs.

## External artefacts deliberately represented by immutable identifiers

The variable-length polygraphic experiment records checkpoint hashes and private object-store identifiers for phase artefacts. Its final 16-row scoring JSON is committed. Redistribution of large checkpoints is unnecessary provided their immutable identifiers and hashes remain resolvable.

## Not a missing-artifact claim

External third-party replication has not been completed. That is an independence limitation, not a data-availability defect and should not be conflated with the repository audit above.
