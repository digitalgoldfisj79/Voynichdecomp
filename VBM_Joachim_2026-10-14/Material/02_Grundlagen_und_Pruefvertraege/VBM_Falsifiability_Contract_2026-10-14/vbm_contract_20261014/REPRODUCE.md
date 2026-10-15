# Reproduce the admission audit

From this runtime with the original user-supplied files mounted at the paths specified inside the script:

```sh
python /mnt/data/vbm_contract_20261014/verify_admission.py
```

The script performs immutable-source and field-specific assertions, then atomically writes:
- stage0_provenance.pkl
- stage1_lineage_checks.pkl
- stage2_identifiability.pkl
- stage3_decision.pkl
- RESULTS.json

No parameter fitting, synthetic generation, fresh Voynich test or independent anchor discovery is attempted. The forum sources are linked in the main contract but not copied into the results file. Q65 and V10 are *prior* conversation findings and were not numerically revalidated here.
