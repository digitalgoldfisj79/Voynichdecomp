# Reproduction

From the repository root on branch `experiment/frozen-mdl-codec-v0.1-20260714`:

```bash
cd experiments/frozen_mdl_codec
python3 bootstrap_source.py --target /tmp/fmdl-expanded
cd /tmp/fmdl-expanded/frozen_mdl_codec
./run_all.sh
python3 synthetic_gate.py --workers 32 --output FULL_SYNTHETIC_RESULT.json
```

Expected source archive SHA-256:

`7dd12696bb6b6e550be325a373cf5c44d60b0ea2429b8e8e9fd5090a6f73999e`

Expected deterministic fuzz aggregate SHA-256:

`fcba2b40abdd963b74492f030c5473b5a94c547e08b45accc0b67df69926acb1`

Expected full-gate verdict:

`PASS_SELECTION_POLICY_ROBUSTNESS`

The full stochastic result is deterministic under the frozen seed and configuration. A result-changing edit requires a new codec version and complete rerun.
