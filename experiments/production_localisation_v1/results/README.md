# Results directory

No primary results have been generated yet.

## Rules

- Never overwrite an existing result file.
- Each run gets a UTC timestamp and git commit SHA in its filename or metadata.
- Primary and exploratory outputs are stored separately.
- Every result file must record:
  - cohort manifest hash
  - feature matrix hash
  - script commit SHA
  - random seed
  - model specification
  - cross-validation metrics
  - missingness summary
  - excluded features and reasons

Recommended structure:

- `results/primary/<timestamp>_cv_metrics.json`
- `results/primary/<timestamp>_vms_classification.json`
- `results/primary/<timestamp>_coefficients.csv`
- `results/sensitivity/...`
- `results/exploratory/...`

A write-up may quote only versioned outputs that can be regenerated from the frozen cohort and feature manifests.
