# Generator clean-rerun cancellation record v0.6.1

- Job: `Digitalgoldfish79/6a70be5f6b79c09949c20d54`
- Source: clean clone of `digitalgoldfisj79/Voynichdecomp` main
- Entry point: `Paper/reproduce_s3.py --skip-ablation --skip-ceiling --force`
- Compute: CPU Basic
- Outcome: cancelled during the Gen-04T metric stage.

The run confirmed that the committed data, generator implementations and scoring pipeline executed in a clean environment. It was stopped before completing all 23 generators because the repository already contains the full cached output (`Paper/s3_all_generators.pkl`) and summary (`Paper/s3_summary.md`), and a complete regeneration would not resolve the principal reproducibility gap, which concerns the later C2ST feature-construction harness. No scientific result from the partial run is used in the manuscript.
