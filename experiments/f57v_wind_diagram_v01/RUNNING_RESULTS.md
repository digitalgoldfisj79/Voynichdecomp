# RUNNING RESULTS — f57v Wind-Diagram Comparator v0.1

## 2026-08-09 final v0.1 execution status

**Formal result: NO TEST.**

The preregistered positive-control gate failed before target inference was evaluated. Therefore v0.1 makes **no empirical claim about f57v**.

### Execution
- 8 independent manuscript strata
- 12 independently labelled wind-positive images
- 29 matched controls
- 3 late-medieval wind positives
- pixels-only blind coding with `Qwen/Qwen2.5-VL-3B-Instruct`
- frozen 12-feature visual codebook
- target randomized into the packet but not analyzed after the qualification failure

One feature value out of the full coded matrix was missing; pairwise Gower handling ignored only that missing value.

### Positive-control qualification
Preregistered thresholds:
- overall leave-one-stratum-out wind classification >= 75%
- late-medieval wind classification >= 60%

Observed:
- overall: **5/12 = 41.67%**
- late medieval: **1/3 = 33.33%**

Qualification therefore failed decisively.

Per preregistration, no target Delta statistic, p-value, margin, or ablation result is reported from this run. The machine coder's latent target feature vector is treated as sealed/uninspected for scientific interpretation.

### Interpretation
This is an instrument/panel failure, not evidence against the wind hypothesis. The fixed visual codebook plus the chosen automated coder did not reliably recognize independently attested wind exemplars against the hard controls. Several source surfaces also require stronger folio-level provenance checking before reuse.

### Next programme
The post-freeze Raff/Obrist lead changes the most useful next experiment. Rather than tune v0.1 against its failures, v0.2 will build a population-based corpus from Thomas Raff's dedicated catalogue of medieval wind personifications plus Obrist's diagrammatic witnesses, with explicit date/place/shelfmark/folio verification and a new feature family covering:
- four principal anthropomorphs;
- concentric cosmological rings;
- hand/arm indication;
- held subordinate heads/round objects;
- paired subordinate directions;
- explicit 4+8=12 organization;
- breath/horn streams as a separate feature rather than a requirement.

The Astronomy of Nemrod witness (Venice, Biblioteca Marciana, MS lat. 2760, fol. 2v, ca. 1200) is especially important: four cardinal wind figures stand on concentric elemental rings and hold collateral wind heads in their outstretched hands. It establishes an attested four-agent representation of a twelve-wind system.

The user-supplied BnF screenshot remains quarantined pending exact shelfmark/folio resolution. Bodl. MS 646 remains a priority late-medieval Padua witness but was not used in the v0.1 execution because its exact image surface was not reproducibly acquired before the coding freeze.
