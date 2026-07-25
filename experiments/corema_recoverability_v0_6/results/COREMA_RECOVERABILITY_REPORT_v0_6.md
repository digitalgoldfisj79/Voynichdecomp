# CoReMA procedural recoverability calibration v0.6

**Formal verdict:** **CALIBRATION_FAILURE**  
**Manuscripts parsed:** 29  
**Procedural units:** 4636  
**Labelled word tokens:** 370980  

## Frozen gates

- Lexical known-role recovery: **FAIL**
- Identity-neutral role recovery: **FAIL**
- Procedural sequence-order signal: **PASS**
- CoReMA calibration admissible for target transfer: **NO**

## Token-role recovery

| Model | Eligible-role macro-F1 | All-role macro-F1 | Weighted F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| majority | 0.0780 | 0.0780 | 0.2052 | 0.1429 |
| lexical | 0.5721 | 0.5721 | 0.5776 | 0.6507 |
| rank | 0.2219 | 0.2219 | 0.3005 | 0.2594 |
| pattern | 0.2547 | 0.2547 | 0.2855 | 0.3597 |
| structural | 0.3689 | 0.3689 | 0.4623 | 0.4566 |
| structural_hmm | 0.3270 | 0.3270 | 0.4963 | 0.3716 |

## Role sequence structure

Mean first-order Markov gain over IID: **0.7572 bits/token**.  
Mean real-order advantage over within-recipe shuffling: **0.9268 bits/token**.

## Interpretation

The corpus does not pass the frozen recoverability calibration. The Voynich target must remain sealed for this route; no nearest-role narrative is admissible.

## Provenance

CoReMA TEI/XML was retrieved from the University of Graz GAMS public endpoints. The semantic model supplies explicit ingredient, instruction, tool, time, dish and advisory annotations. Manuscripts are the cross-validation groups.
