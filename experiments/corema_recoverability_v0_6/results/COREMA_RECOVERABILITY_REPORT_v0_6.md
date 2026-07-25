# CoReMA procedural recoverability calibration v0.6

**Formal verdict:** **CALIBRATION_FAILURE**  
**Manuscripts parsed:** 27  
**Procedural units:** 4193  
**Labelled word tokens:** 335464  

## Frozen gates

- Lexical known-role recovery: **FAIL**
- Identity-neutral role recovery: **FAIL**
- Procedural sequence-order signal: **PASS**
- CoReMA calibration admissible for target transfer: **NO**

## Token-role recovery

| Model | Eligible-role macro-F1 | All-role macro-F1 | Weighted F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| majority | 0.0699 | 0.0699 | 0.1726 | 0.1077 |
| lexical | 0.5622 | 0.5622 | 0.5665 | 0.6368 |
| rank | 0.2167 | 0.2167 | 0.2923 | 0.2535 |
| pattern | 0.2567 | 0.2567 | 0.2931 | 0.3611 |
| structural | 0.3735 | 0.3735 | 0.4711 | 0.4577 |
| structural_hmm | 0.3304 | 0.3304 | 0.5026 | 0.3746 |

## Role sequence structure

Mean first-order Markov gain over IID: **0.8092 bits/token**.  
Mean real-order advantage over within-recipe shuffling: **0.9179 bits/token**.

## Interpretation

The corpus does not pass the frozen recoverability calibration. The Voynich target must remain sealed for this route; no nearest-role narrative is admissible.

## Provenance

CoReMA TEI/XML was retrieved from the University of Graz GAMS public endpoints. The semantic model supplies explicit ingredient, instruction, tool, time, dish and advisory annotations. Manuscripts are the cross-validation groups.
