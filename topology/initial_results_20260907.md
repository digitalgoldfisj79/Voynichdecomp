# Initial topology results — 2026-09-07

## RETRACTION / STOP RULE

**Do not use maximum textual similarity to infer original Voynich reading order.**

Tier-A recovery on known continuous Latin fails badly enough that all textual path outputs below are explicitly unlicensed as order.

## Physical registry

Supabase protocol: `vms_prod_topology_20260907_v01`.

Registry after whole-codex expansion:

- 58 physical bifolium/singulion nodes
- 6 wholly missing nodes
- 2 part-surviving nodes (f12/f13; f73/f74)
- all surviving conjunctions come from explicit Stolfi `Bifolio: ... = fX+fY` declarations rather than inferred folio arithmetic

Wholly missing nodes explicitly represented include Q8 f59/f64, f60/f63, f61/f62; singulions f91/f92 and f97/f98; and Q20 f109/f110.

## Hard/physical evidence currently encoded

- Q13 f76/f83: cross-fold spraying-liquid continuity.
- Q13 f78/f81: cross-fold pipe/water continuity.
- Q9 f67/f68: earlier sewing holes identify a prior sewing valley/orientation; this also restores the quiremark to a final-verso position.
- f33/f40: cross-gutter illustration evidence retained as bifolium-functional evidence, not sequence evidence.
- early upper-margin spill: retained as a potential physical ordering field; no stain-shape ordering score is yet licensed.

## Soft production-proximity edges

These are NOT reading-order edges.

Q13:
- f75/f84 <-> f78/f81: 8/8 channel-optimal path support
- f76/f83 <-> f77/f82: 7/8

Q20:
- f106/f113 <-> f107/f112: 8/8
- f103/f116 <-> f108/f111: 8/8
- f107/f112 <-> f108/f111: 7/8
- f104/f115 <-> f105/f114: 7/8
- f104/f115 <-> f106/f113: 6/8

The Q20 best path after excluding the f104/f115 Scribe-2-intervention unit is:
`103/116 -- 108/111 -- 107/112 -- 106/113 -- 105/114`.
Again: this is a proximity summary only.

## Tier-A calibration

Control: Caesar, *De Bello Gallico* I-IV, Project Gutenberg #218.
Frozen SHA-256: `976c130b6637c8b643617e66e039b8fddcb610eda02c3b68db1bdce87bd09866`.
20,561 strict alphabetic tokens.

Windows use the exact ZL token-length profiles of the Voynich units.

### Q13 shape
- 57 windows
- mean true-adjacency recall: 0.5921
- median recall: 0.5000
- exact path recovery: 0.0702
- mean false-edge rate: 0.4079

### Q20 shape
- 42 windows
- mean true-adjacency recall: 0.6286
- median recall: 0.6000
- exact path recovery: 0.0952
- mean false-edge rate: 0.3714

**Calibration result: FAIL_FOR_LINEAR_ORDER.**

A Tier-B page/bifolium-resolved historical manuscript control remains mandatory before any later ordering method can be licensed.

## Consequence

The programme now targets a **partial production topology**, not a forced manuscript sequence. Hard physical constraints and stable undirected production-proximity clusters may be recovered even when linear reading order is unknowable.
