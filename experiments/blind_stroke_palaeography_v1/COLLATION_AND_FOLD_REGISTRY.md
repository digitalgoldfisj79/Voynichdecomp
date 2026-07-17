# Collation and fold registry

**Phase-I compatible:** yes  
**Contains Davis hand labels:** no  
**Contains f115r target boundary:** no

## Sources and scope

The extant physical grouping is based on the formal collation in Lisa Fagin Davis, “How Many Glyphs and How Many Scribes? Digital Paleography and the Voynich Manuscript” (Manuscript Studies 5.1, 2020), using only its codicological collation statement and not its hand assignments. Foldout and historical gathering-mark descriptions were cross-checked against René Zandbergen’s quire descriptions, including the documented folio ranges and gathering marks for quires 15, 17, 19 and 20.

The registry distinguishes:

- `extant_gathering_index`: the 18 current extant physical gatherings used for primary leave-one-gathering-out validation;
- `historical_quire_mark`: the surviving historical gathering mark, with gaps at 16 and 18;
- `physical_bifolium_id`: conjoint-leaf grouping used as the minimum train/test exclusion unit.

Panel suffixes such as `f89r1`, `f89r2`, `f102v1` and `f102v2` inherit the base folio’s physical bifolium. A panel is never an independent split unit.

## Missing material

- Folio 12 is represented as the missing conjoint leaf of extant folio 13.
- Folios 59–64 are represented as three missing central bifolia in gathering 8.
- Folio 74 is represented as the missing conjoint leaf of extant folio 73.
- Folios 109–110 are represented as the missing central bifolium of the final gathering.
- Folios 91–92 and 97–98 remain deliberately unassigned because their historical gathering interpretation is disputed and no extant image depends on those assignments.

Missing leaves are never samples; their records exist only to make the collation assumptions explicit.

## Frozen validation folds

Primary validation uses 18 leave-one-extant-gathering-out folds. Secondary validation uses five physical-bifolium group folds.

The secondary assignment is deterministic. Within each gathering, bifolia are assigned to successive cyclic folds so that gatherings of five or fewer bifolia do not place two bifolia in the same secondary fold. The cyclic offset is selected by a fixed hash seed, `blind-pal-v1-secondary`, subject to minimizing global fold-count range and squared load. The resulting fold counts are 10, 10, 11, 10 and 11 physical bifolia.

Files:

- `config/physical_bifolium_registry.csv`: 102 extant folios, 52 extant physical bifolia; SHA-256 `958e662c25e112a17827340f95b92500aa1f8fd600d3e7f20b553e95d75909ae`.
- `config/fold_registry_v1.csv`: 52 bifolia; SHA-256 `9479bdd2cfccab8ab291d078f7817cb33ced509d9770cfbb70ee4bdb01ee5a79`.

Any mismatch between these hashes and execution-time files is a hard stop.