# VSN-B4-v1 Diplomatic Transcription Schema

## General rule

Preserve what the manuscript writes. Normalisation is a separate field. Do not expand an abbreviation merely because a likely Latin word is known from Cacopardo or from another part of the manuscript.

Use `[?]` only in the diplomatic field to mark uncertain characters and `U` in confidence when no defensible reading exists.

## B4-A codeword table fields

- `group_id`: stable B4A01... identifier in manuscript reading order.
- `folio`: always 8v for v1 source core.
- `side`: L or R page half.
- `side_order`: top-to-bottom order within half.
- `source1_dipl` ... `source4_dipl`: manuscript readings.
- `source1_norm` ... `source4_norm`: normalized lexical readings only where defensible.
- `syll1` ... `syll4`: first syllables implied/explicitly written; never generated automatically for the binding corpus.
- `compound_dipl`: compact form exactly as written after the immediate bracket.
- `compound_norm`: conservative lowercase/letters-only normalization for metrics.
- `immediate_bracket`: 1 if a direct graphical bracket links the four source entries to the compound.
- `parent_graphic_id`: identifier of the next larger brace/grouping, without semantic interpretation.
- `group_conf`: confidence that the graphical group itself is correctly segmented.
- `text_conf`: minimum confidence across textual fields.
- `evidence`: PRIMARY / SECONDARY / PRIMARY+SECONDARY.
- `notes`: unresolved palaeographic detail, abbreviation, correction etc.

## f.8v graphical relation table

Keep graphical structure separate from word transcription.

Fields:
- `edge_id`
- `folio`
- `level` (0 immediate four-name bracket; 1,2... higher graphical nesting)
- `sources`
- `target`
- `side`
- `relation_type` (`BRACKET`, `BRACE`, `LINE`, `LABEL_ASSOCIATION`)
- `semantic_status` (`UNINTERPRETED` by default)
- `confidence`
- `notes`

Only a textual instruction can promote `semantic_status` to a stronger interpretation such as `CODEWORD_FORMATION` or `SPATIAL_GROUPING`.

## B4-B De numeris ficticiis table

Fields:
- `item_id`
- `folio`
- `order`
- `value_dipl`
- `value_norm`
- `sign_or_expression_dipl`
- `body_or_spatial_locus_dipl`
- `body_or_spatial_locus_norm`
- `laterality` (`LEFT`, `RIGHT`, `NONE`, `UNRESOLVED`)
- `grade_or_magnitude`
- `rule_text_dipl`
- `rule_text_norm`
- `confidence`
- `notes`

No linear string serialization is to be constructed from these fields unless explicitly present in the manuscript.

## Confidence propagation

- Binding source corpus: A+B fields only and only rows whose required fields are all A/B.
- A-only robustness corpus.
- A+B+C sensitivity corpus.
- U fields excluded, never inferred.

Every correction to a frozen diplomatic field after source freeze requires a new corpus version; the prior version is retained for audit.
