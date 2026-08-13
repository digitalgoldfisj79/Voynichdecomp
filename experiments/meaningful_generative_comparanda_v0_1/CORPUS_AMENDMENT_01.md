# Corpus amendment 01 — LLCT1 unit definition

Frozen: 2026-08-13, **before any LLCT target statistic was calculated**.

The schema-only validation of the exact LLCT1 v1.2 XML established facts not visible in the literature-level preregistration:

- 9,227 sentence-root `LM` nodes;
- token `seg` values are `formulaic` (156,075 raw nodes), `free` (56,314), and `subs` (13,445);
- sentence label sets: 6,112 formulaic-only, 1,483 free-only, 1,431 subs-only, 129 formulaic+subs, and 72 formulaic+free;
- therefore `seg` is a passage/token annotation, not universally sentence-level;
- dependency-tree XML traversal is not written order; lexical nodes carry integer token IDs that recover written order;
- a small number of dependency representations repeat the same integer token ID in the XML. Repeated occurrences observed in validation have identical surface attributes and are technical tree representation, not repeated written tokens.

The primary corpus definition is therefore amended as follows, without inspecting any target score:

1. For each sentence root, collect descendant lexical nodes carrying `form` and an integer `id`.
2. Collapse duplicate occurrences of the same integer `id`; abort if duplicates disagree on `form`, `lemma`, or `seg`.
3. Sort unique nodes by integer `id` to recover written order.
4. Apply the preregistered lexical filter (surface form contains at least one Unicode letter), NFC normalization, and lowercase.
5. Split the resulting stream at every change in the author-provided `seg` label and at every removed `subs` passage.
6. Retain maximal contiguous `formulaic` and `free` runs as analysis loci; discard `subs` entirely.
7. Runs shorter than 2 lexical tokens contribute no adjacency statistic; runs shorter than 3 contribute no E2 statistic, exactly as implied by the metric definitions.
8. Charter block ID is `(document_id, subdoc)` from the sentence-root metadata.

This amendment is stricter than assigning mixed sentences wholesale to one class and is faithful to the corpus authors' formulaic/non-formulaic passage annotation. All H2/H3 thresholds and all statistic/null definitions remain unchanged.
