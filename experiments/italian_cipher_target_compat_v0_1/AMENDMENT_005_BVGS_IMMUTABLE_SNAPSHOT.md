# Target Compatibility Amendment 005 — immutable BVGS snapshot after live-site HTML drift

Date: 2026-08-15
Status: **pre-target source-reproducibility repair**. No target data were downloaded or scored in the failed run described below.

GitHub Actions run `31907289957` passed runner reconstruction, hashes and self-tests, then stopped while verifying the external sources because the live `Buch von guter Speise` webpage had changed its raw HTML bytes. The parent entropy-transfer run had recorded raw SHA-256 `03ada8b2...`; the later live download returned `0e3ea822...`. External calibration and every target step were skipped.

Raw HTML is not the scientific object used by the frozen parent runner: `parse_bvgs()` strips markup, extracts the edition's locus-delimited text, tokenises it, and then deterministically splits the parsed units into training and test sets.

An earlier immutable GitHub Actions artifact from hostile run `31893199016` (artifact `9249108075`, digest `sha256:8296cc5a12b7b37f5859efcceba53ccfdbee6c6e7025c06a458fd8d780a10298`) contains `external/bvgs.htm` with raw SHA-256:

`c793d8d7ba3c08a745cded86f3ad19a15099b298ade1af8bba17d2ac51a9be03`

Before target access, that snapshot was tested with the exact frozen parent entropy runner against the successful parent run `31906169329`:

- all 1,020 BVGS mechanism/seed/representation rows were regenerated;
- categorical/key fields matched exactly;
- maximum absolute numeric difference across the generated metrics was `8.881784197001252e-16`, below the `1e-12` audit tolerance;
- the parsed all/train/test unit hashes are frozen in `BVGS_PARENT_FINGERPRINT.json`.

Therefore this snapshot is scientifically equivalent to the BVGS corpus actually used by the parent entropy-transfer analysis, despite differing in irrelevant surrounding HTML bytes.

The target workflow is amended to retrieve this immutable artifact snapshot rather than the mutable live webpage, verify its raw snapshot hash and parsed all/train/test hashes, and archive the snapshot in the new target-stage result artifact.

This changes no parsed BVGS training/test data, generated mechanism output, historical parameter, metric, null, target representation, threshold, or verdict logic. The repair is strictly a source-immutability correction made before target access.
