# Parser correction — superseding first legacy-chain execution

Date: 2026-08-14

## Status

GitHub Actions run `31816720864` is scientifically invalid for the Timm arms and must not be used for final adjudication.

## Defect

The pinned Timm generator writes a metadata preamble beginning with `#Properties` and `#Statistics` before the generated text.  The first replay runner applied its lowercase-token regex to every nonblank physical file line, so alphabetic words in the metadata preamble were admitted as if they were generated tokens.

This defect is independently detectable from generator-format invariants, without reference to the Voynich target:

- published configuration: `text.lines_to_create=1200`;
- raw seed-19 output: 1,235 physical file lines = 35 `#` metadata lines + exactly 1,200 generated text lines;
- corrected seed-19 parse: exactly **1,200 generated lines and 10,832 tokens**, matching the earlier faithful Timm audit.

The defective parser instead produced 1,228 eligible pseudo-lines because metadata lines containing two or more alphabetic strings were tokenized.

## Correction

`run_legacy_chain_replay_corrected.py` imports the frozen base runner and replaces only `load_timm_file`:

1. ignore blank lines and lines whose first non-space character is `#`;
2. tokenize every remaining generated line exactly as before;
3. assert exactly 1,200 generated lines;
4. leave all target parsing, ED1 definitions, null generation, G-prime construction, seeds, parameters, equivalence margin and adjudication untouched.

The scientific protocol SHA-256 remains:
`d464cdc717e55d4233e2e5700be85b14fa2bc62a7691ac024b9e9bf98949533f`.

No numeric result from the defective Timm parse is admissible.  A corrected workflow execution supersedes run `31816720864` for all Timm values and for the formal final result.
