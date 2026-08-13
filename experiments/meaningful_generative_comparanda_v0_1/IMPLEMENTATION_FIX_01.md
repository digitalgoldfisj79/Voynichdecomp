# Implementation fix 01 — pre-target execution failures

Frozen: 2026-08-13.

The first full GitHub Actions execution at workflow commit `a9509e38c10edd854cda3b49f6f4b3954032249d` did **not** score either external target corpus.

Two implementation failures occurred before target calculation:

1. **LLCT job:** the clean Python 3.11 runner did not include NumPy. `run_llct_formulaic_profile.py` stopped on import with `ModuleNotFoundError: No module named 'numpy'` immediately after the LLCT XML passed its frozen MD5 check.
2. **Cantus job:** pinned `chant21` 0.4.6 installed successfully, but importing the package executed its GABC registration code. Current `music21` exposes `registerSubConverter`, whereas the 2020 package calls `registerSubconverter`; import therefore stopped with `AttributeError` before any Cantus row was parsed.

A workflow defect also masked both Python failures: the analysis commands were piped to `tee` without `pipefail`, so the jobs inherited `tee`'s zero exit code. The uploaded artifacts contained only the tiny empty run logs; this exposed the problem before any result JSON existed.

## Frozen corrections

Only execution/compatibility plumbing is changed:

- install `numpy<2` explicitly in the LLCT job;
- execute analysis steps with `set -o pipefail`;
- for Cantus, continue to use the exact pinned `chant21` commit `ad52f6084efce4a440d083b588d7b51ff6973730`, but load its self-contained `chant21/cantus/parser_volpiano.py` together with `cantus_volpiano.peg` directly via `importlib`, avoiding package `__init__` and the unrelated obsolete GABC `music21` registration side effect;
- retain Arpeggio, which is the published parser dependency;
- no change to Volpiano preprocessing/parsing rules, note-position mapping, corpora, normalization, statistic definitions, permutations, seeds, controls, decision thresholds, or hypotheses.

No LLCT or Cantus target values had been produced or inspected before this correction.
