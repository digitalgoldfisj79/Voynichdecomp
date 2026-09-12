# Yiddish qualification v03 — short-message search repair

Parent failure: v02 `RECOVERY_UNQUALIFIED` on fresh 512-word Lev Tov cells. Ten of eleven failed trials had a better true-key fitting objective; one showed objective misalignment. This version therefore changes **search budget only**. The bigram objective, atomisation, recovery thresholds, key family, erasure rates, fit/audit split, decoder and score remain unchanged.

## Consumed sources

Consumed by v01/v02 and therefore unavailable as fresh confirmation in v03: every 1600–1750 file used in either prior training manifest; Bovo 1507; Shir 1579; Ester 1589; Lev Tov 1620; Purim 1697; their named prefaces where applicable.

## BUILD/DEVELOPMENT

BUILD_SOLVER: prior-consumed 1600–1750 Penn sources excluding the v03 development works below. No v03 confirmation source is admitted to BUILD.

DEVELOPMENT works, all already consumed in prior versions and therefore safe to use for tuning:
- `1507w-bovo.psd`
- `1648w-kine.psd`
- `1666w-messiah.psd`
- `1675e-ashkenaz-un-polak.psd`

Development conditions: 512 fitting + 32-word buffer + 512 audit, 32 independent keys, at 0% and 1% marked atom erasure. Registered recovery pass remains >=90% audit atoms AND >=80% complete audit words; work/cell gate >=29/32.

## At most two algorithm candidates

Both retain the v02 conditional character-bigram objective and move set.

- Candidate S1: 8 restarts × 5,000 annealing steps; 60 greedy passes.
- Candidate S2: 16 restarts × 6,000 annealing steps; 80 greedy passes.

Selection rule frozen before development outcomes: choose S1 if **every** DEVELOPMENT work/erasure cell reaches >=29/32. Otherwise choose S2 if every cell reaches >=29/32. If neither passes, terminate `RECOVERY_UNQUALIFIED` without reading v03 confirmation plaintext into the solver stage. No objective change or third candidate is permitted in this version.

## Fresh confirmation, unopened until candidate freeze

Transfer-panel short confirmation (1501–1600; 512 cells only):
- `1588e-letters-cracow.psd`
- `1590e-sam-hayyim.psd`

Out-of-window robustness panel (cannot fill medieval/transfer cells):
- `1834e-ukraine-2.psd` — 512 and 2,048 cells if length permits.

These filenames are selected from source date and machine-readable word-count feasibility only, before any v03 recovery outcome. No confirmation source may be moved into development.

## Interpretation

Even if all fresh v03 cells pass, the result is a `FINITE_PANEL_ONLY` normalized-representation recovery result. Penn Romanisation is lossy and the fresh 1501–1600 confirmation works do not populate the 2,048+2,048 transfer cell. Therefore v03 cannot by itself authorise L or T. The historical long cell remains `CORPUS_LIMITED` unless a separate admitted source supplies it.
