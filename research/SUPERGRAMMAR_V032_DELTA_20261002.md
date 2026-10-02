# Super-Grammar v03.2 delta candidate — SGT12 + SGT13

Date: 2026-10-02  
Parent: `supergrammar_v031_20260928` (candidate; v03 frozen release remains immutable)  
Status: **DELTA CANDIDATE — not certified release**

This file records post-v03.1 theorem additions for the next Super-Grammar release. It does not modify the frozen v03 release or the v03.1 correction record.

## SGT12 — within-line exact-recurrence geometry

Status: CANDIDATE_BOUNDED_PROMOTED, external review open.

Licensed statement:

> Within the tested physical-line controls, Voynich running text shows an exact-repetition geometry in which immediate identical-token repetition is not detectably suppressed while identical-token recurrence at distance two is enriched. Recipe controls can also show lag-2 enrichment, so lag-2 alone is not diagnostic.

Primary ZLZI: lag1 effect +0.000226 / null SD 0.000479 = +0.47 SD (unresolved); lag2 +0.002141 / 0.000515 = +4.16 SD. Full closeout: Supabase handoff `voynich_supergrammar_xd1_closeout_20260928`.

## SGT13 — vertical opener grammar

Status: **CANDIDATE_BOUNDED_PROMOTED**, external adversarial review + independent implementation replay open.

Frozen theorem statement:

> Within ordinary paragraph text, the opener atom of physical line i+1 carries substantial information about the opener atom of physical line i beyond section, Currier, Davis hand and exact paragraph-line position. Identical consecutive openers are strongly suppressed. A directed d→q→y→d asymmetry is a reproducible adjacent-line substructure; the corresponding d/q/y directional contrast does not resolve at lag 2. This theorem does not identify a literal wheel mechanism.

### Source contract

Primary: Zandbergen–Landini `ZL3b-n.txt`, version 3b, 2025-05-13.  
Parsed paragraph lines: 4,130.  
Unambiguous opener atoms: 4,111.  
Within-paragraph adjacent-line pairs: 3,326.  
Lag-2 within-paragraph pairs: 2,624.

Compound EVA openers `ch sh cth ckh cph cfh` are treated as atoms; ambiguous initial `?` / `@` forms are excluded.

### SGK21A — adjacent-opener dependency

Observed adjacent-opener MI = 0.184656 bits.

Position-stratified null (1,000 permutations; exact section × Currier × hand × paragraph-line-position strata):  
null mean 0.074143, null SD 0.005057; effect +0.110514 bits = **+21.85 SD**.

Independent within-paragraph composition null:  
null mean 0.064203, null SD 0.004725; effect +0.120453 bits = **+25.49 SD**.

### SGK21B — identical-opener suppression

Observed same-opener rate 7.2159%.

Position-stratified null: mean 12.2788%, null SD 0.5053 pp; effect −5.0630 pp = **−10.02 SD**.

Within-paragraph null: mean 12.4746%, null SD 0.4496 pp; effect −5.2587 pp = **−11.70 SD**.

### SGK21C — d/q/y direction

Observed within-paragraph adjacent transitions:

| edge | forward | reverse |
|---|---:|---:|
| d↔q | d→q 124 | q→d 74 |
| q↔y | q→y 75 | y→q 34 |
| y↔d | y→d 119 | d→y 82 |

Combined forward = 318; reverse = 190; forward share = 62.598%; directional difference = +128.

Position-stratified null: mean +16.410, null SD 19.4456; effect +111.590 = **+5.74 SD**.  
Within-paragraph composition null: mean −2.113, null SD 20.4733; effect +130.113 = **+6.36 SD**.

Each edge independently resolves under the position-stratified null:
- d→q − q→d: effect +27.321, null SD 9.4227 = **+2.90 SD**
- q→y − y→q: effect +43.088, null SD 9.0464 = **+4.76 SD**
- y→d − d→y: effect +41.181, null SD 10.0101 = **+4.11 SD**

### SGK21D — lag-2 bound

Lag 2: forward 152, reverse 170, directional difference −18.

Position-stratified null: mean −6.141, null SD 16.9007; effect −11.859 = **−0.70 SD**.  
Within-paragraph null: mean −2.308, null SD 18.3036; effect −15.692 = **−0.86 SD**.

**The metric does not resolve a lag-2 d/q/y direction.**

### Paragraph-first / top-row control

Removing every paragraph line1→line2 transition:
forward 312, reverse 178, difference +134.

Position-stratified null mean +18.949, null SD 19.0300; effect +115.051 = **+6.05 SD**.

Thus paragraph-opening gallows/top-row behaviour does not explain the d/q/y direction.

### Metadata completeness control

Pairs with known section + Currier + hand only: forward 277, reverse 169, difference +108.  
Matched-null mean +13.961, null SD 18.6796; effect +94.039 = **+5.03 SD**.

### Transcription replication

- RF1b-er: lag1 forward 330 / reverse 197 = 62.62%; lag2 187 / 218 = 46.17%.
- IT2a-n (Takahashi via Stolfi): lag1 318 / 186 = 63.10%; lag2 150 / 174 = 46.30%.
- VT0e-n (Takahashi / voynichese.com): lag1 331 / 195 = 62.93%; lag2 184 / 221 = 45.43%.

### Selection / mechanism bound

The unrestricted max-over-triplets search is **not** evidence that d/q/y is uniquely privileged: paragraph-first `p` effects dominate that search.

After removing all line1→line2 transitions, d/q/y ranks 1/109 eligible cycles and exceeds the max-cycle null by +3.167 standardized-cycle units with null SD 0.518 = +6.11 SD. Because this exclusion was motivated after inspecting the unrestricted competitor list, this is **exploratory only** and is excluded from SGT13 certification.

SGT13 licenses a vertical adjacent-line opener state. It does **not** license a literal physical wheel, cyclic device, or any specific historical mechanism.

## Open obligations before release freeze

1. Independent implementation replay of SGT13 from the frozen source contract.
2. External adversarial review of SGT13.
3. Preregistered replay of the top-row-excluded max-cycle test if cycle privilege is to be claimed.
4. No change to the frozen v03 release until a new release manifest is built.

Canonical analytical handoff: `voynich_vertical_opener_transition_program_20261002_v01`.
