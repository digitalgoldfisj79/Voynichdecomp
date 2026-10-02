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


## SGT13 architecture decomposition — independent VoynichStats replay (2026-10-02)

This follow-up asks what SGT13 actually represents rather than merely whether the opener correlation exists. It uses VoynichStats full physical-line dumps as an implementation-separate corpus route, with ZL3b metadata joined only for paragraph/Currier/hand controls.

### Corrections / discarded analysis

- An initial folio-held-out categorical likelihood comparison was discarded because adding high-cardinality opener/end states with naive Laplace smoothing caused systematic sparse-cell overfitting. Its negative held-out gains are not evidence against SGT13 and are not used below.
- The independent alignment yields 4,117 mapped paragraph lines and 3,343 adjacent within-paragraph pairs rather than the primary parser's 4,130 loci / 3,326 eligible pairs. Thirteen ZL3b paragraph lines did not map to the VoynichStats line dump and the independent opener eligibility/order implementation differs slightly. All architecture conclusions below are therefore treated as an independent sensitivity layer, not a replacement of the frozen primary counts.

### SG13-A — the information channel is opener→opener, not ordinary line-end continuity

Condition on section × Currier × hand × exact paragraph-line position, previous line ending, and previous line-length bin:

- previous opener → next opener: effect **+0.0238688 bits**, null SD **0.0064337**, **+3.71 SD**, empirical upper p=.001.
- previous ending → next opener after previous opener is known: **the metric does not resolve this**; effect **+0.0013833 bits**, null SD **0.0061667**, **+0.22 SD**.

Removing all paragraph line1→line2 transitions still leaves previous opener → next opener at effect **+0.0249125 bits**, null SD **0.0082133**, **+3.03 SD**.

Interpretation licensed: the adjacent vertical information is not explained by the physically preceding line ending and is not a paragraph-top artefact.

### SG13-B — the channel is mostly first-order in vertical distance

Given the immediately previous opener plus section/Currier/hand/position:

- opener two lines back → current opener: **the metric does not resolve this**. Exact-position effect **+0.0132953 bits**, null SD **0.0078209**, **+1.70 SD**; capped-position sensitivity **+0.0123757 bits**, null SD **0.0093033**, **+1.33 SD**.

This bounds SGT13 to a mostly first-order vertical process at current resolution; it does not prove all longer-range opener dependence is zero.

### SG13-C — the previous opener does not control the next line body once the new opener is known

Condition on section/Currier/hand/position, previous line ending, previous line length, and the current line opener.

Residual previous-opener effects:
- next-line length: **the metric does not resolve this**; +0.0031335 bits / null SD 0.0039328 = **+0.80 SD**.
- second-token opener: **the metric does not resolve this**; +0.0014404 / 0.0039957 = **+0.36 SD**.
- first-token ending: **the metric does not resolve this**; +0.0000417 / 0.0041366 = **+0.01 SD**.
- line ending: no positive residual; observed CMI is below its matched-null mean by -0.0105542 bits / null SD 0.0041655 = **-2.53 SD**.

Likewise, once the previous opener is known, extra features of the previous first word do not predict the next opener:
- previous first-word second atom: **the metric does not resolve this**, +0.0016866 / 0.0040262 = **+0.42 SD**.
- previous first-word ending: **the metric does not resolve this**, -0.0047399 / 0.0045662 = **-1.04 SD**.
- previous first-word length: **the metric does not resolve this**, -0.0051303 / 0.0046549 = **-1.10 SD**.

And once the current opener is known, the previous opener does not resolve details of the current first word:
- current first-word second atom: **the metric does not resolve this**, +0.0020968 / 0.0036691 = **+0.57 SD**.
- current first-word ending: **the metric does not resolve this**, -0.0002337 / 0.0041259 = **-0.06 SD**.
- current first-word length: **the metric does not resolve this**, +0.0031650 / 0.0042599 = **+0.74 SD**.

Licensed interpretation: the cross-line state is astonishingly narrow — opener atom → next opener atom — rather than first-word→first-word or previous line-state→next line-body.

### SG13-D — simple anti-repetition is insufficient

Independent VoynichStats representation, with every observed same-opener transition frozen exactly and section/Currier/hand/position opener marginals preserved:

- d/q/y directional excess: effect **+106.522 transitions**, null SD **19.4847**, **+5.47 SD**.
- total antisymmetric directional-flow Frobenius norm: effect **+17.2182**, null SD **2.42755**, **+7.09 SD**.

Therefore the d/q/y direction and broader directional flow cannot be generated merely by avoiding identical consecutive initials.

### SG13-E — d/q/y is a dominant subcycle, not the whole directed system

After matched-stratum marginal removal, the antisymmetric residual transition matrix across 11 common opener states has:

- total directional residual norm: effect **+37.194**, null SD **4.1148**, **+9.04 SD**.
- leading rank-2 antisymmetric concentration: **the metric does not resolve a single low-rank wheel**; effect **+0.1388**, null SD **0.08743**, **+1.59 SD**.
- d/q/y share of directional residual energy: observed **45.46%** vs null mean **14.45%**; effect **+31.01 percentage points**, null SD **9.98 pp**, **+3.11 SD**.

After removing all paragraph line1→line2 transitions:
- total directional residual: **+8.57 SD**.
- leading-mode concentration: **the metric does not resolve this**, **+1.53 SD**.
- d/q/y residual-energy share observed **52.70%** vs null **16.12%**; effect **+36.58 pp**, null SD **10.44 pp**, **+3.50 SD**.

Thus a literal three-state wheel is not identified. The data support a broader directed opener-transition network in which d/q/y is an unusually strong subcycle.

### Pre-fixed medieval prose controls

Two multi-page witnesses from the pre-existing CREMMA prose control set were run with the same start→start versus end→start decomposition:

- BnF fr. 1728: **the metric does not resolve vertical opener dependence**; effect **-0.007113 bits**, null SD **0.010792**, **-0.66 SD**.
- KBR 9232 Examens Moraux: **the metric does not resolve vertical opener dependence**; effect **+0.004198 bits**, null SD **0.010637**, **+0.39 SD**.

This is preliminary external bounding only: two controls do not establish Voynich-specificity.

### Architecture verdict

The strongest currently licensed model is:

`previous physical-line opener atom → next physical-line opener atom`

with a mostly first-order transition law.

The following stronger models are not supported by the present tests:

- normal sequential end-of-line carryover as the source of SGT13;
- whole previous first-word morphology as the transmitted state;
- a previous-opener state that directly governs the rest of the next line after its opener is chosen;
- simple same-initial avoidance;
- a unique three-state d/q/y wheel.

The d/q/y cycle remains a real, dominant directed substructure inside a broader opener-state transition network.


## SGT13 architecture refinement — 2026-10-02

Subsequent independent VoynichStats and paragraph-aware ZL3b tests sharpen SGT13 from a generic vertical dependency to a bounded architectural statement:

> **SGT13 is best modelled as a paragraph-local, first-order, left-margin opener state process.** The previous physical line's opener predicts the next line's opener after the previous line ending and line length are known; the reverse conditional contribution of the previous line ending does not resolve. The state resets at paragraph boundaries, does not require order-2 memory, and does not propagate as a direct vertical dependence through token positions 2–5 or into the remainder of the next first word once the current opener is known.

Key bounds:
- True-paragraph opener unique contribution after previous ending + line length: +0.033347 bits / null SD 0.008373 = **+3.98 SD**.
- Previous ending after previous opener + length: -0.006207 / 0.007975 = **-0.78 SD; metric does not resolve**.
- Across paragraph boundaries: opener MI effect -0.004769 / 0.018583 = **-0.26 SD; metric does not resolve**; d/q/y direction = **+1.78 SD**, unresolved.
- True-paragraph order-2 residual I(O_i;O_{i+2}|O_{i+1},section): +0.015763 bits / 0.012044 = **+1.31 SD; metric does not resolve**.
- Same-column positions 2–5 after conditioning horizontal-left neighbours: ZLZI **-0.41, +0.90, -0.06, -0.14 SD**, all unresolved; replicated as unresolved across ZLZB/TTLI/VDRB/TTIA.
- Previous opener -> next first-word remainder after current opener/previous ending/length: ZLZI **+0.16 SD**, unresolved; first-word final **-0.43 SD**, unresolved; alternate transcriptions likewise unresolved.
- d/q/y under a fixed-diagonal degree-preserving null (all self-transitions fixed; off-diagonal source/target marginals preserved within Currier×hand×paragraph-position): effect +109.571 transitions / null SD 21.0859 = **+5.20 SD**.
- Frequent-opener Hodge decomposition: total directed energy excess **+11.58 SD**; global-order/gradient component excess **+20.91 SD**, but it explains only 12.64% of observed energy; non-gradient/cyclic residual excess **+9.86 SD**. Thus a monotone opener hierarchy exists but does not explain most directed structure.
- Historical physical-line control Enikel Weltchronik ONB Cod. 2921 (1397), N=3231 adjacent lines: opener-after-ending/length effect -0.000554 bits / null SD 0.011258 = **-0.05 SD; metric does not resolve**.

This refinement does **not** add a new mechanism claim. It narrows the licensed SGT13 architecture. The frozen Nuremberg Letterbooks 2–5 control remains the decisive external control still to replay; the public Zenodo dataset is 262 MB and could not be re-fetched in the current runtime.

Canonical deep-dive handoff: `voynich_vertical_opener_architecture_deepdive_20261002_v01`.


## Literal d/q/y wheel simulation — 2026-10-02

A deliberately literal visible-state wheel was built and tested on true ZL3b paragraph sequences. From d/q/y, the process can stay, rotate clockwise d→q→y→d, rotate anticlockwise, or exit to a background opener state X; X can re-enter d/q/y or remain background. Paragraphs are independently seeded.

Five-fold held-out by folio number mod 5:
- IID: 3.348850 bits/opener
- literal wheel: 3.009618 bits/opener
- unrestricted 19-state first-order Markov: 2.962800 bits/opener

The literal wheel captures **87.87%** of the unrestricted Markov gain over IID while using ~39 vs ~360 free parameters (~10.8%). Fitted all-data moves: stay .09337, clockwise .22963, anticlockwise .13735, exit .53965. Clockwise > anticlockwise in every held-out training fold.

In 1,000 generated corpora preserving paragraph lengths:
- observed d/q/y directional difference +128; simulated mean +129.226, SD 21.510; effect -1.226 = **-0.057 SD, unresolved**.
- observed forward share .625984; simulated .627588, SD .020437; effect = **-0.078 SD, unresolved**.
- observed same-opener rate .072159; simulated .079957, SD .004683; effect = **-1.67 SD, unresolved**.
- observed full opener-transition MI .184656 bits; simulated .074655, SD .006147; excess **+0.110001 bits = +17.90 SD**.

Adjudication:
- **Literal d/q/y wheel as a compact component: strongly supported.**
- **Three-anchor wheel as the complete SGT13 mechanism: falsified.**

The correct next mechanism class is a constrained multi-state / multi-ring cyclic opener machine, because the d/q/y wheel captures most held-out predictive gain but cannot reproduce the full directed transition matrix.

Code: `research/sgt13_literal_wheel_sim_20261002.py` (commit `10f186ca8f858d974c299182050cb60b9804d2d5`).
Canonical result handoff: `voynich_sgt13_literal_wheel_simulation_20261002_v01`.
