# RETRACTION / CORRECTION — POSITION STATE — 2026-10-02

RETRACT any unqualified wording that line position is fully reducible to INITIAL/MEDIAL/FINAL. The earlier closure was conditional on current root. Independent five-transcription replay shows strong fine-position effects on raw token edge distributions after I/M/F alone (ZLZI opener effect +0.013695 bits / null SD 0.0003063 = +44.71 SD; final +0.002327 / 0.0003223 = +7.22 SD; all alternate transcriptions replicate). Correct architecture: **fine line position controls root/repertoire selection; conditional on the chosen root, surface morphology is adequately represented by I/M/F at current resolution.**

# Super-Grammar small-state architecture programme — 2026-10-02

Status: exploratory architecture freeze; not a certified theorem.

## Core proposition

The current Super Grammar is parsimoniously represented as a composition of small local state machines rather than a large lexical/page transition codebook.

## Frozen module specification

### 1. Global mode
- Section
- Currier A/B
- Davis hand is not an independent mode after section+Currier.

These global coordinates may shift root/edge/operator frequencies, but do not get separate token-internal order-2 grammars.

### 2. Paragraph / line opener
- Paragraph reset.
- First-order left-margin state.
- d/q/y clockwise-biased core wheel.
- Anchor-specific exit spokes.
- Off-wheel source-specific re-entry routing.
- At line break: horizontal junction channel attenuates/resets while vertical opener state carries.

### 3. Within-line position
Exactly three states:
- INITIAL
- MEDIAL
- FINAL

Finer relative position is not resolved after this state for token initial/final atoms.

### 4. Within-token graphotactics
- Order-2 atom memory.
- No order-3 state.
- Certified hard local forbidden transitions.
- q-present boolean switch for k/t choice.
- short i-run-length counter for termination.

Independent 5-transcription replay:
ZLZI order2 gain +0.169802 bits/atom; order3 increment -0.012224.
ZLZB +0.169802 / -0.012094.
TTLI +0.195848 / -0.012421.
VDRB +0.167700 / -0.016998.
TTIA +0.179611 / -0.012412.

### 5. Within-line token junction
Candidate compact state:
- previous initial atom
- previous final atom
- previous token length S/M/L
- previous position I/M/F
- previous-current root edit relation ED0/1/2/3/>3

On current token-work representation, exact previous-token identity after these states:
effect +0.003440 bits / null SD 0.001810 = +1.90 SD.
The metric does not resolve this.

Caveat: certified SG11 records a small TTLI exact-identity residual, so this closure is representation-local pending replay.

### 6. Within-line recurrence / neighbourhood clock
Four states:
- lag 1
- lag 2
- lag 3-5
- lag 6-12

After this band:
exact-repeat exact-lag residual = +1.21 SD, unresolved.
ED0/1/2/3/>3 exact-lag residual = +0.80 SD, unresolved.
ED<=3 exact-lag residual = -0.35 SD, unresolved.

Do not conflate with page-scale SG09 R64.

### 7. Page-scale recurrence
Retain certified SG09 banded same-page R64 channel as a separate module until independently compressed.
Certified evidence:
banded R64 vs no reuse +0.00567153 / null SD 0.00253780 = +2.2348 SD, positive 5/5.
banded vs uniform +0.00208411 / null SD 0.000937989 = +2.2219 SD, positive 5/5.

## Cross-module closure results

- Previous line ending after previous opener known: -0.78 SD, unresolved.
- Order-2 opener memory after current opener: +1.31 SD, unresolved.
- Vertical positions 2-5 after horizontal-left conditioning: all <1 SD.
- Paragraph-boundary opener carry: unresolved.
- Fine line position after I/M/F: ~1.80 SD initial, ~0.91 SD final.
- Exact previous token after compact junction state: 1.90 SD on current layer.
- Hand after section+Currier/local controls: ~-0.21 SD.
- Token-internal section-specific order2 kernel worsens heldout prediction by 0.005776 bits/atom.
- Currier-specific order2 kernel worsens by 0.009922 bits/atom.

## Certified small switches already in v03

Q switch:
q-presence changes k/t beyond exact continuation tail + section.
+0.0203372 bits/event / null SD 0.00463778 = +4.385 SD.

I-run counter:
run length changes termination beyond local raw/atomic context.
~6.05-6.24 null SD across frozen controls.

## Unified production topology

GLOBAL MODE(section, Currier)

PARAGRAPH:
    reset opener state
    seed O1

FOR EACH LINE:
    choose O_i from O_{i-1} via wheel+spokes
    set horizontal position state = INITIAL
    generate token using:
        order2 atom machine
        q switch
        i-run counter
        junction morphology state if not first token
        recurrence clock
        global mode parameter shifts
    transition position -> MEDIAL
    ...
    final token uses FINAL state
    line break:
        discard/attenuate horizontal junction state
        retain O_i as vertical state for next line

PARAGRAPH BREAK:
    discard vertical state and reseed.

## Falsifiers

1. A detailed variable must add >=2 null SD after its proposed compact parent state to require another state.
2. New module must improve held-out data, not only training fit.
3. Alternate transcription/decomposition must not materially reverse a claimed closure.
4. A forward generator must recover multiple independent SG metrics without metric-specific patches.
5. Page-scale R64 remains outside the compressed architecture until a valid closure is run.

## Next executable experiment

Modify the existing PGCS/Gen-SP architecture under a strict ablation:
- replace static line-start sampling with frozen wheel+spokes;
- replace suffix->prefix lexical lookup with compact junction state;
- retain shared order-2 atom grammar;
- use I/M/F only;
- add frozen recurrence clock and the two certified local switches;
- do not add page/folio codebooks;
- score on the existing 84-metric harness.

Baseline Gen-SP: 59/84 (historical S3 run).
The new generator must be evaluated against that baseline and the unrestricted richer models without per-metric tuning.
