# VBM v8 — bounded prequential / MDL comparison protocol

Date: 2026-08-11
Branch: `experiment/vbm-prequential-mdl-v8-20260811`
Namespace: `VBMPREQMDLV8`

## Scope

This is the final bounded experiment in the current VBM Bavarian/German homophonic-transducer programme. There will be no successor chase from this branch. The experiment either qualifies and is scored exploratorily on the already-consumed Voynich FIT corpus, or it fails synthetic qualification and closes without target access.

No plaintext, per-symbol decode map, or C1/H1 target scoring is permitted.

## Question

Does a historically constrained Bavarian/German latent homophonic-transducer model achieve a shorter block-prequential predictive description than flexible surface-only models, after both are selected on the same warm-up material and repeatedly refitted only on prior blocks?

This is an explanatory-compression test, not an identifiability test.

## Data and controls

Synthetic positive families:
- `BAV_GLOBAL`: reusable Bavarian CYCLE homophone key.
- `GER_GLOBAL`: reusable German CYCLE homophone key.
- `BAV_GLOBAL_SWAP`: reusable Bavarian key plus sparse same-type adjacent swaps.

Synthetic negative / adversarial families:
- `BAV_FRESH`, `GER_FRESH`: fresh key per pseudo-folio.
- `MARKOV1`, `MARKOV2`, `MARKOV3`: stable non-language n-gram generators fitted to an independently generated reusable-key surface stream.
- `SLOT5`: stable periodic surface generator.

The Markov adversaries are binding. A latent model that compresses them as if they were ciphertext is not a qualified cipher-specific explanatory instrument.

## Block-prequential code

Each synthetic replicate has 12 pseudo-folios. Their order is frozen by a namespace hash.

Warm-up:
- folios 0–1: architecture/language fitting seed.
- folios 2–3: model-family selection validation.
- folios 0–3 are common warm-up and excluded from the comparative predictive code length.

Latent model selection:
- candidates: Bavarian and German fixed external language transition models;
- fit the emission transducer on folios 0–1;
- choose the language with better held warm-up likelihood on folios 2–3;
- freeze that language for the rest of the replicate.

Surface model selection:
- candidates: hierarchical surface Markov orders 1–5, typed and untyped; periodic slot models p=2..8;
- fit on folios 0–1 and choose by predictive likelihood on folios 2–3;
- freeze only the architecture/order, not its parameters.

Coding stage:
For each folio i=4..11:
1. refit latent emission parameters on folios 0..i-1, keeping the selected language fixed;
2. refit the selected surface architecture on folios 0..i-1;
3. score folio i under both models without updating on it;
4. accumulate negative log-likelihood in nats.

Primary statistic:

`PREQ_ADV = (NLL_surface - NLL_latent) / coded_events`

Positive values favour the latent transducer. A selector-header-adjusted version is secondary only:

`PREQ_ADV_HEADER = PREQ_ADV + (ln(17)-ln(2))/coded_events`.

No BIC/AIC parameter-count term is added; the prequential held-block loss is the binding complexity control.

## Synthetic qualification

Smoke is nonbinding.

Formal CAL uses 3 fresh replicates per family. It qualifies only if:
- the minimum positive-family median `PREQ_ADV` exceeds the maximum negative/adversarial median `PREQ_ADV`;
- a threshold `TAU_PREQ` can therefore be frozen at their midpoint;
- at least 8/9 positive CAL replicates have `PREQ_ADV > 0`;
- Bavarian/German global-key controls select their true language in at least 5/6 CAL replicates combined.

Untouched VAL uses 3 new replicates per family and passes only if:
- at least 8/9 positives exceed `TAU_PREQ`;
- each positive family has at least 2/3 passes;
- all 18 negative/adversarial replicates remain below `TAU_PREQ`;
- at least 5/6 BAV_GLOBAL + GER_GLOBAL VAL replicates select the true language.

Any binding Markov false positive closes the programme before Voynich access.

## Voynich disposition

Only after synthetic CAL+VAL qualification, run the same frozen block-prequential comparison on the existing 181-folio FIT corpus. FIT is already consumed, so this is explicitly exploratory, not confirmatory.

Voynich folios are placed in one deterministic namespace-hash order. The first 20% are warm-up, split equally for fit/selection; the remaining 80% are coded in eight approximately equal chronological-by-hash blocks. Language and surface architecture are frozen after warm-up selection; parameters are refit cumulatively before each block.

No C1/H1 access. No plaintext or mapping extraction.

## Closeout rule

Close this branch after the first complete disposition: synthetic failure, or synthetic qualification plus one exploratory FIT result. Do not design or run a successor experiment from this result.