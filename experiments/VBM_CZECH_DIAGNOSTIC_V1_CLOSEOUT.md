# VBM Czech diagnostic v1 — closeout

Date: 2026-08-12
Namespace: `VBMCZECHDIAGV1`
Branch: `experiment/vbm-language-extension-cz-v1-20260812`
Frozen execution commit: `be13c2a03e4aa430cb96b1614cade767e0208716`
GitHub Actions run: `31578853569`
Job: `94057079511`

## Binding diagnostic disposition

**CLOSED: CZECH QUALIFIES SYNTHETICALLY BUT IS STRONGLY DISFAVOURED ON VOYNICH FIT; SURFACE MARKOV BEATS ALL TESTED LATENT LANGUAGES IN 6/6 FOLDS.**

This is an exploratory language extension of an already-closed VBM programme. It does not reopen the latent-transducer cipher hypothesis and it does not alter the established latent-vs-stable-surface identifiability limitation.

## Czech source

Pinned source: Universal Dependencies `UD_Czech-CAC`, commit `798f89716ae5a96e86042df7d394d56787e2e213`.

Frozen VBM normalization was applied unchanged.

- train: 23,478 sentences; 2,237,650 normalized characters;
- control: 1,231 sentences; 105,325 normalized characters;
- train SHA-256: `b089aeebc4a2a32b0cc07767e2704dc2ea61410970c45f0ff8b01e223c12d154`;
- dev SHA-256: `06ab3b41d0192641a063048c58f27ecc10640b61c0bae3112dd778ea1f4201f7`;
- test SHA-256: `71c03ef1ab14451e294bbc0d7896ea4f8a3faf26790fc2f04cc473e62e43162c`.

This is modern normalized Czech under the frozen 19-state VBM alphabet, not a dedicated Old Czech model.

## Q0 synthetic recognisability

Both preregistered same-key Czech controls qualified under the inherited v4 language-vs-surface-null rule.

### UNIFORM + FLAT

- Czech held-out score: -4.0343455016 nats/event;
- Bavarian: -4.2196454709;
- German: -4.2101988800;
- Czech margin over runner-up: +0.1758533784;
- best surface null: iid, -4.3868580145;
- Czech delta vs surface null: +0.3525125128;
- paired score gap: 0.0008039892;
- winner A/B/mean: Czech/Czech/Czech;
- qualified: PASS.

Hidden-state recovery was only 0.04939 in this deliberately flat homophone control; that quantity is diagnostic and was not an inherited v4 qualification gate. Language discrimination itself was strong and stable.

### FREQ_PROP + CYCLE

- Czech held-out score: -4.1783725288;
- Bavarian: -4.3497167455;
- German: -4.3518922119;
- Czech margin over runner-up: +0.1713442167;
- best surface null: typed_hier_o1, -4.5213346598;
- Czech delta vs surface null: +0.3429621309;
- paired score gap: 0.0000642919;
- hidden-state recovery: 0.9687017544;
- winner A/B/mean: Czech/Czech/Czech;
- qualified: PASS.

Thus Q0 passed and exploratory FIT access was permitted.

## Q1 exploratory Voynich FIT

Exact inherited v6 deterministic sixfold FIT split, all 181 retained FIT folios.

Per-fold latent winner:

- fold 0: German -3.0201995693; Czech -3.2810026488; best surface null -2.0783342429.
- fold 1: German -3.0107600144; Czech -3.2557922046; best surface null -2.0410560027.
- fold 2: Bavarian -3.0293239507; Czech -3.2691814505; best surface null -2.0647822420.
- fold 3: Bavarian -3.0431208528; Czech -3.3061864071; best surface null -2.1247548953.
- fold 4: German -2.9927303863; Czech -3.2574162101; best surface null -1.9905337868.
- fold 5: Bavarian -3.0224605714; Czech -3.2816384604; best surface null -2.0805715013.

Aggregate:

- latent wins: Bavarian 3/6, German 3/6, Czech 0/6;
- Czech beats best(Bavarian,German): 0/6;
- Czech beats surface null: 0/6;
- any tested latent language beats surface null: 0/6;
- median Czech score: -3.2750920497;
- median best Bavarian/German score: -3.0213300703;
- median Czech minus best(Bavarian,German): **-0.2599904843 nats/event**;
- median Czech minus surface null: **-1.2035338072 nats/event**;
- median best-latent minus surface null: **-0.9532153894 nats/event**.

The best surface model was `markov_hier_o2` in every fold.

All target latent HMM fits were formally nonconverged under the inherited stopping criterion. This reproduces the optimisation weakness already seen in the earlier VBM language-vs-null target work and is an additional reason not to overinterpret relative latent scores. It cannot rescue Czech because the negative margins are large and uniform across all six folds.

## Interpretation

Under the frozen VBM representation and transducer:

1. the instrument can recognise Czech when Czech actually generated the synthetic ciphertext;
2. actual Voynich FIT does not prefer Czech — Czech loses to Bavarian/German in every fold by a median ~0.260 nats/event;
3. more importantly, a second-order hierarchical surface Markov model predicts Voynich substantially better than Czech, Bavarian or German in every fold.

Therefore there is **no VBM Czech signal**. This result also reinforces the pre-existing conclusion that relative latent-language ranking inside VBM is not cipher-specific evidence when a strong surface process explains the observations materially better.

## Disposition

- Q0: PASS.
- exploratory FIT: COMPLETE, strongly negative for Czech.
- H1: not run.
- C1: not run.
- decoding/mapping/key interpretation: not run.
- no parameter tuning or rescue.
- no successor experiment proposed or run.

Diagnostic closed.
