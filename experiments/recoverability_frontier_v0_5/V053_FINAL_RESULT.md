# Recoverability frontier v0.5.3 — final neural/classical homophonic bake-off result

Date: 2026-07-16

Verdict: **NO FRESH-KEY ARM PASSED THE COMPLETE LOCKED PROGRAMME. CLOSE THIS HOMOPHONIC SOLVER LINE AND MOVE TO A DIFFERENT CIPHER FAMILY.**

No Voynich text was scored.

## Frozen question

Can bounded frequency-adaptive homophonic substitution with a fresh independently sampled key be recovered reliably from 384 normalized English plaintext characters, using only train-partition language information, first-occurrence recurrence input and an inferred rather than true homophone inventory?

Development used corpus `dev` replicates 0–7. Eligible methods were to be tested once on untouched corpus `test` replicates 128–147.

Development gate:

- mean recovery at least 70%;
- median recovery at least 90%;
- at least 7/8 trials at least 70%.

Locked test gate:

- mean recovery at least 70%;
- median recovery at least 90%;
- at least 16/20 trials at least 70%.

## Arm A — strict classical restart curve

Development job: `Digitalgoldfish79/6a585bafb1669a49bf076b69`

Development SHA-256: `bdb967dfbfbacdfe62b4417c251026cc3f54a3d75ac3fb9808416ea869e16e12`

Architecture:

- fixed inferred homophone-label multiset;
- CrypTool-style exhaustive pair sweeps;
- train-only quadgram objective;
- calibrated initial acceptance 0.05;
- 3,000,000 proposals per restart;
- no inventory mutation.

| Restart prefix | Mean | Median | Trials ≥70% | Development gate |
|---:|---:|---:|---:|---|
| 12 | 55.5339% | 57.0313% | 4/8 | fail |
| 24 | 56.9661% | 57.6823% | 4/8 | fail |
| 48 | 67.5456% | 98.5677% | 5/8 | fail |
| 96 | 67.8711% | 98.5677% | 5/8 | fail |
| 192 | **89.6159%** | **99.3490%** | **7/8** | **pass** |

### Locked test

Job: `Digitalgoldfish79/6a585dd685d9643ce16d5cd3`

Scientific SHA-256: `fa568d3e473729ae55abd1e60e3a0d8cd04a879adec115805fe8e98da5d44055`

At 192 restarts:

- mean recovery: **70.7943%**;
- median recovery: **99.0885%**;
- trials at least 70%: **13/20**;
- trials at least 90%: **13/20**;
- trials at least 95%: **13/20**.

The mean and median passed, but reliability failed the frozen 16/20 requirement. The result remains bimodal: successful basin hits recover essentially the whole plaintext, while seven ciphertexts remain poor despite 192 independent trajectories.

**Locked verdict: fail.**

## Arm B — neural-language-model beam search

### One-layer character LSTM

Job: `Digitalgoldfish79/6a585bbab1669a49bf076b6d`

SHA-256: `5d95943663bff845fe48173bb7f62d42f9558b17951e5bc0d71b846169baa97a`

Best development result, beam 2048:

- mean recovery: 26.2695%;
- median: 21.8750%;
- trials at least 70%: 0/8.

### Two-layer character LSTM

Job: `Digitalgoldfish79/6a585bc485d9643ce16d5cb4`

SHA-256: `beeaaaf5bef5a7a254c31368d05504a897eea667c9f059c34a39afcb31db02aa`

Best development result, beam 2048:

- mean recovery: 24.2188%;
- median: 20.9635%;
- trials at least 70%: 0/8.

The language models learned low-loss character distributions, but first-occurrence incremental assignment pruned the correct key paths.

**Development verdict: fail; no locked test.**

## Arm C/E — fresh-key recurrence Transformer and neural-seeded classical hybrid

### Short architecture diagnostic

Job: `Digitalgoldfish79/6a585bd785d9643ce16d5cb6`

SHA-256: `cfe953b5a57149de601ddba0eba218ecfb0f446721e6030f9e4a27b18fac9c54`

Configuration:

- d_model 384;
- six encoder and two decoder layers;
- 2,500 updates × batch 48;
- 32 posterior-derived fixed-inventory key seeds;
- 250,000 classical refinement proposals per seed.

Results:

- direct greedy recovery: 19.1732%;
- inventory-constrained recovery: 28.4831%;
- hybrid recovery: 99.3164% mean, 99.4792% median;
- 8/8 trials at least 95%.

This passed as a short architecture diagnostic but did not satisfy the preregistered requirement for at least one million fresh-key training examples.

### Preregistered million-example confirmation

Job: `Digitalgoldfish79/6a585cb0b1669a49bf076b8d`

Scientific SHA-256: `be575f677c8b7ff337c174901fca14f23fb89e055d7a8af6313c0d1f32c55443`

Configuration:

- same selected architecture;
- 22,000 updates × batch 48 = 1,056,000 fresh-key examples;
- unchanged 32 posterior seeds and 250,000-proposal refinement.

Results:

- greedy recovery: **70.8659% mean**, 76.0417% median;
- constrained recovery: **74.7070% mean**, 79.8177% median;
- hybrid recovery: **87.9883% mean**, **89.3229% median**;
- hybrid trials at least 70%: 7/8;
- hybrid trials at least 90%: 4/8;
- hybrid trials at least 95%: 2/8.

Longer training materially improved the standalone network but did not preserve the exceptional short-diagnostic basin-seeding result. The frozen median threshold was missed.

**Development verdict: fail; no locked test.**

## Arm D — shared-code-pool positive control

Job: `Digitalgoldfish79/6a585be1b1669a49bf076b73`

SHA-256: `ac1380e8461431c9c04427405526fc44e0076374f61811355fedc31525ead1d4`

- mean recovery: 100%;
- exact recovery: 20/20.

This confirms that the implementation reproduces the easier reused/stable code-pool setting. It is explicitly ineligible as fresh-key evidence.

## Final scientific conclusion

Bounded fresh-key homophonic substitution is **recoverable in favourable search basins**, often at 95–100% character accuracy. Under the tested model class and compute, however, basin discovery is not reliably calibrated across unseen ciphertexts.

The following claims are supported:

1. the true plaintext is often a strong and discoverable optimum under the train-only language objective;
2. successful classical or hybrid runs recover almost the entire message;
3. stable shared code pools make the problem dramatically easier;
4. fresh arbitrary keys create a materially different generalisation problem;
5. neither 192 blind trajectories nor the preregistered million-example neural hybrid achieved the locked reliability requirement.

The following claims are not supported:

- a general homophonic decipherer has passed;
- null-bearing homophonic substitution should now be attempted;
- an unknown text can be identified as homophonically enciphered;
- the Voynich Manuscript should be scored.

## Programme decision

Per the frozen v0.5.3 protocol, stop further ad hoc homophonic search modifications. Do not retrospectively combine failed test outputs or relax the reliability gate.

Proceed to a structurally different family with its own oracle and family-known recoverability programme. The selected next target is **nomenclator substitution**, because it is historically relevant and decomposes into separately testable character-substitution and whole-word code recovery components.
