# L recovery v0.1b — calibration diagnosis

## RETRACTIONS / NON-CONTROLLING EXPLORATION

- v0.1 remains permanently quarantined for the ReF token-unit defect.
- v0.1b controlling result remains `L_RECOVERY_DEVELOPMENT_NOT_RESOLVED`. Nothing below changes that result.
- A same-random-map paired contrast gives 10/10 family calls at |z|>=1, but this is **not an admissible replacement metric**: the two language solvers return distinct mappings (median cross-model positional agreement about 7/26), so a 100%-shared-mapping null understates the relevant contrast variance. It is retained only as an upper-bound diagnostic.
- A previous three-image pagination read for Wagenseil scan 00395 as p.292 was wrong; sequential audit shows 00394=p.292 Hebrew and 00395=p.293 Latin.

## Frozen v0.1b result

Primary separately-standardized margin `z_Y - z_G`:
- family accuracy 7/10, with 3 abstentions and 0 wrong-language calls;
- exact balanced-label effect +0.350 with null SD 0.130171 = 2.689 null SD; exact one-sided p=1/252;
- correct-language M0 recovery gate passed in all 10 relationship groups; wrong-language recovery gate passed in all 10;
- unigram and within-word-shuffle nuisance gates passed;
- deployable-accuracy and leave-one-family-out gates failed.

German abstentions were F037 (-0.9490), F018 (-0.9775), F034 (-0.9941) against the frozen -1.0 cutoff.

## Source of the calibration compression

For each model/case let `s_L` be returned-map audit score, `mu_L` the random-map null mean, and `sd_L` its null SD. v0.1b used `(s_Y-mu_Y)/sd_Y - (s_G-mu_G)/sd_G`.

The null-centered **raw** contrast

`D = (s_Y-mu_Y) - (s_G-mu_G)`

has the truth-consistent sign in **320/320 primary planted-key cases** across all ten development families. This is not a new gate; it is a diagnosis of where the information is lost.

Median family D values:
- German: F016 -0.508; F018 -0.309; F034 -0.398; F037 -0.403; F148 -0.344.
- Yiddish: bovo +0.685; cracow +0.774; kine +0.815; lev_tov +0.773; sam_hayyim +0.710.

The model-specific random-map baselines systematically favour the German model in raw score units. Subtracting those baselines is sensible, but subsequently scaling the two models separately and subtracting z scores is not the only possible contrast calibration.

## Principled alternative contrasts (diagnostic only)

### Independence-conservative variance

`Z_ind = D / sqrt(sd_Y^2 + sd_G^2)`

This treats the two returned mappings as independent under the null. At the unchanged |Z|>=1 threshold it gives 8/10 family calls, with two German abstentions (F018, F148), 0 wrong calls. Exact balanced-label effect is 0.400 over null mean with null SD 0.145297 = 2.753 null SD; exact one-sided p=1/252. It still fails the original leave-one-family-out rule, so it cannot rescue v0.1b.

### Mapping-overlap-matched variance

The two returned mappings share only about 5–8 of 26 cipher→plaintext assignments per family. A Monte Carlo null conditioning random map pairs on the observed positional-overlap count gives 9/10 at |Z|>=1; only F018 abstains. This is exploratory because the null construction was devised after outcomes.

### Same-map paired variance

Using the identical random map under both language models gives 10/10 at |Z|>=1, but is too optimistic for two distinct returned mappings and is therefore rejected as a controlling calibration.

## Nuisance comparison under D / Z_ind

- Primary D sign: 320/320 truth-consistent key cases.
- Unigram control: 160/320 truth-consistent; it points Yiddish on every family.
- Within-word-shuffle: 209/320 truth-consistent; German discrimination largely collapses.

Thus the stable directional signal is sequence-dependent and is not explained by the unigram nuisance.

## Controlling interpretation

The v0.1b failure is **partly calibration-driven but not fully removable from the same data**. F018 is the persistent borderline German family under more realistic contrast nulls. No threshold or statistic is changed retrospectively.

Next permissible step: freeze a new v0.2 transfer statistic before any new-family scoring, then test it on source-disjoint-from-L families. This transfer panel may use works consumed elsewhere in R, but must be labelled as such and cannot issue final L qualification. A genuinely fresh external historical-Yiddish family remains necessary for confirmation.

Voynich/C7 remains sealed.