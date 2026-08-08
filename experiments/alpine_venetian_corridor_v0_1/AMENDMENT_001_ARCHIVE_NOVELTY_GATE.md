# AMENDMENT 001 — Voynich Archive Prior-Art / Novelty Gate

Date: 2026-08-08  
Status: **MANDATORY**  
Effect on preregistered endpoint: **NONE**. This amendment changes research workflow and novelty claims, not geography, chronology, feature families, thresholds, controls, or statistics.

## Requirement

Before external catalogue research is interpreted, and before any candidate/evidence item is described as new, the programme MUST search Ed's internal Voynich archive (`public.sources` + `public.source_passages`) for:

1. the manuscript title and shelfmark, including spelling variants;
2. production locality and historic/modern place-name variants;
3. named people/workshops/institutions attached to the item;
4. the specific visual or documentary feature being claimed;
5. the proposed transmission/provenance relationship;
6. obvious synonym formulations of the claim.

The archive search is proactive. It is not postponed until after an interesting external result is found.

## Why this amendment exists

A build-time archive scan shows that the broad geographical idea is already substantial Voynich prior art. The programme must not rediscover it and call it novel.

### Prior-art examples already found

- **1997 — Venice/Padua as a plausible North-Italian intellectual crossroads.** Jorge Stolfi discussed Venice as a fifteenth-century cosmopolitan centre with Padua nearby (`voynich_nu_list_a1997h`, para 48).
- **1999/2002 — northern-Italian / Alpine castle morphology.** René Zandbergen and others discussed Verona, Padua, Lake Garda and Alpine/North-Italian swallowtail-merlon geography (`voynich_nu_list_a1999i`, para 12; `voynich_net_arch_2002_04`, paras 62–69).
- **2001 — Carrara Herbal + Venetian Liber de Simplicibus/Rinio as VMS botanical comparanda.** Nick Pelling cited the two Veneto herbals and Cathleen Hoeniger's work on naturalism around Padua/Venice (`voynich_nu_list_a2001k`, paras 42–43).
- **2011 — explicit Tyrol/Trento German–Italian synthesis.** René Zandbergen argued that German marginal/annotation evidence and a North-Italian origin were compatible, explicitly naming Tyrol and noting an alchemical herbal then kept in Trento (`voynich_nu_list_a2011a`, para 46). He later developed the Italian/German borderland argument and Runkelstein/Bolzano (`voynich_nu_list_a2011b`, para 164).
- **2018–2021 — Trento/Bolzano visual culture.** Runkelstein, Buonconsiglio, the Trento–Bolzano/Val d'Adige connection, costume/fresco comparisons and related Tacuinum material were discussed repeatedly (`voynich_ninja_thread-2495`, para 20; `voynich_ninja_thread-3339`, paras 15–17).
- **2021 — Brenner Pass explicitly identified as the principal Italy–Northern Europe route** in the merlon discussion (`voynich_ninja_thread-3643`, para 81).
- **2022 — published southern-German/northern-Italian context.** Keagan Brewer explicitly treated the rosettes merlons as evidence for those contexts and cited Padua Cod. 74 among the comparanda (`brewer_2022_emotions_encipherment`, para 6).
- **2024 — Brixen institutional/person lead.** Ulrich Putsch, Bishop of Brixen and servant of Friedrich IV of Tyrol, was raised as a person of interest (`voynich_ninja_thread-4298`, para 23).
- **2026 — the exact modern corridor intuition is already being articulated.** Archive discussions connect Alpine/German background with Italy/Padua/Venice, and one post explicitly frames Bozen/Bolzano and the Brenner axis as the intersection of Alpine and North-Italian architectural signals (`voynich_ninja_thread-5355`, para 30; `voynich_ninja_thread-3643`, para 503).

Therefore **"the VMS may come from an Alpine–Venetian German/Italian interface" is PRIOR ART, not a programme discovery.**

## What can still be genuinely new

The programme should preferentially seek:

1. **Previously undiscussed manuscripts** satisfying the neutral 1350–1500 corridor census.
2. **Previously unseen folios/images** in known manuscripts, especially incidental drawings not described by catalogues.
3. **New documentary links**: ownership, commissions, loans, copying relationships, itineraries, university records, book inventories, workshop links, correspondence, or bindings connecting route nodes in 1390–1450.
4. **New transmission edges**, e.g. a securely documented illustrated exemplar moving Trento/Bolzano → Verona/Padua/Venice or the reverse.
5. **New negative evidence**, such as systematic absence of a supposedly diagnostic feature where the corridor hypothesis predicts it.
6. **New measurements** on known material: blind feature coding, image-normalised morphology, manuscript-level effect sizes, control-region comparisons.
7. **A genuinely controlled corridor test.** The archive contains many qualitative comparisons and geographic arguments, but the present programme's bounded neutral census + matched controls + blind multi-family scoring + manuscript-level permutation design is treated as a methodological extension unless an archive search finds an earlier equivalent.

## Operational novelty classes

Every interesting item is entered in `corridor_novelty_register` as one of:

- `new_manuscript`
- `new_image`
- `new_documentary_link`
- `new_transmission_link`
- `new_feature_measurement`
- `new_negative_evidence`
- `new_quantitative_result`
- `replication`
- `prior_art`

And receives one prior-art verdict:

- `no_prior_art_found`
- `partially_anticipated`
- `already_discussed`
- `replication_or_extension`

`no_prior_art_found` means only **not found in the archived corpus under the recorded searches**. It is never phrased as globally unprecedented.

## Candidate gate

No candidate may move from `needs_review` to `included` until its archive scan has run. Archive discussion does NOT exclude a candidate from the statistical census; doing that would create selection bias. Instead it changes only the novelty classification and research priority.

Priority order for manual/external follow-up:

1. eligible candidate with no archive prior art;
2. known candidate with an untested documentary/image lead;
3. known candidate useful for replication/control;
4. heavily discussed candidate with no new evidential angle.

## Reporting rule

Final reports have two separate columns:

- **Evidence for/against corridor hypothesis**
- **Novelty relative to the Voynich archive**

A strong result can be non-novel; a novel item can be weak evidence. The two must never be conflated.
