# Rhine–Bodensee–Upper Rhine Corridor v0.2 — Frozen Protocol

Run label: `rb_corridor_v02_20260904_run01`
Frozen: 2026-09-04
Seed: `20260904`
Base programme: `experiment/alpine-venetian-corridor-v0.1-20260808` at `9596e174bf187bd14ed44b7bfc00889877bf3da9`

## Research question

Does the Rhine–Bodensee–Upper Rhine system provide a better evidenced production ecology for the Voynich Manuscript than matched alternative regions, while still providing documented mechanisms for its Italian-derived learned inputs?

This is not a search for a literal provenance, castle, patron, or named author. A candidate region receives no credit merely for being cosmopolitan or geographically convenient.

## Prior state and leakage register

Before this freeze, the following were already known/exposed:

1. The closed Alpine–Venetian corridor v0.1 programme and its `NONRESOLVING_REFERENCE_SET` outcome.
2. The manuscript-registry metadata were queried neutrally by date/place. This surfaced several possible corridor witnesses, including registry records describing Bodleian Canon. Misc. 378 as Basel 1436 with French/Italian production elements, Würzburg M.ch.f.232 as Padua/Basel, and multiple Basel/Hagenau/Konstanz records. These are metadata leads only and must be independently verified before evidential use.
3. `cmp_monthname_manuscripts` was identified and queried for `Aberil`, `Augst`, `Octembre`, and `Novembre`; no inferential result was interpreted before freeze.
4. Known historical leads include Montfort–Brandis links, Basel, Hagenau/Lauber, and the uploaded Upper Rhine/Alsace corridor summary. These are hypothesis-generating exposures and cannot be treated as blind discoveries.

No visual or Voynich-similarity score from the above exposures is used to define the candidate cohorts.

## Chronology

- Primary production window: 1404–1438.
- Antecedent context: 1390–1403.
- Immediate reception/context: 1439–1450.

Only the primary window can support the production-environment endpoint. Earlier/later evidence may establish transmission or institutional continuity only.

## Geographic strata

### Candidate corridor

- `RB_BODENSEE`: Konstanz and the Bodensee rim; St Gallen/Schaffhausen where institutionally connected.
- `RB_ALPINE_RHINE`: Chur, Sargans, Vaduz/Schellenberg, Feldkirch, Bregenz and documented connecting routes.
- `RB_UPPER_RHINE`: Basel, Breisgau/Freiburg, Strasbourg, Hagenau/Haguenau, Upper Alsace/Sundgau.

### Matched alternatives

- `IT_NORTH`: Padua, Venice, Verona, Milan, Pavia.
- `DE_SOUTH`: southern German centres outside the corridor, including Augsburg, Nuremberg, Munich and Regensburg.
- `SAVOY_BURG`: Savoy/Burgundy and adjacent Franco-Alpine contact zones.
- `EAST_ALPINE`: Tyrol, Salzburg, Vienna and comparable east-Alpine networks.

## Evidence families

### A. Calendar geography / orthography — primary

Test the *joint month-name sequence* rather than cherry-picked single spellings. Compare dated 1390–1450 witnesses across the strata. Required controls: genre/function, language, date, scribal copying versus local composition where knowable, and manuscript-family dependence.

Primary test: whether the Voynich-like combination of German-compatible and French-compatible forms is more concentrated in the candidate corridor than matched alternatives. Single-form matches cannot carry the endpoint.

### B. Documentary / prosopographic connectivity — primary

Build dated person–institution–place–manuscript edges from charters, colophons, accounts, conciliar records and catalogues. A valid mechanism requires explicit documentary movement or co-presence. Genealogical or political adjacency without movement counts only as contextual evidence.

Primary test: candidate-corridor nodes must exhibit direct, dated links joining at least two otherwise independent requirements (e.g. German/Alemannic scribal milieu + Italian-trained scribe; north–south manuscript transfer; learned-text circulation).

### C. Manuscript ecology — primary

Construct a neutral date/place census of manuscripts from the candidate and control strata before selecting visually Voynich-like items. Characterise languages, materials, collaborative production, secular/scientific genres, illustration practices and cross-regional hands.

Primary test: compare prevalence of the required production constraints using matched controls. Holdings and modern digitisation cannot substitute for production origin.

### D. Learned-source transmission — primary

Trace pre-1439 transmission into or through candidate/control strata for independently motivated source traditions already identified in prior Voynich work: `De Balneis Puteolanis` / balneological illustration; `La Sfera` / related cosmography; German herbal/Kräuterbuch traditions and Italian-derived herbals; Kyeser/`Bellifortis` and closely related technical imagery. Taccola/Oresme and other source families may be tracked but are secondary unless independent witness-level links are demonstrated.

Primary test: dated, source-specific transmission evidence. Generic intellectual traffic does not count.

### E. Production/codicology — constraint, not localisation endpoint

Known multiple-hand evidence motivates a coordinated-production constraint. Fair-copy/exemplar production remains an explicit hypothesis, not an assumption. Testable signatures include hand interleaving, correction/eye-skip patterns, shared layout rules, ink/quire phase relationships and consistency of unusual script forms.

### F. Architecture / visual typology — exploratory only

No literal castle identification is allowed as primary evidence. Architectural material may be used only as a broad compatibility constraint or as a predeclared comparison against matched visual corpora. The v0.1 image-confound gate failed; no embedding-based visual result enters the primary endpoint until a new instrument passes its own source/confound gate.

## Null and alternative

`H0`: after matching on chronology, genre/function, material, manuscript-family dependence, preservation/holding and digitisation opportunity, the Rhine–Bodensee–Upper Rhine corridor does not outperform alternative production ecologies on the independent evidential constraints.

`H1`: the candidate corridor shows reproducible convergence across at least three independent **nonvisual** evidence families, including at least one direct documentary/transmission mechanism, and the result survives control-region, leave-one-node-out and leave-one-family-out tests.

## Dependency / anti-circularity rules

1. A manuscript or person discovered because it resembles Voynich cannot enter the neutral ecology denominator.
2. Source traditions used to motivate the corridor cannot count both as candidate selection criteria and as independent confirmation without a held-out witness/test.
3. Multiple manuscripts from one textual family or workshop are clustered as one dependency unit for inferential tests unless independence is demonstrated.
4. Modern holding institution is never treated as medieval geography.
5. A council, court or city is not credited for 'cosmopolitanism'; named/datable people, manuscripts or transfers are required.
6. Italian-derived content and Germanic production are separate variables. Evidence for one cannot be used as evidence against the other.
7. Missing evidence is not negative evidence unless the relevant archival/catalogue coverage is measured and comparable.

## Required robustness sequence

Before interpretation: circularity → leakage → confounds → matched nulls → control fairness → measurement degeneracy → representation dependence → decision-rule fragility → audit completeness.

For every numerical comparison report effect size and matched-null standard deviation in the same sentence. If `effect_size / null_SD < 2`, lead with: **the metric does not resolve this**.

## Decision rule

Primary support requires all of:

1. >=3 independent nonvisual evidence families favour the corridor under their preregistered tests.
2. >=1 family supplies a direct dated mechanism rather than compatibility alone.
3. No single node (e.g. Basel) or single source family accounts for the conclusion under leave-one-out analysis.
4. North Italy remains explicitly tested as a production alternative and not merely as a source reservoir.
5. Negative/retracted findings remain visible at the top of `RUNNING_RESULTS.md`.

Otherwise verdict is `NONRESOLVING`, `H0_NOT_REJECTED`, or `H1_REJECTED` as appropriate. No provenance claim is permitted under this protocol.
