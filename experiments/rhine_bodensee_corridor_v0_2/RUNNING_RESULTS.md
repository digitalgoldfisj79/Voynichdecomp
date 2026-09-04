# Rhine–Bodensee–Upper Rhine Corridor v0.2 — Running Results

## RETRACTED / DOWNGRADED FINDINGS

These remain at the top of the file for the life of the programme.

1. **Literal castle identification:** downgraded. Prior castle work supports only broad architectural compatibility/typology; no specific building identification is evidential.
2. **Jakob I von Lichtenberg as a major documented Lauber patron / esoteric-medical scholar:** retracted pending primary evidence. Remove from the evidential spine.
3. **Pfirt chancery scribes demonstrably adopting French calendar orthography:** unverified; retain only as a testable mechanism.
4. **Wilhelm V von Montfort-Tettnang personally negotiating face-to-face with Jean de Neuchâtel in 1430:** unverified; do not use.
5. **Alpine–Venetian corridor v0.1:** `NONRESOLVING_REFERENCE_SET`; it neither supported nor rejected the old corridor hypothesis. Do not reinterpret the gate failure as substantive negative evidence.
6. **Vicenza, Biblioteca Bertoliana MS 362 as a 1404–1438 Italian/German herbal bridge:** retracted from the primary window. ManuComp’s loose 1400–1450 metadata made it initially look ideal (Italian alchemical herbal, German material/annotations), but repository-level and specialist dating place the relevant manuscript around the later 15th century, approximately the 1470s. The hybrid phenomenon is real; its date is too late for this test.

## AUDIT NOTES

- The v0.1 database records a closed/nonresolving run, while the inherited GitHub `RUNNING_RESULTS.md` remained stale and described the programme as not yet run. Database/result records are treated as the substantive run state; v0.2 is isolated rather than overwriting v0.1.
- Pre-freeze metadata exposure is recorded in `PROTOCOL.md`. No exposed Voynich similarity result is used for candidate selection.
- **Graph-first rule:** reuse/reference the existing Supabase Siena graph, Paper 3 Padua/Venice and Pavia/Milan environments, and legacy `corridor_*` person/manuscript/movement edges. `rhine_bodensee_v02` stores only new/corrected delta edges.
- Vaduz/Schellenberg is a formal graph component, not prose context. Architecture remains non-primary; the territorial/person network is retained for independent documentary tests.
- **Herbal illustration geography statistic is not admissible as localisation evidence.** In release `2026-08-25-r14`, possible-by-1438 Northern Italy vs German/HRE gives RD = 1.00, exact-permutation null SD = 0.333, effect/null-SD = 3.0; secure-by-1438 gives RD = 1.00, null SD = 0.50, ratio = 2.0. However the Italian rows are heavily drawn from illustration-selected Beasley/Segre-Rutz and Collins traditions, while the German rows come from a broad HSP herbal sweep. The source-ascertainment mechanism is therefore geography-dependent. These numbers describe corpus construction as well as manuscript ecology and are **rejected for geographic inference**.

## CURRENT STATUS

Run: `rb_corridor_v02_20260904_run01`
Stage: `stage2_source_transmission`
Protocol: **FROZEN 2026-09-04**
Primary result: **PROMISING BUT NONRESOLVING. H1 NOT YET PASSED.**

Current graph snapshot after herbal traversal: **50 nodes / 44 edges / 8 environment bridges; 7 bridges verified or qualified-verified at their stated level.**

Current source-family scoreboard:

- Kyeser / `Bellifortis`: **POSITIVE — strong primary-window Upper-Rhine transmission; survives leave-Basel-out.**
- Herbal / materia medica: **QUALIFIED POSITIVE — primary-window German/Salernitan textual-pharmacological integration in the corridor; Italian-derived visual transfer mechanism independently supported, but pre-1439 corridor-local illustrated herbal access remains unresolved.**
- `De Balneis Puteolanis`: **NEGATIVE FOR CORRIDOR SOURCE TRANSMISSION — H0 not rejected.**
- `La Sfera`: **NONRESOLVING — strong person/institution route via Fra Leonardo Dati, but no manuscript custody/transfer edge.**

No source family is double-counted through multiple manuscript witnesses.

## RUNNING FINDINGS

### R00 — Programme inheritance

The prior Alpine–Venetian programme had 12 corridor manuscripts and 10 controls but failed its preregistered Voynich reference-set gate; no primary similarity test was permitted. This is an audit constraint, not evidence for or against the present Rhine–Bodensee hypothesis.

### R01 — Neutral registry census exposure

A neutral 1390–1450 date/place query of the existing manuscript registry surfaced multiple Basel, Hagenau, Konstanz/Bodensee and Upper-Rhine records before v0.2 freeze. Two metadata records were flagged for primary verification: Bodleian Canon. Misc. 378 and Würzburg M.ch.f.232.

Canon. Misc. 378 has since cleared verification sufficiently to serve as a **production/transmission mechanism**, not a Voynich analogue: Pietro Donato's existing legacy graph records Padua 1428–33, Basel 1434–36 and Speyer 1436, while the existing manuscript edge records Canon. Misc. 378 copied for him at Basel in 1436 from the Speyer exemplar.

### R02 — Architecture status

Architecture is removed from the primary endpoint. It can only constrain broad visual compatibility unless a new predeclared visual instrument passes a source/confound gate.

### R03 — Formal Bodensee / Alpine-Rhine / Upper-Rhine graph

Peer views exist in Supabase:

- `public.bodensee_graph_nodes_v01`
- `public.bodensee_graph_edges_v01`
- `public.bodensee_domain_assessment_v01`
- `public.bodensee_environment_bridges_v01`

The Vaduz component formally includes the Montfort–Brandis–Werdenberg chain, Vaduz, Eschnerberg, Alt-/Neu-Schellenberg, and the 27 June 1437 settlement recorded in Staatsarchiv Schwyz HA.II.408. Wilhelm V's 1429 Breisgau/Upper-Alsace and 1433 Basel offices are retained as secondary-verified edges pending primary-document strengthening.

A separate 20 January 1416 edge links Heinrich Stoll, chaplain of St Florin in Vaduz, directly to Pfäfers through his witness role in the abbey's convent statutes. This supports a Vaduz–Pfäfers institutional/balneological environment only; it does not imply `De Balneis` transmission.

### R04 — Kyeser / Bellifortis: source-family positive

The authority census plus catalogue checking identifies primary-window `Bellifortis` transmission in the candidate region:

- BAV Pal. lat. 1994: Southwest Germany (Strasbourg?), c.1410.
- Budapest MTAK K 465 fragment: illuminated in Southern Germany / Upper Rhine region, 1411–1437.
- Karlsruhe Cod. Durlach 11: c.1420–30, Alemannic writing language; no exact production place inferred from language.

Verdict: **verified primary-window Upper-Rhine transmission**. It survives removal of Basel. The witnesses count as one source family, not three independent votes.

### R05 — De Balneis Puteolanis: corridor source-family negative

Targeted witness/provenance traversal found no verified pre-1439 Bodensee/Upper-Rhine witness or custody/transfer edge for Petrus de Ebulo's `De Balneis Puteolanis`.

- BnF fr. 1313 is French but was manufactured in Naples.
- Morgan G.74 was written and illuminated in southern Italy c.1400.
- Edinburgh MS 176 is broadly 15th-century; its catalogue describes an Italian Gothic hand and Italian scenery. The old binding label `1413` is not accepted as a production date.
- Bodleian Digby 129 has no recovered medieval production origin usable for the corridor test.
- Pavia Aldini 488 is later and does not establish a Voynich-window Pavia anchor.

Verdict: **H0_NOT_REJECTED for pre-1439 corridor source transmission.** Pfäfers balneological ecology remains a separate positive context and cannot substitute for Petrus de Ebulo transmission.

### R06 — La Sfera: high-information person route, source transfer unresolved

No verified pre-1439 Bodensee/Upper-Rhine `La Sfera` manuscript has been recovered. The Basel Comites Latentes 194 witness dates 1460–90 and is excluded.

Fra Leonardo Dati nevertheless supplies a real person/institution route:

- Constance 1414–18;
- personally presided at the Dominican general chapter at Strasbourg in 1417;
- Dominican chapter sequence under his mastership continued Freiburg im Breisgau 1419, Metz 1421 and Pavia 1423.

Because a minority of `La Sfera` witnesses explicitly attribute the work to Leonardo, the source/person association is historically real, but modern scholarship generally prefers Gregorio Dati. No evidence shows that Leonardo carried, owned, taught from or deposited a `La Sfera` copy in the corridor.

Verdict: **NONRESOLVING** for source transmission.

### R07 — Comparative control check after source-family traversal

Padua/Venice, Pavia/Milan and Upper Rhine all have hard astronomy/cosmology and copying-infrastructure anchors. These broad capabilities are non-discriminating. The programme therefore relies on source-specific transmission and graph intersections rather than generic learned culture.

### R08 — Herbal census and quantitative confound gate

The existing ManuComp herbal audit (`2026-08-25-r14`) contains 224 rows, 153 possible-by-1438 and 101 secure-by-1438. The pre-existing origin/illustration test nominally separates Northern Italy from German/HRE herbals:

- possible-by-1438: RD = 1.00; null SD = 0.333; effect/null-SD = 3.0;
- secure-by-1438: RD = 1.00; null SD = 0.50; effect/null-SD = 2.0.

This apparent result **fails the ascertainment-confound gate**. Italian rows are disproportionately selected from known illustrated traditions (Beasley/Segre-Rutz/Collins), whereas the German comparison comes from a broad HSP sweep. The metric is therefore rejected for localisation. It is not used anywhere in the corridor score.

### R09 — German herbal with Italian-derived visual mechanism: Debrecen R 459

Debrecen R 459 is a second-quarter-15th-century illustrated German Pseudo-Apuleius herbal in a Middle Bavarian writing language. It is the only illustrated witness among three independent 15th-century Austrian-Bavarian German Pseudo-Apuleius translations.

The crucial source-family result is visual: scholarship on the manuscript concludes that its naturalistic plant images do not simply continue the inherited Pseudo-Apuleius cycle. The most likely source class is the Italian `Tractatus de herbis` illustrated tradition, itself an expanded illustrated development of `Circa instans`; no individual exemplar has been identified and a systematic image-family comparison remains incomplete.

Verdict: **POSITIVE for the mechanism “German-language Kräuterbuch can incorporate Italian-derived herbal imagery in the Voynich-era period.”**

Limitations: date is 1426–1450 rather than securely pre-1439, and the manuscript is Middle Bavarian rather than Bodensee/Upper Rhine. It is therefore **not** a corridor-localisation vote.

### R10 — Basel UBH D III 2: source-specific corridor integration

A stronger corridor object was recovered after the internal census: Basel UBH **D III 2**, a paper medical composite catalogued c.1420.

Relevant contents are unusually diagnostic:

- `Circa instans / Liber de simplici medicina` (the Salernitan materia-medica tradition);
- `Synonyma Antidotario Nicolai Salernitani applicata`, Latin–German;
- `Vocabularium herbarum`, Latin–German;
- further practical medical material including Guido de Cauliaco.

Handschriftencensus dates the composite from a 1420 internal date and identifies parts as the hand of **Johannes Burkardi de Monasterio Grandisvallis**. Earlier Basel Dominican-library scholarship likewise lists D III 2 among the convent medical manuscripts and attributes part of it to Burkardi. Steinmann documents Burkardi copying in Heidelberg in 1423 and being in Basel by 1437; his private library became one of the substantial fifteenth-century book collections associated with the Basel Dominican environment.

This is the first source-family result in this round that directly satisfies the user-level historical mechanism: **Salernitan/Italian-derived materia-medica knowledge and German vernacular plant nomenclature co-exist in the same primary-window medical book embedded in a documented Heidelberg→Basel scribal/book network.**

Bounding cautions:

1. The repository dates the composite c.1420 but does not give a single secure production place for the whole object. Modern Basel custody is not treated as Basel manufacture.
2. “Partly in Burkardi's hand/from his possession” does not identify the `Circa instans` and Latin–German vocabulary folios individually as his autograph.
3. D III 2 does not establish an illustrated plant programme.

Verdict: **QUALIFIED PRIMARY-WINDOW POSITIVE for herbal/materia-medica transmission and German–Italian-derived integration; NOT a positive for corridor-local illustrated herbal access.**

This has been encoded in Supabase as two separate domain assessments:

- `herbal_transmission`: `qualified_primary_window_positive_textual_integration_visual_localisation_unresolved`, primary-window positive = TRUE;
- `paper3_D1_illustrated_medicinal_plant`: `textual_source_transfer_positive_illustrated_corridor_access_unresolved`, primary-window positive = FALSE.

## CURRENT INTERSECTION PICTURE

Documented or formally represented bridges now include:

- Padua/Venice → Basel/Speyer through Pietro Donato and Canon. Misc. 378.
- Salernitan `Circa instans` source family → Heidelberg/Basel book network through UBH D III 2 and Burkardi (qualified: production origin / exact herbal-folio hand unresolved).
- German-language herbal production ← Italian-derived `Tractatus de herbis` visual tradition through Debrecen R 459 (mechanism only; Middle Bavarian, not corridor-local).
- Siena ↔ wider Bodensee/Upper-Rhine network through shared canonical node Ciriaco d'Ancona (context only).
- Vaduz/Schellenberg → Upper Rhine/Basel through the Montfort–Brandis documentary network.
- Vaduz → Pfäfers through Heinrich Stoll in 1416.
- Florence → Constance → Strasbourg through Fra Leonardo Dati, with Dominican institutional continuation to Freiburg and Pavia.
- `Bellifortis` → Upper Rhine independently of Basel through early Strasbourg?/Upper-Rhine witnesses.

The graph now contains **two source-specific positives** relevant to the production model (`Bellifortis`; herbal/materia-medica, qualified), one clean negative (`De Balneis Puteolanis`), and one high-information but nonresolving route (`La Sfera`). No single person or institution yet joins all traditions.

## NEXT BOUNDED TESTS

1. Codicological-unit test on UBH D III 2: determine whether the `Circa instans` + Latin-German herbal vocabulary block is contemporary with the 1420 dated unit and whether Burkardi's hand/ownership can be assigned to that block specifically. Do not infer this from whole-codex metadata.
2. Neutral manuscript-ecology comparison across candidate and control environments, especially collaborative/secular/scientific production behaviour.
3. Joint calendar-orthography geography with a newly bounded, non-selected corpus.
4. Primary-document strengthening for Wilhelm V's Breisgau/Upper-Alsace and Basel offices.
5. Targeted archive/library follow-up on the Leonardo Dati route only if a source can address actual manuscript custody.
6. Prior-art audit against Voynich Archive before any novelty claim.