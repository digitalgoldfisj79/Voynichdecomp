# Rhine–Bodensee–Upper Rhine Corridor v0.2 — Running Results

## RETRACTED / DOWNGRADED FINDINGS

These remain at the top of the file for the life of the programme.

1. **Literal castle identification:** downgraded. Prior castle work supports only broad architectural compatibility/typology; no specific building identification is evidential.
2. **Jakob I von Lichtenberg as a major documented Lauber patron / esoteric-medical scholar:** retracted pending primary evidence. Remove from the evidential spine.
3. **Pfirt chancery scribes demonstrably adopting French calendar orthography:** unverified; retain only as a testable mechanism.
4. **Wilhelm V von Montfort-Tettnang personally negotiating face-to-face with Jean de Neuchâtel in 1430:** unverified; do not use.
5. **Alpine–Venetian corridor v0.1:** `NONRESOLVING_REFERENCE_SET`; it neither supported nor rejected the old corridor hypothesis. Do not reinterpret the gate failure as substantive negative evidence.

## AUDIT NOTES

- The v0.1 database records a closed/nonresolving run, while the inherited GitHub `RUNNING_RESULTS.md` remained stale and described the programme as not yet run. Database/result records are treated as the substantive run state; v0.2 is isolated rather than overwriting v0.1.
- Pre-freeze metadata exposure is recorded in `PROTOCOL.md`. No exposed Voynich similarity result is used for candidate selection.
- **Graph-first rule:** reuse/reference the existing Supabase Siena graph, Paper 3 Padua/Venice and Pavia/Milan environments, and legacy `corridor_*` person/manuscript/movement edges. `rhine_bodensee_v02` stores only new/corrected delta edges.
- Vaduz/Schellenberg is a formal graph component, not prose context. Architecture remains non-primary; the territorial/person network is retained for independent documentary tests.

## CURRENT STATUS

Run: `rb_corridor_v02_20260904_run01`
Stage: `stage2_source_transmission`
Protocol: **FROZEN 2026-09-04**
Primary result: **PROMISING BUT NONRESOLVING. H1 NOT YET PASSED.**

Current source-family scoreboard:

- Kyeser / `Bellifortis`: **POSITIVE — strong primary-window Upper-Rhine transmission; survives leave-Basel-out.**
- `De Balneis Puteolanis`: **NEGATIVE FOR CORRIDOR SOURCE TRANSMISSION — H0 not rejected.**
- `La Sfera`: **NONRESOLVING — strong person/institution route via Fra Leonardo Dati, but no manuscript custody/transfer edge.**

No source family is double-counted through multiple manuscript witnesses.

## RUNNING FINDINGS

### R00 — Programme inheritance

The prior Alpine–Venetian programme had 12 corridor manuscripts and 10 controls but failed its preregistered Voynich reference-set gate; no primary similarity test was permitted. This is an audit constraint, not evidence for or against the present Rhine–Bodensee hypothesis.

Bounding status: complete as a statement about the old programme; substantive geography remains unbounded.

### R01 — Neutral registry census exposure

A neutral 1390–1450 date/place query of the existing manuscript registry surfaced multiple Basel, Hagenau, Konstanz/Bodensee and Upper-Rhine records before v0.2 freeze. Two metadata records were flagged for primary verification: Bodleian Canon. Misc. 378 and Würzburg M.ch.f.232.

Canon. Misc. 378 has since cleared verification sufficiently to serve as a **production/transmission mechanism**, not a Voynich analogue: Pietro Donato's existing legacy graph records Padua 1428–33, Basel 1434–36 and Speyer 1436, while the existing manuscript edge records Canon. Misc. 378 copied for him at Basel in 1436 from the Speyer exemplar.

Bounding test: comparable mixed-production witnesses remain required in matched controls; no uniqueness claim is made.

### R02 — Architecture status

Architecture is removed from the primary endpoint. It can only constrain broad visual compatibility unless a new predeclared visual instrument passes a source/confound gate.

Bounding status: enforced by protocol.

### R03 — Formal Bodensee / Alpine-Rhine / Upper-Rhine graph

Peer views now exist in Supabase:

- `public.bodensee_graph_nodes_v01`
- `public.bodensee_graph_edges_v01`
- `public.bodensee_domain_assessment_v01`
- `public.bodensee_environment_bridges_v01`

The graph reuses canonical legacy keys and references existing edges rather than copying them.

The Vaduz component formally includes the Montfort–Brandis–Werdenberg chain, Vaduz, Eschnerberg, Alt-/Neu-Schellenberg, and the 27 June 1437 settlement recorded in Staatsarchiv Schwyz HA.II.408. Wilhelm V's 1429 Breisgau/Upper-Alsace and 1433 Basel offices are retained as secondary-verified edges pending primary-document strengthening.

A separate 20 January 1416 edge links Heinrich Stoll, chaplain of St Florin in Vaduz, directly to Pfäfers through his witness role in the abbey's convent statutes. This supports a Vaduz–Pfäfers institutional/balneological environment only; it does not imply `De Balneis` transmission.

### R04 — Kyeser / Bellifortis: source-family positive

The existing Supabase authority census, followed by independent catalogue checking, identifies primary-window `Bellifortis` transmission in the candidate region:

- BAV Pal. lat. 1994: Southwest Germany (Strasbourg?), c.1410.
- Budapest MTAK K 465 fragment: illuminated in Southern Germany / Upper Rhine region, 1411–1437.
- Karlsruhe Cod. Durlach 11: c.1420–30, Alemannic writing language; no exact production place inferred from language.

Verdict: **verified primary-window Upper-Rhine transmission**.

Robustness: this result survives removal of Basel. The witnesses form one textual/source dependency family and therefore count as **one** evidence family, not three votes.

### R05 — De Balneis Puteolanis: corridor source-family negative

Targeted witness/provenance traversal found no verified pre-1439 Bodensee/Upper-Rhine witness or custody/transfer edge for Petrus de Ebulo's `De Balneis Puteolanis`.

Relevant checks:

- BnF fr. 1313 is a French translation but was manufactured in Naples; French language is not evidence of French/Upper-Rhine production.
- Morgan G.74 was written and illuminated in southern Italy c.1400.
- Edinburgh MS 176 is only broadly 15th-century; its catalogue describes a good Italian Gothic hand and Italian scenery. The old binding label `1413` is not accepted as a production date.
- Bodleian Digby 129 is 15th-century but no usable medieval production origin has been recovered.
- Pavia Aldini 488 is later and does not establish a Voynich-window Pavia `De Balneis` anchor.

Verdict: **H0_NOT_REJECTED for pre-1439 corridor source transmission.**

Important separation: Pfäfers provides genuine Alpine-Rhine bathing/balneological ecology, including a direct Vaduz institutional edge, but this cannot be substituted for the missing Petrus de Ebulo manuscript-transmission evidence.

### R06 — La Sfera: high-information person route, source transfer unresolved

The current witness census does not supply a verified pre-1439 Bodensee/Upper-Rhine `La Sfera` manuscript. The Basel Comites Latentes 194 witness dates only to 1460–90 and is excluded from the production window. Pavia Aldini 90 is only broadly dated to the 15th century in the current project metadata; modern holding location is not medieval provenance.

A much stronger non-manuscript route has been established through **Fra Leonardo Dati (c.1360–1425)**, the Dominican and historical rival claimant to `La Sfera` authorship:

- documented at the Council of Constance from November 1414 to early 1418 as Dominican leader and representative of Florence;
- personally left Constance to preside over the Dominican general chapter at Strasbourg in June 1417, then returned;
- Dominican general chapters under his mastership were held at Freiburg im Breisgau in 1419, Metz in 1421, and Pavia in 1423.

Modern `La Sfera` scholarship generally prefers Gregorio/Goro Dati as author; the manuscript tradition nevertheless contains a minority of explicit Leonardo attributions. The person/source association is therefore historically real but disputed.

Verdict: **NONRESOLVING** for source transmission. The route Florence → Constance → Strasbourg (and institutional sequence onward to Freiburg/Pavia) is direct and highly relevant, but no evidence currently shows that Leonardo carried, owned, taught from or deposited a `La Sfera` copy in the corridor. It does not score as a primary source-family positive.

### R07 — Comparative control check after source-family traversal

The frozen Paper 3 environments remain active controls. On the broad D2 bath/process domain, Padua/Venice, Pavia/Milan and Upper-Rhine/Alsace were all previously unresolved under the matched census; the new `De Balneis` audit therefore does not create an artificial Italian-control win at the broad-domain level.

On D4 astronomy/cosmology and D6 copying infrastructure, Padua/Venice, Pavia/Milan and Upper-Rhine/Alsace all have independent hard pre-window/primary-window anchors. These broad capabilities are therefore **non-discriminating**. The present programme accordingly relies on source-specific transmission and graph intersections, not generic learned culture.

## CURRENT INTERSECTION PICTURE

Documented or formally represented bridges now include:

- Padua/Venice → Basel/Speyer through Pietro Donato and Canon. Misc. 378.
- Siena ↔ wider Bodensee/Upper-Rhine network through shared canonical node Ciriaco d'Ancona (context only; no Basel residence inferred).
- Vaduz/Schellenberg → Upper Rhine/Basel through the Montfort–Brandis documentary network.
- Vaduz → Pfäfers through Heinrich Stoll in 1416.
- Florence → Constance → Strasbourg through Fra Leonardo Dati, with Dominican institutional continuation to Freiburg and Pavia.
- `Bellifortis` → Upper Rhine independently of Basel through early Strasbourg?/Upper-Rhine witnesses.

The graph therefore has real intersections, but no single person or institution yet joins all required Voynich-relevant source traditions.

## NEXT BOUNDED TESTS

1. German/Italian herbal transmission into the Bodensee/Upper-Rhine graph, using the existing ManuComp herbal corpus before external search.
2. Neutral manuscript-ecology comparison across candidate and control environments, especially collaborative/secular/scientific production behaviour.
3. Joint calendar-orthography geography with a newly bounded, non-selected corpus.
4. Primary-document strengthening for Wilhelm V's Breisgau/Upper-Alsace and Basel offices.
5. Targeted archive/library follow-up on the Leonardo Dati route only if a source can address actual manuscript custody; do not browse generic Dominican networks further.
6. Prior-art audit against Voynich Archive before any novelty claim.
