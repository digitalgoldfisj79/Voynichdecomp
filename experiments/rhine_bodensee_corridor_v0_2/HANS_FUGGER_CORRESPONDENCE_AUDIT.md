# Hans Fugger correspondence audit — Montfort estate and book-custody test

Date: 2026-09-04
Programme: Rhine–Bodensee–Upper Rhine Corridor v0.2
Corpus: Christl Karnehm (ed.), *Die Korrespondenz Hans Fuggers von 1566 bis 1594. Regesten der Kopierbücher aus dem Fuggerarchiv* (2003), c.4,700 outgoing-letter regesta in three volumes.
Status: **VERIFIED POST-MORTEM OBJECT FLOW / BOOK-CUSTODY EDGE NOT FOUND**

## Question

Does Hans Fugger’s correspondence document any movement of books, manuscripts, Kunstbücher, or other identifiable written material from the estate/collection of Ulrich von Montfort after 1574 into the Fugger/Augsburg network, creating a plausible material route toward the later Widemann market?

This is a bounded corpus audit. It does not ask whether the Fuggers broadly collected books or whether the Montforts were culturally connected to Augsburg.

## 1. Hans Fugger’s Montfort role is direct, not contextual

Karnehm’s corpus and modern network analysis show Hans Fugger acting as close mentor/family representative to the sons of his sister Katharina Fugger and Jakob von Montfort. After Ulrich von Montfort’s death, Hans and Marx Fugger were deeply involved in securing the Montfort inheritance and in the ensuing legal/administrative work.

This confirms the existing graph treatment: Hans is not an external collector who happens to know the family; he is inside the Montfort succession network.

## 2. Immediate physical access to Montfort estate records, 1574

The 1574 regesta record a consultation at Tettnang over **Graf Ulrichs Hinterlassenschaft**. The participants intended, in the presence of Dr Bürglin, to open:

> `ein Gewölb, darinn allerlay alt Montfortische Documenten und Schrifften verwart ligen`

and inspect whether the contents supported their succession case.

This establishes direct access by the Fugger/Montfort legal network to old Montfort documentary holdings after Ulrich’s death. It concerns archival/legal documents, not Ulrich’s Kunstbücher.

## 3. First verified movable-object chain from Ulrich’s collection through Hans Fugger

The 1577–78 regesta supply a stronger material result.

Karnehm’s linked entries (esp. II 1219 and II 1290, with cross-references II 1189, 1210, 1211, 1226, 1249, 1250, 1287) show:

1. Duke Wilhelm V of Bavaria pressing Hans Fugger to obtain **antiquities of the deceased Count Ulrich von Montfort**.
2. Ulrich’s widow **Ursula von Solms-Lich** keeping the relevant antiquities in a **Truhe** at Tettnang.
3. The key being held by the `Aigenthumbs Erben`, separating physical custody from legal/access control.
4. Hans Fugger corresponding with Count Heinrich von Fürstenberg and the Tettnang Landschreiber about access and specific objects.
5. The Landschreiber supplying information about a **versteinerte Schüssel** and other antiquities from Ulrich’s possession.
6. Hans actually forwarding the **Schüssel** and a **Hundsköpfchen** to Duke Wilhelm.
7. Karnehm II 1290, Hans Fugger to Hans Leutholt, Landschreiber at Tettnang, Augsburg, 25 February 1578, records that Wilhelm did not like them and that Hans therefore sent them back.

This establishes a complete, demonstrable post-mortem object route:

`Ulrich collection / Tettnang estate → controlled chest access → Hans Fugger / Augsburg → Wilhelm V / Munich → return`

It is materially stronger than the previously documented social/legal network because physical collection objects actually travelled through Hans Fugger.

## 4. The estate inventory was not exhaustive for every movable object

At least one antiquity pursued in this sequence is described in the regesta as **not recorded in the Nachlassinventar**. Hans nevertheless treats it as an object belonging to Ulrich’s possession and attempts to facilitate its acquisition through the widow/family network.

Therefore absence of a named Kräuterbuch, Sammelband or Voynich-like manuscript from the surviving 1574 inventory cannot be treated as an exhaustive exclusion test for every movable object in Ulrich’s collection.

This does **not** license the converse inference that an unrecorded Voynich manuscript existed. It merely downgrades inventory silence from negative evidence to weak/non-exclusionary evidence.

## 5. Bounded book/manuscript term matrix

The Karnehm regesta were searched across both relevant chronological blocks (1574–81 and 1582–94) using Montfort/Tettnang/Ulrich/estate terms constrained against:

- `Bücher`, `Buch`;
- `Manuskript` / manuscript;
- `Kunstbuch`;
- `Bibliothek`;
- `Hinterlassenschaft`;
- `Nachlass` / `Nachlaßinventar`;
- `Truhe`;
- `Antiquitäten`;
- `Sammlung`;
- `Documenten` / `Schrifften`.

### Result

No regest was recovered that identifies a **book, manuscript, Kunstbuch, herbal, Sammelband, or library unit from Ulrich’s estate** passing through Hans Fugger.

Hits for `Montfort + Bücher/Buch` in later correspondence concern other book-procurement tasks or the education/activity of the younger Montforts, not Ulrich’s estate.

The later 1582–94 volume produced no continuation of the 1577–78 Ulrich-antiquities sequence under the constrained collection terms.

Verdict: **bounded negative at regesta level**.

## 6. Important coverage caveat

Karnehm is a regesta edition, not a complete transcription. A negative keyword result means that no such transfer is visible in the edited summaries; it is not proof that the full original letters omit books.

Hans Fugger’s broader correspondence demonstrably includes book procurement and collecting, and scholarship notes that his own library cannot be fully reconstructed from the correspondence. Consequently the correct follow-up, if this line is pursued further, is the small set of original/copybook entries surrounding the verified Ulrich-object sequence — not another broad keyword search.

## 7. Supabase encoding

The verified chain has been encoded separately from the pre-existing social/legal provenance network.

New graph object group:

`ulrich_estate_antiquities_subset`

New bridge:

`bridge_ulrich_estate_hans_fugger_object_flow`

Status:

`verified_postmortem_object_transfer_nonbook`

The late-provenance domain is now:

`verified_postmortem_object_transfer_chain_book_custody_unresolved`

Current graph after this audit: **98 nodes / 99 edges / 10 environment bridges**.

A permanent negative edge records that no Ulrich-estate book/manuscript transfer was identified in the bounded Karnehm-regesta audit.

## 8. One pending lead discovered during the audit

Older Bavarian court scholarship records Ulrich von Montfort writing from Tettnang that a recently deceased learned man in Venice had assembled an enormous collection containing antiquities **and books**, reportedly requiring three or four wagons, and offering to acquire it for Duke Albrecht V.

This would establish Ulrich as an active **Venice → Tettnang → Bavarian court collection broker**, including books, rather than merely a passive collector.

The OCR of the surviving secondary source obscures the exact year of the 27 April letter. The event predates Ulrich’s death in 1574, but it is **not yet promoted into the graph** until the date/reference can be checked cleanly against the underlying Bavarian correspondence.

## 9. Decision

The Hans Fugger correspondence **upgrades the late Vaduz/Montfort provenance network one category**:

- before: very strong social/legal/collecting network, no demonstrated post-mortem material flow;
- now: **verified post-mortem movement of Ulrich collection objects through Hans Fugger/Augsburg**, but no demonstrated book/manuscript movement.

The missing discriminator is narrower than before: determine whether the same Tettnang estate-access and object-transfer machinery handled any written material.

## 10. Highest-value next documents

1. Full copybook/original letters for Karnehm II 1189, 1211, 1219, 1226, 1249, 1250, 1287 and 1290, especially correspondence with Hans Leutholt, Jörg von Montfort, Heinrich von Fürstenberg and Duke Wilhelm V.
2. Any surviving detailed inventory/list attached to the Tettnang chest or the estate consultation.
3. Bavarian `Antiquitäten` / Kunstkammer correspondence that generated the Ulrich→Wilhelm object sequence.
4. Verification of the undated-year Venetian learned-collection offer by Ulrich.

Until one of these names a written object, the correct conclusion is:

**Hans Fugger is now a verified physical intermediary for Ulrich’s post-mortem collection, but not yet for Ulrich’s books.**