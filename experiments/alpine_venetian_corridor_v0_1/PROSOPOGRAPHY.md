# Prosopography / signed-colophon arm

Added 2026-08-08 before any corridor-vs-control similarity analysis.

## Purpose

Use signed medieval colophons as an **independent name-first discovery stream** for the Alpine–Venetian corridor. This arm asks who copied, illuminated, bound, commissioned, received, owned, or otherwise participated in manuscripts made in or moving through the corridor, and whether those people connect otherwise separate manuscript witnesses.

This is deliberately distinct from the image-similarity arm. A person or manuscript enters this arm because of documentary colophon evidence, not because it resembles the Voynich Manuscript.

## Core source

Bénédictins du Bouveret, *Colophons de manuscrits occidentaux des origines au XVIe siècle*.

Priority source supplied for this programme:

- Tome II, *Colophons signés E–H* (entries 3562–7391), Fribourg, 1967.
- Seed entry: **no. 5912**.
- Internet Archive item: `colophonsdemanus0002sain`.

Entry 5912 is seeded as `pending_transcription`. No person name, role, manuscript, place, or date is inferred until the entry itself is directly inspected.

## Critical role rule

A Bouveret signed-colophon entry is **not automatically a scribe record**. The named person can be a copyist, illuminator, binder, recipient, commissioner, owner, or another participant. Therefore:

1. ingest the raw colophon entry;
2. preserve the exact person-name form and colophon wording;
3. classify the person's role only from explicit documentary evidence or a reliable catalogue interpretation;
4. normalise the person only after role verification;
5. retain uncertainty rather than coercing every named person into `scribe`.

## Census rule

For the primary corridor prosopography census, extract all signed-colophon records meeting the **frozen geography and chronology criteria**, not merely names that look German, Italian, familiar, or Voynich-relevant.

Search order:

1. entries explicitly localised to corridor-core nodes;
2. entries localised to the corridor buffer / Tyrol–Veneto interface;
3. matched control-region colophons using the same date rules;
4. records whose named person's independently documented mobility crosses corridor nodes;
5. records linked to manuscripts in relevant practical/scientific/medical/astronomical/botanical/technical genres.

The colophon stream may generate new manuscript candidates. Those candidates must still pass the normal illustration/content inclusion rules before entering the primary image cohort.

## Data model

### `corridor_colophon_records`
Raw source-level record. Stores Bouveret volume/entry, raw person name, role, colophon text, manuscript reference, date/place, source link, and verification state.

### `corridor_people`
Normalised person entities. `role_tags` may include `scribe` only after verification.

### `corridor_person_manuscript_links`
Evidence edges between people and manuscripts. Relationships can include:

- `copied`
- `illuminated`
- `bound`
- `commissioned`
- `received`
- `owned`
- `corrected`
- `rubricated`
- `worked_on`
- `uncertain`

## Network questions

This arm is designed to test questions not answerable by visual similarity alone:

1. Do named scribes/book-workers recur across more than one corridor node?
2. Can a single person be documented working on both sides of the Alpine linguistic boundary?
3. Do book-workers associated with practical/scientific manuscripts move along Brixen/Bolzano → Trento → Verona → Padua → Venice?
4. Are there identifiable workshop, university, monastic, episcopal, or patronage networks linking the relevant manuscripts?
5. Do apparently Germanic and Veneto visual/textual traits co-occur in manuscripts made by the same named person or network?
6. Does the corridor show more cross-node scribe mobility than matched control ecologies?

## Evidence tiers

- **P0:** raw Bouveret entry only; not transcribed/verified.
- **P1:** colophon directly transcribed; named person and manuscript secure.
- **P2:** person's role securely identified from colophon/catalogue.
- **P3:** same person securely linked to multiple manuscripts or places.
- **P4:** cross-corridor documentary network with independently dated/place-authorised manuscripts.

P3/P4 evidence can materially strengthen the corridor ecology hypothesis, but it is analysed independently of VMS visual affinity. It cannot by itself identify a Voynich scribe.

## Novelty gate

Every named person, manuscript, and proposed mobility/network edge is searched against the internal Voynich archive before being labelled new. A Bouveret-derived name may be new to the programme but already known to the Voynich community; novelty and evidential strength remain separate fields.

## Immediate action

1. Obtain/directly inspect Tome II no. 5912.
2. Transcribe it exactly.
3. Identify the manuscript and named person's role without inference from the corridor hypothesis.
4. Resolve the present shelfmark/catalogue authority.
5. Search the internal Voynich archive for that person, shelfmark, place, and relationship.
6. Expand outward to all eligible Tome II corridor records rather than cherry-picking no. 5912.
