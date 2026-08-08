# Amendment 002 — Signed-colophon / prosopography arm

Date: 2026-08-08
Status: **PREREGISTERED BEFORE ANY CORRIDOR-VS-CONTROL SIMILARITY RESULT**

## Trigger

User supplied Bénédictins du Bouveret, *Colophons de manuscrits occidentaux des origines au XVIe siècle*, Tome II, no. 5912, as a source that records named manuscript personnel.

Independent verification confirms that Tome II is *Colophons signés E–H*, covering entries 3562–7391 (Fribourg, 1967). The Bouveret signed-colophon series contains names of persons associated with manuscripts, but such a name is not necessarily the copyist: published reviews explicitly note copyists, illuminators, binders, recipients and other named persons. Therefore role verification is mandatory.

## What changes

A documentary prosopography stream is added between archive-prior-art review and image comparison.

New source priority:

- Bouveret Tome II, including seed no. 5912;
- later, the remaining Bouveret signed-colophon volumes where frozen geography/time criteria justify expansion;
- institutional manuscript catalogues and person authority records used to verify Bouveret entries.

New Supabase tables:

- `corridor_colophon_records`
- `corridor_people`
- `corridor_person_manuscript_links`

No existing table is altered by this amendment.

## What does NOT change

This amendment does **not** change:

- H0/H1;
- frozen corridor/control geography;
- chronology bins;
- primary image-cohort inclusion rules;
- feature families or image models;
- confound gates;
- primary permutation statistic, alpha, or coverage thresholds.

Prosopographical evidence is an independent documentary/codicological corroboration arm. It cannot rescue a failed primary visual/ecology test by changing weights or sample membership.

## Anti-cherry-picking rule

No. 5912 may generate a lead, but the inferential prosopography arm cannot consist of selected interesting entries. The programme must enumerate all recoverable signed-colophon records satisfying the same frozen date/geography rules for the relevant source scope, then compare the corridor against matched controls.

A Bouveret-derived manuscript enters the primary image cohort only if it independently satisfies the neutral manuscript/illustration inclusion criteria.

## Role-verification rule

Raw Bouveret entries are ingested before entity resolution. `scribe` is assigned only when the colophon itself or a reliable catalogue explicitly identifies copying/scribal activity. Ambiguous named persons remain `unknown_person_role` or their evidenced non-scribal role.

## Novelty rule

Every person/manuscript/network edge is checked against the internal Voynich archive before being labelled new. Prior-art status affects novelty claims and research priority, not documentary truth.

## Planned outputs

1. corridor/control signed-colophon census;
2. verified named book-worker list;
3. person ↔ manuscript ↔ place network;
4. cross-node mobility edges;
5. manuscripts newly discovered through named people;
6. archive prior-art status for each lead;
7. a documentary-network result reported independently from visual similarity.

See `PROSOPOGRAPHY.md` for the operational specification.
