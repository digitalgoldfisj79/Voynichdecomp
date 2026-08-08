# Amendment 004 — Pre-outcome VMS architecture/diagram reference expansion

Date: 2026-08-08
Programme: Alpine–Venetian Corridor Illustration Programme v0.1
Run: `corridor_v01_20260808_run01`

## Reason

After the blind zodiac review passed 12/12, the existing reviewed VMS reference set supplied plant, root and zodiac families. The sealed corridor/control cohort nevertheless contains cartographic and technical manuscripts that cannot be scored under the protocol's strict class-matching rule against those classes. Before any corridor/VMS similarity is computed, this amendment defines fixed additional VMS reference classes from the already designated Voynich rosettes foldout.

## Frozen source

Yale Beinecke digitisation, catalogue folio `85v and 86r (foldout)`, `cat_folios.seq=158`, IIIF service `https://collections.library.yale.edu/iiif/2/1006231`.

No corridor/control image or score is available to the reference extractor.

## Classes

Only the two already-preregistered visual classes may be populated:

- `architecture_cartography`: towers, battlements, roofed structures, enclosure/city-like structures or clearly cartographic architectural motifs.
- `diagram_geometry`: circular/radial/nested-ring/connector structures with clear geometric organisation.

## Extraction and acceptance

A blind VLM may propose at most four tight, non-overlapping regions per class using normalised 0..1000 coordinates. A proposal is rejected if any coordinate lies outside 0..1000, the box is degenerate, text-only, or class identity is unclear. Invalid boxes are not clipped or repaired by hand.

Initial extractor: `Qwen/Qwen2.5-VL-3B-Instruct`, deterministic generation, bounded HF job `6a773f093e1f34a7e32bedd7`.

Initial output produced three valid proposals and one invalid architecture box. The invalid box (`x1=1098`) is rejected by rule. This amendment records the procedure and does not retrospectively rescue it.

## Outcome firewall

At amendment time:

- no corridor-to-VMS similarity has been computed;
- no comparandum image was supplied to the extractor;
- cohort/matcher seal v2 remains `0b499356c5f901d9b1ac825c0657e494`;
- these references may only enable strict class-matched scoring; they cannot change candidate membership, geography, date, control matching or content tags.

## Sensitivity

Any primary result involving `architecture_cartography` or `diagram_geometry` must be repeated with these newly created VMS reference classes removed. The overall Tier-2/Tier-3 interpretation may not depend solely on this amendment.