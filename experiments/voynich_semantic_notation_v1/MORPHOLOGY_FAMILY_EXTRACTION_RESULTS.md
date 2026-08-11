# VSN-v1 Morphology-Family Extraction Results

Date: 2026-08-12
Status: DISCOVERY infrastructure; visual outcomes were not used to construct this inventory.

## RF census

Audited RF occurrence snapshot:

- 36,680 raw dot-delimited occurrences;
- 35,314 exact-letter primary occurrences (`^[a-z]+$`);
- 7,893 primary RF token types;
- 227 folios;
- 18 populated quire labels plus explicitly missing quire metadata.

The snapshot cardinality exactly matches direct RF splitting after a foldout-metadata join duplication was identified and corrected. No target analysis was retained from the inflated intermediate table.

## Edit graph

Exhaustive deletion-signature construction yields **28,435 exact Levenshtein-distance-1 token pairs**. Each pair records:

- insertion/deletion or substitution;
- edit position;
- prefix / suffix / internal position class;
- changed component(s);
- residual core/family signature;
- type and occurrence support.

This graph was constructed before joining any visual target.

## Affix/component census

All prefixes and suffixes of lengths 1–4 were enumerated. The frozen support gate is:

- at least 50 RF occurrences;
- at least 20 token types;
- at least 10 residual cores with an attested alternative component in the same slot.

**361 component candidates** qualify. Their matched alternative-form expansion contains 401,615 component/core/type contrast rows.

Representative high-support candidates (not privileged a priori):

| component | occurrences | token types | contrast cores |
|---|---:|---:|---:|
| suffix `-r` | 5,289 | 1,118 | 537 |
| suffix `-dy` | 6,199 | 1,094 | 502 |
| prefix `ch-` | 5,691 | 1,082 | 500 |
| prefix `s-` | 3,967 | 846 | 491 |
| prefix `o-` | 7,763 | 1,703 | 477 |
| suffix `-l` | 5,316 | 894 | 461 |
| suffix `-y` | 14,330 | 3,074 | 446 |
| suffix `-ey` | 3,808 | 663 | 357 |
| prefix `sh-` | 3,045 | 550 | 357 |
| prefix `qo-` | 5,052 | 737 | 331 |
| suffix `-aiin` | 3,195 | 494 | 252 |
| suffix `-edy` | 4,045 | 424 | 232 |

Thus examples discussed before the experiment (`qo-`, `-dy`, `-ey`, `-edy`, `-aiin`) emerge from an exhaustive outcome-blind inventory rather than from manual candidate selection.

## Cross-family support on herbal DISCOVERY pages

After restricting to DISCOVERY blocks and folios with an existing whole-plant embedding, there is substantial matched within-quire support. Examples:

- `ch-`: 88 residual cores, 177 matched core×quire strata;
- `-r`: 76 cores, 171 strata;
- `-dy`: 74 cores, 136 strata;
- `-ey`: 65 cores, 126 strata;
- `qo-`: 47 cores, 72 strata;
- `-edy`: 34 cores, 57 strata.

These counts establish that cross-family testing is possible without reducing the experiment to repeated full-token associations.

## Caveats

The affix candidates are formal surface components. They are **not** assumed to be true morphemes. Nested components are highly correlated (`-edy` contains `-dy`, for example), so apparently distinct candidates must not be counted as independent semantic operators without de-nesting evidence.

The present populated database has no token-level x/y word table and no populated Davis/Currier hand fields. Those variables remain missing rather than imputed.
