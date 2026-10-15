#!/usr/bin/env python3
"""Compress materialised state tables to one fixed inventory and public derivation rule."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
m=json.loads((P/'04_pair_bridge_manifest.json').read_text());base=json.loads((P/'parent_v141_manifest.json').read_text())
compact={'format':'vbm142b_pair_compact_derived','unit_list':m['unit_list'],
 'cell_inventory':base['cells'],
 'bridge_for_vowel_and_next_key':m['bridge_for_vowel_and_next_key'],
 'vowel_by_pair':m['vowel_by_pair'],
 'initial_state':'e',
 'derivation_algorithm':'filter inventory by permitted next key; retain only cells not proper prefixes of other allowed cells; sort shortest-first then preserve initial inventory order; if capacity below source-unit inventory reserve final code as escape and encode high shared-rank units as escape+cell[index-direct]; else use first U cells',
 'word_boundary':'whitespace',
 'historical_claim':'none; B inventory fitted to the manuscript by original author'}
(P/'06_pair_compact_manifest.json').write_text(json.dumps(compact,indent=2,sort_keys=True)+'\n')
print('compressed bytes', (P/'06_pair_compact_manifest.json').stat().st_size, 'explicit entries',len(compact['unit_list'])+len(compact['cell_inventory'])+len(compact['vowel_by_pair']))
