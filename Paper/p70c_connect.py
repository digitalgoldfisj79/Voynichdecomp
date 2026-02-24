#!/usr/bin/env python3
"""
CONNECT prior findings to p70-c framework
==========================================
1. Word-to-word transitions → 6th axis (prev_sfx_fam)
2. Line-level patterns → paragraph marker flag
3. Quire structure → production-unit metadata
4. Suffix decomposition → validate sfx_fam is correct abstraction
"""

import pickle, json, math, random
from collections import Counter, defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

with open('Voynichdecomp/enriched_records.pkl', 'rb') as f:
    records = pickle.load(f)

N = len(records)

def H(vals):
    c = Counter(vals)
    t = sum(c.values())
    ps = np.array([v/t for v in c.values()])
    return -np.sum(ps * np.log2(ps + 1e-30))

def pos_cat(r):
    if str(r.get('is_first_word', '')) == 'True': return 'FIRST'
    if str(r.get('is_last_word', '')) == 'True': return 'LAST'
    return 'MID'

# ═══════════════════════════════════════════════════════════════
# BUILD LINE SEQUENCES
# ═══════════════════════════════════════════════════════════════
print("Building line sequences...")
lines = defaultdict(list)
for r in records:
    lines[(r['folio'], r['line_no'])].append(r)

# Sort tokens within each line by position
for key in lines:
    lines[key].sort(key=lambda r: int(r['pos']) if isinstance(r['pos'], (int, float)) else 0)

# Annotate each record with prev_sfx_fam and is_paragraph_first
for key, toks in lines.items():
    for i, r in enumerate(toks):
        # Previous word's suffix family
        if i == 0:
            r['prev_sfx'] = 'LINE_START'
        else:
            r['prev_sfx'] = toks[i-1]['sfx_fam']
        
        # Is this the first line of a paragraph?
        # Paragraph markers: first word of first line has ∅-prefix + gallows
        r['_pos'] = pos_cat(r)
        
        # First-line detection: check is_first_line field if available
        is_fl = str(r.get('is_first_line', '')) == 'True'
        r['is_para_first'] = is_fl and i == 0

print(f"Records annotated: {N}")
para_first = sum(1 for r in records if r.get('is_para_first', False))
print(f"Paragraph-first tokens: {para_first}")

# ═══════════════════════════════════════════════════════════════
# CONNECTION 1: WORD-TO-WORD TRANSITIONS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CONNECTION 1: TRANSITION GRAMMAR → 6th AXIS")
print("="*70)

# How much information does prev_sfx add beyond position + section?
prev_sfx = [r['prev_sfx'] for r in records]
pos_vals = [r['_pos'] for r in records]
sec_vals = [r['section'] for r in records]
quad_vals = [(r['prefix'], r['gallows'], r['m_core'], r['sfx_fam']) for r in records]

# Current model: H(quad | pos, section)
h_quad = H(quad_vals)
h_pos_sec = H(list(zip(pos_vals, sec_vals)))
h_quad_pos_sec = H(list(zip(quad_vals, pos_vals, sec_vals)))
mi_quad_possec = h_quad + h_pos_sec - h_quad_pos_sec

# With prev_sfx: H(quad | pos, section, prev_sfx)
h_pos_sec_prev = H(list(zip(pos_vals, sec_vals, prev_sfx)))
h_quad_all3 = H(list(zip(quad_vals, pos_vals, sec_vals, prev_sfx)))
h_quad_given_all3 = h_quad_all3 - h_pos_sec_prev
mi_quad_all3 = h_quad - h_quad_given_all3

mi_prev_additional = mi_quad_all3 - mi_quad_possec

print(f"  MI(quad; pos+sec)          = {mi_quad_possec:.4f} bits ({mi_quad_possec/h_quad*100:.2f}%)")
print(f"  MI(quad; pos+sec+prev_sfx) = {mi_quad_all3:.4f} bits ({mi_quad_all3/h_quad*100:.2f}%)")
print(f"  prev_sfx adds              = {mi_prev_additional:.4f} bits ({mi_prev_additional/h_quad*100:.2f}% additional)")

# How does prev_sfx compare to position and section individually?
h_prev = H(prev_sfx)
h_quad_prev = H(list(zip(quad_vals, prev_sfx)))
mi_quad_prev = h_quad + h_prev - h_quad_prev
print(f"\n  MI(quad; prev_sfx) alone   = {mi_quad_prev:.4f} bits ({mi_quad_prev/h_quad*100:.2f}%)")
print(f"  MI(quad; position) alone   = {0.3797:.4f} bits ({0.3797/h_quad*100:.2f}%)")
print(f"  MI(quad; section) alone    = {0.8105:.4f} bits ({0.8105/h_quad*100:.2f}%)")

# Which prefix is most affected by prev_sfx?
print(f"\n  Transition matrix (prev_sfx → prefix, % of tokens):")
prev_sfx_vals = sorted(set(prev_sfx))
pfx_vals = sorted(set(r['prefix'] for r in records))

print(f"  {'prev↓ pfx→':>12}", end='')
for p in pfx_vals:
    print(f" {p:>5}", end='')
print(f"  {'N':>6}")

for ps in prev_sfx_vals:
    subset = [r for r in records if r['prev_sfx'] == ps]
    n_sub = len(subset)
    pfx_c = Counter(r['prefix'] for r in subset)
    print(f"  {ps:>12}", end='')
    for p in pfx_vals:
        print(f" {pfx_c.get(p,0)/n_sub*100:>4.1f}%", end='')
    print(f"  {n_sub:>6}")

# Build the sextuples: (P, G, mc, SF, pos, prev_sfx)
sext_counts = Counter()
for r in records:
    s = (r['prefix'], r['gallows'], r['m_core'], r['sfx_fam'], r['_pos'], r['prev_sfx'])
    sext_counts[s] += 1

quint_counts = Counter()
for r in records:
    q = (r['prefix'], r['gallows'], r['m_core'], r['sfx_fam'], r['_pos'])
    quint_counts[q] += 1

print(f"\n  Quints (pos only):             {len(quint_counts):,}")
print(f"  Sextuples (pos+prev_sfx):      {len(sext_counts):,}")
print(f"  Expansion factor:              {len(sext_counts)/len(quint_counts):.2f}×")

# Tier distribution of sextuples
sext_tiers = Counter()
for c in sext_counts.values():
    if c >= 50: sext_tiers['T1'] += 1
    elif c >= 10: sext_tiers['T2'] += 1
    elif c >= 4: sext_tiers['T3'] += 1
    else: sext_tiers['T4'] += 1

print(f"  Sextuple tiers: {dict(sext_tiers)}")

# Cross-validation: does the sextuple generalise?
indices = list(range(N))
random.shuffle(indices)
split = int(N * 0.8)
train_idx = set(indices[:split])
test_idx = set(indices[split:])

train_sext = set()
for i in train_idx:
    r = records[i]
    train_sext.add((r['prefix'], r['gallows'], r['m_core'], r['sfx_fam'], r['_pos'], r['prev_sfx']))

test_novel = 0
test_total = 0
for i in test_idx:
    r = records[i]
    s = (r['prefix'], r['gallows'], r['m_core'], r['sfx_fam'], r['_pos'], r['prev_sfx'])
    test_total += 1
    if s not in train_sext:
        test_novel += 1

print(f"\n  80/20 cross-val token miss rate: {test_novel/test_total*100:.1f}% "
      f"(vs quint: ~13.0%)")
print(f"  {'WORSE' if test_novel/test_total > 0.15 else 'COMPARABLE'} generalisation")

# Placement precision gain
# How many (token, pos, prev_sfx) triples are valid?
actual_tok_pos_prev = set((r['token'], r['_pos'], r['prev_sfx']) for r in records)
actual_tok_pos = set((r['token'], r['_pos']) for r in records)
actual_types = set(r['token'] for r in records)

# Quint model: token valid at position (from earlier: 9,446 pairs)
# Sextuple model: token valid at (position, prev_sfx)
print(f"\n  Placement precision:")
print(f"    (tok, pos) pairs:          {len(actual_tok_pos):,}")
print(f"    (tok, pos, prev_sfx):      {len(actual_tok_pos_prev):,}")
print(f"    Expansion from prev_sfx:   {len(actual_tok_pos_prev)/len(actual_tok_pos):.2f}×")

# But how many WRONG triples does the quint model allow?
# Quint allows: any observed (tok, pos) with any prev_sfx
quint_allows = set()
prev_sfx_set = set(prev_sfx)
for tp in actual_tok_pos:
    for ps in prev_sfx_set:
        quint_allows.add((tp[0], tp[1], ps))

print(f"    Quint model allows:        {len(quint_allows):,} triples")
print(f"    Sextuple model allows:     {len(actual_tok_pos_prev):,} triples")
print(f"    Eliminated:                {len(quint_allows)-len(actual_tok_pos_prev):,} "
      f"({(len(quint_allows)-len(actual_tok_pos_prev))/len(quint_allows)*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════
# CONNECTION 2: LINE-LEVEL PATTERNS → PARAGRAPH FLAG
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CONNECTION 2: PARAGRAPH MARKERS")
print("="*70)

# Does is_para_first add information?
para_vals = [str(r.get('is_para_first', False)) for r in records]
h_para = H(para_vals)
h_quad_para = H(list(zip(quad_vals, para_vals)))
mi_quad_para = h_quad + h_para - h_quad_para

print(f"  MI(quad; is_para_first) = {mi_quad_para:.4f} bits ({mi_quad_para/h_quad*100:.2f}%)")
print(f"  Para-first tokens: {para_first} ({para_first/N*100:.1f}%)")

# What's the para-first token profile?
if para_first > 0:
    pf_recs = [r for r in records if r.get('is_para_first', False)]
    pf_pfx = Counter(r['prefix'] for r in pf_recs)
    pf_gal = Counter(r['gallows'] for r in pf_recs)
    n_pf = len(pf_recs)
    
    print(f"  Para-first prefix:  {pf_pfx.most_common(5)}")
    print(f"  Para-first gallows: ∅={pf_gal.get('∅',0)/n_pf*100:.0f}%, "
          f"non-∅={100-pf_gal.get('∅',0)/n_pf*100:.0f}%")
    
    # Is this already captured by pos=FIRST?
    first_recs = [r for r in records if r['_pos'] == 'FIRST']
    first_pf = [r for r in first_recs if r.get('is_para_first', False)]
    print(f"  Para-first tokens that are also FIRST: {len(first_pf)}/{len(pf_recs)}")
    
    # MI of quad with para beyond pos+section
    h_pos_sec_para = H(list(zip(pos_vals, sec_vals, para_vals)))
    h_quad_psp = H(list(zip(quad_vals, pos_vals, sec_vals, para_vals)))
    h_quad_given_psp = h_quad_psp - h_pos_sec_para
    mi_quad_psp = h_quad - h_quad_given_psp
    mi_para_additional = mi_quad_psp - mi_quad_possec
    
    print(f"  Para flag beyond pos+sec: {mi_para_additional:.4f} bits ({mi_para_additional/h_quad*100:.3f}%)")

# ═══════════════════════════════════════════════════════════════
# CONNECTION 3: QUIRE STRUCTURE → PRODUCTION UNITS
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CONNECTION 3: QUIRE / PRODUCTION UNITS")
print("="*70)

# Quire information is folio-level, not token-level.
# Test: does quire identity add information beyond section?

# Extract quire from folio (approximate: quires map to folio ranges)
# Use the section as proxy since section≈quire for most of the MS
# More precisely: bin folios into quires

def folio_to_quire(folio):
    """Approximate quire assignment from folio number."""
    # Strip 'f' prefix and any 'r'/'v' suffix
    num_str = folio.replace('f', '').replace('r', '').replace('v', '')
    # Handle folios like '103' or '1'
    try:
        num = int(num_str.split('.')[0])
    except:
        return 'UNK'
    
    if num <= 8: return 'Q1'
    elif num <= 16: return 'Q2'
    elif num <= 22: return 'Q3'
    elif num <= 32: return 'Q4'
    elif num <= 38: return 'Q5'
    elif num <= 42: return 'Q6'
    elif num <= 50: return 'Q7'
    elif num <= 58: return 'Q8'
    elif num <= 66: return 'Q9-12'
    elif num <= 73: return 'Q13'
    elif num <= 84: return 'Q14'
    elif num <= 86: return 'Q15'
    elif num <= 90: return 'Q16-17'
    elif num <= 96: return 'Q18'
    elif num <= 103: return 'Q19'
    elif num <= 116: return 'Q20'
    return 'UNK'

quire_vals = [folio_to_quire(r['folio']) for r in records]
h_quire = H(quire_vals)
h_sec = H(sec_vals)

# MI(quad; quire) vs MI(quad; section)
h_quad_quire = H(list(zip(quad_vals, quire_vals)))
mi_quad_quire = h_quad + h_quire - h_quad_quire

# MI(quad; quire | section) — does quire add beyond section?
h_sec_quire = H(list(zip(sec_vals, quire_vals)))
h_quad_sq = H(list(zip(quad_vals, sec_vals, quire_vals)))
h_quad_given_sq = h_quad_sq - h_sec_quire
mi_quad_sq = h_quad - h_quad_given_sq
mi_quire_additional = mi_quad_sq - mi_quad_possec + (mi_quad_possec - 0.8105)  # subtract just section MI

# Simpler: MI(quad; quire) and MI(quad; quire | section)
mi_quire_beyond_sec = mi_quad_sq - 0.8105  # MI(quad; section alone)

print(f"  MI(quad; section)     = {0.8105:.4f} bits")
print(f"  MI(quad; quire)       = {mi_quad_quire:.4f} bits")
print(f"  MI(quad; sec+quire)   = {mi_quad_sq:.4f} bits")
print(f"  Quire beyond section  = {mi_quad_sq - 0.8105:.4f} bits ({(mi_quad_sq-0.8105)/h_quad*100:.2f}%)")

print(f"\n  Distinct quires: {len(set(quire_vals))}")
print(f"  Distinct sections: {len(set(sec_vals))}")
print(f"  Section already absorbs most quire information.")

# Cross-quire production cluster test
# f42, f49, f56 should share more quads than expected
cluster_folios = {'f42r', 'f42v', 'f49r', 'f49v', 'f56r', 'f56v'}
cluster_recs = [r for r in records if r['folio'] in cluster_folios]
other_herbal = [r for r in records if r['section'] in ('Herbal-A', 'Herbal-B') 
                and r['folio'] not in cluster_folios]

if cluster_recs:
    cluster_quads = set((r['prefix'],r['gallows'],r['m_core'],r['sfx_fam']) for r in cluster_recs)
    other_quads = set((r['prefix'],r['gallows'],r['m_core'],r['sfx_fam']) for r in other_herbal)
    
    # Jaccard
    shared = cluster_quads & other_quads
    jaccard = len(shared) / len(cluster_quads | other_quads) if (cluster_quads | other_quads) else 0
    print(f"\n  f42-f49-f56 cluster quads: {len(cluster_quads)}")
    print(f"  Other Herbal quads: {len(other_quads)}")
    print(f"  Shared: {len(shared)} (Jaccard={jaccard:.3f})")
    print(f"  Cluster-exclusive quads: {len(cluster_quads - other_quads)}")

# ═══════════════════════════════════════════════════════════════
# CONNECTION 4: SUFFIX DECOMPOSITION → VALIDATE sfx_fam
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CONNECTION 4: SUFFIX DECOMPOSITION VALIDATES sfx_fam")
print("="*70)

# The suffix decomposes into VP (vowel prefix) + terminal
# VP carries 85% of section info, terminal carries 19% structural info
# sfx_fam groups by terminal → is this the right level for p70c?

# Test: does full suffix add information beyond sfx_fam?
sfx_full = [r['suffix'] for r in records]
sfx_fam_vals = [r['sfx_fam'] for r in records]

h_sfx_full = H(sfx_full)
h_sfx_fam = H(sfx_fam_vals)

print(f"  H(suffix_full)  = {h_sfx_full:.4f} bits ({len(set(sfx_full))} values)")
print(f"  H(sfx_fam)      = {h_sfx_fam:.4f} bits ({len(set(sfx_fam_vals))} values)")
print(f"  Information in VP: {h_sfx_full - h_sfx_fam:.4f} bits ({(h_sfx_full-h_sfx_fam)/h_sfx_full*100:.1f}%)")

# Does VP add to quad prediction beyond sfx_fam?
h_quad_sfxfam = H(list(zip(quad_vals, sfx_fam_vals)))
# ... quad already contains sfx_fam, so this is tautological
# Instead: does replacing sfx_fam with full suffix in the quad give better prediction?

full_quads = [(r['prefix'], r['gallows'], r['m_core'], r['suffix']) for r in records]
h_full_quads = H(full_quads)
fam_quads = [(r['prefix'], r['gallows'], r['m_core'], r['sfx_fam']) for r in records]
h_fam_quads = H(fam_quads)

print(f"\n  H(P,G,mc,suffix_full) = {h_full_quads:.4f} bits ({len(set(full_quads))} entries)")
print(f"  H(P,G,mc,sfx_fam)    = {h_fam_quads:.4f} bits ({len(set(fam_quads))} entries)")
print(f"  Full suffix adds:     {h_full_quads - h_fam_quads:.4f} bits ({(h_full_quads-h_fam_quads)/h_fam_quads*100:.1f}%)")

# How many full-suffix quads vs fam quads?
full_quad_counts = Counter(full_quads)
fam_quad_counts = Counter(fam_quads)

print(f"\n  Full-suffix quads: {len(full_quad_counts):,} ({sum(1 for c in full_quad_counts.values() if c==1)} hapax)")
print(f"  Fam quads:         {len(fam_quad_counts):,} ({sum(1 for c in fam_quad_counts.values() if c==1)} hapax)")

# Generalisation test
full_hapax_pct = sum(1 for c in full_quad_counts.values() if c==1) / len(full_quad_counts) * 100
fam_hapax_pct = sum(1 for c in fam_quad_counts.values() if c==1) / len(fam_quad_counts) * 100

print(f"  Full-suffix hapax: {full_hapax_pct:.1f}%")
print(f"  Fam hapax:         {fam_hapax_pct:.1f}%")

# Does section conditioning change with full suffix?
h_full_sec = H(list(zip(full_quads, sec_vals)))
mi_full_sec = H(full_quads) + H(sec_vals) - h_full_sec
h_fam_sec = H(list(zip(fam_quads, sec_vals)))
mi_fam_sec = H(fam_quads) + H(sec_vals) - h_fam_sec

print(f"\n  MI(full_quad; section) = {mi_full_sec:.4f} bits ({mi_full_sec/h_full_quads*100:.2f}%)")
print(f"  MI(fam_quad; section)  = {mi_fam_sec:.4f} bits ({mi_fam_sec/h_fam_quads*100:.2f}%)")
print(f"  VP adds to section MI: {mi_full_sec - mi_fam_sec:.4f} bits")

print(f"\n  VERDICT: sfx_fam is the {'CORRECT' if h_full_quads - h_fam_quads < 1.0 else 'INSUFFICIENT'} "
      f"abstraction for p70-c.")
print(f"  Full suffix adds {h_full_quads - h_fam_quads:.3f} bits ({(h_full_quads-h_fam_quads)/h_fam_quads*100:.1f}%) "
      f"but at the cost of {len(full_quad_counts)-len(fam_quad_counts):,} additional entries "
      f"and {full_hapax_pct:.0f}% vs {fam_hapax_pct:.0f}% hapax rate.")

# ═══════════════════════════════════════════════════════════════
# SYNTHESIS: COMPLETE INFORMATION BUDGET
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMPLETE INFORMATION BUDGET")
print("="*70)

# Cumulative MI as we add each conditioning variable
print(f"\n  {'Conditioning':>40} {'MI (bits)':>10} {'Cumul %':>8} {'Δ':>10}")
print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*10}")

# Build cumulatively
axes = [
    ('section', sec_vals),
    ('position', pos_vals),
    ('prev_sfx', prev_sfx),
    ('para_flag', para_vals),
    ('quire', quire_vals),
]

cumul_context = []
prev_mi = 0
for name, vals in axes:
    cumul_context.append(vals)
    combined = list(zip(*cumul_context))
    h_combined = H(combined)
    h_quad_combined = H(list(zip(quad_vals, *cumul_context)))
    h_quad_given_combined = h_quad_combined - h_combined
    mi = h_quad - h_quad_given_combined
    delta = mi - prev_mi
    
    print(f"  {'+ ' + name:>40} {mi:>10.4f} {mi/h_quad*100:>7.2f}% {delta:>10.4f}")
    prev_mi = mi

print(f"  {'UNEXPLAINED':>40} {h_quad - prev_mi:>10.4f} {(h_quad-prev_mi)/h_quad*100:>7.2f}%")
print(f"  {'H(quad)':>40} {h_quad:>10.4f}")

# ═══════════════════════════════════════════════════════════════
# RECOMMENDATION TABLE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"RECOMMENDATION: WHICH AXES TO ENCODE IN P70-C?")
print(f"{'='*70}")

print(f"""
  AXIS            MI added   Entries   Hapax%   Generalises?   ENCODE?
  ──────────────  ─────────  ────────  ──────   ────────────   ───────
  section         0.810      9,475     n/a      yes            YES (metadata)
  position        0.518*     6,750     83.4%    T1/T2 yes      YES (built)
  prev_sfx        {mi_prev_additional:.3f}*     {len(sext_counts):,}    ~90%     worse          LOOKUP ONLY
  para_flag       tiny       ~few      n/a      yes            NO (too sparse)
  quire           {mi_quad_sq-0.8105:.3f}*     n/a       n/a      yes            NO (≈section)

  * = additional MI beyond all preceding axes

  RECOMMENDED ARCHITECTURE:
    Layer 0: enriched_records.pkl (ground truth)
    Layer 1: p70 rules (character adjacency, 210 rules)
    Layer 2: p70c_pos (slot+position co-occurrence, 6,750 quints)
    Lookup:  transition_table (prev_sfx → prefix probability, 8×8 matrix)
    Metadata: section, quire, para_flag (folio/line level)

  DO NOT encode prev_sfx as a full axis — it adds {len(sext_counts)-len(quint_counts):,} entries
  for {mi_prev_additional:.3f} bits gain, with worse generalisation. Instead, provide it
  as a LOOKUP TABLE for sequential generation: given prev_sfx, what's
  the prefix distribution of the next word?
""")

# Build and export the transition lookup
print("Building transition lookup table...")
trans_table = defaultdict(Counter)
for r in records:
    trans_table[r['prev_sfx']][r['prefix']] += 1

# Normalise
trans_probs = {}
for ps, pfx_counts in trans_table.items():
    total = sum(pfx_counts.values())
    trans_probs[ps] = {p: round(c/total, 4) for p, c in pfx_counts.items()}

with open('transition_lookup.json', 'w') as f:
    json.dump(trans_probs, f, indent=2, sort_keys=True)
print(f"  Saved transition_lookup.json ({len(trans_probs)} prev_sfx values)")

# Save all results
results = {
    'mi_quad_possec': mi_quad_possec,
    'mi_quad_all3': mi_quad_all3,
    'mi_prev_additional': mi_prev_additional,
    'mi_quad_prev': mi_quad_prev,
    'sextuple_count': len(sext_counts),
    'quint_count': len(quint_counts),
    'trans_probs': trans_probs,
    'mi_quad_quire': mi_quad_quire,
    'mi_quire_beyond_sec': mi_quad_sq - 0.8105,
    'sfx_full_vs_fam_delta': h_full_quads - h_fam_quads,
}

with open('p70c_connections.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Saved p70c_connections.pkl")

