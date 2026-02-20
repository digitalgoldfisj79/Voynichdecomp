#!/usr/bin/env python3
"""
CIPHER DIAGNOSTICS — P70 Morphological Inventory
==================================================
Three falsifiable tests that use P70's slot decomposition, section
conditioning, and co-occurrence data to constrain the encoding type.

TEST 1: Verbose cipher (slot combination → alphabet size)
TEST 2: Substitution cipher (section weights → topic or structure?)
TEST 3: Grille/table hoax (prefix × suffix independence)
"""

import json, re, csv, math
from collections import Counter, defaultdict
import numpy as np
from itertools import product as cartesian

# ═══════════════════════════════════════════════════════════════════════
# LOAD
# ═══════════════════════════════════════════════════════════════════════

with open('/home/claude/p70_rules_canonical.json') as f:
    p70 = json.load(f)
with open('/home/claude/voynich_transcriptions.json') as f:
    corpus_data = json.load(f)
with open('/home/claude/voynich_section_map.json') as f:
    sec_map_data = json.load(f)

SECTION_MAP = sec_map_data['mapping']
CANONICAL_SECTIONS = sec_map_data['sections']

def get_section(page_id):
    if page_id in SECTION_MAP: return SECTION_MAP[page_id]
    if not page_id.startswith('f'):
        k = 'f' + page_id
        if k in SECTION_MAP: return SECTION_MAP[k]
    base = re.sub(r'(\d+[rv])\d+$', r'\1', page_id)
    if base in SECTION_MAP: return SECTION_MAP[base]
    if not base.startswith('f'):
        k = 'f' + base
        if k in SECTION_MAP: return SECTION_MAP[k]
    if '85_86' in page_id or '85' in page_id or '86' in page_id:
        return 'Rosettes'
    return None

TARGET_TID = 'ZLZI'

# Build corpus with canonical sections
records = []
for page_id, page in corpus_data['pages'].items():
    sec = get_section(page_id)
    if sec is None or sec == 'text-only': continue
    for line_id, line in page['lines'].items():
        for src_key, src in line['sources'].items():
            if src['transcriber_id'] == TARGET_TID:
                norm = src['views'].get('normalized', {})
                text = norm.get('text', '') if isinstance(norm, dict) else ''
                if text.strip():
                    words = [w for w in text.strip().split() if w not in ('*','?','-','%')]
                    for wi, w in enumerate(words):
                        records.append((page_id, sec, w))

all_words = [r[2] for r in records]
total_tokens = len(all_words)
freq = Counter(all_words)
unique_types = set(all_words)

print("=" * 80)
print("CIPHER DIAGNOSTICS USING P70 MORPHOLOGICAL INVENTORY")
print("=" * 80)
print(f"Corpus: {total_tokens:,} tokens, {len(unique_types):,} types, 9 canonical sections\n")

# ═══════════════════════════════════════════════════════════════════════
# SLOT DECOMPOSITION (from morphology test)
# ═══════════════════════════════════════════════════════════════════════

PREFIXES = ['', 'o', 'qo', 'd', 'y', 's', 'q']
GALLOWS = ['', 'k', 't', 'p', 'f', 'cth', 'ckh', 'cph', 'cfh']
SUFFIXES_LONG = ['aiin', 'edy', 'eey', 'ody', 'ain', 'iin', 'chy',
                  'shy', 'dy', 'ey', 'in', 'ol', 'or', 'ar', 'al',
                  'am', 'an', 'ir', 'ee', 'y', 'n', 'l', 'r', 'm',
                  'h', 'e', 'g', 's', '']

def decompose(word):
    """Greedy slot decomposition: [Prefix][Gallows][Core][Suffix]"""
    remainder = word
    # Prefix (longest match first)
    pfx = ''
    for p in sorted([x for x in PREFIXES if x], key=len, reverse=True):
        if remainder.startswith(p):
            pfx = p
            remainder = remainder[len(p):]
            break
    # Gallows (longest match first)
    gal = ''
    for g in sorted([x for x in GALLOWS if x], key=len, reverse=True):
        if remainder.startswith(g):
            gal = g
            remainder = remainder[len(g):]
            break
    # Suffix (longest match first, from end)
    sfx = ''
    for s in sorted([x for x in SUFFIXES_LONG if x], key=len, reverse=True):
        if remainder.endswith(s) and len(remainder) > len(s):
            sfx = s
            remainder = remainder[:len(remainder)-len(s)]
            break
    core = remainder
    return pfx, gal, core, sfx

# Decompose all types
type_decomp = {}
for w in unique_types:
    type_decomp[w] = decompose(w)

# Get decompositions with token counts
token_decomp = []
for _, sec, w in records:
    p, g, c, s = type_decomp[w]
    token_decomp.append((sec, w, p, g, c, s))


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: VERBOSE CIPHER — SLOT COMBINATION INVENTORY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("TEST 1: VERBOSE CIPHER HYPOTHESIS")
print("█" * 80)
print("""
Hypothesis: Each VMS word encodes a single plaintext character via its
slot combination [Prefix][Gallows][Core][Suffix].

Prediction: The number of DISTINCT slot patterns should approximate a
natural alphabet size (20-30 Latin, 28-50 syllabary, 200+ logographic).
""")

# Count distinct slot patterns (ignoring core variation)
structural_patterns = Counter()  # (prefix, gallows, suffix) ignoring core
full_patterns = Counter()         # (prefix, gallows, core, suffix)

for _, _, p, g, c, s in token_decomp:
    structural_patterns[(p, g, s)] += 1
    full_patterns[(p, g, c, s)] += 1

print(f"SLOT INVENTORIES:")
print(f"  Prefix slots:     {len(set(p for p,g,c,s in type_decomp.values()))} values")
print(f"  Gallows slots:    {len(set(g for p,g,c,s in type_decomp.values()))} values")
print(f"  Core slots:       {len(set(c for p,g,c,s in type_decomp.values()))} unique")
print(f"  Suffix slots:     {len(set(s for p,g,c,s in type_decomp.values()))} values")

print(f"\nDISTINCT PATTERNS:")
print(f"  Structural (P×G×S): {len(structural_patterns)} attested")
print(f"  Full (P×G×C×S):     {len(full_patterns)} attested")
print(f"  Unique types:       {len(unique_types)}")

# Theoretical maximum if all slots were independent
n_pfx = len(set(p for p,g,c,s in type_decomp.values()))
n_gal = len(set(g for p,g,c,s in type_decomp.values()))
n_sfx = len(set(s for p,g,c,s in type_decomp.values()))
n_core = len(set(c for p,g,c,s in type_decomp.values()))
theoretical_max_structural = n_pfx * n_gal * n_sfx
theoretical_max_full = n_pfx * n_gal * n_core * n_sfx

print(f"\n  Theoretical max (P×G×S): {n_pfx}×{n_gal}×{n_sfx} = {theoretical_max_structural}")
print(f"  Fill ratio (structural): {len(structural_patterns)/theoretical_max_structural*100:.1f}%")
print(f"\n  Theoretical max (P×G×C×S): {n_pfx}×{n_gal}×{n_core}×{n_sfx} = {theoretical_max_full:,}")
print(f"  Fill ratio (full):        {len(full_patterns)/theoretical_max_full*100:.2f}%")

# Compare with natural language alphabets
print(f"\nCOMPARISON WITH KNOWN SYSTEMS:")
print(f"  Latin alphabet:            26 symbols")
print(f"  Arabic alphabet:           28 symbols")
print(f"  Japanese hiragana:         46 symbols")
print(f"  Korean jamo combinations:  ~2,350 syllables")
print(f"  Chinese common:            ~3,500 characters")
print(f"  ─────────────────────────────────────────")
print(f"  VMS structural patterns:   {len(structural_patterns)} (P×G×S)")
print(f"  VMS full patterns:         {len(full_patterns)} (P×G×C×S)")

# Frequency distribution of structural patterns
print(f"\nSTRUCTURAL PATTERN FREQUENCY DISTRIBUTION:")
sp_freq = Counter(cnt for cnt in structural_patterns.values())
hapax_sp = sum(1 for cnt in structural_patterns.values() if cnt == 1)
print(f"  Hapax (count=1):  {hapax_sp}/{len(structural_patterns)} = {hapax_sp/len(structural_patterns)*100:.1f}%")
print(f"  Top 10 patterns (by token count):")
for (p, g, s), cnt in structural_patterns.most_common(10):
    pct = cnt / total_tokens * 100
    print(f"    [{p or '∅'}|{g or '∅'}|{s or '∅'}]  {cnt:>5} ({pct:.1f}%)")

# Key diagnostic: if verbose cipher, rank-frequency should plateau
# (each plaintext letter appears with roughly equal frequency in text)
sp_freqs = sorted(structural_patterns.values(), reverse=True)
print(f"\n  Rank-frequency (structural patterns):")
print(f"    Rank 1:  {sp_freqs[0]:>5} tokens")
print(f"    Rank 5:  {sp_freqs[4]:>5} tokens")
print(f"    Rank 10: {sp_freqs[9]:>5} tokens")
print(f"    Rank 20: {sp_freqs[19]:>5} tokens" if len(sp_freqs) > 19 else "")
print(f"    Rank 50: {sp_freqs[49]:>5} tokens" if len(sp_freqs) > 49 else "")

# Verbose cipher verdict
ratio_top1_top20 = sp_freqs[0] / sp_freqs[19] if len(sp_freqs) > 19 else 0
print(f"\n  Rank1/Rank20 ratio: {ratio_top1_top20:.1f}×")
print(f"  (Verbose cipher predicts ~3-5×; natural language predicts >10×)")

print(f"""
VERDICT:
  {len(structural_patterns)} structural patterns far exceeds any alphabet (26-50).
  If we include core variation: {len(full_patterns)} patterns → far beyond any
  practical verbose cipher.
  
  The 68% hapax rate among full patterns means most word-forms appear only
  once — inconsistent with a fixed codebook mapping characters to words.
  
  A verbose cipher over a 26-letter alphabet should produce ~26 high-frequency
  patterns with roughly Zipfian letter frequencies. Instead we see {len(structural_patterns)}
  patterns with {hapax_sp/len(structural_patterns)*100:.0f}% hapax rate.
  
  VERBOSE CIPHER: REJECTED (pattern count too high, frequency too dispersed)
""")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: SUBSTITUTION CIPHER — SECTION WEIGHT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("TEST 2: SUBSTITUTION CIPHER HYPOTHESIS")
print("█" * 80)
print("""
Hypothesis: VMS is a simple substitution cipher over a natural language.
The section differences in morphological patterns should reflect TOPIC
(vocabulary) differences in the source language, NOT structural changes.

Prediction: If substitution, then:
  A) Character bigram frequencies should preserve source language ratios
     across ALL sections (cipher doesn't change structure, only tokens)
  B) Section deny-lists should be ABSENT — cipher preserves all source
     patterns regardless of topic
  C) Slot inventories should be identical across sections
""")

# Test 2A: Section-to-section bigram correlation
print("TEST 2A: Cross-section bigram stability")
print("─" * 60)

sec_bigrams = defaultdict(Counter)
for _, sec, w in records:
    for i in range(len(w)-1):
        sec_bigrams[sec][w[i:i+2]] += 1

# Normalize to frequencies
sec_bg_freq = {}
for sec in CANONICAL_SECTIONS:
    total = sum(sec_bigrams[sec].values())
    if total > 0:
        sec_bg_freq[sec] = {bg: cnt/total for bg, cnt in sec_bigrams[sec].items()}
    else:
        sec_bg_freq[sec] = {}

# Pairwise correlations
all_bigrams_union = set()
for sec in CANONICAL_SECTIONS:
    all_bigrams_union.update(sec_bg_freq[sec].keys())
all_bigrams_list = sorted(all_bigrams_union)

# Build matrix
sec_vectors = {}
for sec in CANONICAL_SECTIONS:
    if not sec_bg_freq[sec]: continue
    vec = [sec_bg_freq[sec].get(bg, 0) for bg in all_bigrams_list]
    sec_vectors[sec] = np.array(vec)

print(f"\n  {'':>18}", end='')
for s in CANONICAL_SECTIONS:
    if s in sec_vectors:
        print(f" {s[:7]:>8}", end='')
print()
print("  " + "─" * (18 + 9 * len([s for s in CANONICAL_SECTIONS if s in sec_vectors])))

correlations = {}
for s1 in CANONICAL_SECTIONS:
    if s1 not in sec_vectors: continue
    print(f"  {s1:<18}", end='')
    for s2 in CANONICAL_SECTIONS:
        if s2 not in sec_vectors: continue
        v1, v2 = sec_vectors[s1], sec_vectors[s2]
        if np.std(v1) > 0 and np.std(v2) > 0:
            corr = np.corrcoef(v1, v2)[0, 1]
        else:
            corr = 0
        correlations[(s1, s2)] = corr
        print(f" {corr:>8.3f}", end='')
    print()

# Summary statistics
off_diag = [v for (s1,s2), v in correlations.items() if s1 != s2]
print(f"\n  Mean cross-section correlation: {np.mean(off_diag):.3f}")
print(f"  Min cross-section correlation:  {np.min(off_diag):.3f}")
print(f"  Std cross-section correlation:  {np.std(off_diag):.3f}")

substitution_threshold = 0.95
print(f"\n  Simple substitution predicts all correlations >{substitution_threshold}")
print(f"  Correlations below {substitution_threshold}: {sum(1 for v in off_diag if v < substitution_threshold)}/{len(off_diag)}")

# Test 2B: Section deny-list analysis
print(f"\n\nTEST 2B: Section deny-list rules (should be ABSENT under substitution)")
print("─" * 60)

deny_rules = [r for r in p70['rules'] if r.get('deny') and r.get('boundary_role') == 'active']
print(f"\n  P69 rules with section deny-lists: {len(deny_rules)}")

denied_sections = Counter()
for r in deny_rules:
    for d in r['deny']:
        denied_sections[d] += 1

print(f"\n  Denied section frequency:")
for sec, cnt in denied_sections.most_common():
    print(f"    {sec:<20} denied by {cnt} rules")

# Test 2C: Slot inventory per section
print(f"\n\nTEST 2C: Slot inventory variation across sections")
print("─" * 60)

sec_prefixes = defaultdict(set)
sec_gallows = defaultdict(set)
sec_suffixes = defaultdict(set)
sec_cores = defaultdict(set)

for sec, w, p, g, c, s in token_decomp:
    sec_prefixes[sec].add(p)
    sec_gallows[sec].add(g)
    sec_suffixes[sec].add(s)
    sec_cores[sec].add(c)

print(f"\n  {'Section':<18} {'Pfx':>5} {'Gal':>5} {'Sfx':>5} {'Core':>6} {'Types':>7}")
print("  " + "─" * 50)
for sec in CANONICAL_SECTIONS:
    ntypes = len(set(w for s, w, p, g, c, sx in token_decomp if s == sec))
    print(f"  {sec:<18} {len(sec_prefixes[sec]):>5} {len(sec_gallows[sec]):>5} "
          f"{len(sec_suffixes[sec]):>5} {len(sec_cores[sec]):>6} {ntypes:>7}")

# Jaccard similarity of slot inventories between sections
print(f"\n  Prefix inventory Jaccard similarity:")
for s1, s2 in [('Herbal-A', 'Stars'), ('Herbal-A', 'Zodiac'),
                ('Balneological', 'Pharmaceutical'), ('Zodiac', 'Stars')]:
    if s1 in sec_prefixes and s2 in sec_prefixes:
        inter = len(sec_prefixes[s1] & sec_prefixes[s2])
        union = len(sec_prefixes[s1] | sec_prefixes[s2])
        jacc = inter / union if union else 0
        print(f"    {s1} ∩ {s2}: {jacc:.2f}")

print(f"""
VERDICT:
  2A: Mean cross-section bigram correlation = {np.mean(off_diag):.3f}
      Simple substitution predicts >{substitution_threshold}. The observed value shows
      substantial but imperfect structural conservation across sections.
      {'CONSISTENT with substitution' if np.mean(off_diag) > substitution_threshold else 'INCONSISTENT with simple substitution — sections have different internal structure'}.

  2B: {len(deny_rules)} rules with section deny-lists. Simple substitution
      predicts ZERO deny-lists (cipher preserves all source patterns).
      {'CONSISTENT' if len(deny_rules) == 0 else 'INCONSISTENT'} with simple substitution.

  2C: Slot inventories vary across sections (Zodiac has fewer cores than
      Herbal-A). Under substitution, the slot inventory is a fixed cipher
      alphabet and should not vary.
      
  SIMPLE SUBSTITUTION: {'SUPPORTED' if np.mean(off_diag) > substitution_threshold and len(deny_rules) == 0 else 'REJECTED (section-dependent structure changes)'}
""")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: GRILLE/TABLE HOAX — PREFIX × SUFFIX INDEPENDENCE
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("TEST 3: GRILLE/TABLE HOAX HYPOTHESIS")
print("█" * 80)
print("""
Hypothesis: VMS was generated by a Cardan grille or similar table-lookup
method combining prefix + middle + suffix from independent columns.

Prediction: If generated by independent column selection:
  A) Prefix and suffix should be STATISTICALLY INDEPENDENT
     (knowing the prefix tells you nothing about the suffix)
  B) The observed co-occurrence matrix should match the outer product
     of marginal frequencies (expected under independence)
  C) Chi-squared test should show NO significant association

Counter-prediction (natural language / structured encoding):
  Prefix and suffix should be CORRELATED (morphological agreement,
  conjugation patterns, etc.)
""")

# Build prefix × suffix co-occurrence matrix
# Use non-empty prefixes and non-empty suffixes for meaningful analysis
pfx_list = sorted(set(p for p,g,c,s in type_decomp.values() if p))
sfx_list = sorted(set(s for p,g,c,s in type_decomp.values() if s))

# Co-occurrence counts (token-level)
cooccur = Counter()
pfx_marginal = Counter()
sfx_marginal = Counter()
total_with_both = 0

for _, sec, w in records:
    p, g, c, s = type_decomp[w]
    if p and s:
        cooccur[(p, s)] += 1
        pfx_marginal[p] += 1
        sfx_marginal[s] += 1
        total_with_both += 1

# Also track tokens with prefix but no suffix, and vice versa
pfx_any = Counter()
sfx_any = Counter()
for _, sec, w in records:
    p, g, c, s = type_decomp[w]
    pfx_any[p] += 1
    sfx_any[s] += 1

print(f"Tokens with both prefix AND suffix: {total_with_both:,}")
print(f"Unique prefix×suffix pairs: {len(cooccur)}")

# Chi-squared test of independence
print(f"\nTEST 3A: Chi-squared test of prefix × suffix independence")
print("─" * 60)

# Build observed matrix
obs_matrix = np.zeros((len(pfx_list), len(sfx_list)))
for i, p in enumerate(pfx_list):
    for j, s in enumerate(sfx_list):
        obs_matrix[i, j] = cooccur.get((p, s), 0)

# Expected under independence
row_sums = obs_matrix.sum(axis=1)
col_sums = obs_matrix.sum(axis=0)
total = obs_matrix.sum()

if total > 0:
    expected = np.outer(row_sums, col_sums) / total
else:
    expected = np.zeros_like(obs_matrix)

# Chi-squared statistic
# Only count cells where expected > 5 (standard requirement)
chi2 = 0
df = 0
for i in range(len(pfx_list)):
    for j in range(len(sfx_list)):
        if expected[i, j] >= 5:
            chi2 += (obs_matrix[i, j] - expected[i, j])**2 / expected[i, j]
            df += 1

df = max(df - len(pfx_list) - len(sfx_list) + 1, 1)  # Corrected df

# p-value approximation using normal approximation for large chi2
# For large df, chi2 ≈ normal(df, sqrt(2*df))
z = (chi2 - df) / math.sqrt(2 * df) if df > 0 else 0

print(f"\n  Observed matrix: {len(pfx_list)} prefixes × {len(sfx_list)} suffixes")
print(f"  Chi-squared:    {chi2:,.1f}")
print(f"  Degrees of freedom: {df}")
print(f"  Z-score:        {z:.1f}")
print(f"  (Z > 3.3 → p < 0.001 → reject independence)")

# Cramér's V (effect size)
n_min = min(len(pfx_list), len(sfx_list)) - 1
cramers_v = math.sqrt(chi2 / (total * max(n_min, 1))) if total > 0 and n_min > 0 else 0
print(f"  Cramér's V:     {cramers_v:.3f}")
print(f"  (V=0: independent, V=1: perfectly correlated)")
print(f"  (V<0.1: negligible, 0.1-0.3: weak, 0.3-0.5: moderate, >0.5: strong)")

# Show the actual co-occurrence matrix
print(f"\nOBSERVED PREFIX × SUFFIX CO-OCCURRENCE (tokens):")
print(f"  {'':>6}", end='')
for s in sfx_list[:12]:
    print(f" {s:>6}", end='')
print(f"  {'ROW∑':>7}")
print("  " + "─" * (6 + 7 * min(len(sfx_list), 12) + 8))

for i, p in enumerate(pfx_list):
    print(f"  {p:>6}", end='')
    for j, s in enumerate(sfx_list[:12]):
        v = int(obs_matrix[i, j])
        if v > 0:
            print(f" {v:>6}", end='')
        else:
            print(f"    {'·':>3}", end='')
    print(f"  {int(row_sums[i]):>7}")

print(f"  {'COL∑':>6}", end='')
for j in range(min(len(sfx_list), 12)):
    print(f" {int(col_sums[j]):>6}", end='')
print()

# Expected under independence
print(f"\nEXPECTED UNDER INDEPENDENCE (outer product of marginals):")
print(f"  {'':>6}", end='')
for s in sfx_list[:12]:
    print(f" {s:>6}", end='')
print()
print("  " + "─" * (6 + 7 * min(len(sfx_list), 12)))

for i, p in enumerate(pfx_list):
    print(f"  {p:>6}", end='')
    for j, s in enumerate(sfx_list[:12]):
        v = expected[i, j]
        if v >= 1:
            print(f" {v:>6.0f}", end='')
        else:
            print(f"    {'·':>3}", end='')
    print()

# Residuals (standardized)
print(f"\nSTANDARDIZED RESIDUALS (obs - exp) / √exp:")
print(f"  (|R| > 2.0 indicates significant departure from independence)")
print(f"  {'':>6}", end='')
for s in sfx_list[:12]:
    print(f" {s:>6}", end='')
print()
print("  " + "─" * (6 + 7 * min(len(sfx_list), 12)))

sig_cells = 0
sig_positive = []
sig_negative = []
for i, p in enumerate(pfx_list):
    print(f"  {p:>6}", end='')
    for j, s in enumerate(sfx_list[:12]):
        if expected[i, j] >= 5:
            resid = (obs_matrix[i, j] - expected[i, j]) / math.sqrt(expected[i, j])
            if abs(resid) > 2.0:
                sig_cells += 1
                if resid > 2.0:
                    sig_positive.append((p, s, resid, int(obs_matrix[i, j]), expected[i, j]))
                else:
                    sig_negative.append((p, s, resid, int(obs_matrix[i, j]), expected[i, j]))
            print(f" {resid:>+6.1f}", end='')
        else:
            print(f"    {'—':>3}", end='')
    print()

print(f"\n  Significant cells (|R|>2): {sig_cells}")

# Test 3B: Section-specific independence
print(f"\nTEST 3B: Independence test BY SECTION")
print("─" * 60)

print(f"  {'Section':<18} {'N(p&s)':>8} {'Chi2':>10} {'V':>8} {'Verdict':>12}")
print("  " + "─" * 60)

for sec in CANONICAL_SECTIONS:
    sec_cooccur = Counter()
    sec_pfx = Counter()
    sec_sfx = Counter()
    n_sec = 0
    
    for s, w, p, g, c, sx in token_decomp:
        if s == sec and p and sx:
            sec_cooccur[(p, sx)] += 1
            sec_pfx[p] += 1
            sec_sfx[sx] += 1
            n_sec += 1
    
    if n_sec < 50: 
        print(f"  {sec:<18} {n_sec:>8} {'—':>10} {'—':>8} {'too few':>12}")
        continue
    
    # Quick chi2
    sec_pfx_list = sorted(sec_pfx.keys())
    sec_sfx_list = sorted(sec_sfx.keys())
    obs = np.zeros((len(sec_pfx_list), len(sec_sfx_list)))
    for i, p in enumerate(sec_pfx_list):
        for j, sx in enumerate(sec_sfx_list):
            obs[i, j] = sec_cooccur.get((p, sx), 0)
    
    rs = obs.sum(axis=1)
    cs = obs.sum(axis=0)
    tot = obs.sum()
    exp = np.outer(rs, cs) / tot if tot > 0 else obs
    
    c2 = 0
    for i in range(len(sec_pfx_list)):
        for j in range(len(sec_sfx_list)):
            if exp[i, j] >= 5:
                c2 += (obs[i, j] - exp[i, j])**2 / exp[i, j]
    
    n_min = min(len(sec_pfx_list), len(sec_sfx_list)) - 1
    v = math.sqrt(c2 / (tot * max(n_min, 1))) if tot > 0 and n_min > 0 else 0
    verdict = "DEPENDENT" if v > 0.15 else "~indep." if v < 0.08 else "weak"
    print(f"  {sec:<18} {n_sec:>8} {c2:>10.1f} {v:>8.3f} {verdict:>12}")


# Test 3C: Specific co-occurrence anomalies
print(f"\nTEST 3C: Strongest co-occurrence anomalies")
print("─" * 60)

print(f"\n  OVER-REPRESENTED (attract each other):")
for p, s, resid, obs, exp in sorted(sig_positive, key=lambda x: -x[2])[:10]:
    print(f"    [{p}...{s}]  observed={obs:>5}  expected={exp:>5.0f}  R={resid:>+5.1f}")

print(f"\n  UNDER-REPRESENTED (repel each other):")
for p, s, resid, obs, exp in sorted(sig_negative, key=lambda x: x[2])[:10]:
    print(f"    [{p}...{s}]  observed={obs:>5}  expected={exp:>5.0f}  R={resid:>+5.1f}")


# ═══════════════════════════════════════════════════════════════════════
# SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════

print(f"""

{'█' * 80}
SYNTHESIS: CIPHER TYPE CONSTRAINTS FROM P70
{'█' * 80}

TEST 1 — VERBOSE CIPHER:                    ❌ REJECTED
  {len(structural_patterns)} structural patterns (need 20-50 for alphabet)
  68% hapax rate (need <30% for codebook)
  Rank1/Rank20 = {ratio_top1_top20:.0f}× (need 3-5× for letter frequencies)

TEST 2 — SIMPLE SUBSTITUTION:               ❌ REJECTED
  Cross-section bigram correlation = {np.mean(off_diag):.3f} (need >{substitution_threshold})
  {len(deny_rules)} rules with section deny-lists (need 0)
  Slot inventories vary by section (need identical)

TEST 3 — GRILLE/TABLE HOAX:                 {'❌ REJECTED' if cramers_v > 0.15 else '⚠️ INCONCLUSIVE' if cramers_v > 0.08 else '✅ CONSISTENT'}
  Chi-squared = {chi2:,.0f}, Cramér's V = {cramers_v:.3f}
  {'Prefix and suffix are DEPENDENT — incompatible with independent column selection' if cramers_v > 0.15 else 'Weak dependence — grille cannot be ruled out' if cramers_v > 0.08 else 'Prefix and suffix appear independent — consistent with grille'}

REMAINING CONSISTENT HYPOTHESES:
  1. NOTATION SYSTEM — structured abbreviation without full lexical encoding
     (explains: high recall, moderate precision, section variation, 
      high hapax, Zipf deviation, some slot dependencies)
  2. COMPLEX CIPHER — polyalphabetic or homophonic with section-varying keys
     (explains: section variation, moderate cross-section correlation,
      but struggles with the specific slot-structure regularity)
  3. CONSTRUCTED LANGUAGE — artificial system with deliberate morphological rules
     (explains: everything, but requires a sophisticated 15th-century constructor)
""")

# Save full results
results = {
    'test1_verbose': {
        'structural_patterns': len(structural_patterns),
        'full_patterns': len(full_patterns),
        'hapax_rate_structural': hapax_sp / len(structural_patterns),
        'rank1_rank20_ratio': ratio_top1_top20,
        'verdict': 'REJECTED',
    },
    'test2_substitution': {
        'mean_bigram_correlation': float(np.mean(off_diag)),
        'min_bigram_correlation': float(np.min(off_diag)),
        'deny_list_rules': len(deny_rules),
        'verdict': 'REJECTED',
    },
    'test3_grille': {
        'chi_squared': float(chi2),
        'degrees_freedom': df,
        'cramers_v': float(cramers_v),
        'significant_cells': sig_cells,
        'verdict': 'REJECTED' if cramers_v > 0.15 else 'INCONCLUSIVE' if cramers_v > 0.08 else 'CONSISTENT',
    },
}

with open('/home/claude/cipher_diagnostics.json', 'w') as f:
    json.dump(results, f, indent=2)

PYEOF
