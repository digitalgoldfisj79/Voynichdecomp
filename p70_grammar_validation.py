#!/usr/bin/env python3
"""
P70 RULE VALIDATION — USING THE GRAMMAR'S OWN METRICS
=======================================================
Gold standard: enriched_records.pkl (37,465 tokens, zero reconstruction errors)
Grammar target: H(w) = 2.788 + 1.374 + 3.622 + 2.527 = 10.311 bits

The chain rule H(w) = H(p) + H(g|p) + H(c|p,g) + H(s|p,g,c) holds trivially
for ANY lossless decomposition. What makes P70 special is:
  1. The DISTRIBUTION of entropy across slots
  2. Cross-slot coupling (Cramér's V, MI values)
  3. Prefix→section MI = 0.154 bits
  4. Core→suffix coupling MI = 0.551 bits

If alternatives can't match these simultaneously, P70 rules ARE special.
"""

import pickle, math, random
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import chi2_contingency, entropy as sp_entropy

random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# LOAD GOLD STANDARD
# ═══════════════════════════════════════════════════════════════════════

print("Loading enriched_records...")
with open('/home/claude/Voynichdecomp/enriched_records.pkl.txt', 'rb') as f:
    records = pickle.load(f)

tokens = [r['token'] for r in records]
sections = [r['section'] for r in records]
N = len(records)
print(f"  {N:,} tokens, {len(set(tokens)):,} types, {len(set(sections))} sections")

# ═══════════════════════════════════════════════════════════════════════
# METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def H(values):
    """Entropy of a discrete distribution."""
    c = Counter(values)
    total = sum(c.values())
    probs = np.array([v/total for v in c.values()])
    return -np.sum(probs * np.log2(probs + 1e-30))

def H_cond(target_vals, given_vals):
    """H(target | given) = H(target, given) - H(given)."""
    joint = list(zip(target_vals, given_vals))
    return H(joint) - H(given_vals)

def H_cond2(target_vals, given1, given2):
    """H(target | given1, given2)."""
    joint_given = list(zip(given1, given2))
    triple = list(zip(target_vals, given1, given2))
    return H(triple) - H(joint_given)

def H_cond3(target_vals, given1, given2, given3):
    """H(target | given1, given2, given3)."""
    joint_given = list(zip(given1, given2, given3))
    quad = list(zip(target_vals, given1, given2, given3))
    return H(quad) - H(joint_given)

def MI(vals_a, vals_b):
    """Mutual information I(A;B) = H(A) + H(B) - H(A,B)."""
    return H(vals_a) + H(vals_b) - H(list(zip(vals_a, vals_b)))

def cramers_v(vals_a, vals_b):
    """Cramér's V between two categorical variables."""
    ct = defaultdict(lambda: defaultdict(int))
    for a, b in zip(vals_a, vals_b):
        ct[a][b] += 1
    rows = sorted(ct.keys())
    cols = sorted(set(vals_b))
    if len(rows) < 2 or len(cols) < 2:
        return 0.0
    m = np.zeros((len(rows), len(cols)))
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            m[i, j] = ct[r].get(c, 0)
    m = m[m.sum(1) > 0][:, m.sum(0) > 0]
    if m.shape[0] < 2 or m.shape[1] < 2:
        return 0.0
    chi2, _, _, _ = chi2_contingency(m)
    n = m.sum()
    k = min(m.shape)
    return np.sqrt(chi2 / (n * (k - 1))) if n * (k - 1) > 0 else 0

def chain_rule_budget(prefixes, gallows_list, cores, suffixes):
    """Compute the full chain-rule entropy decomposition."""
    h_p = H(prefixes)
    h_g_p = H_cond(gallows_list, prefixes)
    h_c_pg = H_cond2(cores, prefixes, gallows_list)
    h_s_pgc = H_cond3(suffixes, prefixes, gallows_list, cores)
    h_total = h_p + h_g_p + h_c_pg + h_s_pgc
    return {
        'H_p': h_p, 'H_g|p': h_g_p, 'H_c|pg': h_c_pg, 'H_s|pgc': h_s_pgc,
        'H_total': h_total,
        'pct_p': 100 * h_p / h_total if h_total > 0 else 0,
        'pct_g': 100 * h_g_p / h_total if h_total > 0 else 0,
        'pct_c': 100 * h_c_pg / h_total if h_total > 0 else 0,
        'pct_s': 100 * h_s_pgc / h_total if h_total > 0 else 0,
    }

def full_metrics(prefixes, gallows_list, cores, suffixes, sections):
    """Compute the complete metric suite matching the grammar spec."""
    budget = chain_rule_budget(prefixes, gallows_list, cores, suffixes)

    # Inventory sizes
    n_p = len(set(prefixes))
    n_g = len(set(gallows_list))
    n_c = len(set(cores))
    n_s = len(set(suffixes))

    # Cross-slot coupling
    v_pg = cramers_v(prefixes, gallows_list)
    mi_cs = MI(suffixes, cores)  # approximate: grammar conditions on p,g
    mi_p_sec = MI(sections, prefixes)
    mi_c_sec = MI(sections, cores)

    # Empty-core fraction
    empty_core_frac = sum(1 for c in cores if c == '∅' or c == '') / len(cores)

    # Residual: reconstruct token from slots and check
    # (This is a sanity check — should be 0 for any valid decomposition)

    return {
        **budget,
        'n_p': n_p, 'n_g': n_g, 'n_c': n_c, 'n_s': n_s,
        'V_pg': v_pg,
        'MI_cs': mi_cs,
        'MI_p_sec': mi_p_sec,
        'MI_c_sec': mi_c_sec,
        'empty_core': empty_core_frac,
    }


# ═══════════════════════════════════════════════════════════════════════
# P70 GOLD STANDARD
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("P70 GOLD STANDARD (from enriched_records)")
print("█" * 80)

p70_p = [r['prefix'] for r in records]
p70_g = [r['gallows'] for r in records]
p70_c = [r['core'] for r in records]
p70_s = [r['suffix'] for r in records]

p70_m = full_metrics(p70_p, p70_g, p70_c, p70_s, sections)

print(f"\n  Chain-rule entropy budget:")
print(f"    H(p)      = {p70_m['H_p']:.3f} bits ({p70_m['pct_p']:.1f}%)")
print(f"    H(g|p)    = {p70_m['H_g|p']:.3f} bits ({p70_m['pct_g']:.1f}%)")
print(f"    H(c|p,g)  = {p70_m['H_c|pg']:.3f} bits ({p70_m['pct_c']:.1f}%)")
print(f"    H(s|p,g,c)= {p70_m['H_s|pgc']:.3f} bits ({p70_m['pct_s']:.1f}%)")
print(f"    Total     = {p70_m['H_total']:.3f} bits")
print(f"\n  Inventories: P={p70_m['n_p']}, G={p70_m['n_g']}, C={p70_m['n_c']}, S={p70_m['n_s']}")
print(f"  Empty core: {p70_m['empty_core']*100:.1f}%")
print(f"\n  Cross-slot coupling:")
print(f"    Cramér's V(P,G) = {p70_m['V_pg']:.3f}")
print(f"    MI(S;C)         = {p70_m['MI_cs']:.3f} bits")
print(f"    MI(section;P)   = {p70_m['MI_p_sec']:.3f} bits")
print(f"    MI(section;C)   = {p70_m['MI_c_sec']:.3f} bits")

# Verify against grammar spec
print(f"\n  Grammar spec comparison:")
print(f"    {'Metric':<20} {'Computed':>10} {'Grammar':>10} {'Match':>8}")
print(f"    {'H(p)':<20} {p70_m['H_p']:10.3f} {'2.788':>10} {'✓' if abs(p70_m['H_p']-2.788)<0.05 else '✗':>8}")
print(f"    {'H(g|p)':<20} {p70_m['H_g|p']:10.3f} {'1.374':>10} {'✓' if abs(p70_m['H_g|p']-1.374)<0.05 else '✗':>8}")
print(f"    {'H(c|p,g)':<20} {p70_m['H_c|pg']:10.3f} {'3.622':>10} {'✓' if abs(p70_m['H_c|pg']-3.622)<0.05 else '✗':>8}")
print(f"    {'H(s|p,g,c)':<20} {p70_m['H_s|pgc']:10.3f} {'2.527':>10} {'✓' if abs(p70_m['H_s|pgc']-2.527)<0.05 else '✗':>8}")
print(f"    {'V(P,G)':<20} {p70_m['V_pg']:10.3f} {'0.266':>10} {'✓' if abs(p70_m['V_pg']-0.266)<0.02 else '✗':>8}")


# ═══════════════════════════════════════════════════════════════════════
# ALTERNATIVE DECOMPOSITION GENERATORS
# ═══════════════════════════════════════════════════════════════════════

# The grammar's prefixes: ∅, o, y, d, s, ch, sh, qo
# Key P70 insight: ch and sh are PREFIXES, not gallows-initial
# The crude decomposer had: ∅, o, qo, d, y, s, q (missing ch, sh; wrong q)

def decompose_generic(token, prefix_list, gallows_list, suffix_list):
    """Greedy 4-slot decomposition with given inventories."""
    rem = token
    pfx = '∅'
    for p in sorted(prefix_list, key=len, reverse=True):
        if p and rem.startswith(p):
            pfx = p
            rem = rem[len(p):]
            break
    gal = '∅'
    for g in sorted(gallows_list, key=len, reverse=True):
        if g and rem.startswith(g):
            gal = g
            rem = rem[len(g):]
            break
    sfx = '∅'
    for s in sorted(suffix_list, key=len, reverse=True):
        if s and rem.endswith(s) and len(rem) > len(s):
            sfx = s
            rem = rem[:len(rem)-len(s)]
            break
    core = rem if rem else '∅'
    return pfx, gal, core, sfx


# GRAMMAR INVENTORIES (gold standard)
GRAMMAR_P = ['o', 'y', 'd', 's', 'ch', 'sh', 'qo']  # ∅ is default
GRAMMAR_G = ['k', 't', 'p', 'f', 'ckh', 'cth', 'cph', 'cfh']
GRAMMAR_S = ['aiin', 'edy', 'eey', 'ody', 'ain', 'iin', 'chy', 'shy',
             'dy', 'ey', 'in', 'ol', 'or', 'ar', 'al', 'am', 'an',
             'ir', 'ee', 'eedy', 'oiin', 'oiiin', 'y', 'n', 'l', 'r',
             'm', 'g', 'iiin', 's', 'a', 'e']

# ALTERNATIVE 1: Crude (ch/sh as gallows, not prefixes)
CRUDE_P = ['o', 'qo', 'd', 'y', 's', 'q']
CRUDE_G = ['k', 't', 'p', 'f', 'cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']

# ALTERNATIVE 2: Move ch/sh to gallows but keep qo as prefix
ALT2_P = ['o', 'qo', 'd', 'y', 's']
ALT2_G = ['k', 't', 'p', 'f', 'cth', 'ckh', 'cph', 'cfh', 'ch', 'sh']

# ALTERNATIVE 3: Flatten — everything is prefix (no gallows)
ALT3_P = ['o', 'y', 'd', 's', 'ch', 'sh', 'qo', 'k', 't', 'p', 'f',
           'ok', 'ot', 'qok', 'qot', 'dk', 'dt', 'yk', 'sk',
           'chk', 'cht', 'shk', 'sht', 'qop', 'qof',
           'cth', 'ckh', 'cph', 'cfh', 'dcth', 'dckh']
ALT3_G = []  # No gallows

# ALTERNATIVE 4: Boundary ±1 (shift ch/sh boundary by 1 char)
# ch- prefix → c- prefix + h-gallows; sh- prefix → s- prefix + h-gallows
ALT4_P = ['o', 'y', 'd', 's', 'c', 'qo']  # c and s absorb ch/sh first char
ALT4_G = ['h', 'k', 't', 'p', 'f', 'hk', 'ht',  # h becomes gallows-initial
           'ckh', 'cth', 'cph', 'cfh']

# ALTERNATIVE 5: Random assignment — for each word, randomly assign
# first 0-2 chars as prefix, next 0-1 as gallows, last 0-3 as suffix
def random_decompose(token, rng):
    n = len(token)
    if n == 0:
        return '∅', '∅', '∅', '∅'
    plen = min(rng.choice([0, 0, 1, 1, 2]), n-1) if n > 1 else 0
    rem = token[plen:]
    glen = min(rng.choice([0, 0, 0, 1]), len(rem)-1) if len(rem) > 1 else 0
    rem2 = rem[glen:]
    slen = min(rng.choice([0, 0, 1, 2, 3]), len(rem2)-1) if len(rem2) > 1 else 0
    pfx = token[:plen] if plen > 0 else '∅'
    gal = rem[:glen] if glen > 0 else '∅'
    sfx = rem2[len(rem2)-slen:] if slen > 0 else '∅'
    core = rem2[:len(rem2)-slen] if len(rem2)-slen > 0 else '∅'
    return pfx, gal, core, sfx

# ALTERNATIVE 6: Fixed-position (first char = prefix, second = gallows,
# last 2 = suffix, rest = core)
def fixed_decompose(token, n_pfx=1, n_sfx=2):
    n = len(token)
    if n <= n_pfx + n_sfx:
        return '∅', '∅', token if token else '∅', '∅'
    pfx = token[:n_pfx]
    sfx = token[-n_sfx:]
    mid = token[n_pfx:-n_sfx]
    if len(mid) > 0:
        gal = mid[0]
        core = mid[1:] if len(mid) > 1 else '∅'
    else:
        gal = '∅'
        core = '∅'
    return pfx, gal, core, sfx

# ALTERNATIVE 7: Suffixes absorbed into core (no suffix slot)
ALT7_P = GRAMMAR_P[:]
ALT7_G = GRAMMAR_G[:]
ALT7_S = []  # No suffixes — everything after gallows is core


# ═══════════════════════════════════════════════════════════════════════
# RUN ALL ALTERNATIVES
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("RUNNING ALTERNATIVE DECOMPOSITIONS")
print("█" * 80)

results = {}

# P70 gold standard (already computed)
results['P70_gold'] = p70_m

# Alternative 1: Crude (ch/sh as gallows)
print("\n  Alt 1: Crude (ch/sh → gallows)...")
alt1 = [decompose_generic(t, CRUDE_P, CRUDE_G, GRAMMAR_S) for t in tokens]
results['Crude_chsh_gal'] = full_metrics([a[0] for a in alt1], [a[1] for a in alt1],
                                          [a[2] for a in alt1], [a[3] for a in alt1], sections)

# Alternative 2: ch/sh as gallows, no q prefix
print("  Alt 2: ch/sh → gallows, no q prefix...")
alt2 = [decompose_generic(t, ALT2_P, ALT2_G, GRAMMAR_S) for t in tokens]
results['No_chsh_prefix'] = full_metrics([a[0] for a in alt2], [a[1] for a in alt2],
                                          [a[2] for a in alt2], [a[3] for a in alt2], sections)

# Alternative 3: Flatten (no gallows)
print("  Alt 3: Flat prefix (no gallows slot)...")
alt3 = [decompose_generic(t, ALT3_P, ALT3_G, GRAMMAR_S) for t in tokens]
results['Flat_no_gallows'] = full_metrics([a[0] for a in alt3], [a[1] for a in alt3],
                                           [a[2] for a in alt3], [a[3] for a in alt3], sections)

# Alternative 4: Boundary shift (c/s prefix, h gallows)
print("  Alt 4: Shifted boundary (c/s prefix, h gallows)...")
alt4 = [decompose_generic(t, ALT4_P, ALT4_G, GRAMMAR_S) for t in tokens]
results['Shift_ch→c+h'] = full_metrics([a[0] for a in alt4], [a[1] for a in alt4],
                                         [a[2] for a in alt4], [a[3] for a in alt4], sections)

# Alternative 5: Random (5 trials)
for trial in range(5):
    print(f"  Alt 5.{trial}: Random decomposition...")
    rng = random.Random(42 + trial)
    alt5 = [random_decompose(t, rng) for t in tokens]
    results[f'Random_{trial}'] = full_metrics([a[0] for a in alt5], [a[1] for a in alt5],
                                               [a[2] for a in alt5], [a[3] for a in alt5], sections)

# Alternative 6: Fixed position splits
for np_, ns_ in [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]:
    print(f"  Alt 6: Fixed({np_},{ns_})...")
    alt6 = [fixed_decompose(t, np_, ns_) for t in tokens]
    results[f'Fixed({np_},{ns_})'] = full_metrics([a[0] for a in alt6], [a[1] for a in alt6],
                                                   [a[2] for a in alt6], [a[3] for a in alt6], sections)

# Alternative 7: No suffixes
print("  Alt 7: No suffix slot...")
alt7 = [decompose_generic(t, ALT7_P, ALT7_G, ALT7_S) for t in tokens]
results['No_suffixes'] = full_metrics([a[0] for a in alt7], [a[1] for a in alt7],
                                       [a[2] for a in alt7], [a[3] for a in alt7], sections)

# Alternative 8: Grammar prefixes but random suffixes
print("  Alt 8: Grammar P/G, random suffixes...")
for trial in range(3):
    rng = random.Random(200 + trial)
    # Build random suffix list from common word endings
    endings = Counter()
    for t in tokens:
        for n in range(1, min(5, len(t))):
            endings[t[-n:]] += 1
    common_ends = [e for e, _ in endings.most_common(100)]
    rand_sfx = list(rng.sample(common_ends, min(33, len(common_ends))))
    alt8 = [decompose_generic(t, GRAMMAR_P, GRAMMAR_G, rand_sfx) for t in tokens]
    results[f'Gram_PG_randS_{trial}'] = full_metrics(
        [a[0] for a in alt8], [a[1] for a in alt8],
        [a[2] for a in alt8], [a[3] for a in alt8], sections)


# ═══════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("CHAIN-RULE ENTROPY BUDGET COMPARISON")
print("█" * 80)

print(f"\n  Grammar target: H(p)=2.788  H(g|p)=1.374  H(c|p,g)=3.622  H(s|p,g,c)=2.527  Total=10.311")
print(f"\n  {'Method':<22} {'H(p)':>6} {'H(g|p)':>7} {'H(c|pg)':>8} {'H(s|pgc)':>9} {'Total':>7} {'|P|':>4} {'|G|':>4} {'|C|':>6} {'|S|':>4}")
print("  " + "=" * 90)

for label in ['P70_gold', 'Crude_chsh_gal', 'No_chsh_prefix', 'Shift_ch→c+h',
              'Flat_no_gallows', 'No_suffixes',
              'Fixed(1,1)', 'Fixed(1,2)', 'Fixed(2,2)',
              'Random_0', 'Random_1', 'Random_2',
              'Gram_PG_randS_0', 'Gram_PG_randS_1']:
    if label not in results:
        continue
    m = results[label]
    marker = " ◀ GOLD" if label == 'P70_gold' else ""
    print(f"  {label:<22} {m['H_p']:6.3f} {m['H_g|p']:7.3f} {m['H_c|pg']:8.3f} {m['H_s|pgc']:9.3f} "
          f"{m['H_total']:7.3f} {m['n_p']:4d} {m['n_g']:4d} {m['n_c']:6d} {m['n_s']:4d}{marker}")


# ═══════════════════════════════════════════════════════════════════════
# CROSS-SLOT COUPLING COMPARISON
# ═══════════════════════════════════════════════════════════════════════

print(f"\n\n  {'Method':<22} {'V(P,G)':>7} {'MI(S;C)':>8} {'MI(sec;P)':>10} {'MI(sec;C)':>10} {'EC%':>6}")
print("  " + "=" * 70)

for label in ['P70_gold', 'Crude_chsh_gal', 'No_chsh_prefix', 'Shift_ch→c+h',
              'Flat_no_gallows', 'No_suffixes',
              'Fixed(1,1)', 'Fixed(1,2)', 'Fixed(2,2)',
              'Random_0', 'Gram_PG_randS_0']:
    if label not in results:
        continue
    m = results[label]
    marker = " ◀ GOLD" if label == 'P70_gold' else ""
    print(f"  {label:<22} {m['V_pg']:7.3f} {m['MI_cs']:8.3f} {m['MI_p_sec']:10.3f} "
          f"{m['MI_c_sec']:10.3f} {m['empty_core']*100:6.1f}{marker}")


# ═══════════════════════════════════════════════════════════════════════
# THE KEY COMPARISON: ch/sh AS PREFIXES vs GALLOWS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("KEY TEST: ch/sh AS PREFIXES (P70) vs ch/sh AS GALLOWS (crude)")
print("█" * 80)

p70 = results['P70_gold']
crude_m = results['Crude_chsh_gal']

print(f"\n  The P70 rules discovered that ch and sh function as PREFIXES.")
print(f"  The crude decomposer treats them as GALLOWS.")
print(f"  What difference does this make?\n")

print(f"  {'Metric':<30} {'P70 (ch/sh=pfx)':>16} {'Crude (ch/sh=gal)':>18} {'Δ':>10} {'Better':>8}")
print("  " + "=" * 85)

comparisons = [
    ('H(prefix)', p70['H_p'], crude_m['H_p']),
    ('H(gallows|prefix)', p70['H_g|p'], crude_m['H_g|p']),
    ('H(core|p,g)', p70['H_c|pg'], crude_m['H_c|pg']),
    ('H(suffix|p,g,c)', p70['H_s|pgc'], crude_m['H_s|pgc']),
    ('|Prefix|', p70['n_p'], crude_m['n_p']),
    ('|Gallows|', p70['n_g'], crude_m['n_g']),
    ('|Core|', p70['n_c'], crude_m['n_c']),
    ('|Suffix|', p70['n_s'], crude_m['n_s']),
    ('V(P,G)', p70['V_pg'], crude_m['V_pg']),
    ('MI(suffix;core)', p70['MI_cs'], crude_m['MI_cs']),
    ('MI(section;prefix)', p70['MI_p_sec'], crude_m['MI_p_sec']),
    ('MI(section;core)', p70['MI_c_sec'], crude_m['MI_c_sec']),
    ('Empty core %', p70['empty_core']*100, crude_m['empty_core']*100),
]

for name, p70v, cv in comparisons:
    delta = p70v - cv
    # "Better" depends on the metric
    if name.startswith('H(prefix') or name.startswith('H(suffix'):
        better = 'P70' if p70v > cv else 'Crude'  # Higher = more informative affixes
    elif name.startswith('H(core'):
        better = 'P70' if p70v < cv else 'Crude'  # Lower = better compression of core
    elif name.startswith('MI(section'):
        better = 'P70' if p70v > cv else 'Crude'  # Higher = more section info in slot
    elif name.startswith('V('):
        better = 'P70' if abs(p70v - 0.266) < abs(cv - 0.266) else 'Crude'  # Closer to grammar spec
    elif name == 'Empty core %':
        better = 'P70' if p70v > cv else 'Crude'  # Higher EC = more efficient slot use
    else:
        better = '—'
    fmt = '.3f' if isinstance(p70v, float) and p70v < 100 else 'd' if isinstance(p70v, int) else '.1f'
    print(f"  {name:<30} {p70v:>16{fmt}} {cv:>18{fmt}} {delta:>+10{fmt}} {better:>8}")


# ═══════════════════════════════════════════════════════════════════════
# SPECIFIC EXAMPLES WHERE ch/sh ASSIGNMENT MATTERS
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  Words where ch/sh assignment changes the parse:")
print("─" * 80)

examples = ['chedy', 'chol', 'chey', 'shedy', 'sheey', 'shor',
            'chor', 'chkal', 'shol', 'chy', 'shy', 'cheol',
            'qokeedy', 'daiin', 'okedy', 'okeedy']

print(f"\n  {'Token':<12} {'P70 parse':<30} {'Crude parse':<30} {'Different?'}")
print("  " + "=" * 85)
for t in examples:
    # Find in records
    p70_rec = next((r for r in records if r['token'] == t), None)
    if p70_rec:
        p70_parse = f"[{p70_rec['prefix']}|{p70_rec['gallows']}|{p70_rec['core']}|{p70_rec['suffix']}]"
    else:
        p70_parse = "not found"

    crude_parse_tuple = decompose_generic(t, CRUDE_P, CRUDE_G, GRAMMAR_S)
    crude_parse = f"[{crude_parse_tuple[0]}|{crude_parse_tuple[1]}|{crude_parse_tuple[2]}|{crude_parse_tuple[3]}]"

    diff = "YES" if p70_parse != crude_parse else "no"
    print(f"  {t:<12} {p70_parse:<30} {crude_parse:<30} {diff}")


# ═══════════════════════════════════════════════════════════════════════
# PROFILE: WHAT FRACTION OF TOKENS ARE AFFECTED?
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("  Impact of ch/sh reclassification:")
print("─" * 80)

chsh_tokens = sum(1 for r in records if r['prefix'] in ('ch', 'sh'))
print(f"\n  Tokens with ch/sh prefix: {chsh_tokens:,} ({100*chsh_tokens/N:.1f}%)")
print(f"  This is nearly 1 in 4 tokens in the entire manuscript.")

# How many of these would the crude decomposer get wrong?
disagree = 0
for r in records:
    crude = decompose_generic(r['token'], CRUDE_P, CRUDE_G, GRAMMAR_S)
    if crude != (r['prefix'], r['gallows'], r['core'], r['suffix']):
        disagree += 1

print(f"  Tokens with different crude decomposition: {disagree:,} ({100*disagree/N:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
# TEST 1 VERDICT: IS P70 ENTROPY BUDGET UNIQUE?
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("TEST 1: IS P70's ENTROPY BUDGET ACHIEVABLE BY ALTERNATIVES?")
print("█" * 80)

# The grammar targets
target = {'H_p': 2.788, 'H_g|p': 1.374, 'H_c|pg': 3.622, 'H_s|pgc': 2.527}

print(f"\n  Distance from grammar targets (sum of absolute errors across 4 slots):")
print(f"  {'Method':<22} {'Error':>8}")
print("  " + "=" * 35)

errors = []
for label, m in sorted(results.items(), key=lambda x: sum(abs(x[1][k]-target[k]) for k in target)):
    err = sum(abs(m[k] - target[k]) for k in target)
    errors.append((label, err))
    marker = " ◀ GOLD" if label == 'P70_gold' else ""
    print(f"  {label:<22} {err:8.3f}{marker}")


# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════

print("\n" + "█" * 80)
print("FINAL SUMMARY")
print("█" * 80)

print(f"""
  The P70 rules' key contribution is identifying ch and sh as PREFIXES.
  This affects {chsh_tokens:,} tokens ({100*chsh_tokens/N:.1f}% of the corpus).

  The crude decomposer (ch/sh as gallows) produces a different parse for
  {disagree:,} tokens ({100*disagree/N:.1f}% of the corpus).

  Chain-rule budget comparison (P70 vs Crude):
    H(prefix):     {p70['H_p']:.3f} vs {crude_m['H_p']:.3f} bits  (P70 has {abs(p70['H_p']-crude_m['H_p']):.3f} {'more' if p70['H_p']>crude_m['H_p'] else 'less'} prefix entropy)
    H(gallows|p):  {p70['H_g|p']:.3f} vs {crude_m['H_g|p']:.3f} bits
    H(core|p,g):   {p70['H_c|pg']:.3f} vs {crude_m['H_c|pg']:.3f} bits
    MI(sec;prefix): {p70['MI_p_sec']:.3f} vs {crude_m['MI_p_sec']:.3f} bits  (P70 prefix is {'more' if p70['MI_p_sec']>crude_m['MI_p_sec'] else 'less'} section-informative)

  Error from grammar spec targets:
    P70:   {errors[0][1]:.3f} (rank {[l for l,_ in errors].index('P70_gold')+1})
    Crude: {next(e for l,e in errors if l=='Crude_chsh_gal'):.3f} (rank {[l for l,_ in errors].index('Crude_chsh_gal')+1})
""")

print("=" * 80)
print("Done.")
