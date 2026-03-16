#!/usr/bin/env python3
"""
REPRODUCE ALL — Paper 2 (Bozzard 2026b) — TERMUX SAFE
=======================================================
Runs on Termux (Android) without scipy for core verification.
Full scoring (§6) still needs scipy via score_85_metrics.py.

Setup (Termux):
    pkg install python
    pip install numpy
    pip install scipy       # needed for §6 scoring only
    
    # Place all files in one directory:
    #   enriched_records.pkl, ci_corpus_parsed.pkl,
    #   ms_ald_211_htr.md, v11_nomenclator.py,
    #   nomenclator_optimizer.py, score_85_metrics.py,
    #   metric_defs.py, cv_folio_reader.py,
    #   battery_v4.pkl (optional)

Usage:
    python reproduce_termux.py                    # all stages
    python reproduce_termux.py --stage 5          # single stage
    python reproduce_termux.py --stage 6 --quick  # 3 seeds not 10

Stages: 2,3,4,5,7,8 need only numpy. Stage 6 needs scipy.
Each stage saves a checkpoint; rerun skips completed stages.

Edward Bozzard · ORCID 0009-0002-4052-0994
"""

import sys, os, pickle, random, json, re, time, math
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════
# PURE-PYTHON SCIPY REPLACEMENTS
# ══════════════════════════════════════════════════════════════

def _log_factorial(n):
    """Log factorial via Stirling for large n, exact for small."""
    if n <= 1:
        return 0.0
    if n <= 20:
        r = 0.0
        for i in range(2, n + 1):
            r += math.log(i)
        return r
    # Stirling
    return n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n)

def _log_comb(n, k):
    if k < 0 or k > n:
        return float('-inf')
    return _log_factorial(n) - _log_factorial(k) - _log_factorial(n - k)

def hypergeom_sf(k, M, n, N):
    """P(X > k) for hypergeometric(M, n, N). Pure Python."""
    # M=population, n=success states, N=draws
    # P(X=i) = C(n,i)*C(M-n,N-i) / C(M,N)
    log_denom = _log_comb(M, N)
    p_greater = 0.0
    lo = max(0, N - (M - n))
    hi = min(n, N)
    for i in range(k + 1, hi + 1):
        log_p = _log_comb(n, i) + _log_comb(M - n, N - i) - log_denom
        p_greater += math.exp(log_p)
    return min(1.0, max(0.0, p_greater))

def chi2_contingency_2xN(table):
    """Chi-squared test for 2×N contingency table. Pure Python.
    Returns (chi2, p, dof).
    table: list of two lists [[row1], [row2]]."""
    r1, r2 = table
    n_cols = len(r1)
    row_totals = [sum(r1), sum(r2)]
    col_totals = [r1[i] + r2[i] for i in range(n_cols)]
    grand = sum(row_totals)
    
    if grand == 0:
        return 0.0, 1.0, 0
    
    chi2 = 0.0
    for i in range(n_cols):
        for j, row in enumerate([r1, r2]):
            expected = row_totals[j] * col_totals[i] / grand
            if expected > 0:
                chi2 += (row[i] - expected) ** 2 / expected
    
    dof = n_cols - 1  # 2 rows, N cols → dof = N-1
    # Chi-squared survival function (upper incomplete gamma)
    p = _chi2_sf(chi2, dof)
    return chi2, p, dof

def _chi2_sf(x, k):
    """Survival function for chi-squared distribution. Approximation."""
    if x <= 0:
        return 1.0
    if k <= 0:
        return 0.0
    # Use Wilson-Hilferty approximation
    z = ((x / k) ** (1.0 / 3) - (1 - 2.0 / (9 * k))) / math.sqrt(2.0 / (9 * k))
    # Standard normal CDF
    p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return max(0.0, min(1.0, 1.0 - p))

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

SEEDS = [42, 404, 501, 606, 808, 909, 101, 202, 303, 505]
QUICK = '--quick' in sys.argv
STAGE_ONLY = None
for i, a in enumerate(sys.argv):
    if a == '--stage' and i + 1 < len(sys.argv):
        STAGE_ONLY = int(sys.argv[i + 1])

CHECKPOINT = 'reproduce_checkpoint.pkl'

def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, 'rb') as f:
            return pickle.load(f)
    return {}

def save_checkpoint(results):
    with open(CHECKPOINT, 'wb') as f:
        pickle.dump(results, f)

def should_run(stage):
    if STAGE_ONLY is not None:
        return STAGE_ONLY == stage
    return True

results = load_checkpoint()

# ══════════════════════════════════════════════════════════════
# LOAD DATA (always needed)
# ══════════════════════════════════════════════════════════════

print("Loading data...")
t0 = time.time()
with open('enriched_records.pkl', 'rb') as f:
    all_records = pickle.load(f)
with open('ci_corpus_parsed.pkl', 'rb') as f:
    ci = pickle.load(f)
ha = [r for r in all_records if r.get('section') == 'Herbal-A']
print(f"  Loaded in {time.time()-t0:.1f}s: {len(all_records)} tokens, "
      f"HA={len(ha)}")

results['total_tokens'] = len(all_records)
results['ha_tokens'] = len(ha)
results['ha_types'] = len(set(r['token'] for r in ha))

# ══════════════════════════════════════════════════════════════
# STAGE 2: σ
# ══════════════════════════════════════════════════════════════

if should_run(2):
    print("\n" + "=" * 50)
    print("STAGE 2: Word-length σ")
    print("=" * 50)
    sigma = float(np.std([len(r['token']) for r in ha]))
    print(f"  σ = {sigma:.2f}  (paper: 1.72)")
    results['sigma'] = round(sigma, 2)
    save_checkpoint(results)

# ══════════════════════════════════════════════════════════════
# STAGE 3: EC/FC
# ══════════════════════════════════════════════════════════════

if should_run(3):
    print("\n" + "=" * 50)
    print("STAGE 3: EC/FC split")
    print("=" * 50)
    ec_all = sum(1 for r in all_records if r['empty_core'])
    ec_ha = sum(1 for r in ha if r['empty_core'])
    print(f"  EC (full MS): {ec_all/len(all_records)*100:.1f}% (paper: 52.7%)")
    print(f"  EC (HA):      {ec_ha/len(ha)*100:.1f}%")
    results['ec_rate_full'] = round(ec_all / len(all_records) * 100, 1)
    results['ec_rate_ha'] = round(ec_ha / len(ha) * 100, 1)
    save_checkpoint(results)

# ══════════════════════════════════════════════════════════════
# STAGE 4: Stickiness + AC_wl (no scipy needed)
# ══════════════════════════════════════════════════════════════

if should_run(4):
    print("\n" + "=" * 50)
    print("STAGE 4: Column stickiness + AC_wl")
    print("=" * 50)
    t0 = time.time()

    tc = {}
    for r in ha:
        mc = r.get('m_core') or r.get('core') or ''
        row = mc[0] if mc and not r['empty_core'] else '∅'
        sf = r.get('sfx_fam', 'BARE')
        if r['token'] not in tc:
            tc[r['token']] = (row, sf)

    sfs = [tc.get(r['token'], ('?', '?'))[1] for r in ha]
    sfx_bi = sum(1 for i in range(len(sfs) - 1)
                 if sfs[i] == sfs[i + 1]) / (len(sfs) - 1)
    fam_counts = Counter(sfs)
    total_sf = sum(fam_counts.values())
    expected = sum((c / total_sf) ** 2 for c in fam_counts.values())

    lens = [len(r['token']) for r in ha]
    m = float(np.mean(lens))
    v = float(np.var(lens))
    ac_wl = float(np.mean([(lens[i] - m) * (lens[i + 1] - m)
                            for i in range(len(lens) - 1)])) / v

    random.seed(42)
    shuf_ac = []
    for _ in range(1000):
        s = list(lens)
        random.shuffle(s)
        sm = float(np.mean(s))
        sv = float(np.var(s))
        shuf_ac.append(float(np.mean(
            [(s[i] - sm) * (s[i + 1] - sm)
             for i in range(len(s) - 1)])) / sv)
    z = (ac_wl - np.mean(shuf_ac)) / np.std(shuf_ac)

    fam_lens = defaultdict(list)
    for r in ha:
        fam_lens[r.get('sfx_fam', 'BARE')].append(len(r['token']))

    print(f"  sfx_bi:   {sfx_bi:.4f} (paper: 0.252)")
    print(f"  expected: {expected:.4f} (paper: 0.204)")
    print(f"  AC_wl:    {ac_wl:.4f} (paper: +0.076)")
    print(f"  Z:        {z:.1f}   (paper: 4.8)")
    print(f"  BARE len: {np.mean(fam_lens['BARE']):.2f} (paper: 3.4)")
    print(f"  N len:    {np.mean(fam_lens['N']):.2f} (paper: 5.7)")
    print(f"  Y len:    {np.mean(fam_lens['Y']):.2f} (paper: 4.9)")
    print(f"  ({time.time()-t0:.1f}s)")

    results['sfx_bi'] = round(sfx_bi, 4)
    results['sfx_bi_expected'] = round(expected, 4)
    results['ac_wl'] = round(ac_wl, 4)
    results['ac_wl_z'] = round(z, 1)
    save_checkpoint(results)

# ══════════════════════════════════════════════════════════════
# STAGE 5: Nomenclator recovery + L≡N (uses pure-Python chi2)
# ══════════════════════════════════════════════════════════════

if should_run(5):
    print("\n" + "=" * 50)
    print("STAGE 5: Nomenclator recovery")
    print("=" * 50)
    t0 = time.time()

    import nomenclator_optimizer as nom_opt
    assignment, final_r = nom_opt.optimize(nom_opt.train_words)
    cv_r = nom_opt.cross_validate(assignment)
    nom_opt.held_out_vms_test(assignment, nom_opt.train_words)
    p = nom_opt.null_model(assignment, nom_opt.train_words)

    print(f"\n  Train r:  {final_r:.4f} (paper: 0.96)")
    print(f"  CI r:     {cv_r:.4f} (paper: 0.89)")
    print(f"  Null p:   {p} (paper: <0.0001)")

    results['nom_train_r'] = round(final_r, 4)
    results['nom_cv_r'] = round(cv_r, 4)
    results['nom_assignments'] = dict(assignment)

    # L≡N (pure Python chi2)
    print("\n  L≡N successor test...")
    lines = defaultdict(list)
    for r in all_records:
        lines[(r['folio'], r['line_no'])].append(r)
    for k in lines:
        lines[k] = sorted(lines[k], key=lambda x: x['pos'])

    cats = ['Y', 'N', 'L', 'R', 'BARE', 'M', 'OTHER']
    counts = {'L': Counter(), 'N': Counter()}
    for line in lines.values():
        for i, r in enumerate(line[:-1]):
            nxt = line[i + 1]
            if r['sfx_fam'] not in {'L', 'N'}:
                continue
            if not r['empty_core'] or not nxt['empty_core']:
                continue
            fam = nxt['sfx_fam']
            if fam in cats:
                counts[r['sfx_fam']][fam] += 1

    L_s = [counts['L'].get(c, 0) for c in cats]
    N_s = [counts['N'].get(c, 0) for c in cats]
    # Remove zero columns
    pairs = [(l, n) for l, n in zip(L_s, N_s) if l + n > 0]
    L_f = [p[0] for p in pairs]
    N_f = [p[1] for p in pairs]
    chi2_val, p_val, dof = chi2_contingency_2xN([L_f, N_f])

    print(f"  χ² = {chi2_val:.4f}, p = {p_val:.4f} (paper: 5.6, 0.47)")
    results['l_n_chi2'] = round(chi2_val, 2)
    results['l_n_p'] = round(p_val, 2)
    print(f"  ({time.time()-t0:.1f}s)")
    save_checkpoint(results)

# ══════════════════════════════════════════════════════════════
# STAGE 6: Forward cipher scoring (NEEDS scipy via scorer)
# ══════════════════════════════════════════════════════════════

if should_run(6):
    print("\n" + "=" * 50)
    print("STAGE 6: Forward cipher v11 scoring")
    print("=" * 50)

    try:
        import score_85_metrics as scorer
        import metric_defs
    except ImportError as e:
        print(f"  SKIPPED — scipy not available: {e}")
        print("  Install: pip install scipy")
        results['v11_note'] = 'skipped (no scipy)'
        save_checkpoint(results)
    else:
        import v11_nomenclator as v11
        print(f"  COPY_ALPHA={v11.COPY_ALPHA} P_STICKY={v11.P_STICKY}")

        toks_vms = [r['token'] for r in ha][:4032]
        lines_vms = [toks_vms[i:i + 84] for i in range(0, 4032, 84)]
        VMS_M = scorer.compute_metrics(toks_vms, lines=lines_vms,
                                        subset_iterations=30, seed=42,
                                        verbose=False)
        seeds = SEEDS if not QUICK else SEEDS[:3]
        all_scores = []

        for seed in seeds:
            t0 = time.time()
            toks = [t[1] for t in v11.run(seed=seed)]
            gl = [toks[i:i + 84] for i in range(0, len(toks), 84)]
            gm = scorer.compute_metrics(toks, lines=gl,
                                         subset_iterations=30,
                                         seed=42, verbose=False)
            sr = scorer.score_against_vms(gm, VMS_M)
            c15 = sum(1 for m in metric_defs.CORE_15
                      if m in sr['details'] and sr['details'][m]['pass'])
            bg42 = sum(1 for m in metric_defs.BG_METRICS
                       if m in sr['details'] and sr['details'][m]['pass'])
            all_scores.append((sr['n_pass'], c15, bg42))
            print(f"  seed {seed}: {sr['n_pass']}/84 "
                  f"C15={c15} BG42={bg42} ({time.time()-t0:.1f}s)")

        n_mean = np.mean([s[0] for s in all_scores])
        n_std = np.std([s[0] for s in all_scores])
        c_mean = np.mean([s[1] for s in all_scores])
        b_mean = np.mean([s[2] for s in all_scores])
        print(f"\n  Mean: {n_mean:.1f}/84 (σ={n_std:.1f}) "
              f"C15={c_mean:.1f} BG42={b_mean:.1f}")

        results['v11_n84'] = round(n_mean, 1)
        results['v11_std'] = round(n_std, 1)
        results['v11_c15'] = round(c_mean, 1)
        results['v11_bg42'] = round(b_mean, 1)
        save_checkpoint(results)

# ══════════════════════════════════════════════════════════════
# STAGE 7: Folio enrichment (pure Python hypergeometric)
# ══════════════════════════════════════════════════════════════

if should_run(7):
    print("\n" + "=" * 50)
    print("STAGE 7: Folio enrichment")
    print("=" * 50)
    t0 = time.time()

    FAM_V = {'Y': 'a', 'R': 'e', 'N': 'i', 'L': 'o',
             'BARE': 'u', 'M': 'u'}
    ROW_CONS = {
        'o': {'c': ['o'], 's': ['od', 'ol', 'ot', 'ok', 'os'],
              'p': ['or', 'octh', 'ockh', 'otch', 'op']},
        'e': {'d': ['e'],
              'f': ['ee', 'eo', 'ek', 'eod', 'es', 'et', 'ees']},
        'a': {'m': ['a', 'ar', 'ai'],
              'l': ['al', 'aii', 'air', 'aiin']},
        'c': {'∅': ['ch', 'che'],
              'v': ['cho', 'chod', 'chos', 'chol']},
        'd': {'r': ['d'], 'n': ['yd', 'da', 'dch']},
        'l': {'t': ['l', 'ld', 'ls']},
        'r': {'b': ['r', 'rch', 'ro']},
    }
    ROW_DEFAULTS = {'o': 'c', 'c': '∅', 'e': 'd', 'a': 'm',
                    'd': 'r', 'l': 't', 'r': 'b'}

    def resolve_cons(row, core):
        if row not in ROW_CONS:
            return '?'
        for cons, cores in ROW_CONS[row].items():
            if core in cores:
                return cons
        return ROW_DEFAULTS.get(row, '?')

    def read_token(r):
        if r['empty_core']:
            return ('EC', 'EC', r.get('sfx_fam', 'BARE'))
        mc = r.get('m_core') or r.get('core') or ''
        row = mc[0] if mc else '?'
        core = r.get('core', '')
        fam = r.get('sfx_fam', 'BARE')
        cons = resolve_cons(row, core)
        vowel = FAM_V.get(fam, '?')
        return ('FC', f"{cons}{vowel}", fam)

    global_cv = Counter()
    global_total = 0
    for r in ha:
        typ, reading, _ = read_token(r)
        if typ == 'FC':
            global_cv[reading] += 1
            global_total += 1

    folio_order = []
    seen = set()
    for r in ha:
        f = r['folio']
        if f not in seen:
            folio_order.append(f)
            seen.add(f)

    bonf_hits = []
    for folio in folio_order:
        folio_recs = [r for r in ha if r['folio'] == folio]
        folio_cv = Counter()
        folio_fc = 0
        for r in folio_recs:
            typ, reading, _ = read_token(r)
            if typ == 'FC':
                folio_cv[reading] += 1
                folio_fc += 1
        if folio_fc < 5:
            continue
        for cv, n in folio_cv.most_common():
            if n < 2 or cv[0] == '?':
                continue
            # P(X >= n) using pure-Python hypergeometric
            p = hypergeom_sf(n - 1, global_total,
                             global_cv[cv], folio_fc)
            if p < 0.001:
                enr = (n / folio_fc) / (global_cv[cv] / global_total)
                bonf_hits.append((folio, cv, n, p, enr))

    print(f"  Bonferroni hits (p < 0.001):")
    for folio, cv, n, p, enr in sorted(bonf_hits, key=lambda x: x[3]):
        print(f"    {folio} {cv} ×{n} ({enr:.1f}×) p={p:.2e}")

    results['bonferroni_hits'] = len(bonf_hits)
    results['f2r_mi_p'] = next(
        (p for f, cv, n, p, e in bonf_hits if f == 'f2r' and cv == 'mi'),
        None)
    print(f"  ({time.time()-t0:.1f}s)")
    save_checkpoint(results)

# ══════════════════════════════════════════════════════════════
# STAGE 8: Language constraints (no scipy needed)
# ══════════════════════════════════════════════════════════════

if should_run(8):
    print("\n" + "=" * 50)
    print("STAGE 8: Language constraints")
    print("=" * 50)

    vowels_eva = set('aeio')
    final_all = Counter(r['token'][-1] for r in all_records
                        if r['token'])
    total_f = sum(final_all.values())
    vowel_f = sum(final_all.get(c, 0) for c in vowels_eva)
    cons_f = total_f - vowel_f
    son_f = sum(final_all.get(c, 0) for c in 'ynlrm')
    son_pct = son_f / cons_f * 100
    print(f"  Sonorant: {son_pct:.1f}% (paper: >93%)")
    results['sonorant_pct'] = round(son_pct, 1)

    if os.path.exists('battery_v4.pkl'):
        with open('battery_v4.pkl', 'rb') as f:
            battery = pickle.load(f)
        r_ci = [r for r in battery['results'] if 'pharma' in r['L']][0]
        print(f"  CI χ²: {r_ci['x']:.4f} (paper: 0.04)")
        results['chi2_ci'] = round(r_ci['x'], 4)
    else:
        print("  battery_v4.pkl not found — CI χ² not verified")
    save_checkpoint(results)

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
for k, v in sorted(results.items()):
    print(f"  {k}: {v}")

with open('paper2_reproduce_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ {len(results)} values saved to paper2_reproduce_results.json")
save_checkpoint(results)
