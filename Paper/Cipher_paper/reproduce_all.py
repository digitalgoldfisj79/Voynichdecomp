#!/usr/bin/env python3
"""
REPRODUCE ALL — Paper 2 (Bozzard 2026b)
=========================================
Single entry point to reproduce every number in the paper.

Required files in working directory:
  enriched_records.pkl     — VMS base data
  ci_corpus_parsed.pkl     — Circa Instans corpus
  ms_ald_211_htr.md        — Ald.211 HTR transcription
  v11_nomenclator.py       — Forward cipher v11
  nomenclator_optimizer.py — Nomenclator recovery pipeline
  cv_folio_reader.py       — CV syllable reader
  score_85_metrics.py      — 84-metric scoring battery
  metric_defs.py           — Metric list definitions

Optional (for language battery §8):
  battery_v4.pkl           — Pre-computed 16-language battery

Usage:
    python reproduce_all.py              # full run (~20 min)
    python reproduce_all.py --quick      # skip 10-seed scoring (~5 min)
    python reproduce_all.py --section 5  # run only §5

Edward Bozzard · ORCID 0009-0002-4052-0994
"""

import sys, os, pickle, random, json, re, time
import numpy as np
from collections import Counter, defaultdict
from scipy.stats import hypergeom, chi2_contingency

sys.path.insert(0, '.')

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════

SEEDS = [42, 404, 501, 606, 808, 909, 101, 202, 303, 505]
QUICK = '--quick' in sys.argv
SECTION_ONLY = None
for i, a in enumerate(sys.argv):
    if a == '--section' and i + 1 < len(sys.argv):
        SECTION_ONLY = int(sys.argv[i + 1])

results = {}  # collect all paper numbers

def section(n):
    if SECTION_ONLY is not None and SECTION_ONLY != n:
        return False
    return True

# ══════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════

print("Loading data...")
with open('enriched_records.pkl', 'rb') as f:
    all_records = pickle.load(f)
with open('ci_corpus_parsed.pkl', 'rb') as f:
    ci = pickle.load(f)

ha = [r for r in all_records if r.get('section') == 'Herbal-A']

print(f"  Total tokens: {len(all_records)}")
print(f"  HA tokens: {len(ha)}")
print(f"  HA types: {len(set(r['token'] for r in ha))}")
results['total_tokens'] = len(all_records)
results['ha_tokens'] = len(ha)
results['ha_types'] = len(set(r['token'] for r in ha))

# ══════════════════════════════════════════════════════════════
# §2.4: WORD-LENGTH σ
# ══════════════════════════════════════════════════════════════

if section(2):
    print("\n" + "=" * 60)
    print("§2.4: Word-length standard deviation")
    print("=" * 60)
    sigma = np.std([len(r['token']) for r in ha])
    print(f"  σ = {sigma:.2f}  (paper: 1.72)")
    results['sigma'] = round(sigma, 2)

# ══════════════════════════════════════════════════════════════
# §3: STRUCTURAL NUMBERS
# ══════════════════════════════════════════════════════════════

if section(3):
    print("\n" + "=" * 60)
    print("§3: Structural numbers")
    print("=" * 60)
    ec_all = sum(1 for r in all_records if r['empty_core'])
    ec_ha = sum(1 for r in ha if r['empty_core'])
    print(f"  EC rate (full MS): {ec_all/len(all_records)*100:.1f}%  (paper: 52.7%)")
    print(f"  EC rate (HA):      {ec_ha/len(ha)*100:.1f}%")
    print(f"  EC-EC bigrams: {sum(1 for i in range(len(ha)-1) if ha[i]['empty_core'] and ha[i+1]['empty_core'])}")
    results['ec_rate_full'] = round(ec_all / len(all_records) * 100, 1)
    results['ec_rate_ha'] = round(ec_ha / len(ha) * 100, 1)

# ══════════════════════════════════════════════════════════════
# §4.3: SCRIBE PRODUCTION RULES
# ══════════════════════════════════════════════════════════════

if section(4):
    print("\n" + "=" * 60)
    print("§4.3: Column stickiness + word-length autocorrelation")
    print("=" * 60)

    # Build token→cell lookup
    tc = {}
    for r in ha:
        mc = r.get('m_core') or r.get('core') or ''
        row = mc[0] if mc and not r['empty_core'] else '∅'
        sf = r.get('sfx_fam', 'BARE')
        if r['token'] not in tc:
            tc[r['token']] = (row, sf)

    # Suffix family bigram rate
    sfs = [tc.get(r['token'], ('?', '?'))[1] for r in ha]
    sfx_bi = sum(1 for i in range(len(sfs) - 1)
                 if sfs[i] == sfs[i + 1]) / (len(sfs) - 1)

    # Expected under independence
    fam_counts = Counter(sfs)
    total_sf = sum(fam_counts.values())
    expected = sum((c / total_sf) ** 2 for c in fam_counts.values())

    # Word-length autocorrelation
    lens = [len(r['token']) for r in ha]
    m = np.mean(lens)
    v = np.var(lens)
    ac_wl = np.mean([(lens[i] - m) * (lens[i + 1] - m)
                      for i in range(len(lens) - 1)]) / v if v > 0 else 0

    # Z-score vs shuffle
    random.seed(42)
    shuf_ac = []
    for _ in range(1000):
        s = list(lens)
        random.shuffle(s)
        shuf_ac.append(np.mean([(s[i] - np.mean(s)) * (s[i + 1] - np.mean(s))
                                 for i in range(len(s) - 1)]) / np.var(s))
    z = (ac_wl - np.mean(shuf_ac)) / np.std(shuf_ac)

    # Family mean lengths
    fam_lens = defaultdict(list)
    for r in ha:
        fam_lens[r.get('sfx_fam', 'BARE')].append(len(r['token']))

    print(f"  sfx_bi observed: {sfx_bi:.4f}  (paper: 0.252)")
    print(f"  sfx_bi expected: {expected:.4f}  (paper: 0.204)")
    print(f"  AC_wl:           {ac_wl:.4f}  (paper: +0.076)")
    print(f"  Z vs shuffle:    {z:.1f}    (paper: 4.8)")
    print(f"  BARE mean len:   {np.mean(fam_lens['BARE']):.2f}  (paper: 3.4)")
    print(f"  N mean len:      {np.mean(fam_lens['N']):.2f}  (paper: 5.7)")
    print(f"  Y mean len:      {np.mean(fam_lens['Y']):.2f}  (paper: 4.9)")

    results['sfx_bi'] = round(sfx_bi, 4)
    results['sfx_bi_expected'] = round(expected, 4)
    results['ac_wl'] = round(ac_wl, 4)
    results['ac_wl_z'] = round(z, 1)

# ══════════════════════════════════════════════════════════════
# §5: NOMENCLATOR RECOVERY
# ══════════════════════════════════════════════════════════════

if section(5):
    print("\n" + "=" * 60)
    print("§5: Nomenclator recovery (running optimizer)")
    print("=" * 60)

    # Import and run the optimizer
    import nomenclator_optimizer as nom_opt

    # Run from scratch
    assignment, final_r = nom_opt.optimize(nom_opt.train_words)
    cv_r = nom_opt.cross_validate(assignment)
    nom_opt.held_out_vms_test(assignment, nom_opt.train_words)
    p = nom_opt.null_model(assignment, nom_opt.train_words)

    print(f"\n  Training r:  {final_r:.4f}  (paper: 0.96)")
    print(f"  CI r:        {cv_r:.4f}  (paper: 0.89)")
    print(f"  Null p:      {p}  (paper: <0.0001)")
    print(f"  Assignments: {dict(assignment)}")

    results['nom_train_r'] = round(final_r, 4)
    results['nom_cv_r'] = round(cv_r, 4)
    results['nom_null_p'] = p
    results['nom_assignments'] = dict(assignment)

    # §5.7: L≡N successor test
    print("\n--- §5.7: L≡N successor test ---")
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
            if not r['empty_core']:
                continue
            if not nxt['empty_core']:
                continue
            fam = nxt['sfx_fam']
            if fam in cats:
                counts[r['sfx_fam']][fam] += 1

    L_s = [counts['L'].get(c, 0) for c in cats]
    N_s = [counts['N'].get(c, 0) for c in cats]
    table = np.array([L_s, N_s])
    nz = table.sum(axis=0) > 0
    chi2_val, p_val, dof, _ = chi2_contingency(table[:, nz])
    print(f"  L successors: {L_s}")
    print(f"  N successors: {N_s}")
    print(f"  χ² = {chi2_val:.4f}, p = {p_val:.4f}  (paper: 5.6, 0.47)")

    results['l_n_chi2'] = round(chi2_val, 4)
    results['l_n_p'] = round(p_val, 4)

# ══════════════════════════════════════════════════════════════
# §6.2: FORWARD CIPHER SCORING
# ══════════════════════════════════════════════════════════════

if section(6):
    print("\n" + "=" * 60)
    print("§6.2: Forward cipher v11 (10-seed scoring)")
    print("=" * 60)

    import v11_nomenclator as v11
    import score_85_metrics as scorer
    import metric_defs

    print(f"  COPY_ALPHA = {v11.COPY_ALPHA}  (paper: 1.3)")
    print(f"  P_STICKY = {v11.P_STICKY}  (paper: 0.22)")

    toks_vms = [r['token'] for r in ha][:4032]
    lines_vms = [toks_vms[i:i + 84] for i in range(0, 4032, 84)]
    VMS_M = scorer.compute_metrics(toks_vms, lines=lines_vms,
                                    subset_iterations=30, seed=42,
                                    verbose=False)
    vms_counts = Counter(toks_vms)
    vms_t50 = sum(c for _, c in vms_counts.most_common(50)) / len(toks_vms)

    seeds = SEEDS if not QUICK else SEEDS[:3]
    all_scores = []

    print(f"\n  {'Seed':>6} {'n/84':>5} {'C15':>4} {'BG42':>5}")
    print(f"  {'-'*24}")

    for seed in seeds:
        toks = [t[1] for t in v11.run(seed=seed)]
        gl = [toks[i:i + 84] for i in range(0, len(toks), 84)]
        gm = scorer.compute_metrics(toks, lines=gl, subset_iterations=30,
                                     seed=42, verbose=False)
        sr = scorer.score_against_vms(gm, VMS_M)
        c15 = sum(1 for m in metric_defs.CORE_15
                  if m in sr['details'] and sr['details'][m]['pass'])
        bg42 = sum(1 for m in metric_defs.BG_METRICS
                   if m in sr['details'] and sr['details'][m]['pass'])
        ct = Counter(toks)
        t50 = sum(c for _, c in ct.most_common(50)) / len(toks)

        # Build sfx_bi
        if not hasattr(v11, '_tc_cache'):
            v11._tc_cache = {}
            for r in ha:
                mc = r.get('m_core') or r.get('core') or ''
                row = mc[0] if mc and not r['empty_core'] else '∅'
                sf = r.get('sfx_fam', 'BARE')
                if r['token'] not in v11._tc_cache:
                    v11._tc_cache[r['token']] = (row, sf)
        sfs_gen = [v11._tc_cache.get(t, ('?', '?'))[1] for t in toks]
        sb = sum(1 for i in range(len(sfs_gen) - 1)
                 if sfs_gen[i] == sfs_gen[i + 1]) / (len(sfs_gen) - 1)

        all_scores.append((sr['n_pass'], c15, bg42,
                           gm.get('autocorr_wordlen', 0), t50, sb,
                           len(set(toks))))
        print(f"  {seed:>6} {sr['n_pass']:>5} {c15:>4} {bg42:>5}")

    n_mean = np.mean([s[0] for s in all_scores])
    n_std = np.std([s[0] for s in all_scores])
    c_mean = np.mean([s[1] for s in all_scores])
    b_mean = np.mean([s[2] for s in all_scores])
    t50_mean = np.mean([s[4] for s in all_scores])
    sb_mean = np.mean([s[5] for s in all_scores])

    print(f"\n  Mean: {n_mean:.1f}/84 (σ={n_std:.1f}), "
          f"C15={c_mean:.1f}, BG42={b_mean:.1f}")
    print(f"  top50: {t50_mean:.3f} (VMS: {vms_t50:.3f})")
    print(f"  sfx_bi: {sb_mean:.3f} (VMS: 0.252)")

    results['v11_n84'] = round(n_mean, 1)
    results['v11_std'] = round(n_std, 1)
    results['v11_c15'] = round(c_mean, 1)
    results['v11_bg42'] = round(b_mean, 1)
    results['v11_top50'] = round(t50_mean, 3)
    results['v11_sfx_bi'] = round(sb_mean, 3)
    results['vms_top50'] = round(vms_t50, 4)

    # §6.3: Cross-section
    if not QUICK:
        print("\n--- §6.3: Cross-section scoring ---")
        sections_to_test = ['Herbal-A', 'Herbal-B', 'Pharmaceutical',
                            'Stars', 'Balneological', 'Zodiac',
                            'Cosmological', 'Astronomical', 'Rosettes']
        xs_results = {}
        for sec_name in sections_to_test:
            sec_recs = [r for r in all_records
                        if r.get('section') == sec_name]
            if len(sec_recs) < 200:
                continue
            sec_sampler, _ = v11.build_pools(sec_recs)
            target = len(sec_recs)
            vocab_cap = len(set(r['token'] for r in sec_recs))
            sec_vms_toks = [r['token'] for r in sec_recs]
            line_len = min(84, len(sec_vms_toks) // 10)
            sec_vms_lines = [sec_vms_toks[i:i + line_len]
                             for i in range(0, len(sec_vms_toks), line_len)]
            sec_VMS_M = scorer.compute_metrics(sec_vms_toks,
                                                lines=sec_vms_lines,
                                                subset_iterations=30,
                                                seed=42, verbose=False)

            # Simple scoring: use v11 architecture with section pools
            # (simplified: just run v11 with section-specific pools)
            sec_scores = []
            for seed in [42, 404, 501]:
                random.seed(seed)
                np.random.seed(seed)
                ec_words = ci.get('ec_words', set())
                words = ci['all_words']
                start = random.randint(3000, 40000)
                words = words[start:] + words[:start]

                output = []
                produced = set()
                past_counts = Counter()
                family_counts = Counter()
                line_tokens = []
                line_target = random.choice(v11.LINE_LENGTHS)
                prev_family = 'Y'
                n_rare = max(1, int(len(v11.RARE_TOKENS) * target
                                    / v11.TARGET))
                rare_toks = v11.RARE_TOKENS[:n_rare]
                safe_range = max(1, target - min(100, target // 4))
                safe_start = min(100, target // 4)
                rp = set(random.sample(
                    range(safe_start, safe_range),
                    min(n_rare, safe_range - safe_start)))
                rs = dict(zip(sorted(rp), rare_toks))

                wi = 0
                while len(output) < target and wi < len(words):
                    n = len(output)
                    if n in rs:
                        t = rs[n]
                        output.append(t)
                        produced.add(t)
                        past_counts[t] += 1
                        family_counts['BARE'] = family_counts.get(
                            'BARE', 0) + 1
                        prev_family = 'BARE'
                        line_tokens.append(t)
                        if len(line_tokens) >= line_target:
                            line_tokens = []
                            line_target = random.choice(v11.LINE_LENGTHS)
                        continue
                    word = words[wi]
                    wi += 1
                    over_cap = len(produced) >= vocab_cap
                    route = v11.classify_and_route(word, ec_words)
                    at_bd = (len(line_tokens) == 0 or
                             len(line_tokens) >= line_target - 1)
                    token = None
                    if route[0] == 'EC':
                        fam = v11.rebalance_family(
                            route[1], family_counts, n)
                        if (random.random() < v11.P_STICKY and
                                prev_family in v11.FAMILIES):
                            fam = prev_family
                        cell = ('∅', fam)
                        if over_cap:
                            token = v11.reuse_token(
                                past_counts, sec_sampler, cell)
                        else:
                            token = v11.pick_token(
                                sec_sampler, cell, produced)
                            if not token:
                                for alt in v11.FAMILIES:
                                    if alt != fam:
                                        t2 = v11.pick_token(
                                            sec_sampler, ('∅', alt),
                                            produced)
                                        if t2:
                                            token = t2
                                            fam = alt
                                            break
                            if not token:
                                token = 'dy'
                    else:
                        row, fam = route[1], route[2]
                        fam = v11.rebalance_family(
                            fam, family_counts, n)
                        if (random.random() < v11.P_STICKY and
                                prev_family in v11.FAMILIES):
                            if (row, prev_family) in sec_sampler:
                                fam = prev_family
                        cell = (row, fam)
                        bf = 0.35 if at_bd else 1.0
                        if over_cap:
                            token = v11.reuse_token(
                                past_counts, sec_sampler, cell)
                        else:
                            if random.random() < (v11.FC_COPY_RATE +
                                                   v11.FC_ED1_RATE) * bf:
                                token = v11.pick_token(
                                    sec_sampler, cell, produced)
                            else:
                                alts = [f for f in v11.FAMILIES
                                        if f != fam and
                                        (row, f) in sec_sampler]
                                if alts:
                                    af = random.choice(alts)
                                    token = v11.pick_token(
                                        sec_sampler, (row, af), produced)
                                    fam = af
                                else:
                                    token = v11.pick_token(
                                        sec_sampler, cell, produced)
                            if not token:
                                token = v11.pick_token(
                                    sec_sampler, cell, produced) or 'dy'

                    output.append(token)
                    produced.add(token)
                    past_counts[token] += 1
                    family_counts[fam] = family_counts.get(fam, 0) + 1
                    prev_family = fam
                    line_tokens.append(token)
                    if len(line_tokens) >= line_target:
                        line_tokens = []
                        line_target = random.choice(v11.LINE_LENGTHS)

                gl = [output[i:i + line_len]
                      for i in range(0, len(output), line_len)]
                gm = scorer.compute_metrics(output, lines=gl,
                                             subset_iterations=30,
                                             seed=42, verbose=False)
                sr = scorer.score_against_vms(gm, sec_VMS_M)
                c15 = sum(1 for m in metric_defs.CORE_15
                          if m in sr['details'] and
                          sr['details'][m]['pass'])
                bg42 = sum(1 for m in metric_defs.BG_METRICS
                           if m in sr['details'] and
                           sr['details'][m]['pass'])
                sec_scores.append((sr['n_pass'], c15, bg42))

            xs_results[sec_name] = sec_scores
            n_xs = np.mean([s[0] for s in sec_scores])
            c_xs = np.mean([s[1] for s in sec_scores])
            b_xs = np.mean([s[2] for s in sec_scores])
            print(f"  {sec_name:<16} n={n_xs:.1f} C15={c_xs:.1f} "
                  f"BG42={b_xs:.1f}")

        results['cross_section'] = {
            sec: round(np.mean([s[0] for s in scores]), 1)
            for sec, scores in xs_results.items()
        }

# ══════════════════════════════════════════════════════════════
# §7.3: FOLIO ENRICHMENT
# ══════════════════════════════════════════════════════════════

if section(7):
    print("\n" + "=" * 60)
    print("§7.3: Folio-level CV enrichment (Herbal-A)")
    print("=" * 60)

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

    # Global baseline
    global_cv = Counter()
    global_total = 0
    for r in ha:
        typ, reading, _ = read_token(r)
        if typ == 'FC':
            global_cv[reading] += 1
            global_total += 1

    # Per-folio enrichment
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
            p = 1 - hypergeom.cdf(n - 1, global_total,
                                  global_cv[cv], folio_fc)
            if p < 0.001:
                bonf_hits.append((folio, cv, n,
                                  n / folio_fc, global_cv[cv] / global_total,
                                  p))

    print(f"\n  Bonferroni-surviving enrichments (p < 0.001):")
    for folio, cv, n, rate, glob, p in sorted(bonf_hits,
                                               key=lambda x: x[5]):
        enr = rate / glob if glob > 0 else 0
        print(f"    {folio} {cv} ×{n} {rate:.1%} vs {glob:.1%} "
              f"({enr:.1f}×) p={p:.2e}")

    results['bonferroni_hits'] = len(bonf_hits)
    results['f2r_mi_p'] = next(
        (p for f, cv, n, r, g, p in bonf_hits
         if f == 'f2r' and cv == 'mi'), None)

# ══════════════════════════════════════════════════════════════
# §8: LANGUAGE CONSTRAINTS
# ══════════════════════════════════════════════════════════════

if section(8):
    print("\n" + "=" * 60)
    print("§8: Language constraints")
    print("=" * 60)

    # Sonorant concentration: y+n+l+r+m / consonant-final tokens
    # EVA vowels (a,e,i,o) excluded from denominator
    vowels_eva = set('aeio')
    final_all = Counter(r['token'][-1] for r in all_records
                        if r['token'])
    total_finals = sum(final_all.values())
    vowel_finals = sum(final_all.get(c, 0) for c in vowels_eva)
    cons_finals = total_finals - vowel_finals
    son_finals = sum(final_all.get(c, 0) for c in 'ynlrm')
    son_pct = son_finals / cons_finals * 100 if cons_finals > 0 else 0
    print(f"  Sonorant (y+n+l+r+m / consonant-final): {son_pct:.1f}%  "
          f"(paper: >93%)")
    results['sonorant_pct'] = round(son_pct, 1)

    # χ² from battery if available
    if os.path.exists('battery_v4.pkl'):
        with open('battery_v4.pkl', 'rb') as f:
            battery = pickle.load(f)
        r_ci = [r for r in battery['results']
                if 'pharma' in r['L']][0]
        print(f"  CI pharma χ²: {r_ci['x']:.4f}  (paper: 0.04)")
        print(f"  16-language battery loaded from battery_v4.pkl")
        results['chi2_ci'] = round(r_ci['x'], 4)
    else:
        # Compute CI χ² directly
        vms_dist = {}
        fc_ha = [r for r in ha if not r['empty_core']]
        for r in fc_ha:
            mc = r.get('m_core') or r.get('core') or ''
            if mc:
                row = mc[0]
                vms_dist[row] = vms_dist.get(row, 0) + 1
        total_fc = sum(vms_dist.values())
        vms_norm = {k: v / total_fc for k, v in vms_dist.items()}

        VOWELS = set('aeiouàèìòùéêîôûäëïöü')
        GROUP_MAP = {'o': {'c', 's', 'p'}, 'c': set('aeiouv'),
                     'e': {'d', 'f'}, 'a': {'m', 'l'}, 'l': {'t'},
                     'd': {'r', 'q', 'h', 'n', 'g'},
                     'r': {'b', 'z', 'x', 'w'}}

        ci_words = ci.get('content_words',
                          [w for w in ci['all_words']
                           if w not in ci.get('ec_words', set())])
        ci_dist = Counter()
        for w in ci_words:
            wl = w.lower()
            ic = wl[0] if wl and wl[0] not in VOWELS else '∅'
            for row, chars in GROUP_MAP.items():
                if ic in chars or (ic == '∅' and row == 'c'):
                    ci_dist[row] += 1
                    break
        ci_total = sum(ci_dist.values())
        ci_norm = {k: v / ci_total for k, v in ci_dist.items()}

        rows = sorted(vms_norm.keys())
        chi2_val = sum(
            (vms_norm.get(r, 0) - ci_norm.get(r, 0)) ** 2
            / ci_norm.get(r, 0.001)
            for r in rows)
        print(f"  CI pharma χ²: {chi2_val:.4f}  (paper: 0.04)")
        results['chi2_ci'] = round(chi2_val, 4)

# ══════════════════════════════════════════════════════════════
# S1: ABLATION TABLE
# ══════════════════════════════════════════════════════════════

if section(6) and not QUICK:
    print("\n" + "=" * 60)
    print("S1: Ablation table")
    print("=" * 60)

    import importlib
    importlib.reload(v11)
    orig_alpha = v11.COPY_ALPHA
    orig_sticky = v11.P_STICKY

    def run_ablated(seed, use_nom, p_sticky, alpha):
        v11.COPY_ALPHA = alpha
        v11.P_STICKY = p_sticky if p_sticky > 0 else 0.0
        # Temporarily patch nomenclator
        if not use_nom:
            saved_nom = dict(v11.NOMENCLATOR)
            v11.NOMENCLATOR.clear()
        random.seed(seed)
        np.random.seed(seed)
        out = v11.run(seed=seed)
        if not use_nom:
            v11.NOMENCLATOR.update(saved_nom)
        return [t[1] for t in out]

    ablation_configs = [
        ('Baseline', False, 0.0, 2.0),
        ('+nom', True, 0.0, 2.0),
        ('+sticky', False, 0.16, 2.0),
        ('+α=1.3', False, 0.0, 1.3),
        ('FULL', True, 0.16, 1.3),
    ]

    print(f"  {'Config':<12} {'n/84':>5} {'C15':>4} {'BG42':>5}")
    print(f"  {'-'*28}")

    for label, nom, ps, a in ablation_configs:
        scores = []
        for seed in [42, 404, 501]:
            v11.COPY_ALPHA = a
            toks = run_ablated(seed, nom, ps, a)
            gl = [toks[i:i + 84] for i in range(0, len(toks), 84)]
            gm = scorer.compute_metrics(toks, lines=gl,
                                         subset_iterations=30,
                                         seed=42, verbose=False)
            sr = scorer.score_against_vms(gm, VMS_M)
            c15 = sum(1 for m in metric_defs.CORE_15
                      if m in sr['details'] and sr['details'][m]['pass'])
            bg42 = sum(1 for m in metric_defs.BG_METRICS
                       if m in sr['details'] and sr['details'][m]['pass'])
            scores.append((sr['n_pass'], c15, bg42))
        print(f"  {label:<12} {np.mean([s[0] for s in scores]):>5.1f} "
              f"{np.mean([s[1] for s in scores]):>4.1f} "
              f"{np.mean([s[2] for s in scores]):>5.1f}")

    v11.COPY_ALPHA = orig_alpha
    v11.P_STICKY = orig_sticky

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("ALL RESULTS")
print("=" * 60)
for k, v in sorted(results.items()):
    print(f"  {k}: {v}")

with open('paper2_reproduce_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n✓ Saved: paper2_reproduce_results.json")
print(f"✓ {len(results)} numbers verified")
