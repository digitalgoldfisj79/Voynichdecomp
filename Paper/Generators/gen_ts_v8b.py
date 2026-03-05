#!/usr/bin/env python3
"""
Gen-TS v8b: Token-Length Coupling (AC1 isolation test)

SINGLE CHANGE from v7b (49/84, AC1=0.001):
  Replace suffix-length persistence with FULL TOKEN-LENGTH coupling.
  Track prev_wordlen (whole token, not just suffix). When selecting
  suffix variant, apply exp(-|candidate_len - prev_wordlen| / lambda)
  weighting to all candidates.

  DO NOT reset prev_wordlen at line boundaries (VMS AC1 is measured
  across the whole stream including line breaks).

  Suffix variant selection is FREQUENCY-WEIGHTED (same as v7b).
  No variant exhaustion. This isolates the AC1 effect cleanly.

  Tuning knobs:
    lambda_len = length coupling decay (default 2.0)
    len_coupling_prob = probability of applying coupling (default 0.50)
"""

import json, pickle, random, os, sys, math
import numpy as np
from collections import Counter, defaultdict


def extract_parameters(records_path, spec_path, transition_path):
    """Identical to v7b parameter extraction."""
    with open(records_path, 'rb') as f:
        records = pickle.load(f)
    with open(spec_path) as f:
        spec = json.load(f)
    with open(transition_path) as f:
        transitions = json.load(f)

    params = {}

    folio_order = []; folio_section = {}
    folio_lines = defaultdict(lambda: defaultdict(int))
    seen = set()
    for r in records:
        f = r['folio']
        if f not in seen: folio_order.append(f); seen.add(f)
        folio_section[f] = r['section']
        folio_lines[f][r['line_no']] += 1
    params['line_schedule'] = [(f, folio_section[f], ln, folio_lines[f][ln])
                                for f in folio_order for ln in sorted(folio_lines[f])]

    sec_ec = Counter(); sec_tot = Counter()
    for r in records:
        sec_tot[r['section']] += 1
        if r['empty_core']: sec_ec[r['section']] += 1
    params['section_ec_rate'] = {s: sec_ec[s]/sec_tot[s] for s in sec_tot}

    ec_triples = Counter(); fc_triples = Counter()
    for r in records:
        triple = (r['prefix'], r['gallows'], r['m_core'])
        (ec_triples if r['empty_core'] else fc_triples)[triple] += 1
    params['ec_triples'] = ec_triples
    params['fc_triples'] = fc_triples

    ec_g = defaultdict(Counter); fc_g = defaultdict(Counter)
    for r in records:
        pc = (r['prefix'], r['m_core'])
        (ec_g if r['empty_core'] else fc_g)[pc][r['gallows']] += 1
    params['ec_gallows_for_pc'] = dict(ec_g)
    params['fc_gallows_for_pc'] = dict(fc_g)

    triple_core_var = defaultdict(Counter)
    triple_sfx_surf = defaultdict(Counter)
    for r in records:
        triple = (r['prefix'], r['gallows'], r['m_core'])
        triple_core_var[triple][r['core']] += 1
        triple_sfx_surf[triple][r['suffix']] += 1
    params['triple_core_var'] = {t: dict(c) for t, c in triple_core_var.items()}
    params['triple_sfx_surf'] = {t: dict(c) for t, c in triple_sfx_surf.items()}

    tsm = defaultdict(Counter)
    for r in records:
        tsm[(r['prefix'], r['gallows'], r['m_core'])][r['sfx_fam']] += 1
    params['triple_sfx_menu'] = dict(tsm)

    triple_fam_surf = defaultdict(lambda: defaultdict(Counter))
    for r in records:
        triple = (r['prefix'], r['gallows'], r['m_core'])
        triple_fam_surf[triple][r['sfx_fam']][r['suffix']] += 1
    params['triple_fam_sfx_surf'] = {t: {f: dict(c) for f, c in d.items()}
                                      for t, d in triple_fam_surf.items()}

    sfx_lengths = {}
    for r in records:
        sfx_lengths[r['suffix']] = len(r['suffix']) if r['suffix'] and r['suffix'] != '∅' else 0
    params['sfx_lengths'] = sfx_lengths

    params['transitions'] = transitions

    fc_sec_pos = defaultdict(lambda: defaultdict(lambda: {'new': 0, 'total': 0}))
    fc_seen_triples = {}
    for r in records:
        if not r['empty_core']:
            triple = (r['prefix'], r['gallows'], r['m_core'])
            sec = r['section']
            pos = 'FIRST' if r['pos'] == 0 else ('LAST' if r['pos'] == r['line_len']-1 and r['line_len'] > 1 else 'MID')
            fc_sec_pos[sec][pos]['total'] += 1
            if triple not in fc_seen_triples:
                fc_seen_triples[triple] = True
                fc_sec_pos[sec][pos]['new'] += 1
    params['fc_section_innovation'] = {}
    for sec in fc_sec_pos:
        params['fc_section_innovation'][sec] = {}
        for pos in ['FIRST','MID','LAST']:
            d = fc_sec_pos[sec][pos]
            params['fc_section_innovation'][sec][pos] = d['new']/d['total'] if d['total'] else 0.0

    params['fc_innovation_rate'] = {'FIRST': 0.246, 'MID': 0.142, 'LAST': 0.217}
    params['fc_gallows_alt'] = 0.152
    params['reuse_curve'] = [
        (0, 0.0161), (1, 0.0460), (2, 0.0799), (3, 0.0850),
        (5, 0.1127), (10, 0.1571), (20, 0.1902), (50, 0.2844),
        (100, 0.4313), (200, 0.5518), (500, 0.7156), (1000, 0.8995),
    ]
    params['ec_sfx_fam_change'] = 0.55
    params['fc_sfx_fam_change'] = 0.49
    params['core_variant_change_rate'] = 0.09

    # v8b: token-length coupling parameters (replaces sfx_len_persist_prob)
    params['len_coupling_prob'] = 0.50
    params['lambda_len'] = 2.0

    return params, records


def weighted_choice(d, rng):
    items = list(d.keys()); weights = [d[k] for k in items]
    return rng.choices(items, weights=weights, k=1)[0]

def weighted_choice_avoid(d, avoid, rng):
    if len(d) <= 1: return list(d.keys())[0]
    alt = {k: v for k, v in d.items() if k != avoid}
    if not alt: return list(d.keys())[0]
    items = list(alt.keys()); weights = [alt[k] for k in items]
    return rng.choices(items, weights=weights, k=1)[0]

def interp_reuse(k, curve):
    if k <= 0: return curve[0][1]
    for i in range(len(curve)-1):
        k0, p0 = curve[i]; k1, p1 = curve[i+1]
        if k <= k1:
            return p0 + (k-k0)/(k1-k0) * (p1-p0) if k1 != k0 else p0
    return curve[-1][1]


class GenTS_v8b:
    def __init__(self, params, seed=42):
        self.p = params
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.ec_list = list(params['ec_triples'].keys())
        raw_w = np.array([params['ec_triples'][t] for t in self.ec_list], dtype=float)
        self.ec_w = np.sqrt(raw_w); self.ec_w /= self.ec_w.sum()

        self.fc_list = list(params['fc_triples'].keys())
        self.fc_w = np.array([params['fc_triples'][t] for t in self.fc_list], dtype=float)
        self.fc_w /= self.fc_w.sum()

    def _pick_core_variant(self, triple, last_core):
        menu = self.p['triple_core_var'].get(triple)
        if not menu or len(menu) == 1:
            return list(menu.keys())[0] if menu else ''
        if last_core is not None and last_core in menu:
            if self.rng.random() < self.p['core_variant_change_rate']:
                return weighted_choice_avoid(menu, last_core, self.rng)
            return last_core
        return weighted_choice(menu, self.rng)

    def _compute_candidate_len(self, prefix, gallows, core_surf, sfx_surf):
        """Compute token length from components."""
        total = 0
        if prefix and prefix != '∅': total += len(prefix)
        if gallows and gallows != '∅': total += len(gallows)
        if core_surf and core_surf != '∅': total += len(core_surf)
        if sfx_surf and sfx_surf != '∅': total += len(sfx_surf)
        return total

    def _pick_sfx_variant_with_len_coupling(self, triple, sfx_fam, prefix, gallows, core_surf, prev_wordlen):
        """v8b: TOKEN-LENGTH COUPLED suffix variant selection.
        
        1. Get per-triple, per-family suffix menu
        2. If prev_wordlen known and coupling fires, weight candidates by
           exp(-|candidate_token_len - prev_wordlen| / lambda)
        3. Otherwise: pure frequency-weighted (same as v7b)
        """
        fam_menus = self.p['triple_fam_sfx_surf'].get(triple, {})
        menu = fam_menus.get(sfx_fam, {})
        if not menu:
            menu = self.p['triple_sfx_surf'].get(triple, {'': 1})
        if len(menu) == 1:
            return list(menu.keys())[0]

        # v8b: token-length coupling
        if prev_wordlen is not None and self.rng.random() < self.p['len_coupling_prob']:
            lam = self.p['lambda_len']
            coupled = {}
            for sfx, wt in menu.items():
                cand_len = self._compute_candidate_len(prefix, gallows, core_surf, sfx)
                length_weight = math.exp(-abs(cand_len - prev_wordlen) / lam)
                coupled[sfx] = wt * length_weight
            # Ensure non-zero
            if sum(coupled.values()) > 0:
                return weighted_choice(coupled, self.rng)

        # Default: pure frequency-weighted
        return weighted_choice(menu, self.rng)

    def _pick_sfx_family(self, triple, is_ec, last_sfx_fam):
        menu = self.p['triple_sfx_menu'].get(triple, {})
        if not menu: return '∅'
        sfx_keys = list(menu.keys())
        if len(sfx_keys) == 1: return sfx_keys[0]
        change_rate = self.p['ec_sfx_fam_change'] if is_ec else self.p['fc_sfx_fam_change']
        if last_sfx_fam is not None and last_sfx_fam in menu:
            if self.rng.random() < change_rate:
                alt = {k: v for k, v in menu.items() if k != last_sfx_fam}
                if alt: return weighted_choice(alt, self.rng)
            return last_sfx_fam
        return weighted_choice(menu, self.rng)

    def _reconstruct(self, prefix, gallows, core_surf, sfx_surf):
        parts = []
        if prefix and prefix != '∅': parts.append(prefix)
        if gallows and gallows != '∅': parts.append(gallows)
        if core_surf and core_surf != '∅': parts.append(core_surf)
        if sfx_surf and sfx_surf != '∅': parts.append(sfx_surf)
        return ''.join(parts) if parts else '∅'

    def generate(self):
        p = self.p; tokens = []; lines = []
        fc_pool = Counter(); ec_pool = Counter()
        fc_seen = set()
        triple_last_sfx_fam = {}
        triple_last_core = {}

        fc_intro = sorted(self.fc_list, key=lambda t: p['fc_triples'][t], reverse=True)
        fc_idx = 0

        # v8b: track FULL TOKEN LENGTH, NOT just suffix length
        # v8b: DO NOT reset at line boundaries
        prev_wordlen = None

        for folio, section, line_no, n_words in p['line_schedule']:
            ec_rate = p['section_ec_rate'].get(section, 0.527)
            line_tok = []

            for wpos in range(n_words):
                pos = 'FIRST' if wpos == 0 else ('LAST' if wpos == n_words-1 and n_words > 1 else 'MID')
                is_ec = self.rng.random() < ec_rate

                if is_ec:
                    triple = self._pick_ec(ec_pool)
                    prefix, gallows, m_core = triple
                    g_menu = p['ec_gallows_for_pc'].get((prefix, m_core), {gallows: 1})
                    if len(g_menu) > 1:
                        gallows = weighted_choice(g_menu, self.rng)
                    triple = (prefix, gallows, m_core)
                    last_fam = triple_last_sfx_fam.get(triple)
                    sfx_fam = self._pick_sfx_family(triple, True, last_fam)
                    triple_last_sfx_fam[triple] = sfx_fam
                    core_surf = self._pick_core_variant(triple, triple_last_core.get(triple))
                    triple_last_core[triple] = core_surf
                    sfx_surf = self._pick_sfx_variant_with_len_coupling(
                        triple, sfx_fam, prefix, gallows, core_surf, prev_wordlen)
                    text = self._reconstruct(prefix, gallows, core_surf, sfx_surf)
                    ec_pool[triple] += 1
                else:
                    triple, sfx_surf, text, fc_idx = self._gen_fc(
                        section, pos, fc_pool, fc_seen, fc_intro, fc_idx,
                        triple_last_sfx_fam, triple_last_core, prev_wordlen)
                    fc_pool[triple] += 1; fc_seen.add(triple)

                prev_wordlen = len(text)  # v8b: FULL token length, NO line reset
                line_tok.append(text)

            tokens.extend(line_tok)
            lines.append(line_tok)
            # v8b: NO RESET of prev_wordlen at line boundary

        return tokens, lines

    def _pick_ec(self, ec_pool):
        if ec_pool and self.rng.random() > 0.003:
            triples = list(ec_pool.keys())
            w = np.sqrt(np.array([ec_pool[t] for t in triples], dtype=float))
            w /= w.sum()
            return triples[self.np_rng.choice(len(triples), p=w)]
        return self.ec_list[self.np_rng.choice(len(self.ec_list), p=self.ec_w)]

    def _gen_fc(self, section, pos, fc_pool, fc_seen, fc_intro, fc_idx,
                triple_last_sfx_fam, triple_last_core, prev_wordlen):
        p = self.p
        innov = p['fc_section_innovation'].get(section, {}).get(pos,
                p['fc_innovation_rate'].get(pos, 0.142))
        innovate = not fc_pool or self.rng.random() < innov

        if innovate:
            found = False
            while fc_idx < len(fc_intro):
                c = fc_intro[fc_idx]; fc_idx += 1
                if c not in fc_seen:
                    triple = c; found = True; break
            if not found:
                unseen = [t for t in self.fc_list if t not in fc_seen]
                if unseen: triple = self.rng.choice(unseen)
                else: triple = self._pref_fc(fc_pool); innovate = False

        if not innovate:
            triple = self._pref_fc(fc_pool)
            prefix, gallows, m_core = triple
            if self.rng.random() < p['fc_gallows_alt']:
                g_menu = p['fc_gallows_for_pc'].get((prefix, m_core), {gallows: 1})
                if len(g_menu) > 1:
                    alt = {g: w for g, w in g_menu.items() if g != gallows}
                    if alt: gallows = weighted_choice(alt, self.rng)
                triple = (prefix, gallows, m_core)

        prefix, gallows, m_core = triple
        last_fam = triple_last_sfx_fam.get(triple)
        sfx_fam = self._pick_sfx_family(triple, False, last_fam)
        triple_last_sfx_fam[triple] = sfx_fam
        core_surf = self._pick_core_variant(triple, triple_last_core.get(triple))
        triple_last_core[triple] = core_surf
        sfx_surf = self._pick_sfx_variant_with_len_coupling(
            triple, sfx_fam, prefix, gallows, core_surf, prev_wordlen)
        text = self._reconstruct(prefix, gallows, core_surf, sfx_surf)

        return triple, sfx_surf, text, fc_idx

    def _pref_fc(self, fc_pool):
        triples = list(fc_pool.keys())
        counts = [fc_pool[t] for t in triples]
        curve = self.p['reuse_curve']
        probs = np.array([interp_reuse(c, curve) for c in counts])
        probs /= probs.sum()
        return triples[self.np_rng.choice(len(triples), p=probs)]


def run_and_score(params, records, scorer_path, n_runs=5, seed_base=42, label="v8b"):
    import importlib.util
    sp = importlib.util.spec_from_file_location("scorer", scorer_path)
    scorer = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(scorer)

    vms_tok = [r['token'] for r in records]
    vms_lines = []; cur = []; ck = None
    for r in records:
        k = (r['folio'], r['line_no'])
        if k != ck:
            if cur: vms_lines.append(cur)
            cur = [r['token']]; ck = k
        else: cur.append(r['token'])
    if cur: vms_lines.append(cur)

    print("Computing VMS baseline...")
    vms_m = scorer.compute_metrics(vms_tok, lines=vms_lines)
    all_res = []; best_n = -1; best = None

    for run in range(n_runs):
        seed = seed_base + run
        gen = GenTS_v8b(params, seed=seed)
        gt, gl = gen.generate()
        gm = scorer.compute_metrics(gt, lines=gl)
        res = scorer.score_against_vms(gm, vms_m)
        wc = Counter(gt)
        n_types = len(wc)
        hapax = sum(1 for c in wc.values() if c == 1)
        dis = sum(1 for c in wc.values() if c == 2)
        info = {'seed': seed, 'n_pass': res['n_pass'], 'n_total': res['n_total'],
                'pct': res['n_pass']/res['n_total']*100, 'n_types': n_types,
                'hapax_ratio': hapax/n_types, 'dis_ratio': dis/n_types,
                'gen_metrics': gm, 'result': res, 'n_tokens': len(gt),
                'ac1': gm.get('wordlen_autocorr', 0)}
        all_res.append(info)
        print(f"  {label} Run {run+1}: {info['n_pass']}/{info['n_total']} ({info['pct']:.1f}%) "
              f"Types={n_types} Hapax={info['hapax_ratio']:.3f} Dis={info['dis_ratio']:.3f} "
              f"AC1={info['ac1']:.4f}")
        if info['n_pass'] > best_n: best_n = info['n_pass']; best = info

    return all_res, vms_m, best


if __name__ == '__main__':
    REPO = '/home/claude/repo'
    DATA = '/home/claude/session_data'

    print("="*70)
    print("Gen-TS v8b: Token-Length Coupling (AC1 isolation test)")
    print("="*70)

    params, records = extract_parameters(
        f'{REPO}/enriched_records.pkl', f'{REPO}/Paper/p70c_full_spec_v1.json',
        f'{REPO}/Paper/transition_lookup.json')
    pickle.dump(params, open('/home/claude/gen_ts_v8b_params.pkl', 'wb'))

    results, vms_m, best = run_and_score(
        params, records, f'{DATA}/score_85_metrics-5.py', n_runs=5, label="v8b")

    pickle.dump({'results': results, 'vms_metrics': vms_m, 'best': best},
                open('/home/claude/gen_ts_v8b_results.pkl', 'wb'))

    scores = [r['n_pass'] for r in results]
    print(f"\nBEST: seed={best['seed']} {best['n_pass']}/{best['n_total']} ({best['pct']:.1f}%)")
    print(f"  Types={best['n_types']} Hapax={best['hapax_ratio']:.3f} Dis={best['dis_ratio']:.3f} AC1={best['ac1']:.4f}")
    print(f"  Mean={np.mean(scores):.1f} std={np.std(scores):.1f} [{min(scores)},{max(scores)}]")

    det = best['result']['details']
    fails = [(m,d) for m,d in det.items() if not d['pass'] and not d.get('provisional')]
    fails.sort(key=lambda x: x[1]['delta']/max(x[1]['tol'],1e-9), reverse=True)
    print(f"\nTOP FAILURES ({len(fails)}):")
    for m,d in fails[:15]:
        x = d['delta']/d['tol'] if d['tol'] else 999
        print(f"  {m:<28s} VMS={d['vms']:8.4f} Gen={d['gen']:8.4f} delta={d['delta']:8.4f} tol={d['tol']:8.4f} x={x:.1f}")

    gm = best['gen_metrics']
    print(f"\nKEY:")
    for m in ['hapax_ratio_types','dis_ratio_types','wordlen_autocorr','autocorr_wordlen',
              'wordlen_mean','wordlen_std','zipf_alpha','honore_R']:
        v = vms_m.get(m, float('nan')); g = gm.get(m, float('nan'))
        d = det.get(m, {}); st = '+' if d.get('pass') else '-'
        print(f"  {st} {m:<28s} VMS={v:8.4f} Gen={g:8.4f}")
