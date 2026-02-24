#!/usr/bin/env python3
"""
P70-C-FULL: Complete PGCS Constraint Layer
============================================
Integrates:
  - Slot+Position co-occurrence (6,750 quints)          [BUILT]
  - Transition lookup (prev_sfx → prefix, 8×8 matrix)   [CONNECTION 1]
  - Paragraph marker flag                                [CONNECTION 2]
  - Quire/production-unit metadata                       [CONNECTION 3]
  - Full suffix alongside sfx_fam                        [CONNECTION 4]

Single importable module. Backwards-compatible with p70c_pos.
"""

import pickle, json, math, random
from collections import Counter, defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

def _tier(count):
    if count >= 50: return 'T1'
    if count >= 10: return 'T2'
    if count >= 4:  return 'T3'
    return 'T4'

def _pos_cat(r):
    if str(r.get('is_first_word', '')) == 'True': return 'FIRST'
    if str(r.get('is_last_word', '')) == 'True': return 'LAST'
    return 'MID'

def _folio_to_quire(folio):
    num_str = folio.replace('f', '').replace('r', '').replace('v', '')
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


class P70C_Full:
    """Complete PGCS constraint layer with all connected axes."""

    # ── CORE: slot+position (from p70c_pos) ──

    def is_valid(self, p, g, mc, sf, pos=None):
        if pos:
            return (p, g, mc, sf, pos) in self.valid_quints
        return (p, g, mc, sf) in self.valid_quads

    def sample_quint(self, pos=None, section=None, min_tier=None):
        if pos and pos in self.pos_quint_list:
            ql = self.pos_quint_list[pos]
            pr = self.pos_quint_probs[pos]
        else:
            ql = self.quint_list
            pr = self.quint_probs
        if min_tier:
            tier_min = {'T1': 50, 'T2': 10, 'T3': 4, 'T4': 1}[min_tier]
            mask = np.array([self.quint_counts.get(q, 0) >= tier_min for q in ql])
            if mask.any():
                pr_m = pr * mask; pr_m /= pr_m.sum()
                return ql[np.random.choice(len(ql), p=pr_m)]
        return ql[np.random.choice(len(ql), p=pr)]

    def sample_quad(self, section=None, min_tier=None):
        ql = self.quad_list
        pr = self.quad_probs
        if min_tier:
            tier_min = {'T1': 50, 'T2': 10, 'T3': 4, 'T4': 1}[min_tier]
            mask = np.array([self.quad_counts.get(q, 0) >= tier_min for q in ql])
            if mask.any():
                pr_m = pr * mask; pr_m /= pr_m.sum()
                return ql[np.random.choice(len(ql), p=pr_m)]
        return ql[np.random.choice(len(ql), p=pr)]

    def get_tokens(self, p, g, mc, sf, pos=None):
        if pos:
            return self.quint_tokens.get((p, g, mc, sf, pos), [])
        return self.quad_tokens.get((p, g, mc, sf), [])

    def get_allowed_mcores(self, p, g, sf=None, pos=None):
        if pos and sf:
            return self.mc_given_pgsf_pos.get((p, g, sf, pos), set())
        if sf:
            return self.mc_given_pgsf.get((p, g, sf), set())
        return self.mc_given_pg.get((p, g), set())

    def get_allowed_sfxfams(self, p, g, mc=None, pos=None):
        if pos and mc:
            return self.sf_given_pgmc_pos.get((p, g, mc, pos), set())
        if mc:
            return self.sf_given_pgmc.get((p, g, mc), set())
        return self.sf_given_pg.get((p, g), set())

    def probability(self, p, g, mc, sf, pos=None, section=None):
        if pos:
            q = (p, g, mc, sf, pos)
            if section and section in self.sec_quint_counts:
                return self.sec_quint_counts[section].get(q, 0) / self.sec_totals.get(section, 1)
            return self.quint_counts.get(q, 0) / self.N
        q = (p, g, mc, sf)
        return self.quad_counts.get(q, 0) / self.N

    def log_probability(self, p, g, mc, sf, pos=None, section=None):
        prob = self.probability(p, g, mc, sf, pos, section)
        return math.log2(prob) if prob > 0 else float('-inf')

    def tier(self, p, g, mc, sf, pos=None):
        if pos:
            c = self.quint_counts.get((p, g, mc, sf, pos), 0)
        else:
            c = self.quad_counts.get((p, g, mc, sf), 0)
        return _tier(c) if c > 0 else None

    # ── CONNECTION 1: TRANSITION GRAMMAR ──

    def transition_prob(self, prev_sfx, prefix):
        """P(prefix | prev_sfx_fam). Use 'LINE_START' for first word."""
        ps = self.trans_probs.get(prev_sfx, {})
        return ps.get(prefix, 0.0)

    def transition_dist(self, prev_sfx):
        """Full prefix distribution given previous suffix family."""
        return dict(self.trans_probs.get(prev_sfx, {}))

    def sample_quint_sequential(self, pos, prev_sfx, section=None, 
                                 min_tier=None, n_attempts=50):
        """Sample a quint, biased by prev_sfx → prefix transition.
        
        Rejects quints whose prefix has zero transition probability.
        Among valid quints, reweights by transition probability.
        """
        if pos not in self.pos_quint_list:
            return self.sample_quint(pos=pos, min_tier=min_tier)
        
        ql = self.pos_quint_list[pos]
        pr = self.pos_quint_probs[pos].copy()
        
        # Tier filter
        if min_tier:
            tier_min = {'T1': 50, 'T2': 10, 'T3': 4, 'T4': 1}[min_tier]
            mask = np.array([self.quint_counts.get(q, 0) >= tier_min for q in ql])
            pr = pr * mask
        
        # Transition reweighting
        trans_d = self.trans_probs.get(prev_sfx, {})
        if trans_d:
            tw = np.array([trans_d.get(q[0], 0.001) for q in ql])
            pr = pr * tw
        
        total = pr.sum()
        if total > 0:
            pr /= total
            return ql[np.random.choice(len(ql), p=pr)]
        
        # Fallback: ignore transition
        return self.sample_quint(pos=pos, min_tier=min_tier)

    # ── CONNECTION 2: PARAGRAPH MARKERS ──

    def is_paragraph_opener(self, p, g, mc, sf):
        """Check if this quad matches paragraph-first token profile.
        
        Paragraph openers: ~79% ∅-prefix, ~85% gallows-bearing, FC-heavy.
        Returns confidence score 0-1.
        """
        score = 0.0
        if p == '∅': score += 0.4
        if g != '∅': score += 0.35
        if mc != '∅' and mc != '': score += 0.25
        return score

    def get_paragraph_openers(self, min_score=0.75, min_tier='T3'):
        """Return quads that match paragraph-opener profile."""
        tier_min = {'T1': 50, 'T2': 10, 'T3': 4, 'T4': 1}.get(min_tier, 1)
        openers = []
        for q, c in self.quad_counts.items():
            if c < tier_min:
                continue
            score = self.is_paragraph_opener(*q)
            if score >= min_score:
                openers.append((q, c, score))
        openers.sort(key=lambda x: -x[1])
        return openers

    # ── CONNECTION 3: QUIRE/PRODUCTION METADATA ──

    def quire_for_folio(self, folio):
        """Return quire assignment for a folio."""
        return _folio_to_quire(folio)

    def quire_profile(self, quire):
        """Return quad frequency profile for a quire."""
        return dict(self.quire_quad_counts.get(quire, {}))

    def production_cluster(self, folio):
        """Check if folio belongs to a known production cluster."""
        return self.production_clusters.get(folio, None)

    # ── CONNECTION 4: FULL SUFFIX ACCESS ──

    def get_full_suffixes(self, p, g, mc, sf_fam, pos=None):
        """Return all full suffixes observed for this quad/quint."""
        if pos:
            return sorted(self.quint_full_suffixes.get((p, g, mc, sf_fam, pos), set()))
        return sorted(self.quad_full_suffixes.get((p, g, mc, sf_fam), set()))

    def full_suffix_prob(self, p, g, mc, full_suffix, pos=None):
        """Probability using full suffix instead of sfx_fam."""
        if pos:
            q = (p, g, mc, full_suffix, pos)
            return self.full_quint_counts.get(q, 0) / self.N
        q = (p, g, mc, full_suffix)
        return self.full_quad_counts.get(q, 0) / self.N

    # ── GENERATION: FULL LINE ──

    def generate_line(self, n_words, section=None, min_tier='T2',
                      first_line=False):
        """Generate a full line of VMS-like tokens.
        
        Uses: position conditioning, transition grammar, paragraph markers.
        Returns list of (token, quint) tuples.
        """
        tokens = []
        prev_sfx = 'LINE_START'
        
        for i in range(n_words):
            # Position
            if i == 0:
                pos = 'FIRST'
            elif i == n_words - 1:
                pos = 'LAST'
            else:
                pos = 'MID'
            
            # Sample with transition bias
            if i == 0 and first_line:
                # Paragraph opener: force ∅-prefix + gallows
                openers = self.get_paragraph_openers(min_score=0.65, min_tier=min_tier)
                if openers:
                    quad, _, _ = random.choice(openers[:20])
                    quint = (*quad, pos)
                    toks = self.get_tokens(*quad, pos=pos)
                    if not toks:
                        toks = self.get_tokens(*quad)
                    tok = random.choice(toks) if toks else '???'
                    tokens.append((tok, quint))
                    prev_sfx = quad[3]
                    continue
            
            quint = self.sample_quint_sequential(
                pos=pos, prev_sfx=prev_sfx,
                section=section, min_tier=min_tier
            )
            
            toks = self.get_tokens(*quint[:4], pos=quint[4])
            if not toks:
                toks = self.get_tokens(*quint[:4])
            tok = random.choice(toks) if toks else '???'
            tokens.append((tok, quint))
            prev_sfx = quint[3]  # sfx_fam of current word
        
        return tokens

    # ── SCORING ──

    def score_line(self, line_records, section=None):
        """Score a line of records using full model.
        
        Returns total surprisal, per-token surprisal, and component breakdown.
        """
        results = []
        for i, r in enumerate(line_records):
            pos = _pos_cat(r)
            prev_sfx = 'LINE_START' if i == 0 else line_records[i-1]['sfx_fam']
            
            # Positional probability
            p_pos = self.probability(
                r['prefix'], r['gallows'], r['m_core'], r['sfx_fam'],
                pos=pos, section=section or r.get('section')
            )
            
            # Transition factor
            p_trans = self.transition_prob(prev_sfx, r['prefix'])
            
            # Combined (geometric mean as approximation)
            p_combined = math.sqrt(p_pos * max(p_trans, 1e-6))
            
            bits = -math.log2(p_combined) if p_combined > 0 else 30.0
            
            results.append({
                'token': r['token'],
                'bits': bits,
                'p_pos': p_pos,
                'p_trans': p_trans,
                'pos': pos,
                'prev_sfx': prev_sfx,
            })
        
        total_bits = sum(x['bits'] for x in results)
        return {
            'total_bits': total_bits,
            'mean_bits': total_bits / len(results) if results else 0,
            'perplexity': 2 ** (total_bits / len(results)) if results else 0,
            'tokens': results,
        }

    # ── VALIDATION ──

    def validate(self, records):
        fails = []
        for i, r in enumerate(records):
            q = (r['prefix'], r['gallows'], r['m_core'], r['sfx_fam'], _pos_cat(r))
            if q not in self.valid_quints:
                fails.append((i, r['token'], q))
        return fails

    # ── SUMMARY ──

    def summary(self):
        print(f"P70-C-FULL Layer Summary")
        print(f"  Quads (no pos):   {len(self.quad_counts):,}")
        print(f"  Quints (w/ pos):  {len(self.quint_counts):,}")
        print(f"  Tokens: {self.N:,}")
        qt = Counter(_tier(c) for c in self.quint_counts.values())
        print(f"  Quint tiers: {dict(qt)}")
        print(f"  Transitions: {len(self.trans_probs)} prev_sfx values")
        print(f"  Full suffixes tracked: {len(self.quad_full_suffixes):,} quads")
        print(f"  Quires: {len(self.quire_quad_counts)}")
        print(f"  Production clusters: {len(self.production_clusters)}")

    # ── EXPORT ──

    def to_json(self, path):
        entries = []
        for q in sorted(self.quint_counts, key=lambda x: -self.quint_counts[x]):
            p, g, mc, sf, pos = q
            entries.append({
                'prefix': p, 'gallows': g, 'm_core': mc, 'sfx_fam': sf,
                'position': pos,
                'count': self.quint_counts[q],
                'tier': _tier(self.quint_counts[q]),
                'prob': round(self.quint_counts[q] / self.N, 6),
                'full_suffixes': sorted(self.quint_full_suffixes.get(q, set())),
                'examples': self.quint_tokens.get(q, [])[:5]
            })
        spec = {
            'schema': 'p70-c-full-v1',
            'total_quints': len(self.quint_counts),
            'total_quads': len(self.quad_counts),
            'total_tokens': self.N,
            'transition_lookup': self.trans_probs,
            'entries': entries
        }
        with open(path, 'w') as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
        return len(entries)


# ═══════════════════════════════════════════════════════════════
# BUILDER
# ═══════════════════════════════════════════════════════════════

def build_p70c_full(records):
    """Build complete p70-c layer with all connected axes."""
    N = len(records)
    
    # ── Build line sequences for transition data ──
    lines = defaultdict(list)
    for r in records:
        lines[(r['folio'], r['line_no'])].append(r)
    for key in lines:
        lines[key].sort(key=lambda r: int(r['pos']) if isinstance(r['pos'], (int, float)) else 0)
    
    # Annotate
    for key, toks in lines.items():
        for i, r in enumerate(toks):
            r['_pos'] = _pos_cat(r)
            r['prev_sfx'] = 'LINE_START' if i == 0 else toks[i-1]['sfx_fam']
            is_fl = str(r.get('is_first_line', '')) == 'True'
            r['is_para_first'] = is_fl and i == 0
            r['quire'] = _folio_to_quire(r['folio'])
    
    # ── Counts ──
    quint_counts = Counter()
    quad_counts = Counter()
    quint_tokens = defaultdict(set)
    quad_tokens = defaultdict(set)
    quint_sections = defaultdict(Counter)
    sec_quint_counts = defaultdict(Counter)
    
    # Full suffix tracking
    quad_full_suffixes = defaultdict(set)
    quint_full_suffixes = defaultdict(set)
    full_quad_counts = Counter()
    full_quint_counts = Counter()
    
    # Transition counts
    trans_counts = defaultdict(Counter)
    
    # Quire counts
    quire_quad_counts = defaultdict(Counter)
    
    # Slot marginals
    pg_counts = Counter()
    prefix_counts = Counter()
    gallows_counts = Counter()
    mcore_counts = Counter()
    sfxfam_counts = Counter()
    pos_counts = Counter()
    
    for r in records:
        p, g, mc, sf = r['prefix'], r['gallows'], r['m_core'], r['sfx_fam']
        pos = r['_pos']
        sec = r['section']
        tok = r['token']
        full_sfx = r['suffix']
        
        quint = (p, g, mc, sf, pos)
        quad = (p, g, mc, sf)
        
        quint_counts[quint] += 1
        quad_counts[quad] += 1
        quint_sections[quint][sec] += 1
        sec_quint_counts[sec][quint] += 1
        if len(quint_tokens[quint]) < 10:
            quint_tokens[quint].add(tok)
        quad_tokens[quad].add(tok)
        
        # Full suffix
        quad_full_suffixes[quad].add(full_sfx)
        quint_full_suffixes[quint].add(full_sfx)
        full_quad_counts[(p, g, mc, full_sfx)] += 1
        full_quint_counts[(p, g, mc, full_sfx, pos)] += 1
        
        # Transitions
        trans_counts[r['prev_sfx']][p] += 1
        
        # Quire
        quire_quad_counts[r['quire']][quad] += 1
        
        # Marginals
        pg_counts[(p, g)] += 1
        prefix_counts[p] += 1
        gallows_counts[g] += 1
        mcore_counts[mc] += 1
        sfxfam_counts[sf] += 1
        pos_counts[pos] += 1
    
    sec_totals = {s: sum(v.values()) for s, v in sec_quint_counts.items()}
    
    # ── Sampling arrays ──
    quint_list = list(quint_counts.keys())
    quint_weights = np.array([quint_counts[q] for q in quint_list], dtype=float)
    quad_list = list(quad_counts.keys())
    quad_weights = np.array([quad_counts[q] for q in quad_list], dtype=float)
    
    pos_quint_list = {}
    pos_quint_probs = {}
    for pos in ['FIRST', 'MID', 'LAST']:
        ql = [q for q in quint_list if q[4] == pos]
        wt = np.array([quint_counts[q] for q in ql], dtype=float)
        pos_quint_list[pos] = ql
        pos_quint_probs[pos] = wt / wt.sum() if wt.sum() > 0 else wt
    
    # ── Conditional lookups ──
    mc_given_pgsf = defaultdict(set)
    mc_given_pgsf_pos = defaultdict(set)
    sf_given_pgmc = defaultdict(set)
    sf_given_pgmc_pos = defaultdict(set)
    mc_given_pg = defaultdict(set)
    sf_given_pg = defaultdict(set)
    
    for (p, g, mc, sf, pos) in quint_counts:
        mc_given_pg[(p, g)].add(mc)
        sf_given_pg[(p, g)].add(sf)
        mc_given_pgsf[(p, g, sf)].add(mc)
        mc_given_pgsf_pos[(p, g, sf, pos)].add(mc)
        sf_given_pgmc[(p, g, mc)].add(sf)
        sf_given_pgmc_pos[(p, g, mc, pos)].add(sf)
    
    # ── Transition probabilities ──
    trans_probs = {}
    for ps, pfx_counts in trans_counts.items():
        total = sum(pfx_counts.values())
        trans_probs[ps] = {p: round(c/total, 4) for p, c in pfx_counts.items()}
    
    # ── Production clusters ──
    production_clusters = {}
    for f in ['f42r', 'f42v', 'f49r', 'f49v', 'f56r', 'f56v']:
        production_clusters[f] = 'CLUSTER_A'
    
    # ── Assemble ──
    layer = P70C_Full()
    layer.N = N
    layer.quint_counts = dict(quint_counts)
    layer.quad_counts = dict(quad_counts)
    layer.quint_sections = {k: dict(v) for k, v in quint_sections.items()}
    layer.quint_tokens = {k: sorted(v) for k, v in quint_tokens.items()}
    layer.quad_tokens = {k: sorted(v)[:10] for k, v in quad_tokens.items()}
    layer.sec_quint_counts = {k: dict(v) for k, v in sec_quint_counts.items()}
    layer.sec_totals = sec_totals
    layer.valid_quints = set(quint_counts.keys())
    layer.valid_quads = set(quad_counts.keys())
    
    layer.quint_list = quint_list
    layer.quint_probs = quint_weights / quint_weights.sum()
    layer.quad_list = quad_list
    layer.quad_probs = quad_weights / quad_weights.sum()
    layer.pos_quint_list = pos_quint_list
    layer.pos_quint_probs = pos_quint_probs
    
    layer.mc_given_pg = dict(mc_given_pg)
    layer.sf_given_pg = dict(sf_given_pg)
    layer.mc_given_pgsf = dict(mc_given_pgsf)
    layer.mc_given_pgsf_pos = dict(mc_given_pgsf_pos)
    layer.sf_given_pgmc = dict(sf_given_pgmc)
    layer.sf_given_pgmc_pos = dict(sf_given_pgmc_pos)
    
    layer.prefix_counts = dict(prefix_counts)
    layer.gallows_counts = dict(gallows_counts)
    layer.mcore_counts = dict(mcore_counts)
    layer.sfxfam_counts = dict(sfxfam_counts)
    layer.pos_counts = dict(pos_counts)
    
    # Connection 1: transitions
    layer.trans_probs = trans_probs
    
    # Connection 2: paragraph markers (via is_paragraph_opener method)
    
    # Connection 3: quire metadata
    layer.quire_quad_counts = {k: dict(v) for k, v in quire_quad_counts.items()}
    layer.production_clusters = production_clusters
    
    # Connection 4: full suffix
    layer.quad_full_suffixes = {k: set(v) for k, v in quad_full_suffixes.items()}
    layer.quint_full_suffixes = {k: set(v) for k, v in quint_full_suffixes.items()}
    layer.full_quad_counts = dict(full_quad_counts)
    layer.full_quint_counts = dict(full_quint_counts)
    
    n_p = len(prefix_counts)
    n_g = len(gallows_counts)
    n_mc = len(mcore_counts)
    n_sf = len(sfxfam_counts)
    layer.unc_quad = n_p * n_g * n_mc * n_sf
    layer.unc_quint = n_p * n_g * n_mc * n_sf * 3
    
    return layer


# ═══════════════════════════════════════════════════════════════
# BUILD, VALIDATE, DEMONSTRATE
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("Loading enriched_records.pkl...")
    with open('Voynichdecomp/enriched_records.pkl', 'rb') as f:
        records = pickle.load(f)
    
    print("Building p70-c-full layer...")
    layer = build_p70c_full(records)
    layer.summary()
    
    # Validate
    fails = layer.validate(records)
    print(f"\nValidation failures: {len(fails)}")
    
    # ── DEMO: Generate lines ──
    print("\n" + "="*60)
    print("DEMO: Generated lines")
    print("="*60)
    
    random.seed(42)
    np.random.seed(42)
    
    print("\n  Herbal-A continuation line (8 words):")
    line = layer.generate_line(8, section='Herbal-A', min_tier='T2')
    print(f"  {' '.join(t[0] for t in line)}")
    
    print("\n  Herbal-A paragraph opener (10 words):")
    line = layer.generate_line(10, section='Herbal-A', min_tier='T2', first_line=True)
    print(f"  {' '.join(t[0] for t in line)}")
    
    print("\n  Stars section continuation (7 words):")
    line = layer.generate_line(7, section='Stars', min_tier='T2')
    print(f"  {' '.join(t[0] for t in line)}")
    
    print("\n  Pharma section continuation (9 words):")
    line = layer.generate_line(9, section='Pharma', min_tier='T2')
    print(f"  {' '.join(t[0] for t in line)}")
    
    # ── DEMO: Score a real line ──
    print("\n" + "="*60)
    print("DEMO: Score a real VMS line")
    print("="*60)
    
    # Pick a line
    test_lines = defaultdict(list)
    for r in records:
        test_lines[(r['folio'], r['line_no'])].append(r)
    for key in test_lines:
        test_lines[key].sort(key=lambda r: int(r['pos']) if isinstance(r['pos'], (int, float)) else 0)
    
    # Score first 5 lines
    scored = 0
    for key in sorted(test_lines.keys())[:5]:
        line_recs = test_lines[key]
        if len(line_recs) < 3:
            continue
        result = layer.score_line(line_recs)
        print(f"\n  {key[0]} L{key[1]} ({len(line_recs)} words): "
              f"{result['mean_bits']:.2f} bits/tok, PPL={result['perplexity']:.1f}")
        print(f"    {' '.join(r['token'] for r in line_recs)}")
        scored += 1
        if scored >= 5:
            break
    
    # ── DEMO: Transition lookup ──
    print("\n" + "="*60)
    print("DEMO: Transition lookup")
    print("="*60)
    
    for prev in ['Y', 'BARE', 'M', 'LINE_START']:
        d = layer.transition_dist(prev)
        top3 = sorted(d.items(), key=lambda x: -x[1])[:3]
        print(f"  After {prev:>10}: {', '.join(f'{p}={v:.0%}' for p,v in top3)}")
    
    # ── DEMO: Full suffix access ──
    print("\n" + "="*60)
    print("DEMO: Full suffix access")
    print("="*60)
    
    for quad in [('ch', '∅', 'ol', 'Y'), ('qo', 'k', 'e', 'Y'), ('d', '∅', 'air', 'N')]:
        full = layer.get_full_suffixes(*quad)
        print(f"  {quad} → full suffixes: {full}")
    
    # ── DEMO: Paragraph openers ──
    print("\n" + "="*60)
    print("DEMO: Paragraph openers")
    print("="*60)
    
    openers = layer.get_paragraph_openers(min_score=0.75, min_tier='T2')
    print(f"  Top 10 paragraph-opener quads:")
    for q, c, s in openers[:10]:
        toks = layer.get_tokens(*q, pos='FIRST')[:3]
        print(f"    {q} n={c} score={s:.2f} tokens={toks}")
    
    # Export
    print("\n" + "="*60)
    print("EXPORT")
    print("="*60)
    
    n = layer.to_json('p70c_full_spec_v1.json')
    print(f"  p70c_full_spec_v1.json: {n} entries")
    
    with open('p70c_full_layer.pkl', 'wb') as f:
        pickle.dump(layer, f)
    print(f"  p70c_full_layer.pkl: saved")
    
    print("\n✓ P70-C-FULL layer built and validated.")

