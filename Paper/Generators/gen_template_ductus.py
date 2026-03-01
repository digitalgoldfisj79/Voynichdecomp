#!/usr/bin/env python3
"""
Generator 9: Scribal Manual + P70-C Templates + Ductus Filter

SAME 6 RULES. Two principled additions:
  A) Template sampling: pick a whole P70-C entry, mutate one slot
     (instead of independent slot sampling — preserves word structure)
  B) Ductus filter: reject words with character bigrams not in VMS
     (a scribe internalises which letter combos "look right")

Both are simple, learnable constraints — not parameter tuning.
"""

import json
import random
import pickle
import numpy as np
import os
from collections import Counter, defaultdict


def build_spec(p70c_path='data/p70c_full_spec_v1.json',
               records_path='enriched_records.pkl'):
    """Build spec with templates and ductus constraints."""

    with open(p70c_path) as f:
        p70c = json.load(f)
    with open(records_path, 'rb') as f:
        records = pickle.load(f)

    entries = p70c['entries']
    transitions = p70c['transition_lookup']
    spec = {}

    # ── TEMPLATE POOL (whole P70-C entries as generation templates) ──
    # A scribe has a reference table of valid word-forms
    templates = []
    for e in entries:
        if e['count'] >= 2:  # entries appearing 2+ times
            templates.append({
                'prefix': e['prefix'],
                'gallows': e['gallows'],
                'core': e['m_core'],
                'sfx_fam': e['sfx_fam'],
                'full_suffixes': e.get('full_suffixes', ['∅']),
                'weight': e['count'],
            })
    spec['templates'] = templates
    spec['template_weights'] = [t['weight'] for t in templates]

    # ── SLOT DISTRIBUTIONS (for mutation targets) ──
    prefix_w = Counter()
    gallows_w = Counter()
    core_w = Counter()
    sfx_fam_w = Counter()
    for e in entries:
        prefix_w[e['prefix']] += e['count']
        gallows_w[e['gallows']] += e['count']
        core_w[e['m_core']] += e['count']
        sfx_fam_w[e['sfx_fam']] += e['count']

    spec['prefix'] = dict(prefix_w)
    spec['gallows'] = dict(gallows_w)
    spec['core'] = dict(core_w)
    spec['sfx_fam'] = dict(sfx_fam_w)

    # ── SUFFIX EXPANSION ──
    suffix_expand = defaultdict(Counter)
    for e in entries:
        fam = e['sfx_fam']
        n_suf = len(e.get('full_suffixes', ['∅']))
        per_suf = e['count'] / max(n_suf, 1)
        for s in e.get('full_suffixes', ['∅']):
            suffix_expand[fam][s] += per_suf
    spec['suffix_expand'] = {f: dict(c) for f, c in suffix_expand.items()}

    # ── TRANSITIONS ──
    spec['transitions'] = transitions

    # ── SECTION PROFILES ──
    section_gallows = defaultdict(Counter)
    section_suffix = defaultdict(Counter)
    section_counts = Counter()
    for r in records:
        sec = r['section']
        section_counts[sec] += 1
        section_gallows[sec][r['gallows']] += 1
        section_suffix[sec][r['sfx_fam']] += 1

    spec['section_gallows'] = {s: dict(c) for s, c in section_gallows.items()}
    spec['section_suffix'] = {s: dict(c) for s, c in section_suffix.items()}
    spec['section_counts'] = dict(section_counts)
    spec['sections'] = sorted(section_counts.keys())

    # ── DUCTUS CONSTRAINT: valid character bigrams ──
    # A scribe knows which letter pairs "look right"
    valid_bigrams = set()
    for r in records:
        token = r['token']
        for i in range(len(token) - 1):
            valid_bigrams.add(token[i:i+2])
    spec['valid_bigrams'] = valid_bigrams

    # ── SECTION-SPECIFIC TEMPLATE POOLS ──
    # Build per-section template indices for faster lookup
    section_templates = defaultdict(list)
    section_template_weights = defaultdict(list)
    
    # Map entries to sections via enriched records
    # Count (prefix, gallows, core, sfx_fam) per section
    section_quad_counts = defaultdict(Counter)
    for r in records:
        quad = (r['prefix'], r['gallows'], r['m_core'], r['sfx_fam'])
        section_quad_counts[r['section']][quad] += 1
    
    for sec in spec['sections']:
        sec_quads = section_quad_counts[sec]
        for i, t in enumerate(templates):
            quad = (t['prefix'], t['gallows'], t['core'], t['sfx_fam'])
            if quad in sec_quads:
                section_templates[sec].append(i)
                section_template_weights[sec].append(sec_quads[quad])
    
    spec['section_templates'] = dict(section_templates)
    spec['section_template_weights'] = dict(section_template_weights)

    return spec


class TemplateScribe:
    """
    Same 6 rules. But:
    - Picks TEMPLATES (whole P70-C entries) instead of independent slots
    - Applies ductus filter (valid bigrams only)
    - Uses section-specific template pool
    """

    def __init__(self, spec, section='Herbal-A', seed=42):
        self.rng = random.Random(seed)
        self.spec = spec
        self.section = section
        self.valid_bigrams = spec['valid_bigrams']

        # Template pool (section-specific if available, else global)
        if section in spec.get('section_templates', {}):
            indices = spec['section_templates'][section]
            weights = spec['section_template_weights'][section]
            self.templates = [spec['templates'][i] for i in indices]
            self.template_weights = weights
        else:
            self.templates = spec['templates']
            self.template_weights = spec['template_weights']

        # Slot distributions for mutations
        self.slot_opts = {}
        self.slot_wts = {}
        for slot in ['prefix', 'core']:
            d = spec[slot]
            self.slot_opts[slot] = list(d.keys())
            self.slot_wts[slot] = [d[k] for k in d]

        # Section-specific gallows
        gal = spec['section_gallows'].get(section, spec['gallows'])
        self.slot_opts['gallows'] = list(gal.keys())
        self.slot_wts['gallows'] = [gal[k] for k in gal]

        # Section-specific suffix families
        sfx = spec['section_suffix'].get(section, spec['sfx_fam'])
        self.slot_opts['sfx_fam'] = list(sfx.keys())
        self.slot_wts['sfx_fam'] = [sfx[k] for k in sfx]

        self.suffix_expand = spec['suffix_expand']
        self.transitions = spec['transitions']

    def _pick(self, slot):
        return self.rng.choices(self.slot_opts[slot],
                                 self.slot_wts[slot], k=1)[0]

    def _pick_template(self):
        """Pick a whole P70-C entry as template."""
        return self.rng.choices(self.templates,
                                 self.template_weights, k=1)[0]

    def _expand_suffix(self, sfx_fam):
        if sfx_fam == 'BARE':
            return ''
        exps = self.suffix_expand.get(sfx_fam, {'∅': 1})
        items = list(exps.keys())
        weights = [exps[k] for k in items]
        r = self.rng.choices(items, weights, k=1)[0]
        return '' if r == '∅' else r

    def _assemble(self, prefix, gallows, core, suffix_str):
        parts = []
        if prefix != '∅': parts.append(prefix)
        if gallows != '∅': parts.append(gallows)
        if core not in ('∅', ''): parts.append(core)
        if suffix_str: parts.append(suffix_str)
        return ''.join(parts) or 'o'

    def _ductus_valid(self, word):
        """Check all character bigrams are valid VMS combinations."""
        if len(word) <= 1:
            return True
        for i in range(len(word) - 1):
            if word[i:i+2] not in self.valid_bigrams:
                return False
        return True

    def _pick_prefix_transition(self, prev_sfx_fam):
        key = prev_sfx_fam if prev_sfx_fam in self.transitions else 'BARE'
        trans = self.transitions.get(key, {})
        if trans:
            items = list(trans.keys())
            weights = [trans[k] for k in items]
            return self.rng.choices(items, weights, k=1)[0]
        return self._pick('prefix')

    def fresh_word(self, prev_sfx_fam='LINE_START'):
        """
        TEMPLATE-BASED fresh word (Rule 1+4):
        Pick a P70-C entry, expand its suffix, apply ductus filter.
        """
        for attempt in range(20):
            tmpl = self._pick_template()
            # Use transition-conditioned prefix override
            prefix = self._pick_prefix_transition(prev_sfx_fam)
            sfx_fam = tmpl['sfx_fam']
            suffix_str = self._expand_suffix(sfx_fam)

            word = self._assemble(prefix, tmpl['gallows'],
                                    tmpl['core'], suffix_str)
            if self._ductus_valid(word):
                return word, (prefix, tmpl['gallows'], tmpl['core'], sfx_fam)

        # Fallback after 20 attempts: use template as-is
        tmpl = self._pick_template()
        sf = self._expand_suffix(tmpl['sfx_fam'])
        word = self._assemble(tmpl['prefix'], tmpl['gallows'],
                                tmpl['core'], sf)
        return word, (tmpl['prefix'], tmpl['gallows'], tmpl['core'],
                       tmpl['sfx_fam'])

    def mutate_word(self, template_slots, prev_sfx_fam='Y'):
        """
        RULE 2: Always change suffix.
        RULE 3: Half the time change one more slot.
        Apply ductus filter.
        """
        p, g, c, sf = template_slots

        for attempt in range(20):
            # Rule 2: new suffix family
            new_sf = sf
            tries = 0
            while new_sf == sf and tries < 10:
                new_sf = self._pick('sfx_fam')
                tries += 1
            new_s_str = self._expand_suffix(new_sf)

            # Rule 3: half the time change one more slot
            new_p, new_g, new_c = p, g, c
            if self.rng.random() < 0.50:
                slot = self.rng.choice(['prefix', 'gallows', 'core'])
                if slot == 'prefix':
                    new_p = self._pick_prefix_transition(prev_sfx_fam)
                elif slot == 'gallows':
                    new_g = self._pick('gallows')
                else:
                    new_c = self._pick('core')

            word = self._assemble(new_p, new_g, new_c, new_s_str)
            if self._ductus_valid(word):
                return word, (new_p, new_g, new_c, new_sf)

        # Fallback: accept without filter
        new_sf = self._pick('sfx_fam')
        new_s_str = self._expand_suffix(new_sf)
        word = self._assemble(p, g, c, new_s_str)
        return word, (p, g, c, new_sf)

    def write_section(self, n_tokens, tokens_per_line=10):
        """Same production process. Same 6 rules."""
        corpus = []
        slots_history = []
        line_pos = 0
        prev_sfx_fam = 'LINE_START'

        for i in range(n_tokens):
            if line_pos == 0:
                # Rule 5: line start
                word, slots = self.fresh_word(prev_sfx_fam='LINE_START')

            elif self.rng.random() < 0.20:
                # Rule 4: fresh word
                word, slots = self.fresh_word(prev_sfx_fam=prev_sfx_fam)

            elif slots_history:
                # Rules 2+3: copy-mutate
                lookback = min(5, len(slots_history))
                template = self.rng.choice(slots_history[-lookback:])
                word, slots = self.mutate_word(template,
                                                prev_sfx_fam=prev_sfx_fam)
            else:
                word, slots = self.fresh_word(prev_sfx_fam=prev_sfx_fam)

            corpus.append(word)
            slots_history.append(slots)
            prev_sfx_fam = slots[3]
            line_pos = (line_pos + 1) % tokens_per_line

        return corpus


def produce_manuscript(spec, n_tokens=37465, seed=42):
    """Multi-section production with section-assigned scribes."""
    rng = random.Random(seed)
    total = sum(spec['section_counts'].values())
    corpus = []

    for idx, section in enumerate(spec['sections']):
        share = spec['section_counts'][section] / total
        n_sec = int(round(share * n_tokens))
        scribe = TemplateScribe(spec, section=section,
                                 seed=seed + idx * 1000)
        corpus.extend(scribe.write_section(n_sec))

    return corpus[:n_tokens] if len(corpus) >= n_tokens else \
        corpus + ['daiin'] * (n_tokens - len(corpus))


def main():
    print("=" * 70)
    print("GENERATOR 9: TEMPLATES + DUCTUS FILTER")
    print("=" * 70)
    print("""
    SAME 6 RULES + 2 PRINCIPLED CONSTRAINTS:
      A) Template sampling (whole P70-C entries, not independent slots)
      B) Ductus filter (reject invalid character bigrams)
    """)

    spec = build_spec()
    print(f"Templates: {len(spec['templates'])} (from P70-C entries with count≥2)")
    print(f"Valid bigrams: {len(spec['valid_bigrams'])}")

    n_tokens = 37465
    n_seeds = 10

    all_corpora = []
    for seed in range(n_seeds):
        corpus = produce_manuscript(spec, n_tokens=n_tokens,
                                     seed=42 + seed * 100)
        all_corpora.append(corpus)
        types = len(set(corpus))
        lens = [len(w) for w in corpus]
        mean_len = sum(lens) / len(lens)
        print(f"  Seed {seed}: {len(corpus)} tok, {types} types, "
              f"TTR={types/len(corpus):.4f}, mean_len={mean_len:.2f}")

    os.makedirs('results', exist_ok=True)
    with open('results/template_ductus_corpora.pkl', 'wb') as f:
        pickle.dump({'corpora': all_corpora, 'n_seeds': n_seeds}, f)

    # Score
    from score_85_metrics import (compute_metrics, score_against_vms,
                                   CORE_15, ALL_85, TOLERANCES)
    with open('results/vms_baseline_85metrics.pkl', 'rb') as f:
        vms_baseline = pickle.load(f)

    all_metrics = []
    for i, corpus in enumerate(all_corpora):
        lines_list = [corpus[j:j+10] for j in range(0, len(corpus), 10)]
        m = compute_metrics(corpus, lines=lines_list, seed=42 + i)
        all_metrics.append(m)

    median_metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
        if vals:
            median_metrics[key] = float(np.median(vals))

    s85 = score_against_vms(median_metrics, vms_baseline, ALL_85, TOLERANCES)
    s15 = score_against_vms(median_metrics, vms_baseline, CORE_15, TOLERANCES)

    print(f"\n{'='*70}")
    print(f"SCORES")
    print(f"{'='*70}")
    print(f"  Core 15: {s15['n_pass']}/{s15['n_total']}")
    print(f"  Full 90: {s85['n_pass']}/{s85['n_total']}")

    print(f"\n{'Generator':<22} {'C15':>5} {'F90':>5}")
    print("-" * 40)
    rows = [
        ('TMPL+DUCTUS', s15['n_pass'], s85['n_pass']),
        ('SM + P70-C', 11, 69),
        ('Section', 10, 67),
        ('Combined', 10, 66),
        ('P70C', 9, 64),
        ('Bigram', 8, 50),
        ('SCRIBAL MANUAL', 7, 43),
    ]
    for name, c, f in rows:
        marker = ' ◄' if name == 'TMPL+DUCTUS' else ''
        print(f"{name:<22} {c:>3}/15 {f:>3}/90{marker}")

    # Core 15 detail
    print(f"\n{'='*70}")
    print(f"CORE 15 DETAIL")
    print(f"{'='*70}")
    for m in sorted(s15['details'].keys()):
        d = s15['details'][m]
        s = '✓' if d['pass'] else '✗'
        print(f"  {s} {m:<22} VMS={d['vms']:>8.4f}  gen={d['gen']:>8.4f}  "
              f"Δ={d['delta']:>8.4f}  tol={d['tol']}")

    # Full 90 fails
    print(f"\nFAILS ({s85['n_total'] - s85['n_pass']}):")
    for m in sorted(s85['fails']):
        d = s85['details'][m]
        tol = float(d['tol'])
        ratio = d['delta']/tol if tol > 0 else 999
        print(f"  ✗ {m:<28} Δ={d['delta']:>8.4f}  tol={tol}  {ratio:.2f}×")

    with open('results/template_ductus_scores.pkl', 'wb') as f:
        pickle.dump({
            'median_metrics': median_metrics,
            'all_metrics': all_metrics,
            'scores_85': s85,
            'scores_15': s15,
        }, f)
    print(f"\nSaved results/template_ductus_scores.pkl")


if __name__ == '__main__':
    main()
