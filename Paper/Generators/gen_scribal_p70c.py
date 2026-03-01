#!/usr/bin/env python3
"""
Generator 8: Scribal Manual + P70-C Grammar Overlay

SAME 6 rules from f57v. SAME production process.
But slot OPTIONS and WEIGHTS from P70-C constrained grammar.

What changes:
  - Slot options: P70-C ledger (6750 quad entries, not 77 f57v tokens)
  - Section profiles: 9 VMS sections (not 4 paradigm units)
  - Prefix transitions: conditioned on previous suffix family
  - Suffix expansion: family → actual suffix strings
  - Position awareness: FIRST/MID/LAST word in line

What stays the same (the scribe's 6 rules):
  1. Words = prefix + gallows + core + suffix
  2. Always change the suffix for the next word
  3. Half the time, also change one other slot
  4. Every ~5th word, write a completely new word
  5. At line starts, use transition-conditioned prefix
  6. Use section-specific gallows profile
"""

import json
import random
import pickle
import numpy as np
import os
from collections import Counter, defaultdict


# ══════════════════════════════════════════════════════════════════════
# STEP 1: BUILD SPEC FROM P70-C + ENRICHED RECORDS
# ══════════════════════════════════════════════════════════════════════

def build_p70c_spec(p70c_path='data/p70c_full_spec_v1.json',
                     records_path='enriched_records.pkl'):
    """Build slot distributions from P70-C ledger."""

    with open(p70c_path) as f:
        p70c = json.load(f)
    with open(records_path, 'rb') as f:
        records = pickle.load(f)

    entries = p70c['entries']
    transitions = p70c['transition_lookup']

    spec = {}

    # ── GLOBAL SLOT DISTRIBUTIONS (from P70-C entry counts) ──
    prefix_w = Counter()
    gallows_w = Counter()
    core_w = Counter()
    sfx_fam_w = Counter()

    for e in entries:
        c = e['count']
        prefix_w[e['prefix']] += c
        gallows_w[e['gallows']] += c
        core_w[e['m_core']] += c
        sfx_fam_w[e['sfx_fam']] += c

    spec['prefix'] = dict(prefix_w)
    spec['gallows'] = dict(gallows_w)
    spec['core'] = dict(core_w)
    spec['sfx_fam'] = dict(sfx_fam_w)

    # ── SUFFIX EXPANSION (family → actual suffix strings) ──
    suffix_expansions = defaultdict(Counter)
    for e in entries:
        fam = e['sfx_fam']
        # Weight each expansion proportionally to entry count
        n_suf = len(e.get('full_suffixes', ['∅']))
        per_suf = e['count'] / max(n_suf, 1)
        for s in e.get('full_suffixes', ['∅']):
            suffix_expansions[fam][s] += per_suf

    spec['suffix_expand'] = {fam: dict(counts) for fam, counts in suffix_expansions.items()}

    # ── POSITION-SPECIFIC PREFIX DISTRIBUTIONS ──
    pos_prefix = defaultdict(Counter)
    for e in entries:
        pos = e.get('position', 'MID')
        pos_prefix[pos][e['prefix']] += e['count']
    spec['pos_prefix'] = {pos: dict(counts) for pos, counts in pos_prefix.items()}

    # ── TRANSITION LOOKUP (prev suffix family → next prefix) ──
    spec['transitions'] = transitions

    # ── SECTION-SPECIFIC GALLOWS (from enriched_records) ──
    section_gallows = defaultdict(Counter)
    section_counts = Counter()
    for r in records:
        sec = r['section']
        section_counts[sec] += 1
        section_gallows[sec][r['gallows']] += 1

    spec['section_gallows'] = {sec: dict(counts)
                                for sec, counts in section_gallows.items()}
    spec['section_counts'] = dict(section_counts)
    spec['sections'] = sorted(section_counts.keys())

    # ── SECTION-SPECIFIC SUFFIX FAMILIES ──
    section_suffix = defaultdict(Counter)
    for r in records:
        section_suffix[r['section']][r['sfx_fam']] += 1
    spec['section_suffix'] = {sec: dict(counts)
                               for sec, counts in section_suffix.items()}

    return spec


# ══════════════════════════════════════════════════════════════════════
# STEP 2: THE SCRIBE (same 6 rules, richer vocabulary)
# ══════════════════════════════════════════════════════════════════════

class P70CScribe:
    """
    Same rules as Scribal Manual. But draws from P70-C grammar
    instead of f57v's 77 tokens.

    The 6 rules (unchanged):
      1. Words = prefix + gallows + core + suffix
      2. Always change the suffix
      3. Half the time, also change one other slot
      4. Every ~5th word, write fresh
      5. At line starts, use transition prefix
      6. Use section-specific gallows
    """

    SLOTS = ['prefix', 'gallows', 'core', 'sfx_fam']

    def __init__(self, spec, section='Herbal-A', seed=42):
        self.rng = random.Random(seed)
        self.spec = spec
        self.section = section

        # Build weighted choice tables
        self.slot_options = {}
        self.slot_weights = {}

        # Prefix: global distribution (transitions override at runtime)
        for slot in ['prefix', 'core']:
            opts = spec[slot]
            items = list(opts.keys())
            weights = [opts[k] for k in items]
            self.slot_options[slot] = items
            self.slot_weights[slot] = weights

        # Gallows: section-specific
        gal = spec['section_gallows'].get(section, spec['gallows'])
        items = list(gal.keys())
        weights = [gal[k] for k in items]
        self.slot_options['gallows'] = items
        self.slot_weights['gallows'] = weights

        # Suffix family: section-specific
        sfx = spec['section_suffix'].get(section, spec['sfx_fam'])
        items = list(sfx.keys())
        weights = [sfx[k] for k in items]
        self.slot_options['sfx_fam'] = items
        self.slot_weights['sfx_fam'] = weights

        # Suffix expansion tables
        self.suffix_expand = spec['suffix_expand']

        # Position-specific prefix tables
        self.pos_prefix = {}
        for pos in ['FIRST', 'MID', 'LAST']:
            if pos in spec['pos_prefix']:
                opts = spec['pos_prefix'][pos]
                self.pos_prefix[pos] = (list(opts.keys()),
                                         [opts[k] for k in opts])

        # Transition table
        self.transitions = spec['transitions']

    def _pick(self, slot):
        """Weighted random choice from slot distribution."""
        return self.rng.choices(
            self.slot_options[slot],
            weights=self.slot_weights[slot],
            k=1
        )[0]

    def _pick_prefix_transition(self, prev_sfx_fam, position='MID'):
        """
        Pick prefix conditioned on previous suffix family.
        This is the P70-C transition grammar.
        """
        # Use transition lookup if available
        key = prev_sfx_fam if prev_sfx_fam in self.transitions else 'BARE'
        trans = self.transitions.get(key, {})
        if trans:
            items = list(trans.keys())
            weights = [trans[k] for k in items]
            return self.rng.choices(items, weights=weights, k=1)[0]

        # Fallback to position-specific
        if position in self.pos_prefix:
            items, weights = self.pos_prefix[position]
            return self.rng.choices(items, weights=weights, k=1)[0]

        return self._pick('prefix')

    def _expand_suffix(self, sfx_fam):
        """Expand suffix family to actual suffix string."""
        if sfx_fam == 'BARE':
            return ''
        expansions = self.suffix_expand.get(sfx_fam, {'∅': 1})
        items = list(expansions.keys())
        weights = [expansions[k] for k in items]
        result = self.rng.choices(items, weights=weights, k=1)[0]
        return '' if result == '∅' else result

    def _assemble(self, prefix, gallows, core, suffix_str):
        """Assemble word from slots."""
        parts = []
        if prefix != '∅':
            parts.append(prefix)
        if gallows != '∅':
            parts.append(gallows)
        if core not in ('∅', ''):
            parts.append(core)
        if suffix_str:
            parts.append(suffix_str)
        word = ''.join(parts)
        return word if word else 'o'

    def fresh_word(self, position='MID', prev_sfx_fam='LINE_START'):
        """Generate completely new word from P70-C distributions (Rule 1+4)."""
        p = self._pick_prefix_transition(prev_sfx_fam, position)
        g = self._pick('gallows')
        c = self._pick('core')
        sf = self._pick('sfx_fam')
        s_str = self._expand_suffix(sf)
        return self._assemble(p, g, c, s_str), sf

    def mutate_word(self, template_slots, prev_sfx_fam='Y'):
        """
        Copy-mutate: keep template, change suffix + maybe one more.
        Rules 2-3.
        """
        p, g, c, sf = template_slots

        # RULE 2: Always change the suffix family
        new_sf = sf
        attempts = 0
        while new_sf == sf and attempts < 10:
            new_sf = self._pick('sfx_fam')
            attempts += 1
        new_s_str = self._expand_suffix(new_sf)

        # RULE 3: Half the time, also change one other slot
        new_p, new_g, new_c = p, g, c
        if self.rng.random() < 0.50:
            slot = self.rng.choice(['prefix', 'gallows', 'core'])
            if slot == 'prefix':
                new_p = self._pick_prefix_transition(prev_sfx_fam)
            elif slot == 'gallows':
                new_g = self._pick('gallows')
            else:
                new_c = self._pick('core')

        return self._assemble(new_p, new_g, new_c, new_s_str), (new_p, new_g, new_c, new_sf)

    def write_section(self, n_tokens, tokens_per_line=10):
        """
        Write a section. Same production process as Scribal Manual.
        Rules 4-5-6 with P70-C grammar.
        """
        corpus = []
        slots_history = []  # track decomposed slots for copy-mutate
        line_pos = 0
        prev_sfx_fam = 'LINE_START'

        for i in range(n_tokens):
            position = 'FIRST' if line_pos == 0 else (
                'LAST' if line_pos == tokens_per_line - 1 else 'MID')

            if line_pos == 0:
                # RULE 5: Line start — use transition from LINE_START
                word, sf = self.fresh_word(position='FIRST',
                                            prev_sfx_fam='LINE_START')
                slots = (word[:2] if len(word) > 2 else '∅', '∅', '∅', sf)

            elif self.rng.random() < 0.20:
                # RULE 4: Every ~5th word, write fresh
                word, sf = self.fresh_word(position=position,
                                            prev_sfx_fam=prev_sfx_fam)
                slots = ('∅', '∅', '∅', sf)

            elif slots_history:
                # RULES 2-3: Copy-mutate from recent word
                lookback = min(5, len(slots_history))
                template = self.rng.choice(slots_history[-lookback:])
                word, slots = self.mutate_word(template,
                                                prev_sfx_fam=prev_sfx_fam)
                sf = slots[3]

            else:
                word, sf = self.fresh_word(position=position,
                                            prev_sfx_fam=prev_sfx_fam)
                slots = ('∅', '∅', '∅', sf)

            corpus.append(word)
            # For copy-mutate, track the actual slot picks
            # (rough decomposition — same imperfection as a real scribe)
            slots_history.append(slots)
            prev_sfx_fam = sf
            line_pos = (line_pos + 1) % tokens_per_line

        return corpus


# ══════════════════════════════════════════════════════════════════════
# STEP 3: MULTI-SECTION PRODUCTION
# ══════════════════════════════════════════════════════════════════════

def produce_manuscript(spec, n_tokens=37465, seed=42):
    """
    Produce manuscript with section-assigned scribes.
    Each section gets tokens proportional to its VMS share.
    """
    rng = random.Random(seed)

    total_vms = sum(spec['section_counts'].values())
    sections = spec['sections']

    corpus = []
    for idx, section in enumerate(sections):
        share = spec['section_counts'][section] / total_vms
        n_sec = int(round(share * n_tokens))

        scribe = P70CScribe(spec, section=section,
                             seed=seed + idx * 1000)
        section_text = scribe.write_section(n_sec)
        corpus.extend(section_text)

    # Trim or pad to exact length
    if len(corpus) > n_tokens:
        corpus = corpus[:n_tokens]
    while len(corpus) < n_tokens:
        corpus.append('daiin')

    return corpus


# ══════════════════════════════════════════════════════════════════════
# STEP 4: RUN AND SCORE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GENERATOR 8: SCRIBAL MANUAL + P70-C GRAMMAR")
    print("=" * 70)
    print("""
    SAME 6 RULES:
      1. Words = prefix + gallows + core + suffix
      2. Always change the suffix
      3. Half the time, also change one other slot
      4. Every ~5th word, write fresh
      5. Line starts use transition-conditioned prefix
      6. Section-specific gallows profile

    NEW VOCABULARY:
      - P70-C ledger: 6750 quad entries (was: 77 f57v tokens)
      - 8 prefix options (was: 15)
      - 9 gallows options incl. compound (was: 5)
      - 1302 core options (was: 9)
      - 7 suffix families → expanded (was: 19 fixed suffixes)
      - Prefix transitions conditioned on previous suffix
    """)

    spec = build_p70c_spec()
    print(f"Sections: {len(spec['sections'])}")
    for sec in spec['sections']:
        n = spec['section_counts'][sec]
        print(f"  {sec:>20}: {n:>5} tokens ({n/37465:.1%})")

    # Generate 10 seeds
    n_tokens = 37465
    n_seeds = 10

    all_corpora = []
    for seed in range(n_seeds):
        corpus = produce_manuscript(spec, n_tokens=n_tokens,
                                     seed=42 + seed * 100)
        all_corpora.append(corpus)

        types = len(set(corpus))
        ttr = types / len(corpus)
        lens = [len(w) for w in corpus]
        mean_len = sum(lens) / len(lens)
        print(f"  Seed {seed}: {len(corpus)} tokens, {types} types, "
              f"TTR={ttr:.4f}, mean_len={mean_len:.2f}")

    # Save corpora
    os.makedirs('results', exist_ok=True)
    with open('results/scribal_p70c_corpora.pkl', 'wb') as f:
        pickle.dump({'spec_keys': list(spec.keys()),
                      'corpora': all_corpora,
                      'n_seeds': n_seeds,
                      'n_tokens': n_tokens}, f)
    print(f"\nSaved corpora.")

    # ── SCORE ──
    from score_85_metrics import (compute_metrics, score_against_vms,
                                   CORE_15, ALL_85, TOLERANCES)

    with open('results/vms_baseline_85metrics.pkl', 'rb') as f:
        vms_baseline = pickle.load(f)

    all_metrics = []
    for i, corpus in enumerate(all_corpora):
        lines = [corpus[j:j+10] for j in range(0, len(corpus), 10)]
        m = compute_metrics(corpus, lines=lines, seed=42 + i)
        all_metrics.append(m)

    median_metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
        if vals:
            median_metrics[key] = float(np.median(vals))

    s85 = score_against_vms(median_metrics, vms_baseline, ALL_85, TOLERANCES)
    s15 = score_against_vms(median_metrics, vms_baseline, CORE_15, TOLERANCES)

    print(f"\n{'=' * 70}")
    print(f"SCORES")
    print(f"{'=' * 70}")
    print(f"  Core 15: {s15['n_pass']}/{s15['n_total']}")
    print(f"  Full 90: {s85['n_pass']}/{s85['n_total']}")

    # Compare
    print(f"\n{'Generator':<20} {'Core15':>8} {'Full90':>8} {'Source'}")
    print("-" * 65)
    print(f"{'SM + P70-C':<20} {s15['n_pass']:>5}/15 {s85['n_pass']:>5}/90  "
          f"f57v rules + P70-C grammar ◄◄◄")
    hierarchy = [
        ('SCRIBAL MANUAL', 7, 43, 'f57v only'),
        ('f57v-ONLY', 5, 38, 'f57v only'),
        ('Bigram', 8, 50, 'Full corpus'),
        ('Scribal', 5, 37, 'Full corpus'),
        ('P70C', 9, 64, 'Full corpus'),
        ('Dual', 7, 58, 'Full corpus'),
        ('Section', 10, 67, 'Full corpus'),
        ('Combined', 10, 66, 'Full corpus'),
    ]
    for name, c, f, src in hierarchy:
        print(f"{name:<20} {c:>5}/15 {f:>5}/90  {src}")

    # Core 15 detail
    print(f"\n{'=' * 70}")
    print(f"CORE 15 DETAIL")
    print(f"{'=' * 70}")
    for m in sorted(s15['details'].keys()):
        d = s15['details'][m]
        status = '✓' if d['pass'] else '✗'
        print(f"  {status} {m:<22} VMS={d['vms']:>8.4f}  gen={d['gen']:>8.4f}  "
              f"Δ={d['delta']:>8.4f}  tol={d['tol']}")

    # Save
    with open('results/scribal_p70c_scores.pkl', 'wb') as f:
        pickle.dump({
            'median_metrics': median_metrics,
            'all_metrics': all_metrics,
            'scores_85': s85,
            'scores_15': s15,
        }, f)
    print(f"\nSaved results/scribal_p70c_scores.pkl")


if __name__ == '__main__':
    main()
