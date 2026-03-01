#!/usr/bin/env python3
"""
Generator 7: Scribal Manual
ALL parameters from f57v. Zero corpus tuning. Simple rules a 15th-century
scribe could learn in an afternoon.

THE SCRIBE'S MANUAL (5 rules):
  1. Every word = prefix + gallows + core + suffix
  2. Always change the suffix when writing the next word
  3. Half the time, also change one other part
  4. Every few words, write a completely new word from scratch
  5. Use your assigned tall-letter pair throughout the section

Architecture: gen_scribal_manual.py
Input:  slim.json (reads ONLY f57v page)
Output: synthetic corpus scored against 90-metric VMS baseline
"""

import json
import random
import pickle
import numpy as np
import os
from collections import Counter


# ══════════════════════════════════════════════════════════════════════
# STEP 1: f57v SPEC EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def load_f57v_spec(slim_path='slim.json'):
    """Extract all parameters from f57v only."""
    with open(slim_path) as f:
        data = json.load(f)

    f57v = data['pages']['f57v']
    lines = {}
    for ln, line_data in f57v.items():
        t = line_data.get('t', {})
        text = t.get('TTLI', '')
        lines[ln] = text.split() if text else []

    # ── SLOT OPTIONS (from f57v output tokens in Rings 3-4) ──
    # These are what appears on the page. Weights = occurrence counts.

    # PREFIXES: how words begin on f57v
    # ∅=31, da=7, ok=6, sh=5, ot=5, d=5, a=5, s=3, of=3, so=2, 
    # qo=1, ch=1, r=1, de=1, v=1
    PREFIX_OPTIONS = {
        '∅': 31, 'o': 0, 'da': 7, 'ok': 6, 'sh': 5, 'ot': 5,
        'd': 5, 'a': 5, 's': 3, 'of': 3, 'so': 2,
        'qo': 1, 'ch': 1, 'r': 1, 'v': 1
    }
    # Note: bare 'o' prefix absorbed into ok/ot/of/∅ in f57v tokens

    # GALLOWS: from Line 3 paradigm + word-internal occurrence
    # Paradigm: k=44%, t=25%, f=12%, p=12%, m=6%
    # In words: ∅=45, k=15, t=11, f=6
    GALLOWS_OPTIONS = {'∅': 45, 'k': 15, 't': 11, 'f': 6, 'p': 3}
    # p rare on f57v but present in paradigm; add minimal weight

    # CORE: medial fillers between gallows and suffix
    CORE_OPTIONS = {
        '∅': 25, 'e': 12, 'o': 5, 'ch': 4, 'ee': 3,
        'ed': 3, 'a': 2, 'ar': 2, 'od': 2
    }

    # SUFFIXES: how words end on f57v
    SUFFIX_OPTIONS = {
        's': 9, 'al': 6, 'o': 5, 'ar': 5, 'aiin': 5,
        'y': 4, 'ey': 3, 'ty': 3, 'r': 3, '∅': 3,
        'iin': 2, 'am': 2, 'ol': 2, 'dy': 2,
        'or': 1, 'l': 1, 'n': 1, 'an': 1, 'ain': 1
    }

    # ── PARADIGM UNITS (Line 3: 4 period-12 patterns) ──
    line3 = lines['3']
    units = [line3[i:i+12] for i in range(0, 48, 12)]
    # Unit 1: k,k,f,t  Unit 2: k,m,f,t  Unit 3: k,k,p,t  Unit 4: k,k,p,t

    # ── 5 SCRIBE GALLOWS PROFILES (from paradigm units) ──
    # Each scribe "prefers" a paradigm unit → gallows weighting
    SCRIBE_GALLOWS = {
        0: {'∅': 45, 'k': 18, 't': 11, 'f': 8, 'p': 0},   # Unit 1: k+f
        1: {'∅': 45, 'k': 14, 't': 11, 'f': 8, 'p': 0},   # Unit 2: k(+m)+f
        2: {'∅': 45, 'k': 18, 't': 11, 'f': 0, 'p': 8},   # Unit 3: k+p
        3: {'∅': 45, 'k': 15, 't': 11, 'f': 4, 'p': 4},   # Mixed: balanced
        4: {'∅': 45, 'k': 18, 't': 11, 'f': 1, 'p': 7},   # Unit 4: mostly p
    }

    # ── LINE OPENERS (common forms from f57v output) ──
    # Words that appear at/near line starts on f57v
    LINE_OPENERS = ['daiin', 'dal', 'okey', 'otey', 'sheky',
                    'aiin', 'ar', 'sar', 'okal']

    return {
        'prefix': PREFIX_OPTIONS,
        'gallows': GALLOWS_OPTIONS,
        'core': CORE_OPTIONS,
        'suffix': SUFFIX_OPTIONS,
        'scribe_gallows': SCRIBE_GALLOWS,
        'line_openers': LINE_OPENERS,
        'n_scribes': 5,
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 2: THE SCRIBE
# ══════════════════════════════════════════════════════════════════════

class Scribe:
    """
    A 15th-century scribe who has learned the f57v system.

    Production rules (the entire "manual"):
      1. Words = prefix + gallows + core + suffix
      2. Each next word: ALWAYS change the suffix
      3. Half the time, ALSO change one other slot
      4. Every ~5th word, write completely fresh
      5. At line starts, use a common opener
    """

    SLOTS = ['prefix', 'gallows', 'core', 'suffix']

    def __init__(self, spec, scribe_id=0, seed=42):
        self.rng = random.Random(seed)
        self.spec = spec
        self.scribe_id = scribe_id

        # Build weighted choice tables for each slot
        self.slot_options = {}
        self.slot_weights = {}

        for slot in ['prefix', 'core', 'suffix']:
            opts = spec[slot]
            items = list(opts.keys())
            weights = [opts[k] for k in items]
            self.slot_options[slot] = items
            self.slot_weights[slot] = weights

        # Gallows: use scribe-specific profile
        gal = spec['scribe_gallows'].get(scribe_id, spec['gallows'])
        items = list(gal.keys())
        weights = [gal[k] for k in items]
        self.slot_options['gallows'] = items
        self.slot_weights['gallows'] = weights

        self.line_openers = spec['line_openers']

    def _pick(self, slot):
        """Weighted random choice for a slot (weights from f57v counts)."""
        return self.rng.choices(
            self.slot_options[slot],
            weights=self.slot_weights[slot],
            k=1
        )[0]

    def _assemble(self, prefix, gallows, core, suffix):
        """Assemble word from 4 slots. Rule: prefix + gallows + core + suffix."""
        parts = []
        if prefix != '∅':
            parts.append(prefix)
        if gallows != '∅':
            parts.append(gallows)
        if core != '∅':
            parts.append(core)
        if suffix != '∅':
            parts.append(suffix)
        word = ''.join(parts)
        return word if word else 'o'  # fallback: simplest possible word

    def _decompose(self, word):
        """
        Rough decomposition of a word into slots.
        Used for copy-mutate: identify what's in each slot so we can change one.
        Not perfect — but a scribe wouldn't be perfect either.
        """
        remaining = word
        prefix = '∅'
        gallows_val = '∅'
        suffix = '∅'

        # Try to identify prefix
        for pf in ['sh', 'ch', 'qo', 'so', 'of', 'ok', 'ot', 'da', 'de']:
            if remaining.startswith(pf) and len(remaining) > len(pf):
                prefix = pf
                remaining = remaining[len(pf):]
                break
        else:
            if len(remaining) > 1 and remaining[0] in 'odsarv':
                prefix = remaining[0]
                remaining = remaining[1:]

        # Try to identify suffix
        for sf in ['aiin', 'iin', 'ain', 'edy', 'ody', 'dy', 'ey', 'ty',
                    'ar', 'or', 'al', 'ol', 'am', 'an',
                    'y', 'n', 'r', 'l', 's', 'o']:
            if remaining.endswith(sf) and len(remaining) > len(sf):
                suffix = sf
                remaining = remaining[:-len(sf)]
                break

        # Identify gallows in remainder
        for i, c in enumerate(remaining):
            if c in 'ktfp':
                gallows_val = c
                remaining = remaining[:i] + remaining[i+1:]
                break

        core = remaining if remaining else '∅'
        return prefix, gallows_val, core, suffix

    def fresh_word(self):
        """Generate a completely new word from slot options (Rule 1)."""
        p = self._pick('prefix')
        g = self._pick('gallows')
        c = self._pick('core')
        s = self._pick('suffix')
        return self._assemble(p, g, c, s)

    def mutate_word(self, template):
        """
        Copy-mutate: keep template structure, change suffix + maybe one more.
        Rules 2-3: Always change suffix. 50% change one other slot.
        """
        p, g, c, s = self._decompose(template)

        # RULE 2: Always change the suffix
        new_s = s
        while new_s == s:
            new_s = self._pick('suffix')
            if len(self.slot_options['suffix']) <= 1:
                break

        # RULE 3: Half the time, also change one other slot
        new_p, new_g, new_c = p, g, c
        if self.rng.random() < 0.50:
            slot = self.rng.choice(['prefix', 'gallows', 'core'])
            if slot == 'prefix':
                new_p = self._pick('prefix')
            elif slot == 'gallows':
                new_g = self._pick('gallows')
            else:
                new_c = self._pick('core')

        return self._assemble(new_p, new_g, new_c, new_s)

    def write_section(self, n_tokens, tokens_per_line=10):
        """
        Write a section of text. This is the full production process.

        Rules 4-5:
          - Every ~5th word is fresh (20% fresh rate)
          - Line starts get common openers
        """
        corpus = []
        line_pos = 0

        for i in range(n_tokens):
            if line_pos == 0:
                # RULE 5: Line start — use opener or fresh word
                if self.rng.random() < 0.60:
                    word = self.rng.choice(self.line_openers)
                else:
                    word = self.fresh_word()
            elif self.rng.random() < 0.20:
                # RULE 4: Every ~5th word, write fresh
                word = self.fresh_word()
            elif corpus:
                # RULES 2-3: Copy-mutate previous word
                # Look at recent word (not always the immediately previous one)
                lookback = min(5, len(corpus))
                template = self.rng.choice(corpus[-lookback:])
                word = self.mutate_word(template)
            else:
                word = self.fresh_word()

            corpus.append(word)
            line_pos = (line_pos + 1) % tokens_per_line

        return corpus


# ══════════════════════════════════════════════════════════════════════
# STEP 3: MULTI-SCRIBE PRODUCTION
# ══════════════════════════════════════════════════════════════════════

def produce_manuscript(spec, n_tokens=37465, seed=42):
    """
    5 scribes produce text in sections, like the real VMS.

    Allocation: roughly equal shares (VMS has ~5 hands across ~7-8 sections).
    Each scribe writes their portion sequentially.
    """
    rng = random.Random(seed)

    # Split tokens across 5 scribes (roughly equal, some variation)
    base = n_tokens // 5
    allocations = [base] * 5
    remainder = n_tokens - base * 5
    for i in range(remainder):
        allocations[i] += 1

    # Shuffle allocation order (scribes don't write in fixed order)
    order = list(range(5))
    rng.shuffle(order)

    corpus = []
    for idx, scribe_id in enumerate(order):
        scribe = Scribe(spec, scribe_id=scribe_id, seed=seed + scribe_id * 1000 + idx)
        section = scribe.write_section(allocations[idx])
        corpus.extend(section)

    return corpus[:n_tokens]


# ══════════════════════════════════════════════════════════════════════
# STEP 4: RUN AND SCORE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GENERATOR 7: SCRIBAL MANUAL (f57v-only, zero tuning)")
    print("=" * 70)
    print("""
    THE SCRIBE'S MANUAL:
      1. Words = prefix + gallows + core + suffix
      2. Always change the suffix for the next word
      3. Half the time, also change one other part
      4. Every ~5th word, write a completely new word
      5. At line starts, use a common opener
      6. Use your assigned tall-letter pair
    """)

    # Load spec from f57v
    spec = load_f57v_spec('slim.json')
    print(f"Slot options from f57v:")
    for slot in ['prefix', 'gallows', 'core', 'suffix']:
        n = len(spec[slot])
        total = sum(spec[slot].values())
        print(f"  {slot}: {n} options, {total} total weight")

    # Generate with 10 seeds
    n_tokens = 37465
    n_seeds = 10

    all_corpora = []
    for seed in range(n_seeds):
        corpus = produce_manuscript(spec, n_tokens=n_tokens, seed=42 + seed * 100)
        all_corpora.append(corpus)

        types = len(set(corpus))
        ttr = types / len(corpus)
        lens = [len(w) for w in corpus]
        mean_len = sum(lens) / len(lens)
        print(f"  Seed {seed}: {len(corpus)} tokens, {types} types, "
              f"TTR={ttr:.4f}, mean_len={mean_len:.2f}")

    # Save for scoring
    results = {
        'spec': spec,
        'corpora': all_corpora,
        'n_seeds': n_seeds,
        'n_tokens': n_tokens,
    }

    os.makedirs('results', exist_ok=True)
    with open('results/scribal_manual_corpora.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nSaved {n_seeds} corpora to results/scribal_manual_corpora.pkl")

    # ── SCORE ──
    from score_85_metrics import compute_metrics, score_against_vms, CORE_15, ALL_85, TOLERANCES

    with open('results/vms_baseline_85metrics.pkl', 'rb') as f:
        vms_baseline = pickle.load(f)

    all_metrics = []
    for i, corpus in enumerate(all_corpora):
        lines = [corpus[j:j+10] for j in range(0, len(corpus), 10)]
        m = compute_metrics(corpus, lines=lines, seed=42 + i)
        all_metrics.append(m)

    # Median metrics
    median_metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
        if vals:
            median_metrics[key] = float(np.median(vals))

    # Score
    s85 = score_against_vms(median_metrics, vms_baseline, ALL_85, TOLERANCES)
    s15 = score_against_vms(median_metrics, vms_baseline, CORE_15, TOLERANCES)

    print(f"\n{'=' * 70}")
    print(f"SCRIBAL MANUAL SCORES")
    print(f"{'=' * 70}")
    print(f"  Core 15: {s15['n_pass']}/{s15['n_total']}")
    print(f"  Full 90: {s85['n_pass']}/{s85['n_total']}")

    # Compare to hierarchy
    print(f"\n{'Generator':<16} {'Core15':>8} {'Full90':>8} {'Source'}")
    print("-" * 55)
    print(f"{'SCRIBAL MANUAL':<16} {s15['n_pass']:>5}/15 {s85['n_pass']:>5}/90  f57v only, zero tuning ◄◄◄")
    hierarchy = [
        ('f57v-ONLY', 5, 38, 'f57v only, strict copy'),
        ('Bigram', 8, 50, 'Full corpus'),
        ('Scribal', 5, 37, 'Full corpus'),
        ('P70C', 9, 64, 'Full corpus'),
        ('Dual', 7, 58, 'Full corpus'),
        ('Section', 10, 67, 'Full corpus'),
        ('Combined', 10, 66, 'Full corpus'),
    ]
    for name, c, f, src in hierarchy:
        print(f"{name:<16} {c:>5}/15 {f:>5}/90  {src}")

    # Core 15 detail
    print(f"\n{'=' * 70}")
    print(f"CORE 15 DETAIL")
    print(f"{'=' * 70}")
    for m in sorted(s15['details'].keys()):
        d = s15['details'][m]
        status = '✓' if d['pass'] else '✗'
        print(f"  {status} {m:<22} VMS={d['vms']:>8.4f}  gen={d['gen']:>8.4f}  "
              f"Δ={d['delta']:>8.4f}  tol={d['tol']}")

    # Save scores
    score_results = {
        'median_metrics': median_metrics,
        'all_metrics': all_metrics,
        'scores_85': s85,
        'scores_15': s15,
    }
    with open('results/scribal_manual_scores.pkl', 'wb') as f:
        pickle.dump(score_results, f)
    print(f"\nSaved results/scribal_manual_scores.pkl")

    return score_results


if __name__ == '__main__':
    main()
