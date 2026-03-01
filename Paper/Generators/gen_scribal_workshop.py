#!/usr/bin/env python3
"""
Generator 8: The Scribal Workshop
═══════════════════════════════════
ALL parameters from f57v. Zero corpus tuning.
Simple rules 5 scribes could learn in a morning.

THE ARCHITECTURE (from f57v):
  - Line 3 teaches: 13 skeleton chars, 4 paradigm units, gallows substitution
  - Line 5 teaches: skeleton frame first, add dressing {a,c,e,h,i,s} around it
  - Rings 3-4 teach: 60 example words (mean length 4.97 ≈ VMS 4.98)

THE SCRIBE'S 5 RULES:
  1. Words are skeleton frames dressed with vowels and connectors
  2. To write the next word: keep the frame, change the ENDING
  3. Sometimes also change the BEGINNING or the TALL LETTER
  4. Every few words, pick a fresh template from the reference sheet
  5. At line starts, use a common opening form

WHAT'S DIFFERENT FROM GEN 7:
  - Templates are f57v's actual output words (correct length distribution)
  - Mutation operates on skeleton/dressing structure (not abstract slots)
  - Word length emerges naturally from template pool, not from assembly
"""

import json
import random
import pickle
import numpy as np
import os
from collections import Counter


# ══════════════════════════════════════════════════════════════════════
# STEP 1: EXTRACT f57v SPEC
# ══════════════════════════════════════════════════════════════════════

SKELETON = set('dfklmnoprtvxy')   # 13 chars from Line 3
DRESSING = set('acehiqs')          # 7 chars: on f57v but NOT in Line 3
GALLOWS  = set('ktfp')             # 4 tall letters (paradigm positions 6-9)
SUFFIX_SKEL = set('dlrmny')        # suffix skeleton chars (Line 3 pos 1-5, 10-11)


def load_f57v_spec(slim_path='slim.json'):
    """Extract everything from f57v — nothing else."""
    with open(slim_path) as f:
        data = json.load(f)

    f57v = data['pages']['f57v']
    lines = {}
    for ln, line_data in f57v.items():
        t = line_data.get('t', {})
        text = t.get('TTLI', '')
        lines[ln] = text.split() if text else []

    # ── TEMPLATE POOL: all multi-char tokens from f57v ──
    templates = []
    for ln in ['1','2','4','5','6','7','8','9','10','11','12','13']:
        for tok in lines.get(ln, []):
            if len(tok) > 1:
                templates.append(tok)
    template_pool = sorted(set(templates))

    # ── SUFFIX ENDINGS: how f57v words end ──
    # Observed endings, ordered longest-first for greedy matching
    suffix_endings = [
        'aiin', 'edy', 'ody', 'aly', 'oly', 'chy',
        'iin', 'ain', 'een',
        'dy', 'ey', 'ty', 'ky', 'ry', 'ly',
        'ar', 'or', 'al', 'ol', 'am', 'an', 'in',
        'as', 'es', 'os',
        'y', 'n', 'r', 'l', 's', 'o', 'd'
    ]
    # Count from f57v output for weights
    suffix_counts = Counter()
    output_toks = []
    for ln in ['2', '4']:
        output_toks.extend(lines[ln])
    for ln in ['5']:
        output_toks.extend(t for t in lines[ln] if len(t) > 1)

    for w in output_toks:
        if len(w) <= 1:
            continue
        for sf in suffix_endings:
            if w.endswith(sf) and len(w) > len(sf):
                suffix_counts[sf] += 1
                break

    # ── PREFIX BEGINNINGS: how f57v words start ──
    prefix_beginnings = [
        'sh', 'ch', 'qo', 'so', 'of', 'ok', 'ot', 'da', 'de',
        'o', 'd', 's', 'a', 'r', 'v'
    ]
    prefix_counts = Counter()
    for w in output_toks:
        if len(w) <= 1:
            continue
        found = False
        for pf in prefix_beginnings:
            if w.startswith(pf) and len(w) > len(pf):
                prefix_counts[pf] += 1
                found = True
                break
        if not found:
            prefix_counts['∅'] += 1

    # ── PARADIGM: gallows weights from Line 3 units ──
    # k=7/16, t=4/16, f=2/16, p=2/16, m=1/16
    paradigm_gallows = {'k': 7, 't': 4, 'f': 2, 'p': 2}

    # ── SCRIBE PROFILES (from 4 paradigm units) ──
    scribe_gallows = {
        0: {'k': 7, 't': 4, 'f': 4, 'p': 1},    # Unit 1: k+f preference
        1: {'k': 5, 't': 4, 'f': 4, 'p': 1},    # Unit 2: less k, still f
        2: {'k': 7, 't': 4, 'f': 1, 'p': 4},    # Unit 3: k+p preference
        3: {'k': 6, 't': 4, 'f': 2, 'p': 2},    # Mixed: balanced
        4: {'k': 7, 't': 4, 'f': 1, 'p': 4},    # Unit 4: k+p (≈Unit 3)
    }

    # ── LINE OPENERS: forms seen at/near line starts on f57v ──
    line_openers = ['daiin', 'dal', 'okey', 'otey', 'sheky',
                    'aiin', 'ar', 'sar', 'okal', 'daram',
                    'okees', 'shes', 'ofchey', 'dkedar']

    return {
        'template_pool': template_pool,
        'suffix_endings': suffix_endings,
        'suffix_counts': dict(suffix_counts),
        'prefix_beginnings': prefix_beginnings,
        'prefix_counts': dict(prefix_counts),
        'paradigm_gallows': paradigm_gallows,
        'scribe_gallows': scribe_gallows,
        'line_openers': line_openers,
    }


# ══════════════════════════════════════════════════════════════════════
# STEP 2: WORD SURGERY (skeleton/dressing operations)
# ══════════════════════════════════════════════════════════════════════

def find_suffix_region(word):
    """
    Find where the suffix starts in a word.
    Suffix = everything from the last skeleton-suffix char onward
    (possibly including trailing dressing).
    A scribe would see this as "the ending part."
    """
    last_skel_pos = -1
    for i in range(len(word) - 1, -1, -1):
        if word[i] in SUFFIX_SKEL:
            last_skel_pos = i
            break

    if last_skel_pos == -1:
        # No suffix skeleton char — suffix is last 1-2 chars
        return max(0, len(word) - 2)

    # Include any dressing before this final skeleton char
    # that is clearly part of the ending (e.g., 'aiin' → the 'a' before 'n')
    start = last_skel_pos
    while start > 0 and word[start-1] in DRESSING:
        # But don't eat into the core — stop if we hit skeleton
        if start > 1 and word[start-2] in SKELETON:
            break
        start -= 1

    # Don't take more than half the word as suffix
    min_stem = max(2, len(word) // 2)
    start = max(start, min_stem)

    return start


def find_prefix_region(word):
    """
    Find where the prefix ends.
    Prefix = initial dressing + possibly first skeleton char.
    A scribe would see this as "the beginning part."
    """
    # Common multi-char prefixes (from f57v)
    for pf in ['sh', 'ch', 'qo', 'so', 'of', 'ok', 'ot', 'da', 'de']:
        if word.startswith(pf) and len(word) > len(pf) + 1:
            return len(pf)

    # Single char prefix if it's a common opener
    if len(word) > 2 and word[0] in 'odsarv':
        return 1

    return 0  # no clear prefix


def find_gallows(word, start_from=0):
    """Find the first gallows char in word after start_from."""
    for i in range(start_from, len(word)):
        if word[i] in GALLOWS:
            return i, word[i]
    return -1, None


# ══════════════════════════════════════════════════════════════════════
# STEP 3: THE SCRIBE
# ══════════════════════════════════════════════════════════════════════

class Scribe:
    """
    A scribe trained on the f57v reference sheet.

    Rules (the complete manual):
      1. Start with a template word from the reference sheet
      2. For each new word: change the ENDING (suffix region)
      3. Half the time, also change the BEGINNING or TALL LETTER
      4. Every ~5 words, pick a completely fresh template
      5. At line starts, use a common opener
    """

    def __init__(self, spec, scribe_id=0, seed=42):
        self.rng = random.Random(seed)
        self.spec = spec
        self.scribe_id = scribe_id
        self.template_pool = spec['template_pool']

        # Build suffix replacement table (weighted by f57v counts)
        self.suffix_options = list(spec['suffix_counts'].keys())
        self.suffix_weights = [spec['suffix_counts'][s] for s in self.suffix_options]
        # Add a few more from the endings list that weren't counted
        for sf in spec['suffix_endings']:
            if sf not in spec['suffix_counts']:
                self.suffix_options.append(sf)
                self.suffix_weights.append(1)  # minimal weight

        # Build prefix replacement table
        self.prefix_options = list(spec['prefix_counts'].keys())
        self.prefix_weights = [spec['prefix_counts'][p] for p in self.prefix_options]

        # Gallows table for this scribe
        gal = spec['scribe_gallows'].get(scribe_id, spec['paradigm_gallows'])
        self.gallows_options = list(gal.keys())
        self.gallows_weights = [gal[g] for g in self.gallows_options]

        self.line_openers = spec['line_openers']

    def _pick_suffix(self):
        return self.rng.choices(self.suffix_options, self.suffix_weights, k=1)[0]

    def _pick_prefix(self):
        p = self.rng.choices(self.prefix_options, self.prefix_weights, k=1)[0]
        return '' if p == '∅' else p

    def _pick_gallows(self):
        return self.rng.choices(self.gallows_options, self.gallows_weights, k=1)[0]

    def _pick_template(self):
        return self.rng.choice(self.template_pool)

    def change_suffix(self, word):
        """
        Rule 2: Change the ending.
        Find the suffix region, replace it with a new suffix.
        """
        cut = find_suffix_region(word)
        stem = word[:cut]
        if not stem:
            stem = word[0] if word else 'o'

        new_suffix = self._pick_suffix()

        # Ensure we don't create empty word or exact duplicate
        result = stem + new_suffix
        return result if result else word

    def change_prefix(self, word):
        """
        Rule 3a: Change the beginning.
        Find the prefix region, replace it.
        """
        cut = find_prefix_region(word)
        remainder = word[cut:]
        if not remainder:
            remainder = word

        new_prefix = self._pick_prefix()
        result = new_prefix + remainder
        return result if result else word

    def change_gallows(self, word):
        """
        Rule 3b: Change the tall letter.
        Find the gallows char, swap it for a different one.
        """
        prefix_end = find_prefix_region(word)
        gpos, old_g = find_gallows(word, prefix_end)

        new_g = self._pick_gallows()

        if gpos >= 0:
            # Replace existing gallows
            result = word[:gpos] + new_g + word[gpos+1:]
        else:
            # No gallows — insert one after prefix
            # (but only sometimes — ~40% of VMS words lack gallows)
            if self.rng.random() < 0.35:
                insert_pos = prefix_end if prefix_end > 0 else min(1, len(word))
                result = word[:insert_pos] + new_g + word[insert_pos:]
            else:
                result = word  # keep gallows-free
        return result if result else word

    def remove_gallows(self, word):
        """Sometimes remove a gallows entirely (make word gallows-free)."""
        prefix_end = find_prefix_region(word)
        gpos, old_g = find_gallows(word, prefix_end)
        if gpos >= 0:
            return word[:gpos] + word[gpos+1:]
        return word

    def mutate(self, template):
        """
        Full mutation: always change suffix, sometimes change one more thing.
        This is what a scribe does for each new word.
        """
        # RULE 2: Always change the suffix
        word = self.change_suffix(template)

        # RULE 3: Half the time, also change one other thing
        if self.rng.random() < 0.50:
            what = self.rng.random()
            if what < 0.35:
                word = self.change_prefix(word)
            elif what < 0.70:
                word = self.change_gallows(word)
            else:
                # Remove gallows (makes shorter/simpler word)
                word = self.remove_gallows(word)

        return word

    def write_section(self, n_tokens, tokens_per_line=10):
        """
        Write a section of text.

        Rules 4-5:
          - ~20% fresh template (Rule 4, from Line 5 ratio: 6/26 assembled)
          - Line starts get openers (Rule 5)
          - Rest: copy-mutate from recent words (lookback ~5)
        """
        corpus = []
        line_pos = 0

        for i in range(n_tokens):
            if line_pos == 0:
                # RULE 5: Line start
                if self.rng.random() < 0.55:
                    word = self.rng.choice(self.line_openers)
                else:
                    word = self.mutate(self._pick_template())

            elif self.rng.random() < 0.20:
                # RULE 4: Fresh template
                word = self._pick_template()
                # Usually mutate it too (don't just copy the reference)
                if self.rng.random() < 0.70:
                    word = self.mutate(word)

            elif corpus:
                # RULES 2-3: Copy-mutate from recent word
                lookback = min(5, len(corpus))
                template = self.rng.choice(corpus[-lookback:])
                word = self.mutate(template)

            else:
                word = self._pick_template()

            # Sanity: ensure non-empty
            if not word:
                word = self._pick_template()

            corpus.append(word)
            line_pos = (line_pos + 1) % tokens_per_line

        return corpus


# ══════════════════════════════════════════════════════════════════════
# STEP 4: MULTI-SCRIBE PRODUCTION
# ══════════════════════════════════════════════════════════════════════

def produce_manuscript(spec, n_tokens=37465, seed=42):
    """
    5 scribes produce text in sections.
    Each scribe gets roughly equal allocation, writes sequentially.
    """
    rng = random.Random(seed)

    # Allocate tokens across 5 scribes
    base = n_tokens // 5
    allocations = [base] * 5
    for i in range(n_tokens - base * 5):
        allocations[i] += 1

    # Scribes write in shuffled order
    order = list(range(5))
    rng.shuffle(order)

    corpus = []
    for idx, scribe_id in enumerate(order):
        scribe_seed = seed + scribe_id * 1000 + idx * 7
        scribe = Scribe(spec, scribe_id=scribe_id, seed=scribe_seed)
        section = scribe.write_section(allocations[idx])
        corpus.extend(section)

    return corpus[:n_tokens]


# ══════════════════════════════════════════════════════════════════════
# STEP 5: RUN AND SCORE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("GENERATOR 8: THE SCRIBAL WORKSHOP")
    print("=" * 70)
    print("""
    f57v-derived, zero tuning, 5 simple rules.
    Templates from f57v output (mean len=4.97 ≈ VMS 4.98).
    Mutation on skeleton/dressing structure.
    5 scribes with paradigm-derived gallows preferences.
    """)

    spec = load_f57v_spec('slim.json')
    print(f"Template pool: {len(spec['template_pool'])} words "
          f"(mean len={np.mean([len(t) for t in spec['template_pool']]):.2f})")
    print(f"Suffix options: {len(spec['suffix_counts'])} weighted")
    print(f"Prefix options: {len(spec['prefix_counts'])} weighted")

    # Generate 10 seeds
    n_tokens = 37465
    n_seeds = 10

    all_corpora = []
    for seed in range(n_seeds):
        corpus = produce_manuscript(spec, n_tokens=n_tokens, seed=42 + seed * 100)
        all_corpora.append(corpus)

        types = len(set(corpus))
        ttr = types / len(corpus)
        lens = [len(w) for w in corpus]
        mean_len = np.mean(lens)
        std_len = np.std(lens)

        # Word-length autocorrelation (quick calc)
        ml = np.array(lens, dtype=float)
        ml_centered = ml - ml.mean()
        if np.std(ml) > 0:
            autocorr = np.corrcoef(ml_centered[:-1], ml_centered[1:])[0, 1]
        else:
            autocorr = 0.0

        print(f"  Seed {seed}: tokens={len(corpus)}, types={types}, "
              f"TTR={ttr:.4f}, μ_len={mean_len:.2f}, σ_len={std_len:.2f}, "
              f"wl_autocorr={autocorr:.3f}")

    # Save corpora
    os.makedirs('results', exist_ok=True)
    with open('results/scribal_workshop_corpora.pkl', 'wb') as f:
        pickle.dump({
            'spec': spec,
            'corpora': all_corpora,
            'n_seeds': n_seeds,
            'n_tokens': n_tokens,
        }, f)

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

    # Median metrics
    median_metrics = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics if isinstance(m[key], (int, float))]
        if vals:
            median_metrics[key] = float(np.median(vals))

    s85 = score_against_vms(median_metrics, vms_baseline, ALL_85, TOLERANCES)
    s15 = score_against_vms(median_metrics, vms_baseline, CORE_15, TOLERANCES)

    print(f"\n{'=' * 70}")
    print(f"SCRIBAL WORKSHOP SCORES")
    print(f"{'=' * 70}")
    print(f"  Core 15: {s15['n_pass']}/{s15['n_total']}")
    print(f"  Full 90: {s85['n_pass']}/{s85['n_total']}")

    # Full comparison
    print(f"\n{'Generator':<18} {'Core15':>7} {'Full90':>7} {'Source'}")
    print("-" * 65)
    print(f"{'SCRIBAL WORKSHOP':<18} {s15['n_pass']:>4}/15 {s85['n_pass']:>4}/90  "
          f"f57v only, skeleton/dressing ◄◄◄")
    hierarchy = [
        ('Scribal Manual', 7, 43, 'f57v only, slot assembly'),
        ('f57v-ONLY', 5, 38, 'f57v only, strict copy'),
        ('Bigram', 8, 50, 'Full corpus'),
        ('Scribal', 5, 37, 'Full corpus'),
        ('P70C', 9, 64, 'Full corpus'),
        ('Dual', 7, 58, 'Full corpus'),
        ('Section', 10, 67, 'Full corpus'),
        ('Combined', 10, 66, 'Full corpus'),
    ]
    for name, c, f, src in hierarchy:
        print(f"{name:<18} {c:>4}/15 {f:>4}/90  {src}")

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
    with open('results/scribal_workshop_scores.pkl', 'wb') as f:
        pickle.dump(score_results, f)
    print(f"\nSaved results/scribal_workshop_scores.pkl")

    # Show a sample of generated text
    print(f"\n{'=' * 70}")
    print(f"SAMPLE OUTPUT (first 50 words, seed 0)")
    print(f"{'=' * 70}")
    sample = all_corpora[0][:50]
    for i in range(0, 50, 10):
        line = ' '.join(sample[i:i+10])
        print(f"  {line}")

    return score_results


if __name__ == '__main__':
    main()
