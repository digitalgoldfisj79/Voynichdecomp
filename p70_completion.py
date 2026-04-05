#!/usr/bin/env python3
"""
P70 COMPLETION: Merge P69 rules + new rules, validate on full corpus
=====================================================================
Takes P69's 109 boundary rules (~60% morphological coverage) and adds
71 new rules derived from full-corpus morphological inventory to reach
~95%+ coverage of all morphological phenomena in Voynichese.
"""

import json
import re
from collections import Counter, defaultdict
import pandas as pd

# ─── LOAD ──────────────────────────────────────────────────────────────

with open('/mnt/user-data/uploads/p69_rules_final.json') as f:
    p69 = json.load(f)
with open('/home/claude/new_rules_raw.json') as f:
    new_rules = json.load(f)
with open('/home/claude/voynich_transcriptions.json') as f:
    corpus_data = json.load(f)

original_rules = p69['rules']

# ─── REBUILD CORPUS ───────────────────────────────────────────────────

def get_section(page_id):
    m = re.match(r'f(\d+)', page_id)
    if not m: return 'Unassigned'
    fnum = int(m.group(1))
    if fnum <= 56: return 'Herbal'
    elif fnum <= 73: return 'Astronomical'
    elif fnum <= 84: return 'Biological'
    elif fnum <= 86: return 'Astronomical'
    elif fnum <= 102: return 'Pharmaceutical'
    elif fnum <= 116: return 'Recipes'
    else: return 'Unassigned'

SECTIONS = ['Herbal', 'Pharmaceutical', 'Recipes', 'Astronomical', 'Biological', 'Unassigned']
TARGET_TID = 'ZLZI'

records = []
for page_id, page in corpus_data['pages'].items():
    sec = get_section(page_id)
    for line_id, line in page['lines'].items():
        for src_key, src in line['sources'].items():
            if src['transcriber_id'] == TARGET_TID:
                norm = src['views'].get('normalized', {})
                text = norm.get('text', '') if isinstance(norm, dict) else ''
                if text.strip():
                    words = [w for w in text.strip().split() if w not in ('*', '?', '-', '%')]
                    for wi, w in enumerate(words):
                        prev_w = words[wi-1] if wi > 0 else ''
                        next_w = words[wi+1] if wi < len(words)-1 else ''
                        records.append((page_id, sec, line_id, wi, w, prev_w, next_w))

all_words = [r[4] for r in records]
total_tokens = len(all_words)
unique_types = set(all_words)
freq = Counter(all_words)

print("=" * 72)
print("P70 RULE SET COMPLETION & VALIDATION")
print("=" * 72)
print(f"Corpus: {total_tokens:,} tokens, {len(unique_types):,} types")
print(f"Original P69 rules: {len(original_rules)}")
print(f"New completion rules: {len(new_rules)}")

# ─── MERGE RULES ──────────────────────────────────────────────────────

merged_rules = list(original_rules) + list(new_rules)
print(f"Merged total: {len(merged_rules)}")

# ─── MORPHOLOGICAL COVERAGE TEST ──────────────────────────────────────
# For each character in each word, check if ANY rule "covers" it
# Coverage = fraction of character positions explained by at least one rule

def rules_covering_word(word, prev_w, next_w, ruleset):
    """Return set of (char_position, rule_id) pairs for all covered positions."""
    covered = set()  # set of character indices in `word`
    matching_rules = []
    
    for r in ruleset:
        pat = r['pattern'].replace('|', '')
        kind = r['kind']
        matched = False
        
        if kind == 'prefix':
            if word.startswith(pat):
                for i in range(len(pat)):
                    covered.add(i)
                matched = True
                
        elif kind == 'suffix':
            if word.endswith(pat):
                start = len(word) - len(pat)
                for i in range(start, len(word)):
                    covered.add(i)
                matched = True
                
        elif kind == 'chargram':
            idx = word.find(pat)
            while idx != -1:
                for i in range(idx, idx + len(pat)):
                    covered.add(i)
                matched = True
                idx = word.find(pat, idx + 1)
                
        elif kind == 'pair':
            # Intra-word match
            if pat in word:
                idx = word.find(pat)
                while idx != -1:
                    for i in range(idx, idx + len(pat)):
                        covered.add(i)
                    matched = True
                    idx = word.find(pat, idx + 1)
            
            # Cross-word boundary match
            if '|' in r['pattern']:
                left, right = r['pattern'].split('|', 1)
                if left and right:
                    if prev_w.endswith(left) and word.startswith(right):
                        for i in range(len(right)):
                            covered.add(i)
                        matched = True
                    if word.endswith(left) and next_w.startswith(right):
                        start = len(word) - len(left)
                        for i in range(start, len(word)):
                            covered.add(i)
                        matched = True
                elif left and not right:
                    # Pattern like "ok|" - left boundary
                    if word.endswith(left):
                        start = len(word) - len(left)
                        for i in range(start, len(word)):
                            covered.add(i)
                        matched = True
                elif right and not left:
                    # Pattern like "|che" - right boundary
                    if word.startswith(right):
                        for i in range(len(right)):
                            covered.add(i)
                        matched = True
        
        if matched:
            matching_rules.append(r['rule_id'])
    
    return covered, matching_rules


print("\n" + "─" * 72)
print("COVERAGE ANALYSIS")
print("─" * 72)

# Test coverage on full corpus (sample for speed)
import random
random.seed(42)

# Use all records for accurate measurement
total_chars = 0
covered_chars_p69 = 0
covered_chars_merged = 0
words_fully_covered_p69 = 0
words_fully_covered_merged = 0
words_partially_covered_p69 = 0
words_partially_covered_merged = 0
words_uncovered_p69 = 0
words_uncovered_merged = 0

# Track which rule kinds cover what
kind_coverage = defaultdict(int)

# Process all tokens (may take a moment)
for idx, (page_id, sec, line_id, wi, w, prev_w, next_w) in enumerate(records):
    wlen = len(w)
    total_chars += wlen
    
    cov_p69, _ = rules_covering_word(w, prev_w, next_w, original_rules)
    cov_merged, rules_matched = rules_covering_word(w, prev_w, next_w, merged_rules)
    
    covered_chars_p69 += len(cov_p69)
    covered_chars_merged += len(cov_merged)
    
    if len(cov_p69) == wlen:
        words_fully_covered_p69 += 1
    elif len(cov_p69) > 0:
        words_partially_covered_p69 += 1
    else:
        words_uncovered_p69 += 1
    
    if len(cov_merged) == wlen:
        words_fully_covered_merged += 1
    elif len(cov_merged) > 0:
        words_partially_covered_merged += 1
    else:
        words_uncovered_merged += 1

print(f"\n{'Metric':<40} {'P69 Only':>12} {'P70 Merged':>12} {'Delta':>10}")
print("─" * 78)

pct_p69_char = covered_chars_p69 / total_chars * 100
pct_merged_char = covered_chars_merged / total_chars * 100
print(f"{'Character coverage':<40} {pct_p69_char:>11.1f}% {pct_merged_char:>11.1f}% {pct_merged_char - pct_p69_char:>+9.1f}%")

pct_p69_full = words_fully_covered_p69 / total_tokens * 100
pct_merged_full = words_fully_covered_merged / total_tokens * 100
print(f"{'Words fully covered':<40} {pct_p69_full:>11.1f}% {pct_merged_full:>11.1f}% {pct_merged_full - pct_p69_full:>+9.1f}%")

pct_p69_any = (words_fully_covered_p69 + words_partially_covered_p69) / total_tokens * 100
pct_merged_any = (words_fully_covered_merged + words_partially_covered_merged) / total_tokens * 100
print(f"{'Words with any coverage':<40} {pct_p69_any:>11.1f}% {pct_merged_any:>11.1f}% {pct_merged_any - pct_p69_any:>+9.1f}%")

pct_p69_none = words_uncovered_p69 / total_tokens * 100
pct_merged_none = words_uncovered_merged / total_tokens * 100
print(f"{'Words with ZERO coverage':<40} {pct_p69_none:>11.1f}% {pct_merged_none:>11.1f}% {pct_merged_none - pct_p69_none:>+9.1f}%")

print(f"\n{'Total characters':<40} {total_chars:>12,}")
print(f"{'Total tokens':<40} {total_tokens:>12,}")
print(f"{'Rules in set':<40} {len(original_rules):>12} {len(merged_rules):>12} {len(new_rules):>+10}")

# ─── FIND REMAINING UNCOVERED WORDS ───────────────────────────────────

print("\n" + "─" * 72)
print("REMAINING UNCOVERED WORDS (top 30 by frequency)")
print("─" * 72)

uncovered_words = Counter()
uncovered_chars_remaining = Counter()

for page_id, sec, line_id, wi, w, prev_w, next_w in records:
    cov, _ = rules_covering_word(w, prev_w, next_w, merged_rules)
    uncovered_positions = set(range(len(w))) - cov
    if uncovered_positions:
        uncovered_words[w] += 1
        for pos in uncovered_positions:
            uncovered_chars_remaining[w[pos]] += 1

print(f"\nWords still partially/fully uncovered: {len(uncovered_words)} types")
print(f"\n{'Word':<16} {'Freq':>6} {'Uncov#':>6}  Uncovered positions")
print("─" * 60)
for w, cnt in uncovered_words.most_common(30):
    cov, _ = rules_covering_word(w, '', '', merged_rules)
    uncov_pos = set(range(len(w))) - cov
    uncov_chars = ''.join(w[i] if i in uncov_pos else '·' for i in range(len(w)))
    print(f"  {w:<14} {freq[w]:>6} {cnt:>6}  {uncov_chars}")

print(f"\nUncovered character residuals:")
for c, cnt in uncovered_chars_remaining.most_common(15):
    print(f"  '{c}': {cnt:>6} positions still uncovered")

# ─── SECTION-LEVEL COVERAGE BREAKDOWN ─────────────────────────────────

print("\n" + "─" * 72)
print("SECTION-LEVEL COVERAGE (P70 merged)")
print("─" * 72)

sec_stats = defaultdict(lambda: {'total_chars': 0, 'covered_chars': 0, 
                                   'total_words': 0, 'full_words': 0})

for page_id, sec, line_id, wi, w, prev_w, next_w in records:
    wlen = len(w)
    sec_stats[sec]['total_chars'] += wlen
    sec_stats[sec]['total_words'] += 1
    
    cov, _ = rules_covering_word(w, prev_w, next_w, merged_rules)
    sec_stats[sec]['covered_chars'] += len(cov)
    if len(cov) == wlen:
        sec_stats[sec]['full_words'] += 1

print(f"\n{'Section':<18} {'Tokens':>8} {'Char Cov%':>10} {'Word Cov%':>10}")
print("─" * 50)
sec_rows = []
for sec in SECTIONS:
    s = sec_stats[sec]
    if s['total_chars'] == 0: continue
    char_pct = s['covered_chars'] / s['total_chars'] * 100
    word_pct = s['full_words'] / s['total_words'] * 100
    print(f"  {sec:<16} {s['total_words']:>8,} {char_pct:>9.1f}% {word_pct:>9.1f}%")
    sec_rows.append({'Section': sec, 'Tokens': s['total_words'],
                     'Char_Coverage%': round(char_pct, 1), 'Word_Coverage%': round(word_pct, 1)})

df_sections = pd.DataFrame(sec_rows)

# ─── RULE-BY-RULE HIT RATES ───────────────────────────────────────────

print("\n" + "─" * 72)
print("RULE-BY-RULE HIT RATES (on full corpus)")
print("─" * 72)

rule_hits = Counter()
for r in merged_rules:
    pat = r['pattern'].replace('|', '')
    kind = r['kind']
    hits = 0
    for w in all_words:
        if kind == 'prefix' and w.startswith(pat): hits += 1
        elif kind == 'suffix' and w.endswith(pat): hits += 1
        elif kind == 'chargram' and pat in w: hits += 1
        elif kind == 'pair' and pat in w: hits += 1
    rule_hits[r['rule_id']] = hits

# New rules sorted by hit rate
print("\nNEW rules by hit rate (top 30):")
rule_rows = []
for r in sorted(new_rules, key=lambda x: -rule_hits.get(x['rule_id'], 0))[:30]:
    h = rule_hits[r['rule_id']]
    print(f"  {r['rule_id']:<35} {h:>6,} hits  w={r['base_weight']}")
    rule_rows.append({'Rule_ID': r['rule_id'], 'Hits': h, 'Weight': r['base_weight'],
                      'Kind': r['kind'], 'Source': 'NEW'})

# Zero-hit rules
zero_new = [r for r in new_rules if rule_hits.get(r['rule_id'], 0) == 0]
print(f"\nNew rules with 0 hits: {len(zero_new)}")
for r in zero_new:
    print(f"  {r['rule_id']}")

# ─── OUTPUT: MERGED RULE SET ──────────────────────────────────────────

# Tag original vs new
for r in original_rules:
    r['_source'] = 'p69_original'

output = {
    'schema': 'P70-completion',
    'created_from': {
        'p69_rules': len(original_rules),
        'p70_new_rules': len(new_rules),
        'total': len(merged_rules),
        'corpus': 'ZLZI (ZL_ivtff_2b) from voynich_transcriptions.json',
        'tokens': total_tokens,
        'types': len(unique_types)
    },
    'coverage_metrics': {
        'p69_only': {
            'char_coverage_pct': round(pct_p69_char, 2),
            'word_full_coverage_pct': round(pct_p69_full, 2),
            'word_any_coverage_pct': round(pct_p69_any, 2),
        },
        'p70_merged': {
            'char_coverage_pct': round(pct_merged_char, 2),
            'word_full_coverage_pct': round(pct_merged_full, 2),
            'word_any_coverage_pct': round(pct_merged_any, 2),
        }
    },
    'rules': merged_rules
}

json_path = '/home/claude/p70_rules_complete.json'
with open(json_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\n\nMerged rules written: {json_path}")

# ─── OUTPUT: XLSX WITH ALL ANALYSIS ───────────────────────────────────

# Build full rule table
all_rule_rows = []
for r in merged_rules:
    all_rule_rows.append({
        'Rule_ID': r['rule_id'],
        'Kind': r['kind'],
        'Pattern': r['pattern'],
        'Pred_Side': r['pred_side'],
        'Base_Weight': r['base_weight'],
        'Hits': rule_hits.get(r['rule_id'], 0),
        'Source': r.get('_source', 'unknown'),
        'Allow': ', '.join(r.get('allow', [])),
        'Deny': ', '.join(r.get('deny', [])),
    })

df_rules = pd.DataFrame(all_rule_rows)

# Coverage comparison
cov_rows = [
    {'Metric': 'Character coverage %', 'P69_Only': round(pct_p69_char, 1), 'P70_Merged': round(pct_merged_char, 1)},
    {'Metric': 'Words fully covered %', 'P69_Only': round(pct_p69_full, 1), 'P70_Merged': round(pct_merged_full, 1)},
    {'Metric': 'Words any coverage %', 'P69_Only': round(pct_p69_any, 1), 'P70_Merged': round(pct_merged_any, 1)},
    {'Metric': 'Words zero coverage %', 'P69_Only': round(pct_p69_none, 1), 'P70_Merged': round(pct_merged_none, 1)},
    {'Metric': 'Rule count', 'P69_Only': len(original_rules), 'P70_Merged': len(merged_rules)},
]
df_coverage = pd.DataFrame(cov_rows)

# Uncovered words
uncov_rows = []
for w, cnt in uncovered_words.most_common(200):
    cov, _ = rules_covering_word(w, '', '', merged_rules)
    uncov_pos = set(range(len(w))) - cov
    uncov_chars = ''.join(w[i] if i in uncov_pos else '·' for i in range(len(w)))
    uncov_rows.append({'Word': w, 'Freq': freq[w], 'Uncov_Tokens': cnt, 'Uncov_Chars': uncov_chars})
df_uncovered = pd.DataFrame(uncov_rows)

xlsx_path = '/home/claude/p70_completion_analysis.xlsx'
with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
    df_coverage.to_excel(writer, sheet_name='Coverage_Comparison', index=False)
    df_rules.to_excel(writer, sheet_name='All_Rules', index=False)
    df_sections.to_excel(writer, sheet_name='Section_Coverage', index=False)
    df_uncovered.to_excel(writer, sheet_name='Uncovered_Words', index=False)

print(f"Analysis XLSX written: {xlsx_path}")

print("\n" + "=" * 72)
print("P70 COMPLETION DONE")
print("=" * 72)
