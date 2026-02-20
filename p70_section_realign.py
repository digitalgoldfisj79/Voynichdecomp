#!/usr/bin/env python3
"""
P70 SECTION REALIGNMENT
========================
Replaces the crude folio-number-range approximation with the canonical
per-page section mapping from Voynich.nu (established in Paper 1).

CANONICAL SECTIONS (from user's specification + Voynich.nu):
  Herbal-A:       Quires 1-7 (f1-f56) + f57r + f65-f66v — Currier A
  Herbal-B:       f87, f90, f93-f96 — Currier B herbal pages
  Astronomical:   f67r, f68r pages (circular star diagrams)
  Cosmological:   f57v, f67v2, f68v1, f68v3, f69, f70r (circular cosmological)
  Zodiac:         f70v-f73 (zodiacal roundels)
  Rosettes:       f85-f86 (the large foldout)
  Balneological:  f75-f84 (bathing nymphs)
  Pharmaceutical: f88-f89, f99-f102 (roots in jars)
  Stars:          f103-f116 (text-heavy with star paragraphs, "Recipes")
  Text-only:      f1r, f58, f66r, f76r, f85r1, f86v5-6, f116v

OLD SECTIONS being replaced:
  Herbal → split into Herbal-A + Herbal-B
  Astronomical → split into Astronomical + Cosmological + Zodiac
  Biological → renamed Balneological
  Pharmaceutical → kept (but pages now correctly assigned)
  Recipes → renamed Stars
  Unassigned → eliminated (all pages now assigned)
"""

import json, re, csv
from collections import Counter, defaultdict
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════
# CANONICAL PER-PAGE SECTION MAP (from Voynich.nu, Paper 1)
# ═══════════════════════════════════════════════════════════════════════

SECTION_MAP = {
    # Quire 1 (f1-f8)
    'f1r': 'text-only', 'f1v': 'Herbal-A',
    'f2r': 'Herbal-A', 'f2v': 'Herbal-A',
    'f3r': 'Herbal-A', 'f3v': 'Herbal-A',
    'f4r': 'Herbal-A', 'f4v': 'Herbal-A',
    'f5r': 'Herbal-A', 'f5v': 'Herbal-A',
    'f6r': 'Herbal-A', 'f6v': 'Herbal-A',
    'f7r': 'Herbal-A', 'f7v': 'Herbal-A',
    'f8r': 'Herbal-A', 'f8v': 'Herbal-A',
    # Quire 2 (f9-f16, f12 missing)
    'f9r': 'Herbal-A', 'f9v': 'Herbal-A',
    'f10r': 'Herbal-A', 'f10v': 'Herbal-A',
    'f11r': 'Herbal-A', 'f11v': 'Herbal-A',
    'f13r': 'Herbal-A', 'f13v': 'Herbal-A',
    'f14r': 'Herbal-A', 'f14v': 'Herbal-A',
    'f15r': 'Herbal-A', 'f15v': 'Herbal-A',
    'f16r': 'Herbal-A', 'f16v': 'Herbal-A',
    # Quire 3 (f17-f24)
    'f17r': 'Herbal-A', 'f17v': 'Herbal-A',
    'f18r': 'Herbal-A', 'f18v': 'Herbal-A',
    'f19r': 'Herbal-A', 'f19v': 'Herbal-A',
    'f20r': 'Herbal-A', 'f20v': 'Herbal-A',
    'f21r': 'Herbal-A', 'f21v': 'Herbal-A',
    'f22r': 'Herbal-A', 'f22v': 'Herbal-A',
    'f23r': 'Herbal-A', 'f23v': 'Herbal-A',
    'f24r': 'Herbal-A', 'f24v': 'Herbal-A',
    # Quire 4 (f25-f32)
    'f25r': 'Herbal-A', 'f25v': 'Herbal-A',
    'f26r': 'Herbal-A', 'f26v': 'Herbal-A',
    'f27r': 'Herbal-A', 'f27v': 'Herbal-A',
    'f28r': 'Herbal-A', 'f28v': 'Herbal-A',
    'f29r': 'Herbal-A', 'f29v': 'Herbal-A',
    'f30r': 'Herbal-A', 'f30v': 'Herbal-A',
    'f31r': 'Herbal-A', 'f31v': 'Herbal-A',
    'f32r': 'Herbal-A', 'f32v': 'Herbal-A',
    # Quire 5 (f33-f40)
    'f33r': 'Herbal-A', 'f33v': 'Herbal-A',
    'f34r': 'Herbal-A', 'f34v': 'Herbal-A',
    'f35r': 'Herbal-A', 'f35v': 'Herbal-A',
    'f36r': 'Herbal-A', 'f36v': 'Herbal-A',
    'f37r': 'Herbal-A', 'f37v': 'Herbal-A',
    'f38r': 'Herbal-A', 'f38v': 'Herbal-A',
    'f39r': 'Herbal-A', 'f39v': 'Herbal-A',
    'f40r': 'Herbal-A', 'f40v': 'Herbal-A',
    # Quire 6 (f41-f48)
    'f41r': 'Herbal-A', 'f41v': 'Herbal-A',
    'f42r': 'Herbal-A', 'f42v': 'Herbal-A',
    'f43r': 'Herbal-A', 'f43v': 'Herbal-A',
    'f44r': 'Herbal-A', 'f44v': 'Herbal-A',
    'f45r': 'Herbal-A', 'f45v': 'Herbal-A',
    'f46r': 'Herbal-A', 'f46v': 'Herbal-A',
    'f47r': 'Herbal-A', 'f47v': 'Herbal-A',
    'f48r': 'Herbal-A', 'f48v': 'Herbal-A',
    # Quire 7 (f49-f56)
    'f49r': 'Herbal-A', 'f49v': 'Herbal-A',
    'f50r': 'Herbal-A', 'f50v': 'Herbal-A',
    'f51r': 'Herbal-A', 'f51v': 'Herbal-A',
    'f52r': 'Herbal-A', 'f52v': 'Herbal-A',
    'f53r': 'Herbal-A', 'f53v': 'Herbal-A',
    'f54r': 'Herbal-A', 'f54v': 'Herbal-A',
    'f55r': 'Herbal-A', 'f55v': 'Herbal-A',
    'f56r': 'Herbal-A', 'f56v': 'Herbal-A',
    # Quire 8 (f57-f66, f59-f64 missing)
    'f57r': 'Herbal-A', 'f57v': 'Cosmological',
    'f58r': 'text-only', 'f58v': 'text-only',  # text/stars
    'f65r': 'Herbal-A', 'f65v': 'Herbal-A',
    'f66r': 'text-only', 'f66v': 'Herbal-A',
    # Quire 9 (f67-f68)
    'f67r1': 'Astronomical', 'f67r2': 'Astronomical',
    'f67v1': 'Astronomical', 'f67v2': 'Cosmological',
    'f68r1': 'Astronomical', 'f68r2': 'Astronomical', 'f68r3': 'Astronomical',
    'f68v1': 'Cosmological', 'f68v2': 'Astronomical', 'f68v3': 'Cosmological',
    # Quire 10 (f69-f70)
    'f69r': 'Cosmological', 'f69v': 'Cosmological',
    'f70r1': 'Cosmological', 'f70r2': 'Cosmological',
    'f70v1': 'Zodiac', 'f70v2': 'Zodiac',
    # Quire 11 (f71-f72)
    'f71r': 'Zodiac', 'f71v': 'Zodiac',
    'f72r1': 'Zodiac', 'f72r2': 'Zodiac', 'f72r3': 'Zodiac',
    'f72v1': 'Zodiac', 'f72v2': 'Zodiac', 'f72v3': 'Zodiac',
    # Quire 12 (f73, f74 missing)
    'f73r': 'Zodiac', 'f73v': 'Zodiac',
    # Quire 13 (f75-f84)
    'f75r': 'Balneological', 'f75v': 'Balneological',
    'f76r': 'text-only', 'f76v': 'Balneological',
    'f77r': 'Balneological', 'f77v': 'Balneological',
    'f78r': 'Balneological', 'f78v': 'Balneological',
    'f79r': 'Balneological', 'f79v': 'Balneological',
    'f80r': 'Balneological', 'f80v': 'Balneological',
    'f81r': 'Balneological', 'f81v': 'Balneological',
    'f82r': 'Balneological', 'f82v': 'Balneological',
    'f83r': 'Balneological', 'f83v': 'Balneological',
    'f84r': 'Balneological', 'f84v': 'Balneological',
    # Quire 14 (f85-f86, Rosettes foldout)
    'f85r1': 'text-only', 'f85r2': 'Rosettes',
    'f85v': 'Rosettes',
    'f85_86r1': 'Rosettes',
    'f85_86v3': 'Rosettes', 'f85_86v4': 'Rosettes',
    'f85_86v5': 'text-only', 'f85_86v6': 'text-only',
    'f86v3': 'Rosettes', 'f86v4': 'Rosettes',
    'f86v5': 'text-only', 'f86v6': 'text-only',
    # Quire 15 (f87-f90)
    'f87r': 'Herbal-B', 'f87v': 'Herbal-B',
    'f88r': 'Pharmaceutical', 'f88v': 'Pharmaceutical',
    'f89r1': 'Pharmaceutical', 'f89r2': 'Pharmaceutical',
    'f89v1': 'Pharmaceutical', 'f89v2': 'Pharmaceutical',
    'f90r1': 'Herbal-B', 'f90r2': 'Herbal-B',
    'f90v1': 'Herbal-B', 'f90v2': 'Herbal-B',
    # Quire 17 (f93-f96, Quire 16 missing)
    'f93r': 'Herbal-B', 'f93v': 'Herbal-B',
    'f94r': 'Herbal-B', 'f94v': 'Herbal-B',
    'f95r1': 'Herbal-B', 'f95r2': 'Herbal-B',
    'f95v1': 'Herbal-B', 'f95v2': 'Herbal-B',
    'f96r': 'Herbal-B', 'f96v': 'Herbal-B',
    # Quire 19 (f99-f102, Quire 18 missing)
    'f99r': 'Pharmaceutical', 'f99v': 'Pharmaceutical',
    'f100r': 'Pharmaceutical', 'f100v': 'Pharmaceutical',
    'f101r': 'Pharmaceutical', 'f101v': 'Pharmaceutical',
    'f101r1': 'Pharmaceutical', 'f101r2': 'Pharmaceutical',
    'f101v1': 'Pharmaceutical', 'f101v2': 'Pharmaceutical',
    'f102r1': 'Pharmaceutical', 'f102r2': 'Pharmaceutical',
    'f102v1': 'Pharmaceutical', 'f102v2': 'Pharmaceutical',
    # Quire 20 (f103-f116, Recipes/Stars)
    'f103r': 'Stars', 'f103v': 'Stars',
    'f104r': 'Stars', 'f104v': 'Stars',
    'f105r': 'Stars', 'f105v': 'Stars',
    'f106r': 'Stars', 'f106v': 'Stars',
    'f107r': 'Stars', 'f107v': 'Stars',
    'f108r': 'Stars', 'f108v': 'Stars',
    'f111r': 'Stars', 'f111v': 'Stars',
    'f112r': 'Stars', 'f112v': 'Stars',
    'f113r': 'Stars', 'f113v': 'Stars',
    'f114r': 'Stars', 'f114v': 'Stars',
    'f115r': 'Stars', 'f115v': 'Stars',
    'f116r': 'Stars', 'f116v': 'text-only',
}

# Canonical section list (excluding text-only)
CANONICAL_SECTIONS = [
    'Herbal-A', 'Herbal-B', 'Astronomical', 'Cosmological',
    'Zodiac', 'Rosettes', 'Balneological', 'Pharmaceutical', 'Stars'
]

# OLD → NEW section mapping (for converting P69 rule allow/deny lists)
OLD_TO_NEW = {
    'Herbal':         ['Herbal-A', 'Herbal-B'],
    'Astronomical':   ['Astronomical', 'Cosmological', 'Zodiac', 'Rosettes'],
    'Biological':     ['Balneological'],
    'Pharmaceutical': ['Pharmaceutical'],
    'Recipes':        ['Stars'],
    'Unassigned':     [],  # eliminated — all pages now assigned
}

NEW_TO_OLD = {}
for old, news in OLD_TO_NEW.items():
    for new in news:
        NEW_TO_OLD[new] = old


def get_section(page_id):
    """
    Map a corpus page ID to its canonical section.
    Uses exact match first, then progressively strips sub-page identifiers.
    """
    # Try exact match
    if page_id in SECTION_MAP:
        return SECTION_MAP[page_id]
    
    # Try with 'f' prefix
    if not page_id.startswith('f'):
        if 'f' + page_id in SECTION_MAP:
            return SECTION_MAP['f' + page_id]
    
    # Strip trailing digits from sub-page (e.g., f67r1 → f67r)
    base = re.sub(r'(\d+[rv])\d+$', r'\1', page_id)
    if base in SECTION_MAP:
        return SECTION_MAP[base]
    if not base.startswith('f') and 'f' + base in SECTION_MAP:
        return SECTION_MAP['f' + base]
    
    # Handle Rosettes foldout variants
    if '85_86' in page_id or '85' in page_id or '86' in page_id:
        return 'Rosettes'
    
    return None  # Unmapped


def convert_section_list(old_list):
    """Convert an old-style section list to new canonical sections."""
    new_list = []
    for old in old_list:
        if old in OLD_TO_NEW:
            new_list.extend(OLD_TO_NEW[old])
        elif old in CANONICAL_SECTIONS:
            new_list.append(old)
    return new_list


# ═══════════════════════════════════════════════════════════════════════
# LOAD CORPUS AND ASSIGN SECTIONS
# ═══════════════════════════════════════════════════════════════════════

with open('/home/claude/voynich_transcriptions.json') as f:
    corpus_data = json.load(f)

TARGET_TID = 'ZLZI'
records = []
unmapped_pages = Counter()

for page_id, page in corpus_data['pages'].items():
    sec = get_section(page_id)
    if sec is None or sec == 'text-only':
        unmapped_pages[page_id] += 1
        continue  # Skip text-only and unmapped pages
    
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

print("=" * 76)
print("SECTION REALIGNMENT: OLD → CANONICAL")
print("=" * 76)

# Section distribution
sec_counts = Counter(r[1] for r in records)
print(f"\nCanonical section distribution ({total_tokens:,} tokens):")
print(f"  {'Section':<18} {'Tokens':>8} {'%':>8} {'Pages':>8}")
print("  " + "─" * 44)

sec_pages = defaultdict(set)
for r in records:
    sec_pages[r[1]].add(r[0])

for sec in CANONICAL_SECTIONS:
    cnt = sec_counts.get(sec, 0)
    pct = cnt / total_tokens * 100 if total_tokens else 0
    npg = len(sec_pages.get(sec, set()))
    print(f"  {sec:<18} {cnt:>8,} {pct:>7.1f}% {npg:>8}")

if unmapped_pages:
    print(f"\n  Unmapped/text-only pages: {len(unmapped_pages)}")
    for p, _ in unmapped_pages.most_common(10):
        print(f"    {p}")


# ═══════════════════════════════════════════════════════════════════════
# CONVERT P69 + P70 RULES TO CANONICAL SECTIONS
# ═══════════════════════════════════════════════════════════════════════

with open('/home/claude/p70_rules_final.json') as f:
    p70_data = json.load(f)

rules = p70_data['rules']

print(f"\n{'='*76}")
print("CONVERTING RULES TO CANONICAL SECTIONS")
print(f"{'='*76}")

converted_rules = []
for r in rules:
    cr = dict(r)
    
    # Convert allow list
    old_allow = r.get('allow', [])
    if old_allow:
        new_allow = convert_section_list(old_allow)
        cr['allow'] = new_allow if new_allow else CANONICAL_SECTIONS[:]
    
    # Convert deny list
    old_deny = r.get('deny', [])
    if old_deny:
        cr['deny'] = convert_section_list(old_deny)
    
    # Convert w_by_section
    old_wbs = r.get('w_by_section', {})
    if old_wbs:
        new_wbs = {}
        for new_sec in CANONICAL_SECTIONS:
            old_sec = NEW_TO_OLD.get(new_sec)
            if old_sec and old_sec in old_wbs:
                new_wbs[new_sec] = old_wbs[old_sec]
            else:
                new_wbs[new_sec] = 1.0  # default
        cr['w_by_section'] = new_wbs
    
    converted_rules.append(cr)

# Verify conversion
sample_rules = [r for r in converted_rules if r.get('deny')]
print(f"\nRules with deny lists: {len(sample_rules)}")
for r in sample_rules[:5]:
    print(f"  {r['rule_id']}: deny={r['deny']}")


# ═══════════════════════════════════════════════════════════════════════
# RECOMPUTE SECTION WEIGHTS FOR ALL 210 RULES ON CANONICAL SECTIONS
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*76}")
print("RECOMPUTING SECTION WEIGHTS ON CANONICAL CORPUS")
print(f"{'='*76}")

for r in converted_rules:
    pat = r['pattern'].replace('|', '')
    kind = r['kind']
    
    sec_hits = Counter()
    sec_totals = Counter()
    
    for _, sec, _, _, w, prev_w, next_w in records:
        sec_totals[sec] += 1
        if kind == 'prefix' and w.startswith(pat): sec_hits[sec] += 1
        elif kind == 'suffix' and w.endswith(pat): sec_hits[sec] += 1
        elif kind == 'chargram' and pat in w: sec_hits[sec] += 1
        elif kind == 'pair' and pat in w: sec_hits[sec] += 1
    
    total_hits = sum(sec_hits.values())
    if total_hits == 0:
        r['w_by_section'] = {s: 0.0 for s in CANONICAL_SECTIONS}
        r['allow'] = []
        r['deny'] = CANONICAL_SECTIONS[:]
        continue
    
    baseline = total_hits / total_tokens
    new_wbs = {}
    new_allow = []
    new_deny = []
    
    for sec in CANONICAL_SECTIONS:
        if sec_totals[sec] == 0:
            new_wbs[sec] = 0.0
            new_deny.append(sec)
            continue
        sec_rate = sec_hits[sec] / sec_totals[sec]
        norm_w = min(sec_rate / max(baseline, 1e-9), 1.0)
        new_wbs[sec] = round(norm_w, 3)
        if norm_w >= 0.3:
            new_allow.append(sec)
        else:
            new_deny.append(sec)
    
    r['w_by_section'] = new_wbs
    r['allow'] = new_allow
    r['deny'] = new_deny

# Show a few rules with interesting section patterns
print("\nRules with strongest section discrimination:")
for r in converted_rules:
    wbs = r['w_by_section']
    if not wbs: continue
    vals = [v for v in wbs.values() if v > 0]
    if not vals: continue
    spread = max(vals) - min(vals) if len(vals) > 1 else 0
    r['_section_spread'] = spread

top_discriminators = sorted(converted_rules, key=lambda x: x.get('_section_spread', 0), reverse=True)[:15]
for r in top_discriminators:
    wbs = r['w_by_section']
    high = max(wbs, key=wbs.get)
    low = min((s for s in wbs if wbs[s] > 0), key=wbs.get, default='—')
    print(f"  {r['rule_id']:<35} high={high}({wbs[high]:.2f}) low={low}({wbs.get(low,0):.2f}) deny={r.get('deny',[])}")

# Clean up temp field
for r in converted_rules:
    r.pop('_section_spread', None)


# ═══════════════════════════════════════════════════════════════════════
# COVERAGE VALIDATION ON CANONICAL SECTIONS
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*76}")
print("COVERAGE ON CANONICAL SECTIONS")
print(f"{'='*76}")

def rules_covering_word(word, prev_w, next_w, ruleset):
    covered = set()
    for r in ruleset:
        pat = r['pattern'].replace('|','')
        kind = r['kind']
        wlen = len(word)
        if kind == 'prefix' and word.startswith(pat):
            covered.update(range(len(pat)))
        elif kind == 'suffix' and word.endswith(pat):
            covered.update(range(wlen-len(pat), wlen))
        elif kind == 'chargram':
            idx = word.find(pat)
            while idx != -1:
                covered.update(range(idx, idx+len(pat)))
                idx = word.find(pat, idx+1)
        elif kind == 'pair':
            if pat in word:
                idx = word.find(pat)
                while idx != -1:
                    covered.update(range(idx, idx+len(pat)))
                    idx = word.find(pat, idx+1)
            if '|' in r['pattern']:
                left, right = r['pattern'].split('|',1)
                if left and right:
                    if prev_w.endswith(left) and word.startswith(right):
                        covered.update(range(len(right)))
                    if word.endswith(left) and next_w.startswith(right):
                        covered.update(range(wlen-len(left), wlen))
                elif left and word.endswith(left):
                    covered.update(range(wlen-len(left), wlen))
                elif right and word.startswith(right):
                    covered.update(range(len(right)))
    return covered

sec_cov = defaultdict(lambda: {'total_chars': 0, 'covered_chars': 0,
                                 'total_words': 0, 'full_words': 0})

for page_id, sec, line_id, wi, w, prev_w, next_w in records:
    wlen = len(w)
    sec_cov[sec]['total_chars'] += wlen
    sec_cov[sec]['total_words'] += 1
    cov = rules_covering_word(w, prev_w, next_w, converted_rules)
    sec_cov[sec]['covered_chars'] += len(cov)
    if len(cov) == wlen:
        sec_cov[sec]['full_words'] += 1

print(f"\n  {'Section':<18} {'Tokens':>8} {'Char%':>8} {'Word%':>8}")
print("  " + "─" * 46)

cov_rows = []
total_chars_all = 0
covered_chars_all = 0
total_words_all = 0
full_words_all = 0

for sec in CANONICAL_SECTIONS:
    s = sec_cov[sec]
    if s['total_words'] == 0: continue
    char_pct = s['covered_chars'] / s['total_chars'] * 100
    word_pct = s['full_words'] / s['total_words'] * 100
    print(f"  {sec:<18} {s['total_words']:>8,} {char_pct:>7.1f}% {word_pct:>7.1f}%")
    cov_rows.append({'Section': sec, 'Tokens': s['total_words'],
                     'Char_Coverage%': round(char_pct, 1), 'Word_Coverage%': round(word_pct, 1)})
    total_chars_all += s['total_chars']
    covered_chars_all += s['covered_chars']
    total_words_all += s['total_words']
    full_words_all += s['full_words']

print(f"  {'─'*46}")
print(f"  {'TOTAL':<18} {total_words_all:>8,} {covered_chars_all/total_chars_all*100:>7.1f}% {full_words_all/total_words_all*100:>7.1f}%")


# ═══════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════

p70_data['rules'] = converted_rules
p70_data['schema'] = 'P70-canonical-sections'
p70_data['sections'] = CANONICAL_SECTIONS
p70_data['section_source'] = 'Voynich.nu per-page descriptions (Paper 1)'
p70_data['section_changes'] = {
    'Herbal → Herbal-A + Herbal-B': 'Split by Currier language (A=Quires 1-8, B=Q15,17)',
    'Astronomical → Astronomical + Cosmological + Zodiac + Rosettes': 'Distinct illustration types',
    'Biological → Balneological': 'Standard nomenclature',
    'Recipes → Stars': 'Per user specification',
    'Unassigned → eliminated': 'All pages now canonically assigned',
}
p70_data['created_from']['canonical_sections_applied'] = True

out_json = '/home/claude/p70_rules_canonical.json'
with open(out_json, 'w') as f:
    json.dump(p70_data, f, indent=2)
print(f"\nSaved: {out_json}")

# Also save section mapping as standalone reference
with open('/home/claude/voynich_section_map.json', 'w') as f:
    json.dump({
        'source': 'Voynich.nu per-page descriptions',
        'sections': CANONICAL_SECTIONS,
        'mapping': SECTION_MAP,
        'old_to_new': OLD_TO_NEW,
    }, f, indent=2)
print("Saved: voynich_section_map.json")

# XLSX with section coverage
df_cov = pd.DataFrame(cov_rows)
xlsx_path = '/home/claude/p70_canonical_coverage.xlsx'
df_cov.to_excel(xlsx_path, index=False)
print(f"Saved: {xlsx_path}")

