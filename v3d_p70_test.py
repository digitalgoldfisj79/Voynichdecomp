#!/usr/bin/env python3
"""
v3d POSITIONAL ENCODING vs P70 MORPHOLOGICAL FINGERPRINT
==========================================================
Tests whether v3d output matches P70's internal morphological
signature, not just surface statistics.
"""

import json, csv, re, math, random
from collections import Counter, defaultdict
import numpy as np

random.seed(42)
np.random.seed(42)

# === 1. LOAD VMS FROM JSON ===

with open('/home/claude/voynich_transcriptions.json') as f:
    corpus_data = json.load(f)
with open('/home/claude/voynich_section_map.json') as f:
    sec_map = json.load(f)
SECTION_MAP = sec_map['mapping']

def get_section(pid):
    for c in [pid, 'f'+pid, re.sub(r'(\d+[rv])\d+$', r'\1', pid),
              'f'+re.sub(r'(\d+[rv])\d+$', r'\1', pid)]:
        if c in SECTION_MAP:
            return SECTION_MAP[c]
    return None

TARGET_TID = 'ZLZI'
vms_words = []
vms_paragraphs = []
vms_by_section = defaultdict(list)

for pid in sorted(corpus_data['pages'].keys()):
    page = corpus_data['pages'][pid]
    sec = get_section(pid)
    if not sec or sec == 'text-only':
        continue
    page_words = []
    for lid in sorted(page['lines'].keys()):
        line = page['lines'][lid]
        for sk, src in line['sources'].items():
            if src['transcriber_id'] == TARGET_TID:
                norm = src['views'].get('normalized', {})
                text = norm.get('text', '') if isinstance(norm, dict) else ''
                for w in text.strip().split():
                    if w not in ('*', '?', '-', '%'):
                        page_words.append(w)
    if page_words:
        vms_paragraphs.append(page_words)
        vms_words.extend(page_words)
        vms_by_section[sec].extend(page_words)

vms_freq = Counter(vms_words)
print("VMS: {} words, {} types, {} pages".format(
    len(vms_words), len(vms_freq), len(vms_paragraphs)))
for sec in sorted(vms_by_section):
    print("  {}: {} words".format(sec, len(vms_by_section[sec])))

# === 2. LOAD CLEANED LATIN ===

with open('/home/claude/clean_latin_recipes.json') as f:
    clean_data = json.load(f)
lat_recipes = clean_data['recipes']
lat_all = [w for r in lat_recipes for w in r]
lat_freq = Counter(lat_all)
print("Latin: {} words, {} types, {} recipes".format(
    len(lat_all), len(lat_freq), len(lat_recipes)))

# === 3. v3d ENCODING ENGINE ===

TEMPLATE_LOGOGRAMS = {
    'recipe': 'p', 'confice': 'f', 'misce': 't', 'fiat': 'k',
    'fac': 'k', 'tere': 't', 'adde': 't', 'coque': 'k',
    'cola': 'k', 'dissolve': 'f', 'ponatur': 'p', 'detur': 't',
    'bibat': 'k',
}

NOMENCLATOR = {
    'et': 'daiin', 'in': 'chedy', 'cum': 'shedy', 'an': 'aiin',
    'ana': 'aiin', 'ad': 'chol', 'de': 'qokeey', 'est': 'qokedy',
    'quod': 'qokeedy', 'vel': 'qokain', 'si': 'okeey', 'ex': 'cheey',
    'per': 'cheol', 'que': 'shey', 'sed': 'otedy', 'aut': 'okaiin',
    'non': 'qokaiin', 'hoc': 'otaiin', 'eius': 'qokal', 'autem': 'qokar',
    'quantum': 'qoteedy', 'sufficit': 'qokchdy', 'partes': 'okeedy',
    'pars': 'okeey', 'libra': 'shedy', 'libras': 'shedy',
    'datur': 'chor', 'valet': 'okedy', 'grana': 'sheey',
    'uncia': 'cheldy', 'uncias': 'cheldy', 'modum': 'chear',
    'super': 'shoky', 'aqua': 'okear', 'aquae': 'okeal',
    'vino': 'sheol', 'mei': 'shor', 'olei': 'otal', 'aceti': 'okal',
    'succi': 'opchol', 'seminis': 'qoteol',
    'bene': 'shol', 'optime': 'otchy', 'bonum': 'shal',
    'or': 'or', 'al': 'al', 'ol': 'ol', 'ar': 'ar', 'dy': 'dy',
    'postea': 'otol', 'deinde': 'dar', 'item': 'dal', 'primo': 'dol',
    'inde': 'otar', 'quam': 'qoar', 'iam': 'oal', 'sic': 'shy',
    'rem': 'chy', 'nec': 'oky', 'bis': 'oty', 'ter': 'shy',
    'omnia': 'otchy', 'misceantur': 'toly', 'fiant': 'koly',
    'terantur': 'tary', 'pulvis': 'okaly', 'electuarium': 'qoteey',
    'unguentum': 'okeoly', 'emplastrum': 'cheldy', 'confectio': 'scheey',
}

INIT_MAP = {
    '': '', 'c': 'ch', 'ch': 'ch', 'cr': 'ch', 'cl': 'ch',
    's': 'sh', 'sp': 'sh', 'st': 'sh', 'sc': 'sh', 'sq': 'sh',
    'str': 'sh', 'squ': 'sh', 'sm': 'sh', 'sn': 'sh', 'sl': 'sh',
    'spr': 'sh', 'scr': 'sh',
    'q': 'qo', 'qu': 'qo', 'l': 'qo',
    'd': 'd', 'dr': 'd', 'm': 'd',
    't': 'ot', 'th': 'ot', 'tr': 'ot',
    'p': 'ok', 'ph': 'ok', 'pr': 'ok', 'pl': 'ok',
    'g': 'o', 'gr': 'o', 'gl': 'o', 'gn': 'o',
    'b': 'o', 'br': 'o', 'bl': 'o', 'v': 'o', 'n': 'o',
    'f': 's', 'fl': 's', 'fr': 's',
    'r': 'qo', 'h': '', 'x': 'sh', 'z': 'sh', 'k': 'ch',
}

END_MAP = {
    '': '', 'um': 'dy', 'is': 'aiin', 'us': 'ey', 'as': 'edy',
    'am': 'am', 'em': 'ain', 'es': 'eey', 'os': 'or', 'ae': 'ol',
    'a': 'al', 'e': 'y', 'i': 'y', 'o': 'ol', 'u': 'ar',
    'ur': 'ar', 'it': 'iin', 'at': 'ain', 'et': 'ey', 'ut': 'edy',
    'er': 'or', 'ar': 'ar', 'nt': 'ain', 'ns': 'aiin',
    'ium': 'eedy', 'uum': 'eedy', 'rum': 'edy', 'orum': 'edy',
    'arum': 'edy', 'ibus': 'eedy', 'atur': 'ar', 'itur': 'ar', 'etur': 'ar',
}

MEDIAL_ATOMS = [
    'e', 'ee', 'k', 'ke', 't', 'te', 'a', 'o',
    'l', 'r', 'ck', 'ch', 'ea', 'eo', 'ok', 'ot',
    'al', 'ar', 'el', 'er',
]

LAT_INITIALS = [
    'str', 'spr', 'scr', 'squ',
    'sp', 'st', 'sc', 'sq', 'sl', 'sm', 'sn',
    'pr', 'pl', 'tr', 'th', 'cr', 'cl', 'ch', 'ph',
    'gr', 'gl', 'gn', 'br', 'bl', 'dr', 'fl', 'fr', 'qu',
    'b', 'c', 'd', 'f', 'g', 'h', 'k', 'l', 'm',
    'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z',
]

LAT_ENDINGS = [
    'ibus', 'orum', 'arum', 'atur', 'itur', 'etur',
    'ium', 'uum', 'rum', 'um', 'us', 'is', 'as', 'os', 'es', 'am', 'em',
    'ae', 'ur', 'it', 'at', 'et', 'ut', 'ar', 'er', 'nt', 'ns',
    'a', 'e', 'i', 'o', 'u',
]

N_POS_VARIANTS = 7

SUFFIX_FAMILIES = {
    '':     ['', 'y', '', 'y', '', '', 'y'],
    'dy':   ['dy', 'edy', 'dy', 'ody', 'eedy', 'dy', 'edy'],
    'aiin': ['aiin', 'oiin', 'aiiin', 'aiin', 'oiiin', 'aiir', 'aiin'],
    'ey':   ['ey', 'eey', 'ey', 'eey', 'oeey', 'ey', 'eey'],
    'edy':  ['edy', 'eedy', 'ody', 'edy', 'eody', 'edy', 'eedy'],
    'am':   ['am', 'am', 'om', 'am', 'eam', 'am', 'om'],
    'ain':  ['ain', 'oain', 'ain', 'eain', 'ain', 'oain', 'ain'],
    'eey':  ['eey', 'oeey', 'eey', 'eeey', 'eey', 'oeey', 'eey'],
    'or':   ['or', 'eor', 'or', 'oor', 'or', 'eor', 'or'],
    'ol':   ['ol', 'eol', 'ol', 'ool', 'ol', 'eol', 'ol'],
    'al':   ['al', 'eal', 'al', 'oal', 'al', 'eal', 'al'],
    'y':    ['y', 'ey', 'y', 'oy', 'y', 'ey', 'y'],
    'in':   ['in', 'oin', 'in', 'ein', 'in', 'oin', 'in'],
    'ar':   ['ar', 'ear', 'ar', 'oar', 'ar', 'ear', 'ar'],
    'iin':  ['iin', 'oiin', 'iin', 'eiin', 'iin', 'oiin', 'iin'],
    'eedy': ['eedy', 'oeedy', 'eedy', 'eeedy', 'eedy', 'oeedy', 'eedy'],
}

def decompose_latin(word):
    initial = ''
    ending = ''
    rest = word
    for ic in LAT_INITIALS:
        if rest.startswith(ic) and len(rest) > len(ic):
            initial = ic
            rest = rest[len(ic):]
            break
    for ce in LAT_ENDINGS:
        if rest.endswith(ce) and len(rest) > len(ce):
            ending = ce
            rest = rest[:-len(ce)]
            break
    return initial, rest, ending

stem_to_medial = {}
stem_counter = 0

def get_medial(stem):
    global stem_counter
    if stem in stem_to_medial:
        return stem_to_medial[stem]
    n = len(MEDIAL_ATOMS)
    idx = stem_counter
    stem_counter += 1
    r = idx % 20
    if r < 9:
        n_atoms = 1
    elif r < 18:
        n_atoms = 2
    else:
        n_atoms = 3
    atoms = []
    remainder = idx
    for _ in range(n_atoms):
        atoms.append(MEDIAL_ATOMS[remainder % n])
        remainder = remainder // n
    medial = ''.join(atoms)
    stem_to_medial[stem] = medial
    return medial

def apply_position_suffix(base_suffix, word_pos):
    if base_suffix in SUFFIX_FAMILIES:
        family = SUFFIX_FAMILIES[base_suffix]
        return family[word_pos % len(family)]
    return base_suffix

def encode_recipe(recipe_words):
    output = []
    pending_logogram = None
    word_pos = 0
    for word in recipe_words:
        if word in TEMPLATE_LOGOGRAMS:
            if word == 'recipe' and word_pos == 0 and hash(str(recipe_words[:3])) % 4 == 0:
                output.append(NOMENCLATOR.get('quantum', 'qoteedy'))
                word_pos += 1
                continue
            pending_logogram = TEMPLATE_LOGOGRAMS[word]
            continue
        if word in NOMENCLATOR:
            vms_word = NOMENCLATOR[word]
            if word == 'et':
                et_variants = ['daiin', 'dain', 'dar', 'dy', 'daiir', 'dal']
                vms_word = et_variants[word_pos % len(et_variants)]
            elif word in ('an', 'ana'):
                an_variants = ['aiin', 'ain', 'aiir', 'air']
                vms_word = an_variants[word_pos % len(an_variants)]
            if pending_logogram:
                vms_word = pending_logogram + vms_word
                pending_logogram = None
            output.append(vms_word)
            word_pos += 1
            continue
        ini, stem, end = decompose_latin(word)
        prefix = INIT_MAP.get(ini, '')
        medial = get_medial(stem)
        base_suffix = END_MAP.get(end, '')
        suffix = apply_position_suffix(base_suffix, word_pos)
        vms_word = prefix + medial + suffix
        if pending_logogram:
            vms_word = pending_logogram + vms_word
            pending_logogram = None
        if len(vms_word) < 2:
            vms_word = vms_word + 'y'
        output.append(vms_word)
        word_pos += 1
    if pending_logogram:
        output.append(pending_logogram + 'ol')
    return output

# Encode all recipes
model_recipes = []
model_all = []
for recipe in lat_recipes:
    encoded = encode_recipe(recipe)
    model_recipes.append(encoded)
    model_all.extend(encoded)

model_freq = Counter(model_all)
print("v3d output: {} words, {} types, {} recipes".format(
    len(model_all), len(model_freq), len(model_recipes)))
print("  Stem lookup: {} unique stems".format(len(stem_to_medial)))

# === 4. LOAD P70 RULES ===

with open('/home/claude/p70_rules_canonical.json') as f:
    p70 = json.load(f)
p69_rules = [r for r in p70['rules'] if r.get('boundary_role') == 'active']
print("P70: {} total rules, {} boundary-active".format(
    len(p70['rules']), len(p69_rules)))

# === P70 SLOT DECOMPOSITION ===

PREFIXES = ['', 'o', 'qo', 'd', 'y', 's', 'q']
GALLOWS_P = ['', 'k', 't', 'p', 'f', 'cth', 'ckh', 'cph', 'cfh']
SUFFIXES_P = ['aiin', 'edy', 'eey', 'ody', 'ain', 'iin', 'chy',
              'shy', 'dy', 'ey', 'in', 'ol', 'or', 'ar', 'al',
              'am', 'an', 'ir', 'ee', 'y', 'n', 'l', 'r', 'm',
              'h', 'e', 'g', 's', '']

def p70_decompose(word):
    rem = word
    pfx = ''
    for p in sorted([x for x in PREFIXES if x], key=len, reverse=True):
        if rem.startswith(p):
            pfx = p
            rem = rem[len(p):]
            break
    gal = ''
    for g in sorted([x for x in GALLOWS_P if x], key=len, reverse=True):
        if rem.startswith(g):
            gal = g
            rem = rem[len(g):]
            break
    sfx = ''
    for s in sorted([x for x in SUFFIXES_P if x], key=len, reverse=True):
        if rem.endswith(s) and len(rem) > len(s):
            sfx = s
            rem = rem[:len(rem)-len(s)]
            break
    return pfx, gal, rem, sfx

def slot_analysis(words, label):
    pfxs = Counter()
    gals = Counter()
    cores = Counter()
    sfxs = Counter()
    pfx_sfx = Counter()
    n_any = 0
    n_both = 0
    clens = []
    for w in words:
        if len(w) < 2:
            continue
        p, g, c, s = p70_decompose(w)
        pfxs[p] += 1
        gals[g] += 1
        cores[c] += 1
        sfxs[s] += 1
        if p and s:
            pfx_sfx[(p, s)] += 1
        if p or s:
            n_any += 1
        if p and s:
            n_both += 1
        clens.append(len(c))
    nt = len([w for w in words if len(w) >= 2])
    return {
        'label': label, 'n_types': nt,
        'pfxs': pfxs, 'gals': gals, 'cores': cores, 'sfxs': sfxs,
        'pfx_sfx': pfx_sfx,
        'any_slot': n_any / nt * 100 if nt else 0,
        'both_slot': n_both / nt * 100 if nt else 0,
        'n_cores': len(cores),
        'mean_core_len': np.mean(clens) if clens else 0,
    }

# === TEST A: SURFACE METRICS ===

def surface_metrics(words, label):
    freq = Counter(words)
    n = len(words)
    types = len(freq)
    wl = [len(w) for w in words]
    hapax = sum(1 for c in freq.values() if c == 1)
    chars = list(''.join(words))
    char_freq = Counter(chars)
    cn = len(chars)
    h_char = -sum((c/cn)*math.log2(c/cn) for c in char_freq.values()) if cn > 0 else 0
    h_word = -sum((c/n)*math.log2(c/n) for c in freq.values()) if n > 0 else 0
    return {
        'label': label, 'N': n, 'V': types,
        'TTR': types/n if n else 0,
        'mean_wl': np.mean(wl),
        'hapax_ratio': hapax/types if types else 0,
        'H_char': h_char, 'H_word': h_word,
        'alpha_size': len(char_freq),
        'wl_dist': Counter(wl),
    }

print("\n" + "="*80)
print("TEST A: SURFACE METRICS")
print("="*80)

vm = surface_metrics(vms_words, "VMS")
mm = surface_metrics(model_all, "v3d")

metrics_list = [
    ('N', 'Tokens'), ('V', 'Types'), ('TTR', 'Type/Token'),
    ('mean_wl', 'Mean word len'), ('hapax_ratio', 'Hapax ratio'),
    ('H_char', 'Char entropy'), ('H_word', 'Word entropy'),
    ('alpha_size', 'Alphabet size'),
]

print("\n  {:25s} {:>10s} {:>10s} {:>10s}".format('Metric', 'v3d', 'VMS', 'Match%'))
print("  " + "-"*58)
matches = []
for key, label in metrics_list:
    mv = mm[key]
    vv = vm[key]
    if vv != 0:
        match = max(0, 100 * (1 - abs(mv - vv)/abs(vv)))
    else:
        match = 100 if mv == 0 else 0
    matches.append(match)
    flag = " ok" if match >= 80 else " XX" if match < 50 else ""
    if isinstance(mv, float):
        print("  {:25s} {:>10.3f} {:>10.3f} {:>9.1f}%{}".format(label, mv, vv, match, flag))
    else:
        print("  {:25s} {:>10d} {:>10d} {:>9.1f}%{}".format(label, mv, vv, match, flag))
print("\n  AVERAGE SURFACE MATCH: {:.1f}%".format(np.mean(matches)))

print("\n  Word length distribution:")
print("  {:>4s} {:>8s} {:>8s} {:>8s}".format('Len', 'v3d%', 'VMS%', 'Delta'))
for wl in range(2, 12):
    mp = 100 * mm['wl_dist'].get(wl, 0) / mm['N']
    vp = 100 * vm['wl_dist'].get(wl, 0) / vm['N']
    print("  {:>4d} {:>7.1f}% {:>7.1f}% {:>+7.1f}".format(wl, mp, vp, mp-vp))

# === TEST B: SLOT DECOMPOSITION ===

print("\n" + "="*80)
print("TEST B: P70 SLOT DECOMPOSITION COMPARISON")
print("="*80)

vms_types_list = list(vms_freq.keys())
v3d_types_list = list(model_freq.keys())

vs = slot_analysis(vms_types_list, "VMS")
ms = slot_analysis(v3d_types_list, "v3d")

print("\n  {:25s} {:>10s} {:>10s}".format('Slot metric', 'v3d', 'VMS'))
print("  " + "-"*48)
print("  {:25s} {:>9.1f}% {:>9.1f}%".format('Any slot (pfx or sfx)', ms['any_slot'], vs['any_slot']))
print("  {:25s} {:>9.1f}% {:>9.1f}%".format('Both slots (pfx & sfx)', ms['both_slot'], vs['both_slot']))
print("  {:25s} {:>10d} {:>10d}".format('Unique cores', ms['n_cores'], vs['n_cores']))
print("  {:25s} {:>10.1f} {:>10.1f}".format('Mean core length', ms['mean_core_len'], vs['mean_core_len']))

print("\n  Prefix distribution:")
all_pfx = sorted(set(list(vs['pfxs'].keys()) + list(ms['pfxs'].keys())))
for p in all_pfx:
    vp = 100 * vs['pfxs'].get(p, 0) / vs['n_types']
    mp = 100 * ms['pfxs'].get(p, 0) / ms['n_types']
    bar = " XX" if abs(vp - mp) > 10 else ""
    pfx_label = "'{}'".format(p) if p else "(none)"
    print("  {:>10s}  v3d={:>5.1f}%  VMS={:>5.1f}%  delta={:>+5.1f}{}".format(
        pfx_label, mp, vp, mp-vp, bar))

print("\n  Top-15 suffix distribution:")
vms_top_sfx = vs['sfxs'].most_common(15)
for s, vc in vms_top_sfx:
    vp = 100 * vc / vs['n_types']
    mc = ms['sfxs'].get(s, 0)
    mp = 100 * mc / ms['n_types'] if ms['n_types'] else 0
    sfx_label = "'{}'".format(s) if s else "(none)"
    print("  {:>10s}  v3d={:>5.1f}%  VMS={:>5.1f}%  delta={:>+5.1f}".format(
        sfx_label, mp, vp, mp-vp))

# === TEST C: PREFIX x SUFFIX CO-OCCURRENCE ===

print("\n" + "="*80)
print("TEST C: PREFIX x SUFFIX CO-OCCURRENCE (Cramers V)")
print("="*80)

def cramers_v(words, freq_map, label):
    cooc = Counter()
    pm = Counter()
    sm = Counter()
    for w in words:
        p, g, c, s = p70_decompose(w)
        if p and s:
            f = freq_map.get(w, 1)
            cooc[(p, s)] += f
            pm[p] += f
            sm[s] += f
    if not pm or not sm:
        return {'V': 0, 'n': 0, 'residuals': {}}
    pl = sorted(pm.keys())
    sl = sorted(sm.keys())
    obs = np.zeros((len(pl), len(sl)))
    for i, p in enumerate(pl):
        for j, s in enumerate(sl):
            obs[i, j] = cooc.get((p, s), 0)
    rs = obs.sum(1)
    cs = obs.sum(0)
    tot = obs.sum()
    if tot == 0:
        return {'V': 0, 'n': 0, 'residuals': {}}
    exp = np.outer(rs, cs) / tot
    residuals = {}
    chi2 = 0
    for i, p in enumerate(pl):
        for j, s in enumerate(sl):
            if exp[i, j] >= 5:
                r = (obs[i, j] - exp[i, j]) / math.sqrt(exp[i, j])
                chi2 += r * r
                residuals[(p, s)] = r
    nmin = min(len(pl), len(sl)) - 1
    v = math.sqrt(chi2 / (tot * max(nmin, 1))) if tot > 0 and nmin > 0 else 0
    return {'V': v, 'n': int(tot), 'residuals': residuals}

vcr = cramers_v(vms_types_list, vms_freq, "VMS")
mcr = cramers_v(v3d_types_list, model_freq, "v3d")

print("\n  {:20s} {:>10s} {:>10s}".format('', 'v3d', 'VMS'))
print("  " + "-"*42)
print("  {:20s} {:>10.3f} {:>10.3f}".format("Cramers V", mcr['V'], vcr['V']))
print("  {:20s} {:>10d} {:>10d}".format("N (pfx+sfx tokens)", mcr['n'], vcr['n']))

vms_sorted = sorted(vcr['residuals'].items(), key=lambda x: x[1], reverse=True)
print("\n  VMS top 5 attractions (v3d residual in parens):")
for (p, s), r in vms_sorted[:5]:
    mr = mcr['residuals'].get((p, s), float('nan'))
    print("    [{}...{}] VMS R={:+.1f}  v3d R={:+.1f}".format(p, s, r, mr))

print("\n  VMS top 5 repulsions:")
for (p, s), r in vms_sorted[-5:]:
    mr = mcr['residuals'].get((p, s), float('nan'))
    print("    [{}...{}] VMS R={:+.1f}  v3d R={:+.1f}".format(p, s, r, mr))

# === TEST D: INVENTORY OVERLAP ===

print("\n" + "="*80)
print("TEST D: SLOT INVENTORY OVERLAP")
print("="*80)

v3d_pfx_set = set(p for p in ms['pfxs'] if p and ms['pfxs'][p] >= 3)
vms_pfx_set = set(p for p in vs['pfxs'] if p and vs['pfxs'][p] >= 3)
print("\n  Prefix inventories (freq >= 3):")
print("    v3d:     {}  {}".format(len(v3d_pfx_set), sorted(v3d_pfx_set)))
print("    VMS:     {}  {}".format(len(vms_pfx_set), sorted(vms_pfx_set)))
print("    Overlap: {}".format(sorted(v3d_pfx_set & vms_pfx_set)))
print("    v3d only: {}".format(sorted(v3d_pfx_set - vms_pfx_set)))
print("    VMS only: {}".format(sorted(vms_pfx_set - v3d_pfx_set)))

v3d_sfx_set = set(s for s in ms['sfxs'] if s and ms['sfxs'][s] >= 3)
vms_sfx_set = set(s for s in vs['sfxs'] if s and vs['sfxs'][s] >= 3)
print("\n  Suffix inventories (freq >= 3):")
print("    v3d:     {}  {}".format(len(v3d_sfx_set), sorted(v3d_sfx_set)))
print("    VMS:     {}  {}".format(len(vms_sfx_set), sorted(vms_sfx_set)))
print("    Overlap: {}".format(sorted(v3d_sfx_set & vms_sfx_set)))

v3d_core_set = set(c for c in ms['cores'] if ms['cores'][c] >= 2)
vms_core_set = set(c for c in vs['cores'] if vs['cores'][c] >= 2)
overlap_cores = v3d_core_set & vms_core_set
print("\n  Core inventories (freq >= 2):")
print("    v3d:  {} cores".format(len(v3d_core_set)))
print("    VMS:  {} cores".format(len(vms_core_set)))
print("    Overlap: {} ({:.1f}% of v3d, {:.1f}% of VMS)".format(
    len(overlap_cores),
    100*len(overlap_cores)/len(v3d_core_set) if v3d_core_set else 0,
    100*len(overlap_cores)/len(vms_core_set) if vms_core_set else 0))
if overlap_cores:
    print("    Shared: {}".format(sorted(overlap_cores)[:30]))

# === TEST E: GALLOWS BEHAVIOUR ===

print("\n" + "="*80)
print("TEST E: GALLOWS BEHAVIOUR")
print("="*80)

gallows_chars = set('tkpf')
v3d_gal = sum(1 for w in model_all if w and w[0] in gallows_chars)
vms_gal = sum(1 for w in vms_words if w and w[0] in gallows_chars)
print("\n  Gallows-initial overall:")
print("    v3d: {}/{} ({:.1f}%)".format(v3d_gal, len(model_all), 100*v3d_gal/len(model_all)))
print("    VMS: {}/{} ({:.1f}%)".format(vms_gal, len(vms_words), 100*vms_gal/len(vms_words)))

v3d_starters = [r[0] for r in model_recipes if r]
vms_starters = [p[0] for p in vms_paragraphs if p]
v3d_gs = sum(1 for w in v3d_starters if w and w[0] in gallows_chars)
vms_gs = sum(1 for w in vms_starters if w and w[0] in gallows_chars)
print("\n  Gallows at paragraph/recipe start:")
print("    v3d: {}/{} ({:.1f}%)".format(v3d_gs, len(v3d_starters),
    100*v3d_gs/len(v3d_starters) if v3d_starters else 0))
print("    VMS: {}/{} ({:.1f}%)".format(vms_gs, len(vms_starters),
    100*vms_gs/len(vms_starters) if vms_starters else 0))

print("\n  Per-gallows breakdown (all positions):")
for g in 'tkpf':
    v3d_c = sum(1 for w in model_all if w and w[0] == g)
    vms_c = sum(1 for w in vms_words if w and w[0] == g)
    print("    '{}': v3d={:.1f}%  VMS={:.1f}%".format(
        g, 100*v3d_c/len(model_all), 100*vms_c/len(vms_words)))

# === SAMPLE OUTPUT ===

print("\n" + "="*80)
print("SAMPLE ENCODED RECIPES (first 5)")
print("="*80)

for i in range(min(5, len(lat_recipes))):
    print("\n  Latin:  {}".format(' '.join(lat_recipes[i][:12])))
    print("  v3d:    {}".format(' '.join(model_recipes[i][:12])))

# === SAVE ===

results = {
    'surface_match': float(np.mean(matches)),
    'cramers_v': {'v3d': mcr['V'], 'vms': vcr['V']},
    'slot_coverage': {
        'v3d_any': ms['any_slot'], 'vms_any': vs['any_slot'],
        'v3d_both': ms['both_slot'], 'vms_both': vs['both_slot'],
    },
    'inventories': {
        'pfx_overlap': sorted(v3d_pfx_set & vms_pfx_set),
        'sfx_overlap': sorted(v3d_sfx_set & vms_sfx_set),
        'core_overlap': len(overlap_cores),
        'v3d_cores': len(v3d_core_set),
        'vms_cores': len(vms_core_set),
    },
}
with open('/home/claude/v3d_p70_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved: v3d_p70_results.json")
