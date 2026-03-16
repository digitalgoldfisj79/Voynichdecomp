"""
V11 + Column Stickiness
========================
After CI routing determines a family, with probability p_sticky,
override to the PREVIOUS token's family instead.

This models a scribe who tends to stay in the same grid column.
Measured VMS sfx_bi = 0.2522; CI raw routing = 0.2090.
The extra 0.043 must come from scribe column preference.
"""
import sys, random, pickle, numpy as np
from collections import Counter
sys.path.insert(0, '/home/claude')
sys.path.insert(0, '/mnt/user-data/uploads')
import v11_nomenclator as v11
import score_85_metrics as scorer
import metric_defs

with open('enriched_records.pkl','rb') as f:
    records = pickle.load(f)
ha = [r for r in records if r.get('section')=='Herbal-A']
toks_vms = [r['token'] for r in ha][:4032]
lines_vms = [toks_vms[i:i+84] for i in range(0,4032,84)]
VMS_M = scorer.compute_metrics(toks_vms, lines=lines_vms,
                                subset_iterations=30, seed=42, verbose=False)

def build_tc(ha):
    lookup = {}
    for r in ha:
        mc = r.get('m_core') or r.get('core') or ''
        row = mc[0] if mc and not r['empty_core'] else '∅'
        sf = r.get('sfx_fam', 'BARE')
        if r['token'] not in lookup:
            lookup[r['token']] = (row, sf)
    return lookup
tc = build_tc(ha)

def run_sticky(seed, p_sticky):
    random.seed(seed)
    np.random.seed(seed)
    
    with open('enriched_records.pkl','rb') as f:
        recs = pickle.load(f)
    with open('ci_corpus_parsed.pkl','rb') as f:
        ci = pickle.load(f)
    
    ha_recs = [r for r in recs if r.get('section')=='Herbal-A']
    sampler, _ = v11.build_pools(ha_recs)
    
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
    prev_family = 'Y'  # starting family
    
    rare_positions = set(random.sample(range(100, v11.TARGET-100),
                                       len(v11.RARE_TOKENS)))
    rare_schedule = dict(zip(sorted(rare_positions), v11.RARE_TOKENS))
    
    i = 0
    while len(output) < v11.TARGET and i < len(words):
        n = len(output)
        
        if n in rare_schedule:
            t = rare_schedule[n]
            output.append(('RC', t, '<rare>'))
            produced.add(t)
            past_counts[t] += 1
            family_counts['BARE'] = family_counts.get('BARE',0)+1
            prev_family = 'BARE'
            line_tokens.append(t)
            if len(line_tokens) >= line_target:
                line_tokens = []; line_target = random.choice(v11.LINE_LENGTHS)
            continue
        
        word = words[i]; i += 1
        over_cap = len(produced) >= v11.VOCAB_CAP
        route = v11.classify_and_route(word, ec_words)
        at_boundary = (len(line_tokens)==0 or
                       len(line_tokens) >= line_target-1)
        
        token = None
        if route[0] == 'EC':
            family = v11.rebalance_family(route[1], family_counts, n)
            # STICKINESS: override family with previous
            if random.random() < p_sticky and prev_family in v11.FAMILIES:
                family = prev_family
            cell = ('∅', family)
            if over_cap:
                token = v11.reuse_token(past_counts, sampler, cell)
            else:
                token = v11.pick_token(sampler, cell, produced)
                if not token:
                    for alt in v11.FAMILIES:
                        if alt != family:
                            t2 = v11.pick_token(sampler, ('∅', alt), produced)
                            if t2:
                                token = t2; family = alt; break
                if not token:
                    token = 'dy'
        else:
            row, family = route[1], route[2]
            family = v11.rebalance_family(family, family_counts, n)
            # STICKINESS: override family with previous
            if random.random() < p_sticky and prev_family in v11.FAMILIES:
                if (row, prev_family) in sampler:
                    family = prev_family
            cell = (row, family)
            bf = 0.35 if at_boundary else 1.0
            if over_cap:
                token = v11.reuse_token(past_counts, sampler, cell)
            else:
                if random.random() < (v11.FC_COPY_RATE + v11.FC_ED1_RATE) * bf:
                    token = v11.pick_token(sampler, cell, produced)
                else:
                    alt_fams = [f for f in v11.FAMILIES
                                if f != family and (row,f) in sampler]
                    if alt_fams:
                        af = random.choice(alt_fams)
                        token = v11.pick_token(sampler, (row,af), produced)
                        family = af
                    else:
                        token = v11.pick_token(sampler, cell, produced)
                if not token:
                    token = v11.pick_token(sampler, cell, produced) or 'dy'
        
        output.append((route[0], token, word))
        produced.add(token)
        past_counts[token] += 1
        family_counts[family] = family_counts.get(family,0)+1
        prev_family = family
        line_tokens.append(token)
        if len(line_tokens) >= line_target:
            line_tokens = []; line_target = random.choice(v11.LINE_LENGTHS)
    
    return output

def score_and_profile(toks):
    gl = [toks[i:i+84] for i in range(0,len(toks),84)]
    gm = scorer.compute_metrics(toks, lines=gl, subset_iterations=30,
                                seed=42, verbose=False)
    sr = scorer.score_against_vms(gm, VMS_M)
    c15 = sum(1 for m in metric_defs.CORE_15
              if m in sr['details'] and sr['details'][m]['pass'])
    bg42 = sum(1 for m in metric_defs.BG_METRICS
               if m in sr['details'] and sr['details'][m]['pass'])
    sfs = [tc.get(t, ('?','?'))[1] for t in toks]
    sfb = sum(1 for i in range(len(sfs)-1)
              if sfs[i]==sfs[i+1]) / (len(sfs)-1)
    return sr['n_pass'], c15, bg42, gm.get('autocorr_wordlen',0), len(set(toks)), sfb

SEEDS = [42, 404, 501, 606, 808]

print(f'{"p_sticky":>10}  {"n/84":>5}  {"C15":>4}  {"BG42":>5}  '
      f'{"AC_wl":>7}  {"sfx_bi":>7}  {"types":>6}')
print('-'*60)

results = {}
for ps in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    scores = []
    for seed in SEEDS:
        toks = [t[1] for t in run_sticky(seed, ps)]
        scores.append(score_and_profile(toks))
    results[ps] = scores
    n = np.mean([s[0] for s in scores])
    c = np.mean([s[1] for s in scores])
    b = np.mean([s[2] for s in scores])
    ac = np.mean([s[3] for s in scores])
    ty = np.mean([s[4] for s in scores])
    sb = np.mean([s[5] for s in scores])
    print(f'{ps:>10.2f}  {n:>5.1f}  {c:>4.1f}  {b:>5.1f}  '
          f'{ac:>+7.4f}  {sb:>7.4f}  {ty:>6.0f}')

print(f'{"VMS":>10}  {"84":>5}  {"15":>4}  {"42":>5}  '
      f'{0.0756:>+7.4f}  {0.2522:>7.4f}  {"1430":>6}')

with open('/home/claude/sticky_sweep_results.pkl','wb') as f:
    pickle.dump(results, f)
print('\nSaved sticky_sweep_results.pkl')
