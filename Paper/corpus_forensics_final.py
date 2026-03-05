"""
Corpus Forensics FINAL — verified against paper §4.6
Slots: prefix, gallows, core, suffix (raw strings)
Null 3: independently permute slot columns globally, folio-bounded window
"""
import pickle, random, math
import numpy as np
from collections import defaultdict, Counter

with open('/home/claude/Voynichdecomp/enriched_records.pkl','rb') as f:
    records = pickle.load(f)

SLOTS = ['prefix', 'gallows', 'core', 'suffix']

def slots_of(r): return tuple(r.get(s,'∅') for s in SLOTS)
def n_shared(a,b): return sum(x==y for x,y in zip(a,b))

def folio_match_rate(records_list, window=10, threshold=2):
    folio_groups = defaultdict(list)
    for r in records_list: folio_groups[r['folio']].append(r)
    hits = total = 0; slot_counts = Counter()
    for folio, recs in folio_groups.items():
        slots = [slots_of(r) for r in recs]
        for i in range(1, len(slots)):
            target = slots[i]
            lookback = slots[max(0,i-window):i]
            best = max(n_shared(target,s) for s in lookback)
            if best >= threshold: hits += 1; slot_counts[best] += 1
            total += 1
    return hits/total if total>0 else 0, hits, total, slot_counts

def folio_match_rate_slotlists(slot_list_by_folio, window=10, threshold=2):
    hits = total = 0
    for folio_slots in slot_list_by_folio:
        for i in range(1, len(folio_slots)):
            target = folio_slots[i]
            lookback = folio_slots[max(0,i-window):i]
            if any(n_shared(target,s) >= threshold for s in lookback): hits += 1
            total += 1
    return hits/total if total>0 else 0

rng = random.Random(42)
N_REPS = 20

# OBSERVED
print("Observed match rates:")
for w in [5,10,20]:
    rate, hits, total, sc = folio_match_rate(records, window=w)
    print(f"  W={w:2d}: {rate*100:.2f}%  ({hits:,}/{total:,})")

rate_obs, hits_obs, total_obs, slot_hist = folio_match_rate(records, window=10)
print(f"\n  Slot histogram (W=10):")
for k in sorted(slot_hist):
    print(f"    exactly {k} slots: {slot_hist[k]:,} ({slot_hist[k]/total_obs*100:.1f}%)")

print("\nPer-section (W=10):")
section_rates = {}
for sec in sorted(set(r['section'] for r in records)):
    sec_recs = [r for r in records if r['section'] == sec]
    r_s,h,t,_ = folio_match_rate(sec_recs, window=10)
    section_rates[sec] = (r_s, t)
    print(f"  {sec:<20} {r_s*100:.1f}%  (n={t:,})")

# NULL 1: within-section shuffle, folio-bounded
print(f"\nNull 1: within-section shuffle ({N_REPS} reps)...")
null1_rates = []
sec_groups = defaultdict(list)
for r in records: sec_groups[r['section']].append(r)
for rep in range(N_REPS):
    shuffled = []
    for sec, recs in sec_groups.items():
        perm = list(recs); rng.shuffle(perm)
        orig_folios = [r['folio'] for r in recs]
        relabelled = [dict(r) for r in perm]
        for i in range(len(relabelled)): relabelled[i]['folio'] = orig_folios[i]
        shuffled.extend(relabelled)
    r_n,_,_,_ = folio_match_rate(shuffled, window=10)
    null1_rates.append(r_n)
n1m,n1s = np.mean(null1_rates), np.std(null1_rates)
print(f"  Mean={n1m*100:.2f}%  SD={n1s*100:.3f}%")

# NULL 2: global shuffle, folio-bounded
print(f"\nNull 2: global shuffle ({N_REPS} reps)...")
null2_rates = []
orig_folios_all = [r['folio'] for r in records]
for rep in range(N_REPS):
    perm = list(records); rng.shuffle(perm)
    relabelled = [dict(r) for r in perm]
    for i in range(len(relabelled)): relabelled[i]['folio'] = orig_folios_all[i]
    r_n,_,_,_ = folio_match_rate(relabelled, window=10)
    null2_rates.append(r_n)
n2m,n2s = np.mean(null2_rates), np.std(null2_rates)
print(f"  Mean={n2m*100:.2f}%  SD={n2s*100:.3f}%")

# NULL 3: independently permute slot columns globally, folio-bounded
print(f"\nNull 3: independent slot column permutation ({N_REPS} reps)...")
all_slots_flat = [slots_of(r) for r in records]
folio_labels = [r['folio'] for r in records]
null3_rates = []
for rep in range(N_REPS):
    cols = [[t[s] for t in all_slots_flat] for s in range(4)]
    for col in cols: rng.shuffle(col)
    new_slots = list(zip(*cols))
    by_folio_new = defaultdict(list)
    for i,f in enumerate(folio_labels): by_folio_new[f].append(new_slots[i])
    r_n = folio_match_rate_slotlists(list(by_folio_new.values()), window=10)
    null3_rates.append(r_n)
n3m,n3s = np.mean(null3_rates), np.std(null3_rates)
print(f"  Mean={n3m*100:.2f}%  SD={n3s*100:.3f}%")

# STATS
print(f"\nStatistical summary (W=10, n={total_obs:,}):")
print(f"  Observed: {rate_obs*100:.2f}%")
for label, nm, ns in [
    ("Null 1 (section shuffle)    ", n1m, n1s),
    ("Null 2 (global shuffle)     ", n2m, n2s),
    ("Null 3 (independent grammar)", n3m, n3s),
]:
    diff = rate_obs - nm
    se = math.sqrt(rate_obs*(1-rate_obs)/total_obs)
    z = diff / se
    h = 2*(math.asin(math.sqrt(rate_obs)) - math.asin(math.sqrt(nm)))
    print(f"  vs {label}: diff={diff*100:+.2f}pp  Z={z:.1f}  h={h:.3f}")

results = {
    'slots': SLOTS,
    'rate_obs': rate_obs, 'hits_obs': hits_obs, 'total_obs': total_obs,
    'slot_hist': dict(slot_hist), 'section_rates': section_rates,
    'null1_mean': n1m, 'null1_std': n1s,
    'null2_mean': n2m, 'null2_std': n2s,
    'null3_mean': n3m, 'null3_std': n3s,
}
with open('/mnt/user-data/outputs/corpus_forensics_results.pkl','wb') as f:
    pickle.dump(results, f)
print("\nSaved.")
