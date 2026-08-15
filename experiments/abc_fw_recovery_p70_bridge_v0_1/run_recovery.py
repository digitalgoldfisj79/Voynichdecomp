#!/usr/bin/env python3
import json, math, os
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np
from scipy.stats import spearmanr

SEED = 20260813
NPERM = 200
CARRIERS = [
    'daiin','ol','chedy','aiin','cshedy','chol','or','ar','chey','dar',
    'qokeey','qokeedy','cshey','qokain','qokedy','dy','qokaiin','al','dal','chor'
]
HIST_ABC = {
    'rho': -0.011244193299352685,
    'n_folios': 218,
    'B': {
        'short': [598, 483.46, 17.377813441281962],
        'mid': [501, 471.405, 18.30549029662959],
        'long': [146, 109.88, 9.551732827084308],
    },
    'C': {
        'accretion': [311,276.06,15.09491305042861],
        'reduction': [329,280.1,15.968406307455982],
        'substitution': [605,508.585,16.647004985882596],
        'sub_first_half': [398,329.74,13.374692519830129],
        'sub_second_half': [207,178.845,10.938965901766036],
    }
}
HIST_FW = {
    'pos_ent': [3.3251263058574407, 3.0981709564179436],
    'breadth': [1.004588720256955, 1.4783722850676964],
    'perm_p': [1.0, 1.0]
}
SLOTS = ('prefix','gallows','core','suffix')

@lru_cache(maxsize=None)
def is_ed1(a, b):
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la-lb) > 1:
        return False
    if la == lb:
        return sum(x != y for x,y in zip(a,b)) == 1
    if la > lb:
        a,b = b,a; la,lb = lb,la
    i=j=diff=0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1; j += 1
        else:
            diff += 1; j += 1
            if diff > 1:
                return False
    return True

def shannon(vals):
    n=len(vals)
    if n == 0: return 0.0
    c=Counter(vals)
    return -sum((v/n)*math.log2(v/n) for v in c.values())

def scalar(obs, sims):
    a=np.asarray(sims,dtype=float)
    mu=float(a.mean()); sd=float(a.std(ddof=1))
    ratio=float(obs/mu) if mu>0 else None
    z=float((obs-mu)/sd) if sd>0 else None
    p=float((1+np.sum(np.abs(a-mu) >= abs(obs-mu)))/(len(a)+1)) if sd>0 else None
    return {'observed':float(obs),'null_mean':mu,'null_sd':sd,'ratio':ratio,'z':z,'empirical_p_two_sided':p}

def ratio_scalar(num, den, sim_num, sim_den):
    obs=float(num/den) if den else float('nan')
    arr=np.asarray([a/b if b else np.nan for a,b in zip(sim_num,sim_den)],dtype=float)
    arr=arr[np.isfinite(arr)]
    mu=float(arr.mean()); sd=float(arr.std(ddof=1))
    z=float((obs-mu)/sd) if sd>0 else None
    p=float((1+np.sum(np.abs(arr-mu) >= abs(obs-mu)))/(len(arr)+1)) if sd>0 else None
    return {'numerator':int(num),'denominator':int(den),'observed_ratio':obs,'null_ratio_mean':mu,'null_ratio_sd':sd,'z':z,'empirical_p_two_sided':p,'n_permutations':int(len(arr))}

def load_records(path='enriched_records.json'):
    obj=json.load(open(path,encoding='utf-8'))
    recs=obj['records']
    lines=defaultdict(list)
    for r in recs:
        lines[(r['folio'],int(r['line_no']))].append(r)
    ordered=[]
    for key,rr in lines.items():
        rr.sort(key=lambda x:int(x['pos']))
        ordered.append((key,rr))
    def folio_key(x):
        f=x[0][0]
        import re
        m=re.match(r'f?(\d+)(.*)',f)
        return (int(m.group(1)) if m else 9999, m.group(2) if m else f, x[0][1])
    ordered.sort(key=folio_key)
    return obj,recs,ordered

def mean_len_class(a,b):
    m=(len(a)+len(b))/2.0
    if m <= 4: return 'short'
    if m <= 6: return 'mid'
    return 'long'

def substitution_half(a,b):
    idx=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None)
    if idx is None: return None
    return 'sub_first_half' if idx < len(a)/2.0 else 'sub_second_half'

def slot_change(r1,r2):
    dif=[s for s in SLOTS if r1[s] != r2[s]]
    if len(dif)==1: return dif[0]
    if len(dif)==0: return 'no_slot_change'
    return 'multi:' + '+'.join(dif)

def core_pair_class(r1,r2):
    a=bool(r1['empty_core']); b=bool(r2['empty_core'])
    if a and b: return 'both_empty'
    if (not a) and (not b): return 'both_nonempty'
    return 'mixed'

def evaluate(lines, permutations=None):
    # Return aggregate plus folio/section counts. If permutations is dict key->index array, use it.
    B=Counter(); C=Counter(); slot=Counter(); corepair=Counter(); e2struct=Counter()
    fol=defaultdict(lambda:Counter({'opp_adj':0})); sec=defaultdict(Counter)
    for (folio,line_no),rr0 in lines:
        rr=rr0 if permutations is None else [rr0[i] for i in permutations[(folio,line_no)]]
        n=len(rr)
        section=rr0[0]['section'] if rr0 else 'UNKNOWN'
        fol[folio]['opp_adj'] += max(0,n-1)
        for i in range(n-1):
            a=rr[i]['token']; b=rr[i+1]['token']
            if is_ed1(a,b):
                fol[folio]['ED1'] += 1; sec[section]['ED1'] += 1
                B[mean_len_class(a,b)] += 1
                if len(b)>len(a): C['accretion'] += 1
                elif len(b)<len(a): C['reduction'] += 1
                else:
                    C['substitution'] += 1
                    h=substitution_half(a,b)
                    if h: C[h] += 1
                slot[slot_change(rr[i],rr[i+1])] += 1
                corepair[core_pair_class(rr[i],rr[i+1])] += 1
        for i in range(n-2):
            if rr[i]['token'] == rr[i+2]['token']:
                fol[folio]['E2'] += 1; sec[section]['E2'] += 1
                e2struct['empty_core' if rr[i]['empty_core'] else 'nonempty_core'] += 1
                e2struct['carrier' if rr[i]['token'] in CARRIERS else 'noncarrier'] += 1
    return {'B':B,'C':C,'slot':slot,'corepair':corepair,'e2struct':e2struct,'folio':fol,'section':sec}

def main():
    obj,recs,lines=load_records()
    rng=np.random.default_rng(SEED)
    obs=evaluate(lines)
    sims=[]
    for _ in range(NPERM):
        perms={key:rng.permutation(len(rr)) for key,rr in lines}
        sims.append(evaluate(lines,perms))

    out={'metadata':{'seed':SEED,'nperm':NPERM,'tokens':len(recs),'lines':len(lines),'folios_total':len(set(r['folio'] for r in recs))},'abc':{},'fw':{},'p70_bridge':{},'audit':{}}

    # ABC A: excess vectors by folio and section.
    eligible=[f for f,c in obs['folio'].items() if c['opp_adj']>=40]
    ed=[]; e2=[]; fol_rows=[]
    for f in eligible:
        mu_ed=float(np.mean([s['folio'][f]['ED1'] for s in sims])); mu_e2=float(np.mean([s['folio'][f]['E2'] for s in sims]))
        xe=obs['folio'][f]['ED1']-mu_ed; x2=obs['folio'][f]['E2']-mu_e2
        ed.append(xe); e2.append(x2); fol_rows.append({'folio':f,'ed1_excess':xe,'e2_excess':x2,'adj_opportunities':obs['folio'][f]['opp_adj']})
    rho,p=spearmanr(ed,e2)
    sections=sorted(obs['section'])
    sed=[]; se2=[]; sec_rows=[]
    for sec in sections:
        mu_ed=float(np.mean([s['section'][sec]['ED1'] for s in sims])); mu_e2=float(np.mean([s['section'][sec]['E2'] for s in sims]))
        xe=obs['section'][sec]['ED1']-mu_ed; x2=obs['section'][sec]['E2']-mu_e2
        sed.append(xe); se2.append(x2); sec_rows.append({'section':sec,'ed1_excess':xe,'e2_excess':x2})
    srho,sp=spearmanr(sed,se2)
    verdict='TWO_PROCESSES' if (rho<0 or abs(rho)<0.15) else ('ONE_PROCESS' if rho>=.3 and p<.05 else 'UNRESOLVED')
    out['abc']['A']={'folio':{'rho':float(rho),'p':float(p),'n':len(eligible),'verdict':verdict,'rows':fol_rows},'section':{'rho':float(srho),'p':float(sp),'n':len(sections),'rows':sec_rows},'historical_rho':HIST_ABC['rho'],'historical_n':HIST_ABC['n_folios']}

    # ABC B
    bout={}
    for k in ('short','mid','long'):
        bout[k]=scalar(obs['B'][k],[s['B'][k] for s in sims])
        bout[k]['historical']={'observed':HIST_ABC['B'][k][0],'null_mean':HIST_ABC['B'][k][1],'null_sd':HIST_ABC['B'][k][2]}
    long=bout['long']; bout['verdict']='CROWDING_FALSIFIED' if long['ratio'] is not None and long['ratio']>=1.15 and long['z'] is not None and long['z']>=2 else 'CROWDING_NOT_FALSIFIED'
    bout['ReM_control']='REQUIRED_CONTROL_UNRECOVERED: exact ReM identity/bytes absent from supplied ABC/FW artefacts and recovery base; no substitute used.'
    out['abc']['B']=bout

    # ABC C marginals + correct ratio distributions
    cout={'marginals':{}}
    for k in ('accretion','reduction','substitution','sub_first_half','sub_second_half'):
        cout['marginals'][k]=scalar(obs['C'][k],[s['C'][k] for s in sims])
        cout['marginals'][k]['historical']={'observed':HIST_ABC['C'][k][0],'null_mean':HIST_ABC['C'][k][1],'null_sd':HIST_ABC['C'][k][2]}
    cout['accretion_reduction_ratio']=ratio_scalar(obs['C']['accretion'],obs['C']['reduction'],[s['C']['accretion'] for s in sims],[s['C']['reduction'] for s in sims])
    cout['substitution_site_ratio']=ratio_scalar(obs['C']['sub_first_half'],obs['C']['sub_second_half'],[s['C']['sub_first_half'] for s in sims],[s['C']['sub_second_half'] for s in sims])
    zvals=[abs(cout['accretion_reduction_ratio']['z'] or 0),abs(cout['substitution_site_ratio']['z'] or 0)]
    cout['verdict']='DIRECTIONAL_SIGNATURE_PRESENT' if max(zvals)>=2 else 'DIRECTIONAL_SIGNATURE_ABSENT'
    out['abc']['C']=cout

    # FW B-slot completion
    lines_tokens=[[r['token'] for r in rr] for _,rr in lines]
    per={c:[] for c in CARRIERS}
    for t in lines_tokens:
        for i in range(len(t)-2):
            if t[i] == t[i+2] and t[i] in per:
                per[t[i]].append(t[i+1])
    fwtypes={}; pooled=[]
    for c in CARRIERS:
        vals=per[c]; pooled.extend(vals); H=shannon(vals); maxH=math.log2(len(vals)) if len(vals)>=2 else 0.0; frac=H/maxH if maxH>0 else 0.0
        fwtypes[c]={'n_aba':len(vals),'H_B_bits':H,'max_H_given_n':maxH,'entropy_fraction':frac,'powered_individually':len(vals)>=30,'distinct_B':len(set(vals))}
    Hp=shannon(pooled); maxHp=math.log2(len(pooled)) if len(pooled)>=2 else 0.0; fracp=Hp/maxHp if maxHp>0 else 0.0
    if fracp>=.80: fwb='FUNCTION_WORD_COMPATIBLE_ON_BSLOT_ONLY'
    elif fracp<.50: fwb='LOW_BSLOT_ENTROPY_REPEATED_PHRASE_LIKE'
    else: fwb='INTERMEDIATE_BSLOT_VARIABILITY'
    # frozen other two legs already point opposite function-word prediction
    formal_fw='FUNCTION_WORD_READING_FALSIFIED' if (HIST_FW['pos_ent'][0]>HIST_FW['pos_ent'][1] and HIST_FW['breadth'][0]<HIST_FW['breadth'][1]) or fracp<.50 else 'MIXED_OR_UNRESOLVED'
    out['fw']['bslot']={'per_carrier':fwtypes,'pooled':{'n':len(pooled),'H_B_bits':Hp,'max_H_given_n':maxHp,'entropy_fraction':fracp,'bslot_reading':fwb},'combined_original_rule_verdict':formal_fw}
    out['fw']['historical_other_legs']=HIST_FW

    # FW post-hoc sensitivity matching + carrier P70 profile
    freq=Counter(r['token'] for r in recs); ranked=[x for x,_ in freq.most_common()]
    pool=ranked[39:200]  # 1-index ranks 40..200
    unused=set(pool); controls=[]
    for c in CARRIERS:
        if not unused: break
        best=min(unused,key=lambda x:(abs(freq[x]-freq[c]),ranked.index(x)))
        controls.append(best); unused.remove(best)
    bytoken=defaultdict(list)
    for r in recs: bytoken[r['token']].append(r)
    def profile(types):
        occ=[r for t in types for r in bytoken[t]]
        empty=sum(bool(r['empty_core']) for r in occ)/len(occ) if occ else 0
        pos_macro=float(np.mean([shannon([r['rel_pos'] for r in bytoken[t]]) for t in types if bytoken[t]]))
        # distinct immediate neighbour types per occurrence, pooled by token then macro mean
        neigh=defaultdict(set); occn=Counter()
        for _,rr in lines:
            for i,r in enumerate(rr):
                tok=r['token']
                if tok in types:
                    occn[tok]+=1
                    if i>0: neigh[tok].add(rr[i-1]['token'])
                    if i+1<len(rr): neigh[tok].add(rr[i+1]['token'])
        breadth_macro=float(np.mean([len(neigh[t])/occn[t] for t in types if occn[t]>0]))
        return {'occurrences':len(occ),'empty_core_fraction':empty,'mean_type_relpos_entropy':pos_macro,'mean_type_neighbor_breadth_per_occurrence':breadth_macro}
    out['fw']['control_sensitivity']={'status':'POST_HOC_NEAREST_FREQUENCY_NO_REPLACEMENT; does not replace underspecified historical matching','controls':controls,'carrier_profile':profile(CARRIERS),'control_profile':profile(controls),'whole_corpus_empty_core_fraction':sum(bool(r['empty_core']) for r in recs)/len(recs)}
    cprofiles={}
    for c in CARRIERS:
        rr=bytoken[c]; parses=Counter((r['prefix'],r['gallows'],r['core'],r['suffix']) for r in rr)
        dominant,n=parse=parses.most_common(1)[0][0],parses.most_common(1)[0][1],None
        cprofiles[c]={'frequency':len(rr),'empty_core_fraction':sum(bool(r['empty_core']) for r in rr)/len(rr) if rr else None,'dominant_parse':{'prefix':dominant[0],'gallows':dominant[1],'core':dominant[2],'suffix':dominant[3]},'dominant_parse_share':n/len(rr) if rr else None}
    out['p70_bridge']['carrier_profiles']=cprofiles

    # P70 ED1 categories
    slot_keys=sorted(set(obs['slot']) | {k for s in sims for k in s['slot']})
    slotout={k:scalar(obs['slot'][k],[s['slot'][k] for s in sims]) for k in slot_keys}
    corekeys=('both_empty','mixed','both_nonempty')
    coreout={k:scalar(obs['corepair'][k],[s['corepair'][k] for s in sims]) for k in corekeys}
    # positive excess shares
    pos_ex={k:max(0.0,v['observed']-v['null_mean']) for k,v in slotout.items()}; den=sum(pos_ex.values())
    for k in slotout: slotout[k]['positive_excess_share']=pos_ex[k]/den if den>0 else 0.0
    out['p70_bridge']['ED1']={'slot_change_categories':slotout,'core_state_pairs':coreout}

    # P70 E2 endpoint classes
    e2out={}
    for k in ('empty_core','nonempty_core','carrier','noncarrier'):
        e2out[k]=scalar(obs['e2struct'][k],[s['e2struct'][k] for s in sims])
    for pair in [('empty_core','nonempty_core'),('carrier','noncarrier')]:
        pos={k:max(0.0,e2out[k]['observed']-e2out[k]['null_mean']) for k in pair}; d=sum(pos.values())
        for k in pair: e2out[k]['positive_excess_share_within_partition']=pos[k]/d if d>0 else 0.0
    out['p70_bridge']['E2']=e2out

    # Reproduction audit
    out['audit']['abc_rho_delta']=float(rho-HIST_ABC['rho'])
    out['audit']['abc_n_folios_delta']=len(eligible)-HIST_ABC['n_folios']
    out['audit']['B_observed_match']={k:int(obs['B'][k])==HIST_ABC['B'][k][0] for k in ('short','mid','long')}
    out['audit']['C_observed_match']={k:int(obs['C'][k])==HIST_ABC['C'][k][0] for k in HIST_ABC['C']}

    os.makedirs('results/abc_fw_recovery_p70_bridge_v0_1',exist_ok=True)
    jpath='results/abc_fw_recovery_p70_bridge_v0_1/RESULTS_20260815.json'
    with open(jpath,'w',encoding='utf-8') as f: json.dump(out,f,indent=2,ensure_ascii=False)

    # concise human-readable synthesis
    A=out['abc']['A']['folio']; B=out['abc']['B']; C=out['abc']['C']; F=out['fw']['bslot']['pooled']; E=out['p70_bridge']['E2']; S=out['p70_bridge']['ED1']['slot_change_categories']
    top_slots=sorted(S.items(),key=lambda kv:kv[1]['positive_excess_share'],reverse=True)[:6]
    powered=[(k,v) for k,v in fwtypes.items() if v['powered_individually']]
    md=[]
    md += ['# ABC/FW recovery + P70 bridge v0.1 — results','',f"Tokens: {len(recs):,}; lines: {len(lines):,}; permutations: {NPERM}; seed: {SEED}.",'']
    md += ['## Reproduction audit',f"- Historical ABC folio rho: {HIST_ABC['rho']:.6f}; recovery rho: **{A['rho']:.6f}** (p={A['p']:.4g}, n={A['n']}).",f"- B observed counts exact match: `{out['audit']['B_observed_match']}`.",f"- C observed counts exact match: `{out['audit']['C_observed_match']}`.",f"- ReM: **{B['ReM_control']}**",'']
    md += ['## ABC-A — one or two legs?',f"**{A['verdict']}**. Folio rho={A['rho']:.4f}, p={A['p']:.4g}, n={A['n']}. Section rho={out['abc']['A']['section']['rho']:.4f}, p={out['abc']['A']['section']['p']:.4g}, n={out['abc']['A']['section']['n']}.",'']
    md += ['## ABC-B — crowding vs structure','| class | observed | null mean | ratio | z |','|---|---:|---:|---:|---:|']
    for k in ('short','mid','long'):
        q=B[k]; md.append(f"| {k} | {q['observed']:.0f} | {q['null_mean']:.3f} | {q['ratio']:.3f} | {q['z']:.2f} |")
    md += ['',f"Verdict: **{B['verdict']}**.",'']
    md += ['## ABC-C — production-order direction',f"- Accretion/reduction: observed {C['accretion_reduction_ratio']['observed_ratio']:.4f}, null {C['accretion_reduction_ratio']['null_ratio_mean']:.4f} ± {C['accretion_reduction_ratio']['null_ratio_sd']:.4f}, z={C['accretion_reduction_ratio']['z']:.2f}, empirical p={C['accretion_reduction_ratio']['empirical_p_two_sided']:.4g}.",f"- First/second substitution site: observed {C['substitution_site_ratio']['observed_ratio']:.4f}, null {C['substitution_site_ratio']['null_ratio_mean']:.4f} ± {C['substitution_site_ratio']['null_ratio_sd']:.4f}, z={C['substitution_site_ratio']['z']:.2f}, empirical p={C['substitution_site_ratio']['empirical_p_two_sided']:.4g}.",f"Verdict: **{C['verdict']}**.",'']
    md += ['## FW — missing B-slot test recovered',f"Pooled carrier ABA events: n={F['n']}; H(B)={F['H_B_bits']:.3f} bits; H/max={F['entropy_fraction']:.3f}. B-slot reading: **{F['bslot_reading']}**.",f"Powered individual carriers (n>=30): {', '.join(k for k,v in powered) if powered else 'none'}.",f"Combined with the frozen positional/breadth directions, original FW decision: **{out['fw']['bslot']['combined_original_rule_verdict']}**.",'']
    md += ['## P70 bridge — ED1 slot localisation','| slot-change class | obs | null | ratio | z | positive excess share |','|---|---:|---:|---:|---:|---:|']
    for k,q in top_slots:
        md.append(f"| {k} | {q['observed']:.0f} | {q['null_mean']:.2f} | {q['ratio']:.3f} | {q['z']:.2f} | {q['positive_excess_share']:.3f} |")
    md += ['','## P70 bridge — lag-2 structural carriers','| endpoint class | obs | null | ratio | z | positive excess share |','|---|---:|---:|---:|---:|---:|']
    for k in ('empty_core','nonempty_core','carrier','noncarrier'):
        q=E[k]; md.append(f"| {k} | {q['observed']:.0f} | {q['null_mean']:.2f} | {q['ratio']:.3f} | {q['z']:.2f} | {q.get('positive_excess_share_within_partition',0):.3f} |")
    cp=out['fw']['control_sensitivity']
    md += ['',f"Carrier occurrence empty-core fraction: **{cp['carrier_profile']['empty_core_fraction']:.3f}**; whole corpus: {cp['whole_corpus_empty_core_fraction']:.3f}; post-hoc frequency sensitivity controls: {cp['control_profile']['empty_core_fraction']:.3f}.",'']
    md += ['## Interpretation guardrail','This branch recovers the August 13 tests and runs a registered-after-the-fact P70 mechanistic bridge. It does not constitute decipherment and the bridge is not an independent confirmation sample. The unidentified ReM control remains explicitly open.']
    mpath='results/abc_fw_recovery_p70_bridge_v0_1/RESULTS_20260815.md'
    with open(mpath,'w',encoding='utf-8') as f: f.write('\n'.join(md)+'\n')
    print('\n'.join(md))

if __name__=='__main__': main()
