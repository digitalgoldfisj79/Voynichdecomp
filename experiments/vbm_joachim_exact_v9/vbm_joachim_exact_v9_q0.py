#!/usr/bin/env python3
from __future__ import annotations
import collections, hashlib, json, math, re, urllib.request

NS='VBMJOACHIMEXACTV9Q0'
DATA_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/gpt56/vbm-bridge-factor-v0.2-20260821/voynich_transcriptions_slim.json'
H1={'f28v','f31v','f88r','f5r','f34r','f81v'}
C1={'f85r1','f53v','f33r','f10r','f23r','f111r'}
VOWELS='aeiou'
CONSONANTS='bcdfghjklmnpqrstvwxyz'
assert len(CONSONANTS)==21

FIXTURE='dcheedy kchedy lcheey ror al chokedy dol qokeeeos qolkeedy qokar ar'
EXPECTED_TRIPLES=[
 ('d','cheed','y'),('k','ched','y'),('l','chee','y'),('r','o','r'),('a','','l'),
 ('ch','oked','y'),('d','o','l'),('qo','keeeo','s'),('qo','lkeed','y'),('qo','ka','r'),('a','','r')]
EXPECTED_BRIDGES=['y|k','y|l','y|r','r|a','l|ch','y|d','l|qo','s|qo','y|qo','r|a']
BRIDGE_MAP={'y|k':'i','y|l':'i','y|r':'a','r|a':'e','l|ch':'u','y|d':'e','l|qo':'e','s|qo':'i','y|qo':'e'}
NUCLEUS_MAP={'cheed':'t','ched':'zs','chee':'chtr','o':'g','oked':'nd','keeeo':'tn','lkeed':'chtsn','ka':'ll'}
EXPECTED_PLAIN='tizsichtrageundegetnichtsnelle'

UA={'User-Agent':'VBMJoachimExactV9/2026-09-01'}

def get_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r: return json.load(r)

def left_half(w):
    if len(w)>=3 and w.startswith('qo'): return 'qo'
    if len(w)>=3 and w.startswith('ch'): return 'ch'
    return w[0]

def parse_token(w,single_shared=True):
    if not re.fullmatch(r'[a-z]+',w): return None
    if len(w)==1:
        if not single_shared: return None
        return (w,'',w)
    L=left_half(w); R=w[-1]
    if len(w)<len(L)+1: return None
    N=w[len(L):-1]
    return (L,N,R)

def parse_words(words,single_shared=True):
    triples=[]; bridges=[]
    for w in words:
        t=parse_token(w,single_shared)
        if t is None: raise ValueError(w)
        triples.append(t)
    for a,b in zip(triples,triples[1:]): bridges.append(a[2]+'|'+b[0])
    return triples,bridges

def fixture_audit():
    words=FIXTURE.split(); triples,bridges=parse_words(words,True)
    parse_ok=(triples==EXPECTED_TRIPLES and bridges==EXPECTED_BRIDGES)
    pieces=[]
    repeated_bridge_ok=True; repeated_nucleus_ok=True
    seenb={}; seenn={}
    for i,(L,N,R) in enumerate(triples):
        if N:
            if N not in NUCLEUS_MAP: raise RuntimeError(('fixture nucleus unmapped',N))
            val=NUCLEUS_MAP[N]; pieces.append(val)
            if N in seenn and seenn[N]!=val: repeated_nucleus_ok=False
            seenn[N]=val
        if i<len(bridges):
            b=bridges[i]
            if b not in BRIDGE_MAP: raise RuntimeError(('fixture bridge unmapped',b))
            val=BRIDGE_MAP[b]; pieces.append(val)
            if b in seenb and seenb[b]!=val: repeated_bridge_ok=False
            seenb[b]=val
    plain=''.join(pieces)
    log5=math.log2(5); log21=math.log2(21)
    bridge_bits=len(BRIDGE_MAP)*log5
    mapped_len=sum(len(x) for x in NUCLEUS_MAP.values())
    nucleus_bits=len(NUCLEUS_MAP)*log5+mapped_len*log21
    raw_n_choices=sum(21**l for l in range(1,6))
    raw_capacity=len(BRIDGE_MAP)*log5+len(NUCLEUS_MAP)*math.log2(raw_n_choices)
    return {
      'parse_ok':parse_ok,'triples':triples,'bridges':bridges,
      'repeat_bridge_consistent':repeated_bridge_ok,'repeat_nucleus_consistent':repeated_nucleus_ok,
      'decoded_from_supplied_values':plain,'matches_supplied_continuous_plaintext':plain==EXPECTED_PLAIN,
      'unique_bridge_mappings':len(BRIDGE_MAP),'unique_nucleus_mappings':len(NUCLEUS_MAP),
      'mapped_unique_nucleus_consonant_chars':mapped_len,
      'bridge_mapping_bits':bridge_bits,'nucleus_mapping_bits_prefix':nucleus_bits,
      'total_key_bits_prefix_no_headers':bridge_bits+nucleus_bits,
      'raw_mapping_space_bits':raw_capacity,
      'produced_plaintext_chars':len(plain),'key_bits_per_plaintext_char_one_line':(bridge_bits+nucleus_bits)/len(plain)
    }

def split_folio(fid):
    h=hashlib.sha256(f'{NS}::{fid}'.encode()).hexdigest()[:8]
    return 'INTERNAL_HOLDOUT' if int(h,16)%5==0 else 'TRAIN'

def fof(counter):
    bins={'1':0,'2':0,'3':0,'4':0,'5':0,'6-10':0,'11-20':0,'21+':0}
    for n in counter.values():
        if n<=5: bins[str(n)]+=1
        elif n<=10: bins['6-10']+=1
        elif n<=20: bins['11-20']+=1
        else: bins['21+']+=1
    return bins

def top(counter,n=30): return [{'type':k,'count':v} for k,v in counter.most_common(n)]

def new_acc():
    return {'folios':set(),'lines':0,'segments':0,'valid_tokens':0,'single_tokens':0,'invalid_tokens':0,
            'empty_nucleus_events':0,'nonempty_nucleus_events':0,'bridge_events':0,'open_edge_halves':0,
            'nucleus':collections.Counter(),'bridge':collections.Counter()}

def process_line(acc,txt,single_shared=True):
    acc['lines']+=1; raw=txt.split(); segment=[]
    def flush():
        nonlocal segment
        if not segment: return
        triples,bridges=parse_words(segment,single_shared)
        acc['segments']+=1; acc['valid_tokens']+=len(segment); acc['bridge_events']+=len(bridges); acc['open_edge_halves']+=2
        for _,N,_ in triples:
            if N: acc['nonempty_nucleus_events']+=1; acc['nucleus'][N]+=1
            else: acc['empty_nucleus_events']+=1
        acc['bridge'].update(bridges); segment=[]
    for w in raw:
        if len(w)==1: acc['single_tokens']+=1
        p=parse_token(w,single_shared)
        if p is None:
            acc['invalid_tokens']+=1; flush(); continue
        segment.append(w)
    flush()

def finalize(acc):
    nc=acc['nucleus']; bc=acc['bridge']
    out={k:(sorted(v) if isinstance(v,set) else v) for k,v in acc.items() if k not in {'nucleus','bridge'}}
    out.update({
      'unique_nucleus_types':len(nc),'unique_bridge_types':len(bc),
      'nucleus_singleton_types':sum(v==1 for v in nc.values()),'bridge_singleton_types':sum(v==1 for v in bc.values()),
      'nucleus_fof':fof(nc),'bridge_fof':fof(bc),'top_nucleus':top(nc),'top_bridge':top(bc)})
    return out

def coverage(train,hold,key):
    tr=train[key]; ho=hold[key]; total=sum(ho.values()); seen=sum(n for t,n in ho.items() if t in tr)
    return {'occurrence_coverage':seen/max(1,total),'seen_occurrences':seen,'total_occurrences':total,
            'unseen_occurrences':total-seen,'type_coverage':sum(t in tr for t in ho)/max(1,len(ho)),
            'seen_types':sum(t in tr for t in ho),'total_types':len(ho),'unseen_types':sum(t not in tr for t in ho)}

def corpus_audit(data,single_shared=True):
    acc={'TRAIN':new_acc(),'INTERNAL_HOLDOUT':new_acc()}
    excluded={'H1':[],'C1':[]}; transcriber='ZLZI'
    for fid,lines in sorted(data['pages'].items()):
        if fid in H1: excluded['H1'].append(fid); continue
        if fid in C1: excluded['C1'].append(fid); continue
        sp=split_folio(fid); A=acc[sp]; A['folios'].add(fid)
        for lnum in sorted(lines.keys(),key=lambda x:int(x) if str(x).isdigit() else 999999):
            txt=lines[lnum].get('t',{}).get(transcriber,'')
            if txt: process_line(A,txt,single_shared)
    cn=coverage(acc['TRAIN'],acc['INTERNAL_HOLDOUT'],'nucleus'); cb=coverage(acc['TRAIN'],acc['INTERNAL_HOLDOUT'],'bridge')
    Kb=len(acc['TRAIN']['bridge']); Kn=len(acc['TRAIN']['nucleus']); log5=math.log2(5); log21=math.log2(21)
    raw_n=sum(21**l for l in range(1,6))
    cap_bridge=Kb*log5; cap_n=Kn*math.log2(raw_n); min_n=Kn*(log5+log21)
    events=acc['TRAIN']['nonempty_nucleus_events']+acc['TRAIN']['bridge_events']
    cost={'train_bridge_types':Kb,'train_nucleus_types':Kn,'bridge_capacity_bits':cap_bridge,
          'nucleus_raw_mapping_capacity_bits':cap_n,'total_raw_mapping_capacity_bits':cap_bridge+cap_n,
          'optimistic_minimum_key_bits_l1_nuclei':cap_bridge+min_n,
          'optimistic_minimum_key_bits_per_train_coded_event':(cap_bridge+min_n)/max(1,events),
          'raw_mapping_capacity_bits_per_train_coded_event':(cap_bridge+cap_n)/max(1,events)}
    return {'primary_rule':'SINGLE_SHARED' if single_shared else 'SINGLE_EXCLUDE','excluded_target_folios':excluded,
            'TRAIN':finalize(acc['TRAIN']),'INTERNAL_HOLDOUT':finalize(acc['INTERNAL_HOLDOUT']),
            'nucleus_heldout_coverage':cn,'bridge_heldout_coverage':cb,'codebook':cost},acc

def main():
    fix=fixture_audit(); data=get_json(DATA_URL)
    primary,raw=corpus_audit(data,True)
    sens,_=corpus_audit(data,False)
    hn=primary['INTERNAL_HOLDOUT']['nonempty_nucleus_events']; hb=primary['INTERNAL_HOLDOUT']['bridge_events']
    nc=primary['nucleus_heldout_coverage']['occurrence_coverage']; bc=primary['bridge_heldout_coverage']['occurrence_coverage']
    gates={'fixture_exact':bool(fix['parse_ok'] and fix['matches_supplied_continuous_plaintext'] and fix['repeat_bridge_consistent'] and fix['repeat_nucleus_consistent']),
           'holdout_nucleus_events_ge_500':hn>=500,'holdout_bridge_events_ge_500':hb>=500,
           'nucleus_occurrence_coverage_ge_090':nc>=.90,'bridge_occurrence_coverage_ge_097':bc>=.97}
    passed=all(gates.values())
    result={'protocol':'VBM_JOACHIM_EXACT_V9_Q0_PROTOCOL.md','namespace':NS,'data_url':DATA_URL,
            'fixture':fix,'primary':primary,
            'single_exclude_sensitivity':{k:sens[k] for k in ['TRAIN','INTERNAL_HOLDOUT','nucleus_heldout_coverage','bridge_heldout_coverage','codebook']},
            'gates':gates,'Q0_pass':passed,'decision':'Q0_PASS_PREREGISTER_Q1' if passed else 'Q0_FAIL_STOP_BEFORE_LANGUAGE'}
    print('VBM_V9_Q0_RESULT='+json.dumps(result,sort_keys=True,separators=(',',':')))

if __name__=='__main__': main()
