#!/usr/bin/env python3
from __future__ import annotations
import collections, hashlib, json, math, re, urllib.request

NS='VBMJOACHIMEXACTV9Q0'
DATA_URL='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/gpt56/vbm-bridge-factor-v0.2-20260821/voynich_transcriptions_slim.json'
H1={'f28v','f31v','f88r','f5r','f34r','f81v'}
C1={'f85r1','f53v','f33r','f10r','f23r','f111r'}
ATOMS=('ckh','cth','cph','cfh','ch','sh','qo')
CONSONANTS='bcdfghjklmnpqrstvwxyz'

FIXTURE='dcheedy kchedy lcheey ror al chokedy dol qokeeeos qolkeedy qokar ar'
EXPECTED_TRIPLES=[('d','cheed','y'),('k','ched','y'),('l','chee','y'),('r','o','r'),('a','','l'),('ch','oked','y'),('d','o','l'),('qo','keeeo','s'),('qo','lkeed','y'),('qo','ka','r'),('a','','r')]
EXPECTED_BRIDGES=['y|k','y|l','y|r','r|a','l|ch','y|d','l|qo','s|qo','y|qo','r|a']
UA={'User-Agent':'VBMJoachimExactV9Q0b/2026-09-01'}

def get_json(url):
    req=urllib.request.Request(url,headers=UA)
    with urllib.request.urlopen(req,timeout=120) as r:return json.load(r)

def left_half(w):
    for a in ATOMS:
        if len(w)>=len(a)+1 and w.startswith(a): return a
    return w[0]

def parse_token(w,single_shared=True):
    if not re.fullmatch(r'[a-z]+',w): return None
    if len(w)==1:
        return (w,'',w) if single_shared else None
    L=left_half(w); R=w[-1]
    if len(w)<len(L)+1:return None
    return (L,w[len(L):-1],R)

def parse_words(words,single_shared=True):
    tr=[]
    for w in words:
        t=parse_token(w,single_shared)
        if t is None:raise ValueError(w)
        tr.append(t)
    br=[a[2]+'|'+b[0] for a,b in zip(tr,tr[1:])]
    return tr,br

def split_folio(fid):
    h=hashlib.sha256(f'{NS}::{fid}'.encode()).hexdigest()[:8]
    return 'INTERNAL_HOLDOUT' if int(h,16)%5==0 else 'TRAIN'

def new_acc():
    return {'folios':set(),'lines':0,'segments':0,'valid_tokens':0,'single_tokens':0,'invalid_tokens':0,'empty_nucleus_events':0,'nonempty_nucleus_events':0,'bridge_events':0,'open_edge_halves':0,'nucleus':collections.Counter(),'bridge':collections.Counter()}

def process_line(A,txt,single_shared=True):
    A['lines']+=1; seg=[]
    def flush():
        nonlocal seg
        if not seg:return
        tr,br=parse_words(seg,single_shared);A['segments']+=1;A['valid_tokens']+=len(seg);A['bridge_events']+=len(br);A['open_edge_halves']+=2
        for _,N,_ in tr:
            if N:A['nonempty_nucleus_events']+=1;A['nucleus'][N]+=1
            else:A['empty_nucleus_events']+=1
        A['bridge'].update(br);seg=[]
    for w in txt.split():
        if len(w)==1:A['single_tokens']+=1
        if parse_token(w,single_shared) is None:A['invalid_tokens']+=1;flush()
        else:seg.append(w)
    flush()

def fof(c):
    b={'1':0,'2':0,'3':0,'4':0,'5':0,'6-10':0,'11-20':0,'21+':0}
    for n in c.values():
        if n<=5:b[str(n)]+=1
        elif n<=10:b['6-10']+=1
        elif n<=20:b['11-20']+=1
        else:b['21+']+=1
    return b

def fin(A):
    n=A['nucleus'];b=A['bridge'];o={k:(sorted(v) if isinstance(v,set) else v) for k,v in A.items() if k not in {'nucleus','bridge'}}
    o.update({'unique_nucleus_types':len(n),'unique_bridge_types':len(b),'nucleus_singleton_types':sum(x==1 for x in n.values()),'bridge_singleton_types':sum(x==1 for x in b.values()),'nucleus_fof':fof(n),'bridge_fof':fof(b),'top_nucleus':n.most_common(30),'top_bridge':b.most_common(30)})
    return o

def cov(tr,ho,key):
    a=tr[key];b=ho[key];tot=sum(b.values());seen=sum(n for t,n in b.items() if t in a)
    return {'occurrence_coverage':seen/max(1,tot),'seen_occurrences':seen,'total_occurrences':tot,'unseen_occurrences':tot-seen,'type_coverage':sum(t in a for t in b)/max(1,len(b)),'seen_types':sum(t in a for t in b),'total_types':len(b),'unseen_types':sum(t not in a for t in b)}

def audit(data,single_shared=True):
    X={'TRAIN':new_acc(),'INTERNAL_HOLDOUT':new_acc()};ex={'H1':[],'C1':[]}
    for fid,lines in sorted(data['pages'].items()):
        if fid in H1:ex['H1'].append(fid);continue
        if fid in C1:ex['C1'].append(fid);continue
        sp=split_folio(fid);A=X[sp];A['folios'].add(fid)
        for ln in sorted(lines,key=lambda x:int(x) if str(x).isdigit() else 999999):
            txt=lines[ln].get('t',{}).get('ZLZI','')
            if txt:process_line(A,txt,single_shared)
    cn=cov(X['TRAIN'],X['INTERNAL_HOLDOUT'],'nucleus');cb=cov(X['TRAIN'],X['INTERNAL_HOLDOUT'],'bridge')
    Kb=len(X['TRAIN']['bridge']);Kn=len(X['TRAIN']['nucleus']);log5=math.log2(5);log21=math.log2(21);raw_n=sum(21**l for l in range(1,6));events=X['TRAIN']['nonempty_nucleus_events']+X['TRAIN']['bridge_events']
    cost={'train_bridge_types':Kb,'train_nucleus_types':Kn,'bridge_capacity_bits':Kb*log5,'nucleus_raw_mapping_capacity_bits':Kn*math.log2(raw_n),'total_raw_mapping_capacity_bits':Kb*log5+Kn*math.log2(raw_n),'optimistic_minimum_key_bits_l1_nuclei':Kb*log5+Kn*(log5+log21),'optimistic_minimum_key_bits_per_train_coded_event':(Kb*log5+Kn*(log5+log21))/max(1,events),'raw_mapping_capacity_bits_per_train_coded_event':(Kb*log5+Kn*math.log2(raw_n))/max(1,events)}
    return {'rule':'PUBLISHED_ATOMS_SINGLE_SHARED' if single_shared else 'PUBLISHED_ATOMS_SINGLE_EXCLUDE','atoms':ATOMS,'excluded_target_folios':ex,'TRAIN':fin(X['TRAIN']),'INTERNAL_HOLDOUT':fin(X['INTERNAL_HOLDOUT']),'nucleus_heldout_coverage':cn,'bridge_heldout_coverage':cb,'codebook':cost}

def main():
    ft,fb=parse_words(FIXTURE.split(),True);fixture=(ft==EXPECTED_TRIPLES and fb==EXPECTED_BRIDGES)
    d=get_json(DATA_URL);p=audit(d,True);s=audit(d,False)
    hn=p['INTERNAL_HOLDOUT']['nonempty_nucleus_events'];hb=p['INTERNAL_HOLDOUT']['bridge_events'];nc=p['nucleus_heldout_coverage']['occurrence_coverage'];bc=p['bridge_heldout_coverage']['occurrence_coverage']
    gates={'fixture_exact':fixture,'holdout_nucleus_events_ge_500':hn>=500,'holdout_bridge_events_ge_500':hb>=500,'nucleus_occurrence_coverage_ge_090':nc>=.90,'bridge_occurrence_coverage_ge_097':bc>=.97}
    out={'protocol':'VBM_JOACHIM_EXACT_V9_Q0B_ATOMS_PROTOCOL.md','namespace':NS,'primary':p,'single_exclude_sensitivity':s,'gates':gates,'Q0b_pass':all(gates.values()),'decision':'Q0B_PASS_Q1_MAY_BE_PREREGISTERED' if all(gates.values()) else 'Q0B_FAIL_STOP_BEFORE_LANGUAGE'}
    print('VBM_V9_Q0B_RESULT='+json.dumps(out,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
