#!/usr/bin/env python3
"""
Gen-SS1: Gen-SP with ONE intervention: paragraph-reset wheel+spokes line openers.

Everything after the first token of each physical line is inherited from
gen_scribal_p70c.P70CScribe unchanged in spirit:
- P70-C slot inventory
- section-specific gallows/suffix distributions
- suffix-changing copy/mutate rule
- 20% fresh-word rule
- FIRST/MID/LAST position

The only changed mechanism is LINE_START:
- paragraph lengths and paragraph seeds come from frozen ZL3b parameters
- continuation opener state follows frozen wheel+spokes parameters
- a valid P70-C FIRST quint is sampled conditional on the requested opener

This is an ablation, not a tuned generator.
"""
from __future__ import annotations
import json, pickle, random, sys
from pathlib import Path
from collections import defaultdict

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
from gen_scribal_p70c import build_p70c_spec, P70CScribe

COMPOUNDS=("cfh","cph","ckh","cth","ch","sh")

def atom_start(s):
    s=''.join(ch for ch in str(s or '').lower() if 'a'<=ch<='z')
    if not s:
        return None
    for a in COMPOUNDS:
        if s.startswith(a):
            return a
    return s[0]

def _nz(x):
    return '' if x in (None,'','∅') else str(x)

def entry_stem(e):
    return _nz(e.get('prefix'))+_nz(e.get('gallows'))+_nz(e.get('m_core'))

def entry_opener(e):
    st=entry_stem(e)
    if st:
        return atom_start(st)
    for s in e.get('full_suffixes',[]) or []:
        if s not in ('','∅',None):
            return atom_start(s)
    return None

def weighted_choice(rng, mapping):
    if not mapping:
        return None
    items=list(mapping)
    weights=[float(mapping[k]) for k in items]
    return rng.choices(items,weights=weights,k=1)[0]

def build_spec(p70c_path='Paper/p70c_full_spec_v1.json',
               records_path='enriched_records.pkl',
               wheel_path='research/sgt13_wheel_spokes_params_20261002.json'):
    spec=build_p70c_spec(p70c_path=p70c_path,records_path=records_path)
    with open(p70c_path,encoding='utf-8') as f:
        p70=json.load(f)
    with open(wheel_path,encoding='utf-8') as f:
        wheel=json.load(f)
    by=defaultdict(list)
    all_first=[]
    for e in p70['entries']:
        if e.get('position')!='FIRST':
            continue
        o=entry_opener(e)
        if not o:
            continue
        by[o].append(e)
        all_first.append(e)
    spec['_line_start_entries']={k:v for k,v in by.items()}
    spec['_all_first_entries']=all_first
    spec['_wheel']=wheel
    return spec

class SmallStateScribe(P70CScribe):
    def __init__(self,spec,section='Herbal-A',seed=42):
        super().__init__(spec,section=section,seed=seed)
        self.wheel=spec['_wheel']
        self.start_entries=spec['_line_start_entries']
        self.all_start_entries=spec['_all_first_entries']

    def _entry_token(self,e):
        suffixes=[s for s in (e.get('full_suffixes') or ['']) if s not in (None,'∅')]
        suffix=self.rng.choice(suffixes) if suffixes else ''
        p=_nz(e.get('prefix')); g=_nz(e.get('gallows')); c=_nz(e.get('m_core'))
        tok=(p+g+c+suffix) or 'o'
        return tok,(e.get('prefix','∅'),e.get('gallows','∅'),
                    e.get('m_core','∅'),e.get('sfx_fam','BARE'))

    def forced_line_start(self,desired):
        cand=self.start_entries.get(desired,[])
        fallback=not bool(cand)
        if fallback:
            # fallback only if the exact opener is absent from valid FIRST quints
            cand=self.all_start_entries
        weights=[max(float(e.get('count',1)),1.0) for e in cand]
        e=self.rng.choices(cand,weights=weights,k=1)[0]
        tok,slots=self._entry_token(e)
        return tok,slots,atom_start(tok),fallback

    def sample_paragraph_length(self):
        return int(weighted_choice(self.rng,self.wheel['paragraph_length']))

    def seed_opener(self):
        return weighted_choice(self.rng,self.wheel['paragraph_initial'])

    def next_opener(self,prev,transition_index):
        R=self.wheel['regimes'][0 if transition_index==0 else 1]
        anchors=set(self.wheel['anchors'])
        if prev in anchors:
            move=weighted_choice(self.rng,R['wheel'])
            if move=='stay':
                return prev
            if move=='cw':
                return self.wheel['cw'][prev]
            if move=='ccw':
                return self.wheel['ccw'][prev]
            dist=R['anchorBg'].get(prev) or R['bg']
            return weighted_choice(self.rng,dist)
        dist=R['xSource'].get(prev) or R['xcat']
        if not dist:
            dist=R['xcat']
        cat=weighted_choice(self.rng,dist)
        if cat!='X':
            return cat
        return weighted_choice(self.rng,R['bg'])

    def write_section_detailed(self,n_tokens,tokens_per_line=10):
        corpus=[]; lines=[]; paragraphs=[]; current_para=[]; opener_requests=[]
        slots_history=[]; prev_sfx='LINE_START'
        para_left=0; para_index=0; prev_opener=None

        while len(corpus)<n_tokens:
            n_line=min(tokens_per_line,n_tokens-len(corpus))
            if para_left<=0:
                desired=self.seed_opener()
                para_left=self.sample_paragraph_length()
                para_index=0
                prev_opener=None
                if current_para:
                    paragraphs.append(current_para)
                current_para=[]
            else:
                desired=self.next_opener(prev_opener,para_index-1)

            line=[]; line_slots=[]
            tok,slots,realized,fallback=self.forced_line_start(desired)
            opener_requests.append({'desired':desired,'realized':realized,'fallback':fallback})
            line.append(tok); line_slots.append(slots)
            slots_history.append(slots)
            prev_sfx=slots[3]

            for j in range(1,n_line):
                position='LAST' if j==n_line-1 else 'MID'
                if self.rng.random()<0.20:
                    word,sf=self.fresh_word(position=position,prev_sfx_fam=prev_sfx)
                    slots=('∅','∅','∅',sf)
                elif slots_history:
                    template=self.rng.choice(slots_history[-min(5,len(slots_history)):])
                    word,slots=self.mutate_word(template,prev_sfx_fam=prev_sfx)
                    sf=slots[3]
                else:
                    word,sf=self.fresh_word(position=position,prev_sfx_fam=prev_sfx)
                    slots=('∅','∅','∅',sf)
                line.append(word);line_slots.append(slots);slots_history.append(slots);prev_sfx=sf

            corpus.extend(line);lines.append(line);current_para.append(line)
            prev_opener=realized
            para_left-=1;para_index+=1
            prev_sfx='LINE_START'

        if current_para:
            paragraphs.append(current_para)
        return {'tokens':corpus,'lines':lines,'paragraphs':paragraphs,'opener_requests':opener_requests}

def produce_manuscript_detailed(spec,n_tokens=37465,seed=42,tokens_per_line=10):
    total_vms=sum(spec['section_counts'].values())
    sections=spec['sections']
    out_tokens=[];out_lines=[];out_paras=[];out_requests=[];remaining=n_tokens
    for idx,section in enumerate(sections):
        if idx==len(sections)-1:
            n_sec=remaining
        else:
            n_sec=int(round(spec['section_counts'][section]/total_vms*n_tokens))
            n_sec=min(n_sec,remaining)
        remaining-=n_sec
        s=SmallStateScribe(spec,section=section,seed=seed+idx*1000)
        d=s.write_section_detailed(n_sec,tokens_per_line=tokens_per_line)
        out_tokens.extend(d['tokens']);out_lines.extend(d['lines']);out_paras.extend(d['paragraphs']);out_requests.extend(d['opener_requests'])
    return {'tokens':out_tokens[:n_tokens],'lines':out_lines,'paragraphs':out_paras,'opener_requests':out_requests}

def produce_manuscript(spec,n_tokens=37465,seed=42):
    return produce_manuscript_detailed(spec,n_tokens=n_tokens,seed=seed)['tokens']
