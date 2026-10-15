#!/usr/bin/env python3
"""Atomic, read-only source/provenance audit for the VBM 2026-10-14 contract.
No Voynich model fitting or historical inference is performed.
"""
from pathlib import Path
from hashlib import sha256
import json,pickle,os,tempfile,zipfile,sys
BASE=Path('/mnt/data'); OUT=BASE/'vbm_contract_20261014'
OUT.mkdir(exist_ok=True)
def atomic(path,data,kind):
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as f:
            if kind=='pickle':pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)
            else:f.write((json.dumps(data,indent=2,ensure_ascii=False)+'\n').encode())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def h(path):return sha256(path.read_bytes()).hexdigest()
def require(cond,msg):
    if not cond:raise RuntimeError('UNVERIFIED: '+msg)
original=BASE/'Generator_v138_Pruefpaket_2026-10-14.zip'
verdict138=BASE/'audit_summary.json'
verdict142=BASE/'vbm_v142_inband/05_pair_bridge_results.json'
contract142=BASE/'vbm_v142_inband/03_pair_bridge_contract.json'
report142=BASE/'vbm_v142_inband/VBM_v142_Admission_Report_2026-10-14.md'
for f in [original,verdict138,verdict142,contract142,report142]:require(f.exists(),f'{f} unavailable')
with zipfile.ZipFile(original) as z:
    original_readme=z.read('gen138/README.md').decode('utf-8')
    original_code=z.read('gen138/vbm_tafel_v138.py').decode('utf-8')
require('VL' not in original_readme or 'Bruecke' in original_readme,'unexpected README')
require('Bruecke am Ende des Tokens' in original_readme and 'Schluessel am Anfang des naechsten' in original_readme,'cross-space bridge original specification')
require('toks.append' in original_code and 'pair_v=' in original_code,'v138 code structure')
contract=json.loads(contract142.read_text()); v142=json.loads(verdict142.read_text()); a=json.loads(verdict138.read_text())
require('inside' in contract['scope'].lower() or 'in-word' in contract['scope'].lower(),'v142 explicit in-word scope')
require(v142['controls']['train_exact'] is True and v142['controls']['hartlieb_exact'] is True,'exact synthetic roundtrip')
require(v142['gate']['0_bad_trigram'] is False,'existing surface gate failure')
require(a['stage1']['n_cluster_types']==722 and a['stage1']['n_output_cells']==443,'original inventory')
require(a['stage1']['n_collision_cells']==147,'original collisions')
assert 'Original bridges are **inside** the word' in report142.read_text()
provenance={
 'source_zip':{'path':str(original),'sha256':h(original)},
 'source_audit':{'path':str(verdict138),'sha256':h(verdict138)},
 'pair_result':{'path':str(verdict142),'sha256':h(verdict142)},
 'pair_contract':{'path':str(contract142),'sha256':h(contract142)},
 'pair_report':{'path':str(report142),'sha256':h(report142)},
 'facts_validated':{'author_spec_cross_space':True,'v142_b_in_word':True,'v142_roundtrip_both':True,'v142_surface_gate_fails':True},
}
atomic(OUT/'stage0_provenance.pkl',provenance,'pickle')
engineering={
 'v138':{'cluster_types':a['stage1']['n_cluster_types'],'cells':a['stage1']['n_output_cells'],'collision_cells':a['stage1']['n_collision_cells'],'generated_tokens':a['stage1']['generated_tokens'],'bad_trigram_tokens':a['stage1']['trigram_violating_tokens']},
 'v142b':{'development_words':v142['train']['words'],'hartlieb_words':v142['hartlieb']['words'],'development_bad_trigram':v142['train']['bad_trigram_count'],'hartlieb_bad_trigram':v142['hartlieb']['bad_trigram_count'],'roundtrip_both':True,'cross_space_bridges_retained':False},
 'comparison_type':'implementation audit, not a historical mechanism comparison',
 'null_sd':'not applicable to deterministic consistency checks; none estimated for mechanism identification',
}
atomic(OUT/'stage1_lineage_checks.pkl',engineering,'pickle')
preflight={
 'question':'Is there an available precommitted, identifiable comparison of plaintext VBM versus autonomous slot generation on existing ciphertext alone?',
 'P':'plaintext source + cipher transformation',
 'S':'autonomous process with exactly the same observable marginal distribution for the ciphertext',
 'mathematical_relation':'P_P(C=c)=sum_m P(M=m) P(C=c|M=m); define P_S(C=c) to equal that marginal for each c',
 'observational_discriminator_without_external_anchor':False,
 'restricted_generator_approach':'Possible only if both mechanisms are fully specified with bounded complexity and held-out predictions, but it cannot establish general historical encoding',
 'data_status':{'external_anchor_verified':False,'author_frozen_essentials_confirmed':False,'prior_q65_v10_identification_passed':False},
 'decision':{'more_plain_ciphertext_generator_tuning':'STOP','v138_engineering_claims':'remain failed','v142_exact_roundtrip':'retain as related engineering proof only','seek_author_specification':'GO','external_anchor_candidate_registration':'GO','historical_test':'NOT ADMITTED'},
 'note':'The absence of independently anchored plaintext is status of available evidence in this investigation, not a claim none exists anywhere.'
}
atomic(OUT/'stage2_identifiability.pkl',preflight,'pickle')
result={'provenance':provenance,'engineering':engineering,'preflight':preflight}
atomic(OUT/'RESULTS.json',result,'json')
atomic(OUT/'stage3_decision.pkl',result,'pickle')
print(json.dumps({'source_sha256':provenance['source_zip']['sha256'],'v138':engineering['v138'],'v142b':engineering['v142b'],'decision':preflight['decision']},indent=2))
