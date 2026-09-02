# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4", "numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2", "triton>=3.0"]
# ///
from __future__ import annotations
import importlib.util, json, runpy, time, urllib.request
from pathlib import Path
import numpy as np

BRANCH='experiment/vbm-v10-terminal-identifiability-20260901'
ROOT=f'https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/{BRANCH}/experiments/vbm_v10_terminal/'
SIZE=2000

# Mandatory software smoke runs in this same paid job.
smoke_path=Path('/tmp/v10_smoke.py')
with urllib.request.urlopen(ROOT+'vbm_v10_gpu_smoke.py',timeout=120) as r:
    smoke_path.write_text(r.read().decode('utf-8'),encoding='utf-8')
runpy.run_path(str(smoke_path),run_name='__main__')

# Load the frozen GPU solver from a real Python file. Triton JIT requires
# inspectable source and therefore must not be defined via exec().
with urllib.request.urlopen(ROOT+'vbm_v10_stage_a_gpu_positive.py',timeout=120) as r:
    src=r.read().decode('utf-8')
repls={
"bp=tl.load(bpos+surf,mask=bm,other=-1).to(tl.int32)":"bp=tl.load(bpos+surf,mask=em & (typ==1),other=-1).to(tl.int32)",
"bv0=tl.load(bmap+surf,mask=bm,other=0).to(tl.int64)":"bv0=tl.load(bmap+surf,mask=em & (typ==1),other=0).to(tl.int64)",
"pw=tl.load(pows+bp,mask=bm & (bp>=0),other=1).to(tl.int64)":"pw=tl.load(pows+bp,mask=em & (typ==1) & (bp>=0),other=1).to(tl.int64)",
"np=tl.load(npos+surf,mask=nm,other=-1).to(tl.int32)":"np=tl.load(npos+surf,mask=em & (typ==2),other=-1).to(tl.int32)",
"nv0=tl.load(nmap+surf,mask=nm,other=0).to(tl.int64)":"nv0=tl.load(nmap+surf,mask=em & (typ==2),other=0).to(tl.int64)",
"pw2=tl.load(pows+np,mask=nm & (np>=0),other=1).to(tl.int64)":"pw2=tl.load(pows+np,mask=em & (typ==2) & (np>=0),other=1).to(tl.int64)",
}
for old,new in repls.items():
    if old not in src:
        raise RuntimeError('expected frozen source fragment missing: '+old)
    src=src.replace(old,new,1)
solver_path=Path('/tmp/vbm_v10_stage_a_gpu_positive_patched3.py')
solver_path.write_text(src,encoding='utf-8')
spec=importlib.util.spec_from_file_location('v10gpu_mod',solver_path)
if spec is None or spec.loader is None:
    raise RuntimeError('failed to create module spec for frozen GPU solver')
G=importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
B=G.B


def random_hold_stats(A,hold,tag):
    vals=[]
    for r in range(20):
        bm,nm=B['init_map'](A,f'{tag}:RAND:{r}')
        vals.append(B['score_lines'](hold,A,{'bmap':bm,'nmap':nm}))
    vals=np.asarray(vals,dtype=float)
    return float(np.median(vals)),float(np.std(vals,ddof=1)),vals.tolist()


def one_o2(lang,rep,A,all_lines,key):
    cur=all_lines[:SIZE];cut=int(SIZE*.8);fit=cur[:cut];hold=cur[cut:]
    bc,nc=G.counts(fit)
    true={'bmap':key['bmap'],'nmap':key['nmap']}
    o0=G.recovery(true,key,hold,bc,nc,A,0)
    o05=G.recovery(true,key,hold,bc,nc,A,5)
    o0.update({'REC_B5':o05['REC_B'],'REC_N5':o05['REC_N'],'REC_CHAR5':o05['REC_CHAR']})
    o0['HOLD_LM']=float(B['score_lines'](hold,A,true))
    base,base_sd,base_vals=random_hold_stats(A,hold,f'FINAL:{lang}:R{rep}:N{SIZE}')
    t=time.time()
    m,chains,bc,nc,fb,fn=G.fit_gpu(A,fit,key,lang,rep,SIZE,'O2')
    if fb or fn:
        raise RuntimeError('O2 unexpectedly revealed key entries')
    r=G.recovery(m,key,hold,bc,nc,A,0)
    r5=G.recovery(m,key,hold,bc,nc,A,5)
    r.update({'REC_B5':r5['REC_B'],'REC_N5':r5['REC_N'],'REC_CHAR5':r5['REC_CHAR']})
    r.update(G.coverage(hold,bc,nc))
    hl=float(B['score_lines'](hold,A,m))
    r.update({
        'HOLD_LM':hl,'RAND_HOLD_LM':base,'RAND_HOLD_SD':base_sd,'HOLD_ADV':hl-base,
        'FIT_LM':float(m['fit_score']),'elapsed_s':time.time()-t,
        'best_chain':max(chains,key=lambda x:(x['fit_score'],-x['chain']))['chain'],
        'chain_fit_scores':[float(x['fit_score']) for x in chains],
        'candidate_evals_total':int(sum(x['candidates_evaluated'] for x in chains)),
    })
    return {'lang':lang,'rep':rep,'size':SIZE,'fit_lines':len(fit),'hold_lines':len(hold),
            'O0_TRUE_KEY':o0,'O2':r,'random_hold_values':base_vals}


def gate(rows):
    o=[r['O2'] for r in rows]
    char_pass=sum(x['REC_CHAR']>=.80 for x in o)
    key_pass=sum(x['REC_B']>=.70 and x['REC_N']>=.70 for x in o)
    frequent_pass=sum(x['REC_CHAR5']>=.90 and x['REC_B5']>=.80 and x['REC_N5']>=.80 for x in o)
    by_lang={}
    for la in ['DE','IT']:
        q=[r['O2'] for r in rows if r['lang']==la]
        by_lang[la]={'char_pass':sum(x['REC_CHAR']>=.80 for x in q),
                     'key_pass':sum(x['REC_B']>=.70 and x['REC_N']>=.70 for x in q),
                     'frequent_pass':sum(x['REC_CHAR5']>=.90 and x['REC_B5']>=.80 and x['REC_N5']>=.80 for x in q)}
    ok=(char_pass>=5 and key_pass>=5 and frequent_pass>=5 and
        all(v['char_pass']>=2 and v['key_pass']>=2 for v in by_lang.values()))
    return {'size':SIZE,'n':len(rows),'char_pass':char_pass,'key_pass':key_pass,
            'frequent_pass':frequent_pass,'by_language':by_lang,'PASS':bool(ok)}


def main():
    import torch
    if torch.cuda.device_count()<4:
        raise RuntimeError(f'expected >=4 GPUs, found {torch.cuda.device_count()}')
    print('V10_FINAL_META='+json.dumps({'protocol':'VBM_V10_FINAL_ONESHOT_PROTOCOL.md','size':SIZE,
        'gpus':torch.cuda.device_count(),'devices':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'bridge_block':G.BRIDGE_BLOCK,'nucleus_block':G.NUC_BLOCK,'sweeps':G.SWEEPS,'chains':G.CHAINS},sort_keys=True),flush=True)
    rows=[];t0=time.time()
    for lang in ['DE','IT']:
        A=B['assets'](lang)
        for rep in range(3):
            all_lines,key=B['make_positive'](lang,rep,A)
            print('V10_FINAL_START='+json.dumps({'lang':lang,'rep':rep,'size':SIZE}),flush=True)
            row=one_o2(lang,rep,A,all_lines,key);rows.append(row)
            print('V10_FINAL_ROW='+json.dumps(row,sort_keys=True),flush=True)
    g=gate(rows)
    verdict='VBM_COMPACT_RECOVERY_GATE_PASSED_FULL_TAIL_REQUIRED' if g['PASS'] else 'VBM_GLOBAL_KEY_NOT_RECOVERABLE_EVEN_COMPACT'
    print('VBM_V10_FINAL_RESULT='+json.dumps({'verdict':verdict,'gate':g,'elapsed_s':time.time()-t0,
        'full_tail_opened':False,'voynich_opened':False},sort_keys=True),flush=True)

if __name__=='__main__':main()
