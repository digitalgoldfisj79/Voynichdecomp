#!/usr/bin/env python3
import urllib.request,json,multiprocessing as mp
BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/72e4b7a197c3283c7f6067fcb89d1a1ff3bd6c81/experiments/bnf_m19_why_german_v1_1/run_atomic_eva.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8');src=src.rsplit("if __name__=='__main__':main()",1)[0]
g={'__name__':'atomic_lib'};exec(compile(src,'run_atomic_eva.py','exec'),g)
# Arabic-only prospective span amendment from 1a90fb6.
_old=g['choose_span']
def choose_span_amended(pool,n,tag):
    if isinstance(tag,tuple) and len(tag)>=2 and tag[0]=='qual' and tag[1]=='arabic':
        target=set(range(g['b']['NV']))
        for k in range(1000):
            sp=_old(pool,n,('qual','arabic','full-repertoire',k));poss=set()
            for c in sp:
                if c!=' ' and c in g['b']['A2I']:
                    poss.update(g['b']['V2I'][int(v)] for v in g['b']['LETTER_VALS'][g['b']['A2I'][c]])
            if poss==target:
                print('ARABIC_SPAN_REPERTOIRE_ATTEMPT',k,flush=True);return sp
        raise RuntimeError('no full-repertoire Arabic span found')
    return _old(pool,n,tag)
g['choose_span']=choose_span_amended

# Globals populated before fork.
LMS=POOLS=COMPS=None

def qual_one(lang):
    b=g['b']; span=g['choose_span'](POOLS[lang],b['TRAIN']+b['HOLD'],('qual',lang));cw,true,attempt=g['gen_control'](span,lang);tr,ho=g['split_words'](cw,b['TRAIN']);symbols=list(range(g['NS']));Str=g['stats'](tr,symbols);Sho=g['stats'](ho,symbols);rows=[];fits={}
    for cand in b['LANGS']:
        sc,m=g['optimize'](Str,COMPS[cand],('qual',lang,cand,1));fw,n,cov=g['forward'](ho,m,symbols,LMS[cand]);rows.append((cand,fw));fits[cand]=m
    rows.sort(key=lambda x:x[1],reverse=True);sc2,m2=g['optimize'](Str,COMPS[lang],('qual',lang,lang,2));wa=g['acc'](Sho['freq'],fits[lang],true);agr=g['agreement'](Str['freq'],fits[lang],m2)
    return {'lang':lang,'top':rows[0][0],'margin':rows[0][1]-rows[1][1],'rank':1+next(i for i,x in enumerate(rows) if x[0]==lang),'map_acc':wa,'agreement':agr,'attempt':attempt}

V_STR=V_HO=V_SYMBOLS=None

def fit_vms(la):
    s1,m1=g['optimize'](V_STR,COMPS[la],('vms',la,1));s2,m2=g['optimize'](V_STR,COMPS[la],('vms',la,2));m=m1 if s1>=s2 else m2;agr=g['agreement'](V_STR['freq'],m1,m2);fw,n,cov=g['forward'](V_HO,m,V_SYMBOLS,LMS[la])
    return {'lang':la,'Hscore':fw,'agreement':agr,'train_score':max(s1,s2),'map':{V_SYMBOLS[i]:g['b']['VALUES'][int(m[i])] for i in range(g['NS'])},'_m':m.tolist()}

def main():
    global LMS,POOLS,COMPS,V_STR,V_HO,V_SYMBOLS
    LMS,POOLS,meta=g['load_fresh']();COMPS={la:g['b']['induced'](LMS[la]) for la in g['b']['LANGS']}
    ctx=mp.get_context('fork')
    with ctx.Pool(processes=6) as pool: controls=pool.map(qual_one,g['QUAL'])
    for r in controls: print('QUAL',json.dumps(r,separators=(',',':')),flush=True)
    gate={'correct':sum(r['top']==r['lang'] for r in controls),'min_margin':min(r['margin'] for r in controls),'median_acc':float(__import__('numpy').median([r['map_acc'] for r in controls])),'min_acc':min(r['map_acc'] for r in controls),'min_agreement':min(r['agreement'] for r in controls)}
    gate['pass']=gate['correct']==6 and gate['min_margin']>=.05 and gate['median_acc']>=.95 and gate['min_acc']>=.85 and gate['min_agreement']>=.90;print('GATE',json.dumps(gate,separators=(',',':')),flush=True);out={'controls':controls,'gate':gate}
    if not gate['pass']:
        out['verdict']='INSTRUMENT NOT QUALIFIED';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    b=g['b'];data=json.loads(b['fetch'](b['SLIM']));sample,hold,pages=g['v09_panels'](data);Tf=[f for f,_,_ in sample];Hf=[f for f,_,_ in hold];Af={f for f,_,_ in pages};Cf=sorted(Af-set(Tf)-set(Hf));tr=g['atoms_for'](data,Tf);ho=g['atoms_for'](data,Hf);symbols=sorted(set(x for w in tr for x in w));V_STR=g['stats'](tr,symbols);V_HO=g['stats'](ho,symbols);V_SYMBOLS=symbols
    census={'symbols':symbols,'ns':len(symbols),'Tfolios':len(Tf),'Hfolios':len(Hf),'Cfolios':len(Cf),'Tunits':sum(map(len,tr)),'Hunits':sum(map(len,ho)),'Cunits':sum(map(len,g['atoms_for'](data,Cf))),'Hcoverage':V_HO['coverage']};print('CENSUS',json.dumps(census,separators=(',',':')),flush=True)
    if len(symbols)!=g['NS'] or V_HO['coverage']<.99:
        out['verdict']='ATOMIC ALPHABET/COVERAGE MISMATCH';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    with ctx.Pool(processes=8) as pool: vres=pool.map(fit_vms,b['LANGS'])
    fits={r['lang']:__import__('numpy').array(r.pop('_m'),dtype='int16') for r in vres}
    for r in vres: print('VMS',json.dumps(r,separators=(',',':')),flush=True)
    rank=sorted(vres,key=lambda x:x['Hscore'],reverse=True);margin=rank[0]['Hscore']-rank[1]['Hscore'];primary=margin>=.05 and rank[0]['agreement']>=.90;cwords=g['atoms_for'](data,Cf);cr=g['rank_fixed'](cwords,fits,symbols,LMS)
    out.update({'vms':vres,'census':census,'Hranking':[(x['lang'],x['Hscore'],x['agreement']) for x in rank],'Hmargin':margin,'Hprimary':primary,'Cranking':cr,'verdict':'ATOMIC SIGNAL '+rank[0]['lang'] if primary else 'EVA REPRESENTATION SENSITIVE / NO STABLE ATOMIC SIGNAL'});print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()
