#!/usr/bin/env python3
import argparse,csv,glob,importlib.util,json,math,os
from collections import Counter,defaultdict
import numpy as np
SEED=20260813
POSITIONS='89abcdefghjklmnopqrs'
LIQ='()ABCDEFGHJKLMNOPQRS'
LIQMAP={ch:POSITIONS[i] for i,ch in enumerate(LIQ)}
NOTE_INDEX={ch:i for i,ch in enumerate(POSITIONS)}

def volpiano_notes(v):
    out=[]
    for ch in v:
        if ch in NOTE_INDEX:out.append(NOTE_INDEX[ch])
        elif ch in LIQMAP:out.append(NOTE_INDEX[LIQMAP[ch]])
    return out

def windows_from_notes(notes):
    iv=np.diff(np.asarray(notes,dtype=np.int16))
    if len(iv)<4:return []
    return [tuple(map(int,iv[i:i+4])) for i in range(len(iv)-3)]
def entropy_codes(codes):
    if len(codes)==0:return float('nan')
    c=np.bincount(codes);p=c[c>0]/len(codes);return float(-(p*np.log2(p)).sum())
def top1_concentration(codes):
    if len(codes)==0:return float('nan')
    c=np.bincount(codes);c=c[c>0];k=max(1,int(math.ceil(.01*len(c))));return float(np.sort(c)[-k:].sum()/len(codes))

def analyse_mode(recs,mode,rng,nperm=1000):
    allw=[w for r in recs for w in r];freq=Counter(allw);types={w:i for i,w in enumerate(freq)};start=np.asarray([types[w] for r in recs for w in r[:3]],dtype=np.int32);end=np.asarray([types[w] for r in recs for w in r[-3:]],dtype=np.int32);interior=np.asarray([types[w] for r in recs for w in r[5:-5]],dtype=np.int32);repeat_type=np.zeros(len(types),dtype=bool)
    for w,i in types.items():repeat_type[i]=freq[w]>=5
    def stat(c):return {'n':int(len(c)),'entropy':entropy_codes(c),'repeat_fraction':float(repeat_type[c].mean()) if len(c) else float('nan'),'top1_concentration':top1_concentration(c)}
    actual={'START':stat(start),'END':stat(end),'INTERIOR':stat(interior)};null={}
    for name,c in [('START',start),('END',end)]:
        k=len(c);ents=np.empty(nperm);reps=np.empty(nperm);tops=np.empty(nperm)
        for p in range(nperm):
            s=rng.choice(interior,size=k,replace=False);ents[p]=entropy_codes(s);reps[p]=repeat_type[s].mean();tops[p]=top1_concentration(s)
        null[name]={'entropy_mean':float(ents.mean()),'entropy_sd':float(ents.std(ddof=1)),'entropy_z':float((actual[name]['entropy']-ents.mean())/ents.std(ddof=1)),'repeat_mean':float(reps.mean()),'repeat_sd':float(reps.std(ddof=1)),'repeat_z':float((actual[name]['repeat_fraction']-reps.mean())/reps.std(ddof=1)),'top1_mean':float(tops.mean()),'top1_sd':float(tops.std(ddof=1)),'top1_z':float((actual[name]['top1_concentration']-tops.mean())/tops.std(ddof=1)),'p_entropy_lower':float((1+np.sum(ents<=actual[name]['entropy']))/(nperm+1)),'p_repeat_higher':float((1+np.sum(reps>=actual[name]['repeat_fraction']))/(nperm+1)),'direction_entropy_lower':bool(actual[name]['entropy']<ents.mean()),'direction_repeat_higher':bool(actual[name]['repeat_fraction']>reps.mean())}
    return {'mode':mode,'chants':len(recs),'unique_windows':len(types),'actual':actual,'null':null}
def find_chant_csv(root):
    c=glob.glob(os.path.join(root,'**','chant.csv'),recursive=True)
    if not c:c=glob.glob(os.path.join(root,'**','chants.csv'),recursive=True)
    if len(c)!=1:raise RuntimeError(f'Expected one chant CSV, found {c}')
    return c[0]
def load_pinned_parser(parser_dir):
    py=os.path.join(parser_dir,'parser_volpiano.py'); grammar=os.path.join(parser_dir,'cantus_volpiano.peg')
    if not os.path.isfile(py) or not os.path.isfile(grammar):raise RuntimeError(f'Pinned parser files missing in {parser_dir}')
    spec=importlib.util.spec_from_file_location('pinned_chant21_parser_volpiano',py);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.ParserCantusVolpiano(grammarPath=grammar,strict=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--parser-dir',required=True);ap.add_argument('--out',default='RESULTS_cantus_boundary_formulaicity_v0_1.json');args=ap.parse_args()
    try:parser=load_pinned_parser(args.parser_dir)
    except Exception as ex:raise RuntimeError(f'Pinned chant21 parser unavailable: {ex}')
    p=find_chant_csv(args.root);by=defaultdict(list);rows=vol=eligible=0;parsefail=0
    with open(p,encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            rows+=1;v=(r.get('volpiano') or '').strip();m=(r.get('mode') or '').strip()
            if not v:continue
            vol+=1
            if m not in set('12345678'):continue
            try:
                pv=parser.preprocess(v,strict=False);parser.parse(pv,strict=False);notes=volpiano_notes(pv)
            except Exception:parsefail+=1;continue
            if len(notes)<20:continue
            w=windows_from_notes(notes)
            if len(w)<10:continue
            by[m].append(w);eligible+=1
    rng=np.random.default_rng(SEED);results={}
    for m in '12345678':
        if len(by[m])>=100:results[m]=analyse_mode(by[m],m,rng,1000);print('MODE',m,results[m],flush=True)
    em=list(results);start_entropy=sum(results[m]['null']['START']['direction_entropy_lower'] for m in em);end_entropy=sum(results[m]['null']['END']['direction_entropy_lower'] for m in em);start_rep=sum(results[m]['null']['START']['direction_repeat_higher'] for m in em);end_rep=sum(results[m]['null']['END']['direction_repeat_higher'] for m in em);threshold=math.ceil(.75*len(em)) if len(em)<8 else 6;H4='SUPPORT' if len(em)>0 and start_entropy>=threshold and end_entropy>=threshold and start_rep>=threshold and end_rep>=threshold else 'UNSUPPORTED';out={'seed':SEED,'chant_csv':p,'rows':rows,'rows_with_volpiano':vol,'eligible_canonical_mode_chants':eligible,'parse_failures':parsefail,'eligible_modes':em,'threshold':threshold,'direction_counts':{'start_entropy_lower':start_entropy,'end_entropy_lower':end_entropy,'start_repeat_higher':start_rep,'end_repeat_higher':end_rep},'H4':H4,'modes':results,'representation':{'pitch_positions':POSITIONS,'liquescent_symbols':LIQ,'interval_window':4,'start_windows':3,'end_windows':3,'boundary_buffer_windows':2},'parser_files':{'parser_volpiano.py':'pinned chant21 commit ad52f6084efce4a440d083b588d7b51ff6973730','cantus_volpiano.peg':'same pinned commit'}};json.dump(out,open(args.out,'w'),indent=2);print('DECISION',H4,out['direction_counts'],'eligible',eligible,'WROTE',args.out,flush=True)
if __name__=='__main__':main()
