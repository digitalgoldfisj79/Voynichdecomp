#!/usr/bin/env python3
import sys, json, csv, io, math, hashlib, urllib.request
from collections import defaultdict, Counter
import numpy as np

BASE='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/ac8fa90f3959fb5346d9837159d194ddb4ab9bd7/experiments/bnf_onomancy_global_key_v0_3/run_global_key.py'
src=urllib.request.urlopen(BASE,timeout=90).read().decode('utf-8')
ns={'__name__':'bnf_v03'}
exec(compile(src,'run_global_key.py','exec'),ns)
ns['SEED0']=20260809
ns['STEPS']=4200; ns['RESTARTS']=3; ns['POLISH']=700; ns['PERM_NULLS']=64

ALPH=ns['ALPH']; A2I=ns['A2I']; N=ns['N']; SPACE=ns['SPACE']; PK=ns['PK']
LANGS=ns['LANGS']; TARGETS=ns['TARGETS']; CAP2=ns['CAP2']
stable_seed=ns['stable_seed']; fetch=ns['fetch']; conllu_sents=ns['conllu_sents']; split_train_hold=ns['split_train_hold']; concat_norm=ns['concat_norm']; build_lm=ns['build_lm']; norm=ns['norm']
optimize=ns['optimize']; QuadAgg=ns['QuadAgg']; score_seq=ns['score_seq']; char_accuracy=ns['char_accuracy']; extract_page=ns['extract_page']; representative_pages=ns['representative_pages']; make_control_cipher=ns['make_control_cipher']; choose_letter_span=ns['choose_letter_span']
LM_URLS=ns['LM_URLS']

MODE=sys.argv[1] if len(sys.argv)>1 else 'currier'
if MODE not in {'currier','section','section_currier','quire'}: raise SystemExit('bad mode')
SLIM='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'
SECMAP='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_section_map.json'
MANIFEST='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/daiin_manifest.csv'
TRAIN_CAP=35000
NULLS=64


def letters_in_text(t): return sum(not c.isspace() for c in t)

def group_name(f,meta,sec):
    if MODE=='currier': return meta[f]['lang']
    if MODE=='section': return sec.get(f,'UNMAPPED')
    if MODE=='section_currier': return sec.get(f,'UNMAPPED')+'|'+meta[f]['lang']
    return 'Q'+meta[f]['quire']

def page_split(group,pages):
    pp=sorted(pages,key=lambda p:stable_seed('split',MODE,group,p[0]))
    n=len(pp); nh=max(1,int(round(.2*n))) if n>=2 else 1
    hold=pp[:nh]; train=pp[nh:]
    return train,hold

def select_train(groups_eval):
    # 2k-character floor per group, then proportional fill to a 35k whole-rung cap.
    totals={g:sum(letters_in_text(t) for _,t in d['train']) for g,d in groups_eval.items()}
    G=len(totals); floor=min(2000, max(800, TRAIN_CAP//max(1,2*G)))
    base={g:min(floor,totals[g]) for g in totals}
    rem=max(0,TRAIN_CAP-sum(base.values())); denom=sum(max(0,totals[g]-base[g]) for g in totals)
    targets={}
    for g in totals:
        add=(rem*max(0,totals[g]-base[g])/denom) if denom else 0
        targets[g]=min(totals[g],int(round(base[g]+add)))
    out={}
    for g,d in groups_eval.items():
        pp=sorted(d['train'],key=lambda p:stable_seed('trsample',MODE,g,p[0]))
        take=[];n=0
        for p in pp:
            take.append(p); n+=letters_in_text(p[1])
            if n>=targets[g]:break
        out[g]=take
    return out

def pages_to_seq(pages, symbols):
    s2i={s:i for i,s in enumerate(symbols)}; arr=[]
    for _,t in pages:
        for c in t:
            if c.isspace(): arr.append(-1)
            elif c in s2i: arr.append(s2i[c])
        arr.append(-1)
    return np.asarray(arr,dtype=np.int16)

def union_symbols(train,hold):
    return sorted(set(c for _,t in train+hold for c in t if not c.isspace()))

def split_control_exact(seq,pa,ntrain_letters):
    pos=np.flatnonzero(seq>=0)
    if ntrain_letters<=0 or ntrain_letters>=len(pos): raise RuntimeError(('bad control split',ntrain_letters,len(pos)))
    cut=int(pos[ntrain_letters])
    return seq[:cut],pa[:cut],seq[cut:],pa[cut:]

def weighted_piece_score(items, maps, logp):
    num=0.0;den=0.0
    for g,d in items.items():
        agg=QuadAgg(d['hold_seq'],len(d['symbols'])); s=agg.score(maps[g],logp); num+=s*agg.total;den+=agg.total
    return num/max(1.0,den)

def piece_perm_stats(items,maps,logp,tag):
    aggs={g:QuadAgg(d['hold_seq'],len(d['symbols'])) for g,d in items.items()}
    obs_num=sum(aggs[g].score(maps[g],logp)*aggs[g].total for g in items); den=sum(aggs[g].total for g in items); obs=obs_num/den
    whole=[]; group_null={g:[] for g in items}
    for j in range(NULLS):
        num=0.0
        for g in items:
            rr=np.random.default_rng(stable_seed('pieceperm',MODE,tag,g,j)); mm=maps[g].copy();rr.shuffle(mm); s=aggs[g].score(mm,logp);group_null[g].append(s);num+=s*aggs[g].total
        whole.append(num/den)
    mu=float(np.mean(whole));sd=float(np.std(whole,ddof=1));z=(obs-mu)/sd if sd>1e-12 else 0.0
    return obs,mu,sd,z,group_null

def mapping_capacity_ok(mp):
    return int(np.bincount(mp,minlength=N).max())<=4

def fit_piece(items,lang,lms,unigs,tag):
    maps={};train_scores={}
    for g,d in items.items():
        sc,mp=optimize(d['train_seq'],CAP2,unigs[lang],lms[lang],(tag,MODE,g,lang)); maps[g]=mp;train_scores[g]=sc
    return maps,train_scores

def vocab_from_sents(sents):
    v=set()
    for s in sents:
        z=norm(s)
        if z:v.update(z.split())
    return v

def lexical_fraction(groups_eval,maps,vocab,which='hold'):
    hit=tot=0
    for g,d in groups_eval.items():
        syms=d['symbols'];s2i={s:i for i,s in enumerate(syms)};mp=maps[g]
        pages=d[which+'_pages']
        for _,t in pages:
            for tok in t.split():
                out=[];ok=True
                for c in tok:
                    i=s2i.get(c)
                    if i is None:ok=False;break
                    out.append(ALPH[int(mp[i])])
                tot+=1
                if ok and ''.join(out) in vocab: hit+=1
    return hit/max(1,tot),hit,tot

def lexical_z(groups_eval,maps,vocab,tag):
    obs,_,_=lexical_fraction(groups_eval,maps,vocab,'hold'); vals=[]
    for j in range(NULLS):
        mm={}
        for g,mp in maps.items():
            rr=np.random.default_rng(stable_seed('lexperm',MODE,tag,g,j));z=mp.copy();rr.shuffle(z);mm[g]=z
        vals.append(lexical_fraction(groups_eval,mm,vocab,'hold')[0])
    mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));z=(obs-mu)/sd if sd>1e-12 else 0.0
    return {'fraction':obs,'null_mean':mu,'null_sd':sd,'z':z}

def group_positive_coverage(items,maps,logp,group_null):
    good=tot=0
    for g,d in items.items():
        agg=QuadAgg(d['hold_seq'],len(d['symbols']));obs=agg.score(maps[g],logp); n=int(np.sum(d['hold_seq']>=0));tot+=n
        if obs>float(np.median(group_null[g])):good+=n
    return good/max(1,tot)

def acc_summary(items,maps):
    correct=tot=0;groupacc={};goodchars=0
    for g,d in items.items():
        mask=d['hold_seq']>=0;n=int(mask.sum());a=char_accuracy(d['hold_seq'],d['hold_pa'],maps[g]);groupacc[g]=a;correct+=a*n;tot+=n
        if a>=.70:goodchars+=n
    return correct/max(1,tot),goodchars/max(1,tot),groupacc

def make_piece_control(groups_eval,train_sample,plain_text,lang):
    # One independent T2 key per group; exact train/hold letter counts match the VMS rung.
    needs={g:(sum(letters_in_text(t) for _,t in train_sample[g]),sum(letters_in_text(t) for _,t in d['hold'])) for g,d in groups_eval.items()}
    need=sum(a+b for a,b in needs.values())
    span,_=choose_letter_span(plain_text,need,('v04',MODE,lang,need))
    # partition span by letter offsets while retaining spaces
    apos=[i for i,c in enumerate(span) if c!=' ']; cursor=0; out={}
    for g in sorted(groups_eval):
        nt,nh=needs[g];nn=nt+nh
        i0=apos[cursor];i1=apos[cursor+nn-1]+1;seg=span[i0:i1].strip();cursor+=nn
        seq,true,pa=make_control_cipher(seg,CAP2,('v04',MODE,lang,g));trseq,trpa,hseq,hpa=split_control_exact(seq,pa,nt)
        syms=list(range(25))
        out[g]={'train_seq':trseq,'hold_seq':hseq,'train_pa':trpa,'hold_pa':hpa,'symbols':syms,'true':true}
    return out

def build_actual_groups(data,meta,sec):
    raw=defaultdict(list)
    for f in sorted(data['pages']):
        if f not in meta:continue
        t=extract_page(data,f,'ZLZI')
        if not t:continue
        raw[group_name(f,meta,sec)].append((f,t))
    info={};total=0;evalletters=0
    for g,pages in raw.items():
        tr,ho=page_split(g,pages);tot=sum(letters_in_text(t) for _,t in pages);hh=sum(letters_in_text(t) for _,t in ho);total+=tot
        ev=len(pages)>=3 and tot>=1500 and hh>=500
        if ev:evalletters+=tot
        info[g]={'pages':pages,'train':tr,'hold':ho,'n_pages':len(pages),'letters':tot,'hold_letters':hh,'evaluable':ev}
    cov=evalletters/max(1,total)
    return info,cov

def build_actual_items(info,train_sample):
    out={}
    for g,d in info.items():
        if not d['evaluable']:continue
        syms=union_symbols(train_sample[g],d['hold']);trseq=pages_to_seq(train_sample[g],syms);hseq=pages_to_seq(d['hold'],syms)
        out[g]={'symbols':syms,'train_seq':trseq,'hold_seq':hseq,'train_pages':train_sample[g],'hold_pages':d['hold']}
    return out

def global_baseline(actual,lang,lms,unigs):
    # exact same rung subset, but one key across all groups
    alltr=[];allho=[]
    for g,d in actual.items():alltr+=d['train_pages'];allho+=d['hold_pages']
    syms=sorted(set(c for _,t in alltr+allho for c in t if not c.isspace()));tr=pages_to_seq(alltr,syms);ho=pages_to_seq(allho,syms)
    sc,mp=optimize(tr,CAP2,unigs[lang],lms[lang],('v04globalbaseline',MODE,lang));agg=QuadAgg(ho,len(syms));return agg.score(mp,lms[lang]),agg.total

def transfer_candidate(data,meta,sec,info,actual,maps,cand,lms,vocabs,tid):
    # Same folio split and same literal label mappings; no refit.
    gpages={g:[] for g in actual}
    total=covered=0
    for g,d in info.items():
        if g not in actual:continue
        for f,_ in d['hold']:
            if f not in data['pages']:continue
            t=extract_page(data,f,tid)
            if t:gpages[g].append((f,t))
    # materialize decoded arrays with unknown labels as breaks; score z under each LM by groupwise mapping permutations.
    rankings=[]
    for la in LANGS:
        obsnum=den=0.0; null=[0.0]*NULLS; nullden=[0.0]*NULLS
        for g,pages in gpages.items():
            if not pages:continue
            syms=actual[g]['symbols'];s2i={s:i for i,s in enumerate(syms)};mp=maps[g]
            arr=[]
            for _,t in pages:
                for c in t:
                    if c.isspace():arr.append(SPACE)
                    else:
                        total+=1 if la==LANGS[0] else 0
                        if c in s2i:
                            if la==LANGS[0]:covered+=1
                            arr.append(int(mp[s2i[c]]))
                        else:arr.append(-2)
                arr.append(SPACE)
            a=np.asarray(arr,dtype=np.int16)
            if len(a)<4:continue
            ok=(a[:-3]>=0)&(a[1:-2]>=0)&(a[2:-1]>=0)&(a[3:]>=0)
            if not np.any(ok):continue
            q=np.stack([a[:-3][ok],a[1:-2][ok],a[2:-1][ok],a[3:][ok]],axis=1);cnt=np.ones(len(q));idx=((q[:,0].astype(np.int64)*PK+q[:,1])*PK+q[:,2])*PK+q[:,3];obsnum+=float(lms[la][idx].sum());den+=len(idx)
            # null by literal mapping permutation then redecode page text
            for j in range(NULLS):
                rr=np.random.default_rng(stable_seed('xferperm',MODE,tid,cand,la,g,j));mm=mp.copy();rr.shuffle(mm);aa=[]
                for _,t in pages:
                    for c in t:
                        if c.isspace():aa.append(SPACE)
                        elif c in s2i:aa.append(int(mm[s2i[c]]))
                        else:aa.append(-2)
                    aa.append(SPACE)
                aa=np.asarray(aa,dtype=np.int16);okk=(aa[:-3]>=0)&(aa[1:-2]>=0)&(aa[2:-1]>=0)&(aa[3:]>=0)
                if np.any(okk):
                    ix=((aa[:-3][okk].astype(np.int64)*PK+aa[1:-2][okk])*PK+aa[2:-1][okk])*PK+aa[3:][okk];null[j]+=float(lms[la][ix].sum());nullden[j]+=len(ix)
        obs=obsnum/max(1,den);vals=[null[j]/max(1,nullden[j]) for j in range(NULLS)];mu=float(np.mean(vals));sd=float(np.std(vals,ddof=1));z=(obs-mu)/sd if sd>1e-12 else 0.0;rankings.append((la,z,obs))
    rankings.sort(key=lambda x:x[1],reverse=True)
    # lexical candidate only, on known shared glyphs; same mapping-permutation null.
    # Reuse a lightweight adapted group object for lexical scorer.
    tmp={}
    for g,pages in gpages.items():
        if pages:tmp[g]={'symbols':actual[g]['symbols'],'hold_pages':pages}
    # custom lexical because lexical_fraction expects train pages too only by key name, which is fine.
    def lex(mm):
        hit=tot=0;v=vocabs[cand]
        for g,d in tmp.items():
            s2i={s:i for i,s in enumerate(d['symbols'])};mpx=mm[g]
            for _,t in d['hold_pages']:
                for tok in t.split():
                    out=[];ok=True
                    for c in tok:
                        if c not in s2i:ok=False;break
                        out.append(ALPH[int(mpx[s2i[c]])])
                    tot+=1
                    if ok and ''.join(out) in v:hit+=1
        return hit/max(1,tot)
    obslex=lex(maps);lv=[]
    for j in range(NULLS):
        mm={}
        for g,mp in maps.items():rr=np.random.default_rng(stable_seed('xferlex',MODE,tid,cand,g,j));x=mp.copy();rr.shuffle(x);mm[g]=x
        lv.append(lex(mm))
    lmu=float(np.mean(lv));lsd=float(np.std(lv,ddof=1));lz=(obslex-lmu)/lsd if lsd>1e-12 else 0.0
    return {'ranking':rankings,'candidate_rank':1+next(i for i,x in enumerate(rankings) if x[0]==cand),'candidate_z':next(x[1] for x in rankings if x[0]==cand),'lexical_fraction':obslex,'lexical_z':lz,'coverage':covered/max(1,total)}

def main():
    # Load language models and heldout positive-control material.
    lms={};unigs={};holds={};vocabs={};lmmeta={};all_sents={}
    for lang in LANGS:
        ss=conllu_sents(fetch(LM_URLS[lang]));tr,ho=split_train_hold(ss) if lang in TARGETS else (ss,[]);all_sents[lang]=tr
        lm,ug,nlet=build_lm(tr);lms[lang]=lm;unigs[lang]=ug;vocabs[lang]=vocab_from_sents(tr)
        if lang in TARGETS:holds[lang]=concat_norm(ho)
        lmmeta[lang]={'lm_letters':nlet,'hold_letters':sum(c!=' ' for c in holds.get(lang,'')),'vocab':len(vocabs[lang])}
        print('LM',lang,lmmeta[lang],flush=True)

    data=json.loads(fetch(SLIM));sec=json.loads(fetch(SECMAP))['mapping'];rows=list(csv.DictReader(io.StringIO(fetch(MANIFEST))));meta={r['folio']:r for r in rows}
    info,coverage=build_actual_groups(data,meta,sec)
    census={g:{k:d[k] for k in ['n_pages','letters','hold_letters','evaluable']} for g,d in sorted(info.items())}
    print('RUNG_CENSUS',MODE,'coverage',coverage,json.dumps(census,separators=(',',':')),flush=True)
    out={'protocol':'v0.4','mode':MODE,'census':census,'evaluable_coverage':coverage,'lm_meta':lmmeta}
    if coverage<.80:
        out['verdict']='UNDERPOWERED: EVALUABLE COVERAGE';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return
    evalinfo={g:d for g,d in info.items() if d['evaluable']};sample=select_train(evalinfo);actual=build_actual_items(evalinfo,sample)
    trletters=sum(int(np.sum(d['train_seq']>=0)) for d in actual.values());hletters=sum(int(np.sum(d['hold_seq']>=0)) for d in actual.values())
    print('SAMPLE',json.dumps({'groups':len(actual),'train_letters':trletters,'hold_letters':hletters,'train_by_group':{g:int(np.sum(d['train_seq']>=0)) for g,d in actual.items()}},separators=(',',':')),flush=True)
    # Ensure all target heldout corpora can supply the synthetic rung.
    if any(sum(c!=' ' for c in holds[l]) < trletters+hletters for l in TARGETS):
        out['verdict']='UNDERPOWERED: CONTROL SOURCE LENGTH';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return

    controls=[]
    for lang in TARGETS:
        ci=make_piece_control(evalinfo,sample,holds[lang],lang)
        ranks=[]; target_maps=None;target_null=None
        for cand in LANGS:
            maps,_=fit_piece(ci,cand,lms,unigs,('control',lang,cand));obs,mu,sd,z,gn=piece_perm_stats(ci,maps,lms[cand],('control',lang,cand));acc,gcov,gacc=acc_summary(ci,maps)
            ranks.append({'cand':cand,'z':z,'acc':acc,'group_acc_coverage':gcov})
            if cand==lang:target_maps,target_null=maps,gn
        ranks.sort(key=lambda x:x['z'],reverse=True);tar=next(x for x in ranks if x['cand']==lang)
        row={'lang':lang,'top':ranks[0]['cand'],'target_rank':1+next(i for i,x in enumerate(ranks) if x['cand']==lang),'target_z':tar['z'],'target_acc':tar['acc'],'group_acc_coverage':tar['group_acc_coverage'],'ranking':[(x['cand'],x['z']) for x in ranks]}
        controls.append(row);print('CONTROL',json.dumps(row,separators=(',',':')),flush=True)
    gate={'correct':sum(r['top']==r['lang'] for r in controls),'median_acc':float(np.median([r['target_acc'] for r in controls])),'min_acc':float(min(r['target_acc'] for r in controls)),'min_group_acc_coverage':float(min(r['group_acc_coverage'] for r in controls)),'median_z':float(np.median([r['target_z'] for r in controls]))}
    gate.update({'P1':gate['correct']==4,'P2':gate['median_acc']>=.90,'P3':gate['min_acc']>=.75,'P4':gate['min_group_acc_coverage']>=.80,'P5':gate['median_z']>=10});gate['pass']=all(gate[k] for k in ['P1','P2','P3','P4','P5'])
    out['controls']=controls;out['control_gate']=gate;print('CONTROL_GATE',json.dumps(gate,separators=(',',':')),flush=True)
    if not gate['pass']:
        out['verdict']='UNDERPOWERED: POSITIVE CONTROL FAIL';print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True);return

    # Voynich primary scoring.
    vres=[];fitted={}
    for lang in LANGS:
        maps,_=fit_piece(actual,lang,lms,unigs,('VMS',MODE,lang));obs,mu,sd,z,gn=piece_perm_stats(actual,maps,lms[lang],('VMS',MODE,lang));lex=lexical_z(actual,maps,vocabs[lang],('VMS',MODE,lang));gcov=group_positive_coverage(actual,maps,lms[lang],gn);capok=all(mapping_capacity_ok(mp) for mp in maps.values());gb,gbn=global_baseline(actual,lang,lms,unigs);piece_n=sum(QuadAgg(d['hold_seq'],len(d['symbols'])).total for d in actual.values());delta=(obs-gb)*piece_n;pen=25*(len(actual)-1)*math.log(23);mdl=delta-pen
        row={'lang':lang,'z':z,'hold_score':obs,'lexical':lex,'group_positive_coverage':gcov,'capacity_ok':capok,'global_hold_score':gb,'mdl_gain_minus_penalty_nats':mdl,'keys':len(actual)};vres.append(row);fitted[lang]=maps;print('VMS',json.dumps(row,separators=(',',':')),flush=True)
    rank=sorted(vres,key=lambda r:r['z'],reverse=True);top,second=rank[0],rank[1];margin=top['z']-second['z']
    primary=bool(top['z']>=10 and margin>=5 and top['lexical']['z']>=5 and top['group_positive_coverage']>=.80 and top['capacity_ok'])
    signal={'top':top['lang'],'top_z':top['z'],'second':second['lang'],'second_z':second['z'],'margin':margin,'primary':primary}
    transfers={}
    if primary:
        for tid in ['TTLI','VDRB']:
            transfers[tid]=transfer_candidate(data,meta,sec,info,actual,fitted[top['lang']],top['lang'],lms,vocabs,tid);print('TRANSFER',tid,json.dumps(transfers[tid],separators=(',',':')),flush=True)
        confirmed=all(transfers[t]['candidate_rank']==1 and transfers[t]['candidate_z']>=7 and transfers[t]['lexical_z']>=3 and transfers[t]['coverage']>=.90 for t in ['TTLI','VDRB'])
        signal['confirmed']=confirmed
    else:signal['confirmed']=False
    if signal['confirmed']:verdict='CONFIRMED PIECEWISE-KEY SIGNAL'
    elif primary:verdict='ZLZI CANDIDATE / TRANSCRIPTION-DEPENDENT'
    else:verdict='NO SIGNAL'
    out.update({'vms':vres,'signal':signal,'transfers':transfers,'verdict':verdict})
    print('RESULT_JSON='+json.dumps(out,separators=(',',':')),flush=True)

if __name__=='__main__':main()
