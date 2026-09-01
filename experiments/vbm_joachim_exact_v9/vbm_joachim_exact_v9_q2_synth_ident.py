# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=1.26,<2.3", "wordfreq>=3.1,<4", "Unidecode>=1.3,<2"]
# ///
from __future__ import annotations
import argparse, collections, hashlib, json, math
import numpy as np
from unidecode import unidecode
from wordfreq import top_n_list, zipf_frequency

NS='VBMJOACHIMEXACTV9Q2'
VOW='aeiou'; NV=5; KB=30; KR=32; KN=96
FAMS=['DE_GLOBAL','IT_GLOBAL','DE_FRESHLINE','IT_FRESHLINE','MARKOV1','SHUFFLED_GLOBAL']
POS={'DE_GLOBAL','IT_GLOBAL'}
NEG=set(FAMS)-POS
ALPHA=.05
LM_CHARS=650_000
PT_CHARS=650_000


def seed(*xs):return int.from_bytes(hashlib.sha256('::'.join(map(str,xs)).encode()).digest()[:8],'big') & 0x7fffffff

def norm(s):return ''.join(c for c in unidecode(s).lower() if 'a'<=c<='z')

def bank(lang,tag,nchars):
    ws=[];wt=[]
    for w in top_n_list(lang,30000):
        q=norm(w)
        if not q or len(q)>24:continue
        z=zipf_frequency(w,lang)
        if not np.isfinite(z):continue
        ws.append(q);wt.append(10.0**(.45*z))
    p=np.asarray(wt,float);p/=p.sum();rng=np.random.default_rng(seed(NS,'BANK',lang,tag));out=[];n=0
    while n<nchars:
        ix=rng.choice(len(ws),4096,p=p);s=''.join(ws[int(i)] for i in ix);out.append(s);n+=len(s)
    return ''.join(out)[:nchars]

def decomp(s):
    runs=[];vs=[];cur=[]
    for ch in s:
        if ch in VOW:runs.append(''.join(cur));cur=[];vs.append(ch)
        else:cur.append(ch)
    runs.append(''.join(cur));return runs,vs

class LM:
    def __init__(self,s):
        self.ctx=collections.defaultdict(collections.Counter);self.tot=collections.Counter();self.uni=collections.Counter(s)
        for i in range(3,len(s)):
            c=s[i-3:i];x=s[i];self.ctx[c][x]+=1;self.tot[c]+=1
    def score(self,s):
        if len(s)<4:return (-20.0,1)
        ll=0.;n=0
        for i in range(3,len(s)):
            c=s[i-3:i];x=s[i];ll+=math.log((self.ctx[c][x]+ALPHA)/(self.tot[c]+26*ALPHA));n+=1
        return ll,n

def assets():
    out={}
    for key,la in [('DE','de'),('IT','it')]:
        lms=bank(la,'LM',LM_CHARS);pts=bank(la,'PT',PT_CHARS);rr,_=decomp(lms);cnt=collections.Counter(r for r in rr if r and len(r)<=5);runs=[r for r,_ in cnt.most_common(KR)]
        if len(runs)!=KR:raise RuntimeError(('run inventory',key,len(runs)))
        out[key]={'lm':LM(lms),'pt':pts,'runs':runs,'run_freq':np.asarray([cnt[r] for r in runs],float)}
        _,vv=decomp(lms);vc=collections.Counter(vv);out[key]['vowel_freq']=np.asarray([vc[v] for v in VOW],float)
        out[key]['run_freq']/=out[key]['run_freq'].sum();out[key]['vowel_freq']/=out[key]['vowel_freq'].sum()
        print('Q2ASSET',json.dumps({'lang':key,'runs':runs[:10],'lm_chars':len(lms),'pt_chars':len(pts)}),flush=True)
    return out

def plaintext_lines(asset,tag,nlines=120):
    runs,vs=decomp(asset['pt']);allow=set(asset['runs']);rng=np.random.default_rng(seed(NS,'PLAIN',tag));valid=[]
    # Randomly propose consonant-boundary starts; accept only representable windows.
    tries=0
    while len(valid)<nlines and tries<500000:
        tries+=1;B=int(rng.integers(8,15));st=int(rng.integers(0,len(vs)-B))
        rs=runs[st:st+B+1];vv=vs[st:st+B]
        if any((r and (len(r)>5 or r not in allow)) for r in rs):continue
        txt=''.join(rs[i]+(vv[i] if i<B else '') for i in range(B+1))
        valid.append({'runs':list(rs),'vowels':list(vv),'plain':txt})
    if len(valid)<nlines:raise RuntimeError(('plaintext shortage',tag,len(valid),tries))
    return valid

def codebook(asset,tag):
    rng=np.random.default_rng(seed(NS,'KEY',tag));bm=np.repeat(np.arange(NV,dtype=np.int16),KB//NV);rng.shuffle(bm)
    nm=np.repeat(np.arange(KR,dtype=np.int16),KN//KR);rng.shuffle(nm)
    bp={v:np.flatnonzero(bm==v) for v in range(NV)};npool={r:np.flatnonzero(nm==r) for r in range(KR)}
    bw={v:rng.dirichlet(np.full(len(bp[v]),.35)) for v in bp};nw={r:rng.dirichlet(np.full(len(npool[r]),.35)) for r in npool}
    return {'bmap':bm,'nmap':nm,'bp':bp,'np':npool,'bw':bw,'nw':nw}

def encode_line(pl,asset,key,tag):
    ridx={r:i for i,r in enumerate(asset['runs'])};rng=np.random.default_rng(seed(NS,'EMIT',tag));ns=[];bs=[]
    for i,r in enumerate(pl['runs']):
        if r=='':ns.append(-1)
        else:
            z=ridx[r];pool=key['np'][z];ns.append(int(rng.choice(pool,p=key['nw'][z])))
        if i<len(pl['vowels']):
            v=VOW.index(pl['vowels'][i]);pool=key['bp'][v];bs.append(int(rng.choice(pool,p=key['bw'][v])))
    return {'n':ns,'b':bs,'plain':pl['plain']}

def positive(lang,phase,rep,A):
    a=A[lang];pl=plaintext_lines(a,f'{phase}:{lang}:R{rep}');key=codebook(a,f'{phase}:{lang}:R{rep}:GLOBAL');lines=[encode_line(x,a,key,f'{phase}:{lang}:R{rep}:L{i}') for i,x in enumerate(pl)]
    return lines,key

def fresh(lang,phase,rep,A):
    a=A[lang];pl=plaintext_lines(a,f'{phase}:{lang}:FRESH:R{rep}');lines=[]
    for i,x in enumerate(pl):
        k=codebook(a,f'{phase}:{lang}:FRESH:R{rep}:K{i}');lines.append(encode_line(x,a,k,f'{phase}:{lang}:FRESH:R{rep}:L{i}'))
    return lines,None

def markov_from(src,tag):
    # Stable separate first-order processes for nucleus and bridge streams.
    SA=KN+1;empty=KN;cn=np.full(SA,.1);tn=np.full((SA,SA),.1);cb=np.full(KB,.1);tb=np.full((KB,KB),.1)
    for L in src:
        n=[empty if x<0 else x for x in L['n']];b=L['b']
        if n:cn[n[0]]+=1
        for x,y in zip(n,n[1:]):tn[x,y]+=1
        if b:cb[b[0]]+=1
        for x,y in zip(b,b[1:]):tb[x,y]+=1
    cn/=cn.sum();tn/=tn.sum(1,keepdims=True);cb/=cb.sum();tb/=tb.sum(1,keepdims=True);rng=np.random.default_rng(seed(NS,'MARKOV',tag));out=[]
    for li,L in enumerate(src):
        nn=len(L['n']);nb=len(L['b']);n=[int(rng.choice(SA,p=cn))]
        for _ in range(1,nn):n.append(int(rng.choice(SA,p=tn[n[-1]])))
        b=[int(rng.choice(KB,p=cb))] if nb else []
        for _ in range(1,nb):b.append(int(rng.choice(KB,p=tb[b[-1]])))
        out.append({'n':[(-1 if x==empty else x) for x in n],'b':b,'plain':None})
    return out

def shuffled(src,tag):
    out=[]
    for i,L in enumerate(src):
        rng=np.random.default_rng(seed(NS,'SHUF',tag,i));n=list(L['n']);b=list(L['b']);rng.shuffle(n);rng.shuffle(b);out.append({'n':n,'b':b,'plain':None})
    return out

def family(fam,phase,rep,A):
    if fam=='DE_GLOBAL':
        z,k=positive('DE',phase,rep,A);return z,k,'DE'
    if fam=='IT_GLOBAL':
        z,k=positive('IT',phase,rep,A);return z,k,'IT'
    if fam=='DE_FRESHLINE':
        z,k=fresh('DE',phase,rep,A);return z,k,'DE'
    if fam=='IT_FRESHLINE':
        z,k=fresh('IT',phase,rep,A);return z,k,'IT'
    src,k=positive('DE',phase,rep,A)
    if fam=='MARKOV1':return markov_from(src,f'{phase}:R{rep}'),None,None
    if fam=='SHUFFLED_GLOBAL':return shuffled(src,f'{phase}:R{rep}'),None,None
    raise ValueError(fam)

def decode_line(L,bmap,nmap,runs):
    s=[]
    for i,n in enumerate(L['n']):
        if n>=0:s.append(runs[int(nmap[n])])
        if i<len(L['b']):s.append(VOW[int(bmap[L['b'][i]])])
    return ''.join(s)

def build_index(lines):
    bi=[set() for _ in range(KB)];ni=[set() for _ in range(KN)]
    for j,L in enumerate(lines):
        for x in set(L['b']):bi[x].add(j)
        for x in set(n for n in L['n'] if n>=0):ni[x].add(j)
    return [sorted(x) for x in bi],[sorted(x) for x in ni]

def init_map(asset,tag):
    rng=np.random.default_rng(seed(NS,'INIT',tag));b=rng.choice(NV,KB,p=asset['vowel_freq']).astype(np.int16);n=rng.choice(KR,KN,p=asset['run_freq']).astype(np.int16);return b,n

def fit_map(lines,asset,tag,restarts=3,passes=5):
    lm=asset['lm'];runs=asset['runs'];bi,ni=build_index(lines);best=None
    for rr in range(restarts):
        bm,nm=init_map(asset,f'{tag}:R{rr}');cache=[];totll=0.;totn=0
        for L in lines:
            ll,nn=lm.score(decode_line(L,bm,nm,runs));cache.append((ll,nn));totll+=ll;totn+=nn
        for pp in range(passes):
            changed=0;rng=np.random.default_rng(seed(NS,'ORDER',tag,rr,pp));items=[('b',i) for i,x in enumerate(bi) if x]+[('n',i) for i,x in enumerate(ni) if x];rng.shuffle(items)
            for typ,t in items:
                affected=bi[t] if typ=='b' else ni[t];old=int(bm[t] if typ=='b' else nm[t]);cand=range(NV) if typ=='b' else range(KR);bestv=old;bestr=totll/max(1,totn);bestpack=None
                for v in cand:
                    if v==old:continue
                    if typ=='b':bm[t]=v
                    else:nm[t]=v
                    dll=0.;dn=0;pack=[]
                    for j in affected:
                        ll0,n0=cache[j];ll1,n1=lm.score(decode_line(lines[j],bm,nm,runs));dll+=ll1-ll0;dn+=n1-n0;pack.append((j,ll1,n1))
                    rat=(totll+dll)/max(1,totn+dn)
                    if rat>bestr+1e-12:bestr=rat;bestv=int(v);bestpack=(dll,dn,pack)
                if typ=='b':bm[t]=old
                else:nm[t]=old
                if bestv!=old and bestpack is not None:
                    if typ=='b':bm[t]=bestv
                    else:nm[t]=bestv
                    dll,dn,pack=bestpack;totll+=dll;totn+=dn
                    for j,ll1,n1 in pack:cache[j]=(ll1,n1)
                    changed+=1
            if changed==0:break
        z={'bmap':bm.copy(),'nmap':nm.copy(),'fit_score':totll/max(1,totn)}
        if best is None or z['fit_score']>best['fit_score']:best=z
    return best

def score_lines(lines,asset,m):
    ll=0.;n=0
    for L in lines:
        a,b=asset['lm'].score(decode_line(L,m['bmap'],m['nmap'],asset['runs']));ll+=a;n+=b
    return ll/max(1,n)

def seen_cov(train,hold):
    sb=set(x for L in train for x in L['b']);sn=set(x for L in train for x in L['n'] if x>=0);hb=[x for L in hold for x in L['b']];hn=[x for L in hold for x in L['n'] if x>=0]
    return sum(x in sb for x in hb)/max(1,len(hb)),sum(x in sn for x in hn)/max(1,len(hn))

def apply_defaults(m,train,asset):
    sb=set(x for L in train for x in L['b']);sn=set(x for L in train for x in L['n'] if x>=0);b=m['bmap'].copy();n=m['nmap'].copy();dv=int(np.argmax(asset['vowel_freq']));dr=int(np.argmax(asset['run_freq']))
    for i in range(KB):
        if i not in sb:b[i]=dv
    for i in range(KN):
        if i not in sn:n[i]=dr
    return {'bmap':b,'nmap':n,'fit_score':m['fit_score']}

def random_baseline(hold,asset,tag):
    z=[]
    for r in range(20):
        bm,nm=init_map(asset,f'{tag}:RAND:{r}');z.append(score_lines(hold,asset,{'bmap':bm,'nmap':nm}))
    return float(np.median(z))

def weighted_agree(ma,mb,hold,kind):
    if kind=='b':ev=[x for L in hold for x in L['b']];a=ma['bmap'];b=mb['bmap']
    else:ev=[x for L in hold for x in L['n'] if x>=0];a=ma['nmap'];b=mb['nmap']
    return sum(int(a[x])==int(b[x]) for x in ev)/max(1,len(ev))

def stability(train,asset,tag,restarts,passes):
    a=train[::2];b=train[1::2];ma=apply_defaults(fit_map(a,asset,tag+':SA',restarts,passes),a,asset);mb=apply_defaults(fit_map(b,asset,tag+':SB',restarts,passes),b,asset)
    ab=weighted_agree(ma,mb,train,'b');an=weighted_agree(ma,mb,train,'n');return .5*(ab+an),ab,an

def truth_recovery(m,key,hold,asset):
    be=[x for L in hold for x in L['b']];ne=[x for L in hold for x in L['n'] if x>=0];rb=sum(int(m['bmap'][x])==int(key['bmap'][x]) for x in be)/max(1,len(be));rn=sum(int(m['nmap'][x])==int(key['nmap'][x]) for x in ne)/max(1,len(ne))
    match=tot=0
    for L in hold:
        dec=decode_line(L,m['bmap'],m['nmap'],asset['runs']);tr=L['plain'];M=max(len(dec),len(tr));tot+=M;match+=sum(i<len(dec) and i<len(tr) and dec[i]==tr[i] for i in range(M))
    return rb,rn,match/max(1,tot)

def one(fam,phase,rep,A,smoke=False):
    lines,key,true_lang=family(fam,phase,rep,A);fit=lines[:60];sel=lines[60:80];hold=lines[80:]
    rest=1 if smoke else 3;pas=2 if smoke else 5;cand={}
    for la in ['DE','IT']:
        m=fit_map(fit,A[la],f'{phase}:{fam}:R{rep}:SEL:{la}',rest,pas);m=apply_defaults(m,fit,A[la]);cand[la]={'map':m,'select_score':score_lines(sel,A[la],m)}
    chosen=max(cand,key=lambda la:(cand[la]['select_score'],la));train=fit+sel;m=fit_map(train,A[chosen],f'{phase}:{fam}:R{rep}:REFIT:{chosen}',rest,pas);m=apply_defaults(m,train,A[chosen]);holdlm=score_lines(hold,A[chosen],m);rand=random_baseline(hold,A[chosen],f'{phase}:{fam}:R{rep}:{chosen}');adv=holdlm-rand;stab,sb,sn=stability(train,A[chosen],f'{phase}:{fam}:R{rep}:STAB:{chosen}',1 if smoke else 2,2 if smoke else 4);cb,cn=seen_cov(train,hold)
    rb=rn=rc=None;langok=None
    if fam in POS:
        langok=(chosen==true_lang)
        # Truth nucleus values are indices in the generating language inventory. Compare strings when solver language differs.
        if chosen==true_lang:rb,rn,rc=truth_recovery(m,key,hold,A[chosen])
        else:
            be=[x for L in hold for x in L['b']];rb=sum(int(m['bmap'][x])==int(key['bmap'][x]) for x in be)/max(1,len(be));ne=[x for L in hold for x in L['n'] if x>=0];rn=sum(A[chosen]['runs'][int(m['nmap'][x])]==A[true_lang]['runs'][int(key['nmap'][x])] for x in ne)/max(1,len(ne));
            match=tot=0
            for L in hold:
                dec=decode_line(L,m['bmap'],m['nmap'],A[chosen]['runs']);tr=L['plain'];M=max(len(dec),len(tr));tot+=M;match+=sum(i<len(dec) and i<len(tr) and dec[i]==tr[i] for i in range(M))
            rc=match/max(1,tot)
    return {'phase':phase,'family':fam,'rep':rep,'selected_language':chosen,'true_language':true_lang,'LANG_OK':langok,'SELECT_DE':cand['DE']['select_score'],'SELECT_IT':cand['IT']['select_score'],'HOLD_LM':holdlm,'RAND_HOLD_LM':rand,'HOLD_ADV':adv,'STAB':stab,'STAB_B':sb,'STAB_N':sn,'REC_B':rb,'REC_N':rn,'REC_CHAR':rc,'COV_B':cb,'COV_N':cn}

def brief(z):return {k:z[k] for k in ['phase','family','rep','selected_language','LANG_OK','HOLD_ADV','STAB','REC_B','REC_N','REC_CHAR','COV_B','COV_N']}

def smoke(A):
    rows=[]
    for f in FAMS:
        z=one(f,'SMOKE',0,A,True);rows.append(z);print('Q2ROW',json.dumps(brief(z),sort_keys=True),flush=True)
    return {'stage':'SMOKE','rows':rows}

def cal(A):
    rows=[]
    for f in FAMS:
        for r in range(3):
            z=one(f,'CAL',r,A,False);rows.append(z);print('Q2ROW',json.dumps(brief(z),sort_keys=True),flush=True)
    pos=[x for x in rows if x['family'] in POS];neg=[x for x in rows if x['family'] in NEG]
    lang=sum(bool(x['LANG_OK']) for x in pos);rchar=sum(x['REC_CHAR']>=.50 for x in pos);rmap=sum(x['REC_B']>=.70 and x['REC_N']>=.40 for x in pos)
    medadv={f:float(np.median([x['HOLD_ADV'] for x in rows if x['family']==f])) for f in FAMS};medst={f:float(np.median([x['STAB'] for x in rows if x['family']==f])) for f in FAMS};minpa=min(medadv[f] for f in POS);maxna=max(medadv[f] for f in NEG);minps=min(medst[f] for f in POS);maxns=max(medst[f] for f in NEG);sep_a=minpa>maxna;sep_s=minps>maxns
    if not (lang>=5 and rchar>=5 and rmap>=5 and sep_a and sep_s):return {'stage':'CAL','pass':False,'reason':'GLOBAL_CODEBOOK_NOT_IDENTIFIABLE_WITH_CURRENT_SOLVER' if not (lang>=5 and rchar>=5 and rmap>=5) else 'ADVERSARIAL_NONSEPARABLE','LANG_OK_count':lang,'REC_CHAR_gate_count':rchar,'REC_MAP_gate_count':rmap,'median_ADV':medadv,'median_STAB':medst,'sep_ADV':sep_a,'sep_STAB':sep_s,'rows':rows}
    ta=(minpa+maxna)/2;ts=(minps+maxns)/2;return {'stage':'CAL','pass':True,'TAU_ADV':ta,'TAU_STAB':ts,'LANG_OK_count':lang,'REC_CHAR_gate_count':rchar,'REC_MAP_gate_count':rmap,'median_ADV':medadv,'median_STAB':medst,'rows':rows}

def val(A,ta,ts):
    rows=[]
    for f in FAMS:
        for r in range(3):
            z=one(f,'VAL',r,A,False);rows.append(z);print('Q2ROW',json.dumps(brief(z),sort_keys=True),flush=True)
    pos=[x for x in rows if x['family'] in POS];neg=[x for x in rows if x['family'] in NEG];joint=lambda x:x['HOLD_ADV']>=ta and x['STAB']>=ts;pc={f:sum(joint(x) for x in rows if x['family']==f) for f in FAMS};lang=sum(bool(x['LANG_OK']) for x in pos);rchar=sum(x['REC_CHAR']>=.50 for x in pos);rmap=sum(x['REC_B']>=.70 and x['REC_N']>=.40 for x in pos);ok=sum(joint(x) for x in pos)>=5 and all(pc[f]>=2 for f in POS) and sum(joint(x) for x in neg)==0 and lang>=5 and rchar>=5 and rmap>=5
    return {'stage':'VAL','pass':bool(ok),'TAU_ADV':ta,'TAU_STAB':ts,'family_joint_pass':pc,'LANG_OK_count':lang,'REC_CHAR_gate_count':rchar,'REC_MAP_gate_count':rmap,'rows':rows}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['smoke','cal','val'],required=True);ap.add_argument('--tau-adv',type=float);ap.add_argument('--tau-stab',type=float);a=ap.parse_args();A=assets()
    if a.mode=='smoke':out=smoke(A)
    elif a.mode=='cal':out=cal(A)
    else:
        if a.tau_adv is None or a.tau_stab is None:raise SystemExit('VAL requires --tau-adv and --tau-stab')
        out=val(A,a.tau_adv,a.tau_stab)
    summary={k:v for k,v in out.items() if k!='rows'};print('VBM_V9_Q2_RESULT='+json.dumps(summary,sort_keys=True,separators=(',',':')))
if __name__=='__main__':main()
