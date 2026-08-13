#!/usr/bin/env python3
import json, math, os, sys
from collections import Counter, defaultdict
import numpy as np

SEED=20260813
UNIQUE_FRAMES=['GCGA','VDRB-1','TTVE','TTIA','ZLZB','ZLZI','TTLI','VDRB','FFSG','FFSG-2','RGVN','PCCA']
DATA_DEFAULT='/mnt/data/joint_lag/voynich_transcriptions_slim.json'

def is_ed1(a,b):
    if a==b: return False
    la,lb=len(a),len(b)
    if abs(la-lb)>1: return False
    if la==lb:
        return sum(x!=y for x,y in zip(a,b))==1
    if la>lb:
        a,b=b,a; la,lb=lb,la
    i=j=diff=0
    while i<la and j<lb:
        if a[i]==b[j]:
            i+=1; j+=1
        else:
            diff+=1; j+=1
            if diff>1: return False
    return True

def load_frame(path, frame, p_only=False):
    obj=json.load(open(path,encoding='utf-8'))
    out=[]
    for page,p in obj['pages'].items():
        def lk(x):
            try: return (0,int(x))
            except: return (1,str(x))
        for lid in sorted(p,key=lk):
            rec=p[lid]
            if p_only:
                u=rec.get('u','')
                if len(u)<2 or u[1] != 'P':
                    continue
            s=rec.get('t',{}).get(frame)
            if not s: continue
            t=s.split()
            if t: out.append((page,str(lid),t))
    return out

def prep(raw, top_sets=None):
    out=[]
    for page,lid,t in raw:
        n=len(t)
        eq=np.zeros((n,n),dtype=np.bool_)
        ed=np.zeros((n,n),dtype=np.bool_)
        for i in range(n):
            eq[i,i]=True
            for j in range(i+1,n):
                e=(t[i]==t[j]); eq[i,j]=eq[j,i]=e
                if not e:
                    d=is_ed1(t[i],t[j]); ed[i,j]=ed[j,i]=d
        tops={}
        if top_sets:
            for k,s in top_sets.items():
                tops[k]=np.array([x in s for x in t], dtype=np.bool_)
        out.append((page,lid,t,eq,ed,tops))
    return out

def counts(prep_lines, rng=None, permute=False, topks=()):
    c=Counter(); phase=Counter(); top=Counter(); rest=Counter()
    den=Counter(); pden=Counter(); tden=Counter(); rden=Counter()
    for _,_,t,eq,ed,tops in prep_lines:
        n=len(t); p=rng.permutation(n) if permute else np.arange(n)
        if n>=2:
            n1=ed[p[:-1],p[1:]]
            e2left=np.zeros(n-1,dtype=np.bool_); e2right=np.zeros(n-1,dtype=np.bool_)
            if n>=3:
                e2right[:n-2]=eq[p[:-2],p[2:]]; e2left[1:]=eq[p[:-2],p[2:]]
            c['N1'] += int(n1.sum()); den['N1'] += n-1
            keep=n1 & ~(e2left|e2right)
            c['N1_NONCLOSURE'] += int(keep.sum()); den['N1_NONCLOSURE'] += n-1
        if n>=3:
            e2=eq[p[:-2],p[2:]]; bridge=ed[p[:-2],p[1:-1]]
            c['E2'] += int(e2.sum()); den['E2'] += n-2
            nb=e2 & ~bridge
            c['E2_NONBRIDGE'] += int(nb.sum()); den['E2_NONBRIDGE'] += n-2
            phase['start'] += int(e2[0]); pden['start'] += 1
            if len(e2)>1:
                phase['interior'] += int(e2[1:].sum()); pden['interior'] += len(e2)-1
            for pos in range(6):
                if pos < len(e2): phase[f'p{pos}'] += int(e2[pos]); pden[f'p{pos}'] += 1
            if len(e2)>1:
                inds=np.arange(1,len(e2)); odd=(inds%2)==1; even=~odd
                phase['int_odd'] += int(e2[1:][odd].sum()); pden['int_odd'] += int(odd.sum())
                phase['int_even'] += int(e2[1:][even].sum()); pden['int_even'] += int(even.sum())
            for k in topks:
                is_top=tops[k][p[:-2]]
                top[k] += int((e2 & is_top).sum()); rest[k] += int((e2 & ~is_top).sum())
                tden[k] += int(is_top.sum()); rden[k] += int((~is_top).sum())
        if n>=5:
            e4=eq[p[:-4],p[4:]]; c['E4'] += int(e4.sum()); den['E4'] += n-4
        if n>=7:
            e6=eq[p[:-6],p[6:]]; c['E6'] += int(e6.sum()); den['E6'] += n-6
    return c,den,phase,pden,top,rest,tden,rden

def scalar(actual, sims):
    a=np.asarray(sims,float); mu=float(a.mean()); sd=float(a.std(ddof=1)) if len(a)>1 else float('nan')
    ratio=float(actual/mu) if mu>0 else float('nan'); z=float((actual-mu)/sd) if sd>0 else float('nan')
    return {'actual':float(actual),'null_mean':mu,'null_sd':sd,'ratio':ratio,'z':z}

def analyze(raw,nperm,seed,topks=(5,20,50)):
    freq=Counter(x for _,_,t in raw for x in t); top_sets={k:set(x for x,_ in freq.most_common(k)) for k in topks}; P=prep(raw,top_sets)
    ac,den,ap,pden,at,ar,tden,rden=counts(P,topks=topks)
    sm={m:[] for m in ['E2','E4','E6','N1','E2_NONBRIDGE','N1_NONCLOSURE']}; sp={k:[] for k in ['start','interior','int_odd','int_even']+[f'p{i}' for i in range(6)]}; st={k:[] for k in topks}; sr={k:[] for k in topks}
    rng=np.random.default_rng(seed)
    for _ in range(nperm):
        cc,_,pp,_,tt,rr,_,_=counts(P,rng,True,topks)
        for m in sm: sm[m].append(cc[m])
        for k in sp: sp[k].append(pp[k])
        for k in topks: st[k].append(tt[k]); sr[k].append(rr[k])
    out={'n_lines':len(raw),'n_tokens':sum(len(t) for _,_,t in raw),'metrics':{},'phase':{},'topk':{},'top_types':freq.most_common(50)}
    for m in sm:
        s=scalar(ac[m],sm[m]); s['denom']=den[m]; s['rate']=ac[m]/den[m] if den[m] else float('nan'); out['metrics'][m]=s
    for k in sp:
        s=scalar(ap[k],sp[k]); s['denom']=pden[k]; s['rate']=ap[k]/pden[k] if pden[k] else float('nan'); out['phase'][k]=s
    for name,a,b in [('boundary','start','interior'),('interior_parity','int_even','int_odd')]:
        obs=ap[a]/pden[a]-ap[b]/pden[b]; arr=np.array(sp[a])/pden[a]-np.array(sp[b])/pden[b]; s=scalar(obs,arr)
        ra=out['phase'][a]['ratio']; rb=out['phase'][b]['ratio']; s['ratio_fold']=max(ra,rb)/min(ra,rb) if min(ra,rb)>0 else float('inf'); out['phase'][name]=s
    for k in topks:
        muT=float(np.mean(st[k])); muR=float(np.mean(sr[k])); obsD=math.log((at[k]+0.5)/(muT+0.5))-math.log((ar[k]+0.5)/(muR+0.5)); Ds=[math.log((x+0.5)/(muT+0.5))-math.log((y+0.5)/(muR+0.5)) for x,y in zip(st[k],sr[k])]; ds=scalar(obsD,Ds)
        exT=at[k]-muT; exR=ar[k]-muR; pos=max(0,exT)+max(0,exR); share=max(0,exT)/pos if pos>0 else float('nan')
        out['topk'][str(k)]={'contrast':ds,'top':scalar(at[k],st[k]),'rest':scalar(ar[k],sr[k]),'top_positive_excess_share':share,'top_opportunities':tden[k],'rest_opportunities':rden[k]}
    return out

def shuffled(raw,seed):
    rng=np.random.default_rng(seed); out=[]
    for p,l,t in raw:
        a=list(t); rng.shuffle(a); out.append((p,l,a))
    return out

def mut1(s): return '~' if not s else ('~' if s[0]!='~' else '^')+s[1:]
def noned(s): return '~~'+s+'~~'

def choose(raw,span,frac,seed,min_i=0,filter_fn=None,blocked=None):
    rng=np.random.default_rng(seed); cand=[]; blocked=blocked or defaultdict(set)
    for li,(_,_,t) in enumerate(raw):
        for i in range(min_i,max(min_i,len(t)-span+1)):
            if filter_fn and not filter_fn(t,i): continue
            if set(range(i,i+span)).isdisjoint(blocked[li]): cand.append((li,i))
    rng.shuffle(cand); target=max(1,int(frac*len(cand))); chosen=[]; occ=defaultdict(set)
    for li,i in cand:
        ss=set(range(i,i+span))
        if occ[li].isdisjoint(ss) and blocked[li].isdisjoint(ss):
            chosen.append((li,i)); occ[li].update(ss)
            if len(chosen)>=target: break
    return chosen,occ

def controls(path):
    raw=load_frame(path,'ZLZI'); base=shuffled(raw,SEED+60000); c={}
    x=[(p,l,list(t)) for p,l,t in base]; sites,_=choose(x,3,.02,SEED+60100,min_i=1)
    for li,i in sites: x[li][2][i+2]=x[li][2][i]
    r=analyze(x,500,SEED+60200); b=r['phase']['boundary']; c['C6']={'n':len(sites),'analysis':r,'pass':r['phase']['interior']['ratio']>=1.15 and r['phase']['interior']['z']>=3 and b['z']<=-3 and b['ratio_fold']>=1.25}
    x=[(p,l,list(t)) for p,l,t in base]; sites,_=choose(x,3,.02,SEED+60300)
    for li,i in sites:
        a=x[li][2][i]; x[li][2][i+1]=mut1(a); x[li][2][i+2]=a
    r=analyze(x,500,SEED+60400); m=r['metrics']; c['C7a']={'n':len(sites),'analysis':r,'pass':m['E2']['z']>=3 and m['N1']['z']>=3 and (m['E2_NONBRIDGE']['ratio']<=1.10 or abs(m['E2_NONBRIDGE']['z'])<2) and (m['N1_NONCLOSURE']['ratio']<=1.10 or abs(m['N1_NONCLOSURE']['z'])<2)}
    x=[(p,l,list(t)) for p,l,t in base]; e2,occ=choose(x,3,.018,SEED+60500)
    for li,i in e2:
        a=x[li][2][i]; x[li][2][i+1]=noned(a); x[li][2][i+2]=a
    n1,_=choose(x,2,.018,SEED+60501,blocked=occ)
    for li,i in n1:
        a=x[li][2][i]; x[li][2][i+1]=mut1(a)
    r=analyze(x,500,SEED+60600); m=r['metrics']; c['C7b']={'n_e2':len(e2),'n_n1':len(n1),'analysis':r,'pass':m['E2_NONBRIDGE']['ratio']>=1.15 and m['E2_NONBRIDGE']['z']>=3 and m['N1_NONCLOSURE']['ratio']>=1.10 and m['N1_NONCLOSURE']['z']>=3}
    x=[(p,l,list(t)) for p,l,t in base]; sites,_=choose(x,7,.02,SEED+60700)
    for li,i in sites:
        a=x[li][2][i]
        for j in (2,4,6): x[li][2][i+j]=a
    r=analyze(x,500,SEED+60800); m=r['metrics']; c['C8']={'n':len(sites),'analysis':r,'pass':m['E4']['ratio']>=1.15 and m['E4']['z']>=3 and m['E6']['ratio']>=1.15 and m['E6']['z']>=3}
    z=[]
    for j in range(20):
        rr=analyze(shuffled(raw,SEED+61000+j),200,SEED+62000+j,topks=(20,)); z.append(rr['topk']['20']['contrast']['z'])
    meanz=float(np.mean([q for q in z if math.isfinite(q)])); nbad=sum(abs(q)>=2 for q in z if math.isfinite(q)); top20=set(x for x,_ in Counter(y for _,_,t in raw for y in t).most_common(20)); x=[(p,l,list(t)) for p,l,t in base]; sites,_=choose(x,3,.03,SEED+63000,filter_fn=lambda t,i:t[i] in top20)
    for li,i in sites: x[li][2][i+2]=x[li][2][i]
    rp=analyze(x,500,SEED+63100,topks=(20,)); q=rp['topk']['20']['contrast']; c['C9']={'null_z':z,'null_mean_z':meanz,'null_n_abs_z_ge2':nbad,'positive_n':len(sites),'positive_analysis':rp,'pass':abs(meanz)<=.5 and nbad<=2 and q['z']>=3}
    c['validity']={'H6':c['C6']['pass'],'H7':c['C7a']['pass'] and c['C7b']['pass'],'H8':c['C8']['pass'],'H9':c['C9']['pass']}; return c

def decide(ctrl,ref,pref,cross):
    D={}; st=ref['phase']['start']; intr=ref['phase']['interior']; bc=ref['phase']['boundary']; ip=ref['phase']['interior_parity']; q1=st['ratio']<=.80; q2=intr['ratio']>=1.20 and intr['z']>=3; q3=abs(bc['z'])>=3 and bc['ratio_fold']>=1.50; q4=abs(ip['z'])<2 or ip['ratio_fold']<1.15; n_dir=sum(cross[f]['phase']['start']['ratio']<cross[f]['phase']['interior']['ratio'] for f in UNIQUE_FRAMES); prok=pref['phase']['interior']['ratio']>=1.15 and pref['phase']['interior']['z']>=2 and pref['phase']['start']['ratio']<pref['phase']['interior']['ratio']
    if ctrl['validity']['H6'] and all([q1,q2,q3,q4]) and n_dir>=10 and prok: v='SUPPORT'
    elif ctrl['validity']['H6'] and all([q1,q2,q3,q4]) and n_dir>=10 and not prok: v='ALL-LOCUS ONLY'
    elif ctrl['validity']['H6'] and ((bc['ratio_fold']<1.15 and abs(bc['z'])<2) or (pref['phase']['start']['ratio']>pref['phase']['interior']['ratio'] and abs(pref['phase']['boundary']['z'])>=2)): v='FALSIFIED'
    else: v='UNRESOLVED'
    D['H6_LINE_START_GATE']={'verdict':v,'primary_conditions':[q1,q2,q3,q4],'cross_direction_n':n_dir,'p_only_ok':prok}
    m=ref['metrics']; a=m['E2_NONBRIDGE']; b=m['N1_NONCLOSURE']; e_dir=sum(cross[f]['metrics']['E2_NONBRIDGE']['ratio']>1 for f in UNIQUE_FRAMES); e_z=sum(cross[f]['metrics']['E2_NONBRIDGE']['z']>=2 for f in UNIQUE_FRAMES); n_dir2=sum(cross[f]['metrics']['N1_NONCLOSURE']['ratio']>1 for f in UNIQUE_FRAMES); n_z=sum(cross[f]['metrics']['N1_NONCLOSURE']['z']>=2 for f in UNIQUE_FRAMES); sup=ctrl['validity']['H7'] and a['ratio']>=1.15 and a['z']>=3 and b['ratio']>=1.10 and b['z']>=3 and e_dir>=10 and n_dir2>=10 and e_z>=8 and n_z>=8; fal=ctrl['validity']['H7'] and ((a['ratio']<=1.05 and abs(a['z'])<2 and e_dir<8) or (b['ratio']<=1.05 and abs(b['z'])<2 and n_dir2<8)); D['H7_SEPARABLE']={'verdict':'SUPPORT' if sup else ('FALSIFIED' if fal else 'UNRESOLVED'),'e2_dir':e_dir,'e2_z':e_z,'n1_dir':n_dir2,'n1_z':n_z}
    e4=m['E4']; e6=m['E6']; sup=ctrl['validity']['H8'] and e4['ratio']>=1.15 and e4['z']>=2 and e6['ratio']>=1.15 and e6['z']>=2; fal=ctrl['validity']['H8'] and e4['ratio']<=1.10 and e6['ratio']<=1.10 and e4['z']<2 and e6['z']<2; D['H8_PARITY_MEMORY']={'verdict':'SUPPORT' if sup else ('FALSIFIED' if fal else 'UNRESOLVED')}
    q=ref['topk']['20']; con=q['contrast']; sup=ctrl['validity']['H9'] and con['z']>=3 and q['top_positive_excess_share']>=.60; fal=ctrl['validity']['H9'] and abs(con['z'])<2 and q['rest']['ratio']>=1.10 and q['rest']['z']>=2; D['H9_LEXICAL_CARRIER']={'verdict':'SUPPORT' if sup else ('FALSIFIED' if fal else 'UNRESOLVED')}; return D

def main():
    path=sys.argv[1] if len(sys.argv)>1 else DATA_DEFAULT; outdir=sys.argv[2] if len(sys.argv)>2 else '/mnt/data/joint_lag/v02'; os.makedirs(outdir,exist_ok=True); print('controls',flush=True); ctrl=controls(path); json.dump(ctrl,open(os.path.join(outdir,'controls.json'),'w'),indent=2); print('control validity',ctrl['validity'],flush=True); cross={}
    for j,f in enumerate(UNIQUE_FRAMES):
        print('frame',f,flush=True); r=analyze(load_frame(path,f),500,SEED+70000+j*997); cross[f]=r; json.dump(r,open(os.path.join(outdir,f'frame_{f}.json'),'w'),indent=2)
    print('reference',flush=True); ref=analyze(load_frame(path,'ZLZI'),2000,SEED+80000); json.dump(ref,open(os.path.join(outdir,'reference_ZLZI.json'),'w'),indent=2); print('P-only',flush=True); pref=analyze(load_frame(path,'ZLZI',True),2000,SEED+81000); json.dump(pref,open(os.path.join(outdir,'reference_ZLZI_P_only.json'),'w'),indent=2); dec=decide(ctrl,ref,pref,cross); out={'seed':SEED,'controls':ctrl,'reference':ref,'reference_P_only':pref,'cross':cross,'decisions':dec}; json.dump(out,open(os.path.join(outdir,'RESULTS_v0_2.json'),'w'),indent=2); print('DECISIONS',json.dumps(dec,indent=2),flush=True)
if __name__=='__main__': main()
