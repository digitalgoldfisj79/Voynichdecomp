#!/usr/bin/env python3
import json, math, os, sys
from collections import Counter
import numpy as np

SEED=20260813
DATA_DEFAULT='/mnt/data/joint_lag/voynich_transcriptions_slim.json'
FRAMES=['GCGA','VDRB-1','TTVE','TTIA','ZLZB','ZLZI','TTLI','VDRB','FFSG','FFSG-2','RGVN','PCCA']
LENGTH_BINS=[(7,9),(10,12),(13,16),(17,10**9)]


def load_frame(path, frame):
    obj=json.load(open(path,encoding='utf-8'))
    vocab={}; nxt=0; lines=[]
    for page,pd in obj['pages'].items():
        def kf(x):
            try:return (0,int(x))
            except:return (1,str(x))
        for lid in sorted(pd,key=kf):
            s=pd[lid].get('t',{}).get(frame)
            if not s: continue
            ids=[]
            for t in s.split():
                if t not in vocab: vocab[t]=nxt; nxt+=1
                ids.append(vocab[t])
            if ids: lines.append(np.asarray(ids,dtype=np.int32))
    return lines


def groups_for(n, null):
    if null=='N0': return [np.arange(n,dtype=np.int32)] if n else []
    if null=='N1':
        if n<5:
            a=np.arange(min(2,n),dtype=np.int32)
            b=np.arange(min(2,n),n,dtype=np.int32)
            return [g for g in (a,b) if len(g)]
        return [np.arange(0,2,dtype=np.int32),np.arange(2,n-2,dtype=np.int32),np.arange(n-2,n,dtype=np.int32)]
    if null=='N2':
        out=[]
        bins=[[] for _ in range(3)]
        for i in range(n): bins[min(2,(3*i)//n)].append(i)
        return [np.asarray(g,dtype=np.int32) for g in bins if g]
    raise ValueError(null)


def permute_once(lines,null,seed):
    rng=np.random.default_rng(seed); out=[]
    for a in lines:
        b=a.copy()
        for g in groups_for(len(a),null):
            if len(g)>1: b[g]=rng.permutation(b[g])
        out.append(b)
    return out


def actual_scores(lines):
    whole=0; interior=0; hot=np.zeros(3,dtype=np.int64); leng=np.zeros(4,dtype=np.int64)
    dwhole=0; dint=0; dhot=np.zeros(3,dtype=np.int64); dleng=np.zeros(4,dtype=np.int64)
    for a in lines:
        n=len(a)
        if n>=3:
            m=(a[:-2]==a[2:])
            whole += int(m.sum()); dwhole += len(m)
        if n>=7:
            starts=np.arange(2,n-4,dtype=np.int32)  # i=2..n-5 inclusive
            if len(starts):
                mm=(a[starts]==a[starts+2])
                interior += int(mm.sum()); dint += len(mm)
                for j,i in enumerate(starts):
                    b=min(2,(3*int(i))//(n-2)); hot[b]+=int(mm[j]); dhot[b]+=1
                for bi,(lo,hi) in enumerate(LENGTH_BINS):
                    if lo<=n<=hi:
                        leng[bi]+=int(mm.sum()); dleng[bi]+=len(mm); break
    return {'whole':whole,'interior':interior,'hotspot':hot.tolist(),'length':leng.tolist(),
            'denom_whole':dwhole,'denom_interior':dint,'denom_hotspot':dhot.tolist(),'denom_length':dleng.tolist()}


def simulate(lines,null,nperm,seed,need_detail=False):
    rng=np.random.default_rng(seed)
    whole=np.zeros(nperm,dtype=np.int32)
    interior=np.zeros(nperm,dtype=np.int32) if need_detail else None
    hot=np.zeros((nperm,3),dtype=np.int32) if need_detail else None
    leng=np.zeros((nperm,4),dtype=np.int32) if need_detail else None
    for a in lines:
        n=len(a)
        if n<3: continue
        mat=np.broadcast_to(a,(nperm,n)).copy()
        for g in groups_for(n,null):
            if len(g)<=1: continue
            # independent row-wise random permutations via random-key ordering
            keys=rng.random((nperm,len(g)))
            order=np.argsort(keys,axis=1)
            vals=mat[:,g].copy()
            mat[:,g]=np.take_along_axis(vals,order,axis=1)
        mm=(mat[:,:-2]==mat[:,2:])
        whole += mm.sum(axis=1,dtype=np.int32)
        if need_detail and n>=7:
            starts=np.arange(2,n-4,dtype=np.int32)
            if len(starts):
                mi=(mat[:,starts]==mat[:,starts+2])
                interior += mi.sum(axis=1,dtype=np.int32)
                for j,i in enumerate(starts):
                    b=min(2,(3*int(i))//(n-2)); hot[:,b]+=mi[:,j]
                for bi,(lo,hi) in enumerate(LENGTH_BINS):
                    if lo<=n<=hi:
                        leng[:,bi]+=mi.sum(axis=1,dtype=np.int32); break
    return {'whole':whole,'interior':interior,'hotspot':hot,'length':leng}


def summ(actual, arr, denom=None):
    arr=np.asarray(arr,dtype=float); mu=float(arr.mean()); sd=float(arr.std(ddof=1))
    ratio=float(actual/mu) if mu>0 else float('nan'); z=float((actual-mu)/sd) if sd>0 else float('nan')
    out={'actual':float(actual),'null_mean':mu,'null_sd':sd,'ratio':ratio,'z':z}
    if denom:
        out['denom']=int(denom); out['rate']=float(actual/denom); out['null_rate']=float(mu/denom)
    return out


def analyse(lines,null,nperm,seed,detail=False):
    act=actual_scores(lines); sim=simulate(lines,null,nperm,seed,detail)
    out={'null':null,'nperm':nperm,'n_lines':len(lines),'n_tokens':int(sum(map(len,lines))),
         'whole':summ(act['whole'],sim['whole'],act['denom_whole'])}
    if detail:
        out['interior']=summ(act['interior'],sim['interior'],act['denom_interior'])
        hs=[]
        for b in range(3): hs.append(summ(act['hotspot'][b],sim['hotspot'][:,b],act['denom_hotspot'][b]))
        rates=np.array([sim['hotspot'][:,b]/act['denom_hotspot'][b] if act['denom_hotspot'][b] else np.zeros(nperm) for b in range(3)]).T
        obs_rates=np.array([act['hotspot'][b]/act['denom_hotspot'][b] if act['denom_hotspot'][b] else 0 for b in range(3)])
        ranges=rates.max(axis=1)-rates.min(axis=1); obs_range=float(obs_rates.max()-obs_rates.min())
        hr=summ(obs_range,ranges)
        ratios=[x['ratio'] for x in hs if math.isfinite(x['ratio']) and x['ratio']>0]
        hr['enrichment_fold']=float(max(ratios)/min(ratios)) if ratios else float('nan')
        out['hotspot']={'bins':hs,'range':hr}
        ls=[]
        for b in range(4): ls.append(summ(act['length'][b],sim['length'][:,b],act['denom_length'][b]) if act['denom_length'][b] else None)
        # concentration criterion, evaluated candidate by candidate with pooled complement null
        excess=np.array(act['length'],dtype=float)-sim['length'].mean(axis=0)
        total_pos=float(max(0,excess).sum()); candidates=[]
        for b in range(4):
            others=[j for j in range(4) if j!=b and act['denom_length'][j]>0]
            pooled_actual=sum(act['length'][j] for j in others); pooled_sim=sim['length'][:,others].sum(axis=1)
            ps=summ(pooled_actual,pooled_sim,sum(act['denom_length'][j] for j in others))
            share=float(max(0,excess[b])/total_pos) if total_pos>0 else 0.0
            candidates.append({'bin':b,'positive_excess_share':share,'pooled_others':ps})
        out['length']={'bins':ls,'candidates':candidates}
    return out


def control_calibration(lines,null,score_kind,seed):
    ratios=[]; zs=[]; reps=[]
    for r in range(20):
        pseudo=permute_once(lines,null,seed+10000*r)
        detail=(score_kind=='interior')
        rr=analyse(pseudo,null,200,seed+10000*r+1,detail=detail)
        s=rr[score_kind]
        ratios.append(s['ratio']); zs.append(s['z']); reps.append(s)
    mr=float(np.nanmean(ratios)); nz=sum(abs(z)>=2 for z in zs if math.isfinite(z))
    return {'mean_ratio':mr,'n_abs_z_ge2':nz,'pass':bool(0.95<=mr<=1.05 and nz<=2),'replicates':reps}


def inject_interior(lines,frac,seed):
    rng=np.random.default_rng(seed); out=[a.copy() for a in lines]
    cand=[]
    for li,a in enumerate(out):
        n=len(a)
        if n>=7:
            for i in range(2,n-4): cand.append((li,i))
    rng.shuffle(cand); target=max(1,int(frac*len(cand))); chosen=[]; occupied={}
    for li,i in cand:
        s=occupied.setdefault(li,set())
        pos={i,i+1,i+2}
        if s.isdisjoint(pos):
            chosen.append((li,i)); s.update(pos)
            if len(chosen)>=target: break
    for li,i in chosen: out[li][i+2]=out[li][i]
    return out,len(chosen)


def decisions(zl,cross,controls):
    n1=zl['N1']['whole']; n2=zl['N2']['whole']; n3=zl['N1']['interior']
    valid=all(controls[k]['pass'] for k in ['C0_N1','C0_N2','C0_N3']) and controls['C1']['pass']
    laafu=valid and all((x['ratio']<1.10 or x['z']<2) for x in [n1,n2,n3])
    cpos=sum(cross[f]['ratio']>1 for f in FRAMES); cstrong=sum(cross[f]['ratio']>=1.10 and cross[f]['z']>=2 for f in FRAMES)
    zstrong=[x['ratio']>=1.10 and x['z']>=3 for x in [n1,n2,n3]]
    inline_strong=valid and all(zstrong) and cpos>=8 and cstrong>=6
    inline_partial=valid and not inline_strong and (sum(zstrong)==2 or (all(zstrong) and not (cpos>=8 and cstrong>=6)))
    hr=zl['N1']['hotspot']['range']; hotspot=valid and hr['z']>=3 and hr['enrichment_fold']>=1.25
    length_support=False; winner=None
    for c in zl['N1']['length']['candidates']:
        p=c['pooled_others']
        if c['positive_excess_share']>0.5 and (p['ratio']<1.05 or abs(p['z'])<2): length_support=True; winner=c['bin']; break
    return {
      'H_LAAFU':{'verdict':'SUPPORT' if laafu else 'NOT_SUPPORTED','valid':valid},
      'H_INLINE':{'verdict':'STRONG_SUPPORT' if inline_strong else ('PARTIAL_SUPPORT' if inline_partial else 'UNSUPPORTED'),'ZLZI_strong_flags':zstrong,'cross_positive':cpos,'cross_strong':cstrong,'nframes':len(FRAMES)},
      'H_HOTSPOT':{'verdict':'SUPPORT' if hotspot else 'UNSUPPORTED','range':hr},
      'H_LENGTH':{'verdict':'SUPPORT' if length_support else 'UNSUPPORTED','winner_bin':winner}
    }


def main():
    path=sys.argv[1] if len(sys.argv)>1 else DATA_DEFAULT
    outdir=sys.argv[2] if len(sys.argv)>2 else '/mnt/data/joint_lag/v02'
    os.makedirs(outdir,exist_ok=True)
    zlines=load_frame(path,'ZLZI')
    print('controls',flush=True)
    controls={}
    controls['C0_N0']=control_calibration(zlines,'N0','whole',SEED+100)
    print(' C0 N0',controls['C0_N0']['pass'],flush=True)
    controls['C0_N1']=control_calibration(zlines,'N1','whole',SEED+200)
    print(' C0 N1',controls['C0_N1']['pass'],flush=True)
    controls['C0_N2']=control_calibration(zlines,'N2','whole',SEED+300)
    print(' C0 N2',controls['C0_N2']['pass'],flush=True)
    controls['C0_N3']=control_calibration(zlines,'N1','interior',SEED+400)
    print(' C0 N3',controls['C0_N3']['pass'],flush=True)
    base=permute_once(zlines,'N1',SEED+500); inj,ns=inject_interior(base,.02,SEED+501)
    c1=analyse(inj,'N1',1000,SEED+502,detail=True)['interior']; controls['C1']={'n_sites':ns,'stat':c1,'pass':bool(c1['ratio']>=1.15 and c1['z']>=3)}
    print(' C1',controls['C1']['pass'],c1['ratio'],c1['z'],flush=True)
    print('ZLZI primary',flush=True)
    zl={'N0':analyse(zlines,'N0',2000,SEED+1000,detail=False),
        'N1':analyse(zlines,'N1',2000,SEED+2000,detail=True),
        'N2':analyse(zlines,'N2',2000,SEED+3000,detail=False)}
    for k in ['N0','N1','N2']:
        print(' ',k,zl[k]['whole']['ratio'],zl[k]['whole']['z'],flush=True)
    print(' N3',zl['N1']['interior']['ratio'],zl['N1']['interior']['z'],flush=True)
    print('cross-frame N1',flush=True)
    cross={}
    for i,f in enumerate(FRAMES):
        rr=analyse(load_frame(path,f),'N1',500,SEED+10000+i*101,detail=False)['whole']; cross[f]=rr
        print(' ',f,rr['ratio'],rr['z'],flush=True)
    dec=decisions(zl,cross,controls)
    out={'seed':SEED,'frames':FRAMES,'controls':controls,'ZLZI':zl,'cross_N1':cross,'decisions':dec}
    jp=os.path.join(outdir,'RESULTS_laafu_conditioned_e2_v0_2_20260813.json'); json.dump(out,open(jp,'w'),indent=2)
    print('DECISIONS',json.dumps(dec,indent=2),flush=True)
    print('WROTE',jp,flush=True)

if __name__=='__main__': main()
