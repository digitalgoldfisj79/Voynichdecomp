#!/usr/bin/env python3
import importlib.util, json, math, os, sys
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
BASE_PATH=os.path.normpath(os.path.join(HERE,'..','laafu_conditioned_e2_v0_2','run_laafu_conditioned_e2.py'))
spec=importlib.util.spec_from_file_location('v02',BASE_PATH); v02=importlib.util.module_from_spec(spec); spec.loader.exec_module(v02)

SEED=20260813
FRAMES=v02.FRAMES
DATA_DEFAULT='/mnt/data/joint_lag/voynich_transcriptions_slim.json'


def subtype(u):
    if not u: return ''
    return u[1:] if len(u)>=2 else ''


def keep_locus(u,mode):
    st=subtype(u)
    if mode=='P0': return st=='P0'
    if mode=='PALL': return st.startswith('P')
    raise ValueError(mode)


def load_frame(path,frame,mode):
    obj=json.load(open(path,encoding='utf-8')); vocab={}; nxt=0; lines=[]
    for page,pd in obj['pages'].items():
        def kf(x):
            try:return (0,int(x))
            except:return (1,str(x))
        for lid in sorted(pd,key=kf):
            rec=pd[lid]
            if not keep_locus(rec.get('u',''),mode): continue
            s=rec.get('t',{}).get(frame)
            if not s: continue
            ids=[]
            for t in s.split():
                if t not in vocab: vocab[t]=nxt; nxt+=1
                ids.append(vocab[t])
            if ids: lines.append(np.asarray(ids,dtype=np.int32))
    return lines


def decisions(zl,cross,controls):
    n1=zl['N1']['whole']; n2=zl['N2']['whole']; n3=zl['N1']['interior']
    valid=all(controls[k]['pass'] for k in ['C0_N1','C0_N2','C0_N3']) and controls['C1']['pass']
    suff=valid and all((x['ratio']<1.10 or x['z']<2) for x in [n1,n2,n3])
    flags=[x['ratio']>=1.10 and x['z']>=3 for x in [n1,n2,n3]]
    cpos=sum(cross[f]['ratio']>1 for f in FRAMES); cstrong=sum(cross[f]['ratio']>=1.10 and cross[f]['z']>=2 for f in FRAMES)
    strong=valid and all(flags) and cpos>=8 and cstrong>=6
    partial=valid and not strong and (sum(flags)==2 or (all(flags) and not(cpos>=8 and cstrong>=6)))
    hr=zl['N1']['hotspot']['range']; hotspot=valid and hr['z']>=3 and hr['enrichment_fold']>=1.25
    length=False; winner=None
    for bi,c in enumerate(zl['N1']['length']['candidates']):
        b=zl['N1']['length']['bins'][bi]; p=c['pooled_others']
        if b and c['positive_excess_share']>0.5 and b['ratio']>=1.15 and b['z']>=2 and (p['ratio']<1.05 or abs(p['z'])<2):
            length=True; winner=bi; break
    return {
      'H_P0_LAAFU_SUFFICIENT':{'verdict':'SUPPORT' if suff else 'NOT_SUPPORTED','valid':valid},
      'H_P0_INLINE':{'verdict':'STRONG_SUPPORT' if strong else ('PARTIAL_SUPPORT' if partial else 'UNSUPPORTED'),'flags':flags,'cross_positive':cpos,'cross_strong':cstrong,'nframes':len(FRAMES)},
      'H_P0_HOTSPOT':{'verdict':'SUPPORT' if hotspot else 'UNSUPPORTED','range':hr},
      'H_P0_LENGTH':{'verdict':'SUPPORT' if length else 'UNSUPPORTED','winner_bin':winner}
    }


def main():
    path=sys.argv[1] if len(sys.argv)>1 else DATA_DEFAULT
    outdir=sys.argv[2] if len(sys.argv)>2 else '/mnt/data/joint_lag/v03'
    os.makedirs(outdir,exist_ok=True)
    p0=load_frame(path,'ZLZI','P0')
    print('P0 lines/tokens',len(p0),sum(map(len,p0)),flush=True)
    controls={}
    print('controls',flush=True)
    controls['C0_N0']=v02.control_calibration(p0,'N0','whole',SEED+100); print(' C0 N0',controls['C0_N0']['pass'],flush=True)
    controls['C0_N1']=v02.control_calibration(p0,'N1','whole',SEED+200); print(' C0 N1',controls['C0_N1']['pass'],flush=True)
    controls['C0_N2']=v02.control_calibration(p0,'N2','whole',SEED+300); print(' C0 N2',controls['C0_N2']['pass'],flush=True)
    controls['C0_N3']=v02.control_calibration(p0,'N1','interior',SEED+400); print(' C0 N3',controls['C0_N3']['pass'],flush=True)
    base=v02.permute_once(p0,'N1',SEED+500); inj,ns=v02.inject_interior(base,.02,SEED+501)
    c1=v02.analyse(inj,'N1',1000,SEED+502,detail=True)['interior']; controls['C1']={'n_sites':ns,'stat':c1,'pass':bool(c1['ratio']>=1.15 and c1['z']>=3)}; print(' C1',controls['C1']['pass'],c1['ratio'],c1['z'],flush=True)
    print('P0 primary',flush=True)
    zl={'N0':v02.analyse(p0,'N0',2000,SEED+1000,detail=False),
        'N1':v02.analyse(p0,'N1',2000,SEED+2000,detail=True),
        'N2':v02.analyse(p0,'N2',2000,SEED+3000,detail=False)}
    for k in ['N0','N1','N2']: print(' ',k,zl[k]['whole']['ratio'],zl[k]['whole']['z'],flush=True)
    print(' N3',zl['N1']['interior']['ratio'],zl['N1']['interior']['z'],flush=True)
    cross={}; print('cross P0 N1',flush=True)
    for i,f in enumerate(FRAMES):
        rr=v02.analyse(load_frame(path,f,'P0'),'N1',500,SEED+10000+i*101,detail=False)['whole']; cross[f]=rr; print(' ',f,rr['ratio'],rr['z'],flush=True)
    print('PALL robustness',flush=True)
    pall=load_frame(path,'ZLZI','PALL')
    rob={'N0':v02.analyse(pall,'N0',1000,SEED+20000,detail=False),
         'N1':v02.analyse(pall,'N1',1000,SEED+21000,detail=True),
         'N2':v02.analyse(pall,'N2',1000,SEED+22000,detail=False)}
    for k in ['N0','N1','N2']: print(' ',k,rob[k]['whole']['ratio'],rob[k]['whole']['z'],flush=True)
    print(' N3',rob['N1']['interior']['ratio'],rob['N1']['interior']['z'],flush=True)
    dec=decisions(zl,cross,controls)
    out={'seed':SEED,'P0_lines':len(p0),'P0_tokens':int(sum(map(len,p0))),'PALL_lines':len(pall),'PALL_tokens':int(sum(map(len,pall))),
         'controls':controls,'P0_ZLZI':zl,'P0_cross_N1':cross,'PALL_ZLZI_robustness':rob,'decisions':dec}
    jp=os.path.join(outdir,'RESULTS_laafu_paragraph_e2_v0_3_20260813.json'); json.dump(out,open(jp,'w'),indent=2)
    print('DECISIONS',json.dumps(dec,indent=2),flush=True); print('WROTE',jp,flush=True)

if __name__=='__main__': main()
