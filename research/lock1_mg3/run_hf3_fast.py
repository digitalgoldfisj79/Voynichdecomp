"""HF CPU-XL driver for Lock-1 MG3 with implementation-only fast scorer amendment."""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys,time,urllib.request,pickle
CODE=Path(__file__).resolve().parent;LAYER=os.environ.get('MG_LAYER','ZLZI')
R=Path('/tmp/ut3');OUT=Path('/tmp')/f'mg3_out_{LAYER}'
shutil.rmtree(R,ignore_errors=True);shutil.rmtree(OUT,ignore_errors=True)
(R/'voynich_repo').mkdir(parents=True);(R/'joint_run').mkdir();OUT.mkdir()
def log(**k):k['t']=time.strftime('%H:%M:%S');print(json.dumps(k),flush=True)
def sh(cmd,**kw):log(stage='cmd',cmd=cmd);subprocess.run(cmd,check=True,**kw)
def nhash(p):return hashlib.sha256(((Path(p).read_bytes().decode().rstrip()+'\n').encode())).hexdigest()
MANIFEST={
'joint_model.py':'0377d7ddd68498ff054f99a3b433aad9b0be71b9996a48f04c5baa726195d762',
'joint_native.cpp':'d98bc0d6d3f81d80634c02652be0ec67bec80cb8bcaf187668dd27aa70ca6b05',
'run_joint.py':'114dc84e3d046ad991f33eedfc66e8552f336faaa7d176fa2dbf8cd4b596a372',
'c2st_kept8.py':'aaf2581812a6f2c214f3328216f75bb275e889c480c74f25dbfcfaa6fa51324e',
'mg2.py':'9a89e05be1616243a3b5aabb8a89e9a5b91de3af630b836d3833cb98b046343d',
'mg3.py':'6a65029391c2bfdce1baf9fc0278c0c365532520481217a53a28d1bf843470d2',
'MG3_PREREG_20261003.md':'78409045b54a45c5b3146ff40c1e90ada6b48338a2a5e217eeee8e70eb48420f',
'mg3_fastscore.py':'c10eba29cfc0b7d89e4ea0bac29ab4a277d40d1062be4ef87b0d2731bb6ce498',
'MG3_SCORING_AMENDMENT_20261003.md':'f158e9f262ab2d0cfb3845a60d591e30580d43a1545bb7796161c1d2714c2e09'}
bad={f:nhash(CODE/f) for f in MANIFEST if nhash(CODE/f)!=MANIFEST[f]};log(stage='manifest',layer=LAYER,bad=bad,manifest=MANIFEST);assert not bad,bad
sh([sys.executable,'-m','pip','install','-q','numpy==2.5.3','scipy==1.18.1','scikit-learn==1.9.1','rapidfuzz==3.14.1'])
for f in ['joint_model.py','joint_native.cpp','run_joint.py','c2st_kept8.py','mg2.py','mg3.py','mg3_fastscore.py']:shutil.copy(CODE/f,R/f)
zl=urllib.request.urlopen(urllib.request.Request('https://www.voynich.nu/data/ZL3b-n.txt',headers={'User-Agent':'Mozilla/5.0 (research job)'}),timeout=120).read()
assert hashlib.sha256(zl).hexdigest()=='bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc';(R/'ZL_source.txt').write_bytes(zl)
url='https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json'
data=urllib.request.urlopen(url,timeout=120).read();h=hashlib.sha256(data).hexdigest();assert h=='26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f',h;(R/'voynich_repo'/'voynich_transcriptions_slim.json').write_bytes(data)
slim=json.loads(data);recs=[]
for fol,ls in slim['pages'].items():
 rr=[(int(re.match(r'\d+',ln).group(0)),x['t'][LAYER].split()) for ln,x in ls.items() if x.get('t',{}).get(LAYER,'')];rr.sort(key=lambda z:z[0]);recs += [dict(folio=fol,line=a,tokens=t) for a,t in rr]
dh=hashlib.sha256(json.dumps(recs,sort_keys=True).encode()).hexdigest();EXP={'TTLI':'29e4b714ffc96d6bd27b44e5baf04a2ae0f65796ead832de8c14b79cad03a2a3','ZLZI':'0e7080e53b307a210929ffddd633e653a7a03811e9c46ece1f7ebf762736c7b8'};assert dh==EXP[LAYER],(LAYER,dh)
(R/f'dump{LAYER}.json').write_text(json.dumps({'structuredContent':{'data':recs}}));log(stage='data_hash',layer=LAYER,slim_sha=h,dump_sha=dh,n_lines=len(recs))
sh(['g++','-O3','-std=c++17','-fPIC','-shared','joint_native.cpp','-o','joint_native_v2.so'],cwd=R)
env=dict(os.environ,MG_LAYER=LAYER,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
chk=subprocess.run([sys.executable,'-c','import os,joint_model as J;rs=J.load_lines(os.environ["MG_LAYER"]);import collections as C;print(len(rs),sum(len(r["tokens"]) for r in rs),sorted(C.Counter(d for r,i,a,b,d,p in J.transition_rows(rs)).items()))'],cwd=R,env=env,capture_output=True,text=True,check=True);log(stage='data_check',out=chk.stdout.strip())
if LAYER=='ZLZI':assert '5162 37465' in chk.stdout and '(4, 23415)' in chk.stdout
if LAYER=='TTLI':assert '5101 34351' in chk.stdout
ps=[subprocess.Popen([sys.executable,'mg3.py','events',str(f)],cwd=R,env=env,stdout=open(OUT/f'events{f}.log','w'),stderr=subprocess.STDOUT) for f in range(5)]
assert all(p.wait()==0 for p in ps);log(stage='events_done')
nw=max(1,min(7,(os.cpu_count() or 5)//5));fenv=dict(env,MG1_WORKERS=str(nw));log(stage='fit_start',workers_per_fold=nw,cpus=os.cpu_count())
ps=[subprocess.Popen([sys.executable,'mg3.py','fit',str(f)],cwd=R,env=fenv,stdout=open(OUT/f'fit{f}.log','w'),stderr=subprocess.STDOUT) for f in range(5)];rc=[p.wait() for p in ps]
if any(rc):
 for f,x in enumerate(rc):
  if x:print((OUT/f'fit{f}.log').read_text()[-5000:],flush=True)
assert not any(rc);log(stage='fit_done')
for f in range(5):
 pr=pickle.load(open(R/'joint_run'/'MG3'/f'params_fold{f}.pkl','rb'));log(stage='fit_summary',fold=f,converged=pr['converged'],nit=pr['nit'],nll_per_event=pr['nll_per_event'],nov_rate=pr['nov_rate_fit'])
NREP=int(os.environ.get('MG3_NREP','32'));per=max(1,(os.cpu_count() or 5)//5);chunks=[(f,a,min(NREP,a+-(-NREP//per))) for f in range(5) for a in range(0,NREP,-(-NREP//per))]
ps=[subprocess.Popen([sys.executable,'mg3.py','gen',str(f),str(a),str(b)],cwd=R,env=env,stdout=open(OUT/f'gen{f}_{a}.log','w'),stderr=subprocess.STDOUT) for f,a,b in chunks];rc=[p.wait() for p in ps]
if any(rc):
 for (f,a,b),x in zip(chunks,rc):
  if x:print((OUT/f'gen{f}_{a}.log').read_text()[-5000:],flush=True)
assert not any(rc);log(stage='gen_done',chunks=len(chunks))
sh([sys.executable,'mg3_fastscore.py',str(NREP)],cwd=R,env=env,stdout=open(OUT/'score.log','w'),stderr=subprocess.STDOUT);print((OUT/'score.log').read_text(),flush=True)
sh([sys.executable,'c2st_kept8.py','MG3',str(NREP)],cwd=R,env=env,stdout=open(OUT/'c2st.log','w'),stderr=subprocess.STDOUT);print((OUT/'c2st.log').read_text(),flush=True)
res=json.load(open(R.parent/'mg3_results.json'));c2=json.load(open(R.parent/'c2st_MG3.json'));gate=c2['real_vs_real_gate'][0] if isinstance(c2['real_vs_real_gate'],list) else c2['real_vs_real_gate'];c2pass=(c2['auc_mean']<0.65 and gate<0.65)
if res['max_abs_z']>10 or len(res['runaway_reps'])>=3:verdict='FALSIFIED'
elif res['core_close'] and c2pass:verdict='CLOSES'
else:verdict='NOT_CLOSED_PARTIAL'
zs=sorted(res['z'].items(),key=lambda kv:-abs(kv[1]))[:15]
final=dict(layer=LAYER,verdict=verdict,core_close=res['core_close'],max_abs_z=res['max_abs_z'],argmax=res['argmax'],null95=res['null95'],runaway_n=len(res['runaway_reps']),top_z=zs,extra={k:dict(obs=res['observed_extra'][k],sim=res['sim_extra_mean'][k],sd=res['sim_extra_sd'][k],z=res['z_extra'][k]) for k in res['z_extra']},c2st_auc=c2['auc_mean'],c2st_sd=c2['auc_sd_across_reps'],real_gate=gate,c2st_pass=c2pass,fit_converged=[pickle.load(open(R/'joint_run'/'MG3'/f'params_fold{f}.pkl','rb'))['converged'] for f in range(5)],prereg_sha=MANIFEST['MG3_PREREG_20261003.md'],mg3_sha=MANIFEST['mg3.py'],scorer_sha=MANIFEST['mg3_fastscore.py'],score_amend_sha=MANIFEST['MG3_SCORING_AMENDMENT_20261003.md'])
print('MG3_FINAL_JSON='+json.dumps(final,sort_keys=True),flush=True)
