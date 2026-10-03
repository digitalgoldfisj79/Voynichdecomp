"""MG3 verified-checkpoint replay on CPU XL.
Reuses exact audited MG2 parameter pickles from /work/out2_{LAYER}; all hashes abort on mismatch.
Writes MG3 outputs to /work/out3_{LAYER}_mg3palette_v01.
"""
from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys,time,urllib.request,pickle,tarfile
CODE=Path(__file__).resolve().parent;LAYER=os.environ.get('MG_LAYER','ZLZI')
R=Path('/tmp/ut3r');OUT=Path('/tmp')/f'mg3r_{LAYER}'
shutil.rmtree(R,ignore_errors=True);shutil.rmtree(OUT,ignore_errors=True)
(R/'voynich_repo').mkdir(parents=True);(R/'joint_run'/'MG3').mkdir(parents=True);OUT.mkdir()
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
'MG3_SCORING_AMENDMENT_20261003.md':'f158e9f262ab2d0cfb3845a60d591e30580d43a1545bb7796161c1d2714c2e09',
'MG3_PARAM_REUSE_AMENDMENT_20261003.md':'3bb820b032f886a06daa6f7ecbd3de791ac237a22d45cd63dc38ae05b52ea5d2'}
PARAM={
'ZLZI':['3e75671f69535a40d0e77c0c7e71f18f9739d579e4937f98b4ce928de7966128','37b6698b47a8275f74015380e78a59fc6e0a6ae6757e1be24f441d254b99b9f6','6d9e7a728c74469bcfef9c2dc2815b4d0c2ca546262aa510dc0491482a8fa2f8','2fb8662cbb1ef2f8de15daf8b1f7ae51440226ae08155cc9e61d5cf5ee442b2b','d9e7208f47959514546ce0bd531c82b9b88822db4c52fdeca402d2dbc731de8f'],
'TTLI':['9478d1dcd39297442d1d3c5f8394de06e19af661fea676cc043ba5122eab9e2a','5e75d245f21d666a38eee4d4366afe0921da97a30348afe992ff87a9c785f6b0','06203f9ad64f50958a9e264e6b1d73cd2f5ccb887ae104d7a98fa0d0dd1b364a','cecb421d6c259207c988963a357b27edb42399bc45c25fb02e6ec36cfd372e49','b1ff6eb423ba6b7e69fa2741a00d94b8fb161829815a81084ded06c4e6842271']}
MG2RES={'ZLZI':'e6d849a89498f33097e822c2105353c37dbe4c1205e12fac51082bd06a055071','TTLI':'77437f7a4435d3673c6e30b84a7894ba2fe4b40c18a00e1c4b789757717ce97e'}
bad={f:nhash(CODE/f) for f in MANIFEST if nhash(CODE/f)!=MANIFEST[f]};log(stage='manifest',layer=LAYER,bad=bad);assert not bad,bad
sh([sys.executable,'-m','pip','install','-q','numpy==2.5.3','scipy==1.18.1','scikit-learn==1.9.1','rapidfuzz==3.14.1'])
for f in ['joint_model.py','joint_native.cpp','run_joint.py','c2st_kept8.py','mg2.py','mg3.py','mg3_fastscore.py']:shutil.copy(CODE/f,R/f)
# Data frozen exactly as MG2
zl=urllib.request.urlopen(urllib.request.Request('https://www.voynich.nu/data/ZL3b-n.txt',headers={'User-Agent':'Mozilla/5.0 (research job)'}),timeout=120).read();assert hashlib.sha256(zl).hexdigest()=='bf5b6d4ac1e3a51b1847a9c388318d609020441ccd56984c901c32b09beccafc';(R/'ZL_source.txt').write_bytes(zl)
data=urllib.request.urlopen('https://raw.githubusercontent.com/digitalgoldfisj79/Voynichdecomp/main/voynich_transcriptions_slim.json',timeout=120).read();assert hashlib.sha256(data).hexdigest()=='26e7490e099b1074ed2ce19356d0ea493aa1791826004e1c551d3f4f9bf8574f';(R/'voynich_repo'/'voynich_transcriptions_slim.json').write_bytes(data)
slim=json.loads(data);recs=[]
for fol,ls in slim['pages'].items():
 rr=[(int(re.match(r'\d+',ln).group(0)),x['t'][LAYER].split()) for ln,x in ls.items() if x.get('t',{}).get(LAYER,'')];rr.sort(key=lambda z:z[0]);recs += [dict(folio=fol,line=a,tokens=t) for a,t in rr]
dh=hashlib.sha256(json.dumps(recs,sort_keys=True).encode()).hexdigest();EXP={'TTLI':'29e4b714ffc96d6bd27b44e5baf04a2ae0f65796ead832de8c14b79cad03a2a3','ZLZI':'0e7080e53b307a210929ffddd633e653a7a03811e9c46ece1f7ebf762736c7b8'};assert dh==EXP[LAYER],(LAYER,dh);(R/f'dump{LAYER}.json').write_text(json.dumps({'structuredContent':{'data':recs}}));log(stage='data_hash',layer=LAYER,dump_sha=dh,n_lines=len(recs))
sh(['g++','-O3','-std=c++17','-fPIC','-shared','joint_native.cpp','-o','joint_native_v2.so'],cwd=R)
env=dict(os.environ,MG_LAYER=LAYER,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
# Provenance-check and copy exact MG2 checkpoints.
src=Path('/work')/f'out2_{LAYER}';m2=src/'mg2_results.json';assert hashlib.sha256(m2.read_bytes()).hexdigest()==MG2RES[LAYER]
old=json.load(open(m2));log(stage='mg2_result_check',max_abs_z=old['max_abs_z'],argmax=old['argmax'],null95=old['null95'],runaways=len(old.get('runaway_reps',[])))
for f,h in enumerate(PARAM[LAYER]):
 p=src/f'params_fold{f}.pkl';got=hashlib.sha256(p.read_bytes()).hexdigest();assert got==h,(f,got,h);x=pickle.load(open(p,'rb'));assert x['converged'] is True
 shutil.copy(p,R/'joint_run'/'MG3'/f'params_fold{f}.pkl');log(stage='checkpoint',fold=f,sha=got,nit=x['nit'],nll_per_event=x['nll_per_event'],nov_rate=x['nov_rate_fit'])
log(stage='checkpoint_reuse_pass')
NREP=32;per=max(1,(os.cpu_count() or 5)//5);chunks=[(f,a,min(NREP,a+-(-NREP//per))) for f in range(5) for a in range(0,NREP,-(-NREP//per))]
ps=[subprocess.Popen([sys.executable,'mg3.py','gen',str(f),str(a),str(b)],cwd=R,env=env,stdout=open(OUT/f'gen{f}_{a}.log','w'),stderr=subprocess.STDOUT) for f,a,b in chunks];rc=[p.wait() for p in ps]
if any(rc):
 for (f,a,b),x in zip(chunks,rc):
  if x:print((OUT/f'gen{f}_{a}.log').read_text()[-5000:],flush=True)
assert not any(rc);log(stage='gen_done',chunks=len(chunks))
sh([sys.executable,'mg3_fastscore.py',str(NREP)],cwd=R,env=env,stdout=open(OUT/'score.log','w'),stderr=subprocess.STDOUT);print((OUT/'score.log').read_text(),flush=True)
sh([sys.executable,'c2st_kept8.py','MG3',str(NREP)],cwd=R,env=env,stdout=open(OUT/'c2st.log','w'),stderr=subprocess.STDOUT);print((OUT/'c2st.log').read_text(),flush=True)
res=json.load(open(R.parent/'mg3_results.json'));c2=json.load(open(R.parent/'c2st_MG3.json'));gate=c2['real_vs_real_gate'][0] if isinstance(c2['real_vs_real_gate'],list) else c2['real_vs_real_gate'];c2pass=(c2['auc_mean']<0.65 and gate<0.65)
verdict='FALSIFIED' if(res['max_abs_z']>10 or len(res['runaway_reps'])>=3) else('CLOSES' if res['core_close'] and c2pass else'NOT_CLOSED_PARTIAL')
zs=sorted(res['z'].items(),key=lambda kv:-abs(kv[1]))[:15]
final=dict(layer=LAYER,verdict=verdict,core_close=res['core_close'],max_abs_z=res['max_abs_z'],argmax=res['argmax'],null95=res['null95'],runaway_n=len(res['runaway_reps']),top_z=zs,extra={k:dict(obs=res['observed_extra'][k],sim=res['sim_extra_mean'][k],sd=res['sim_extra_sd'][k],z=res['z_extra'][k]) for k in res['z_extra']},c2st_auc=c2['auc_mean'],c2st_sd=c2['auc_sd_across_reps'],real_gate=gate,c2st_pass=c2pass,param_hashes=PARAM[LAYER],checkpoint_reuse=True)
print('MG3_FINAL_JSON='+json.dumps(final,sort_keys=True),flush=True)
log(stage='done_readonly_checkpoint_replay')
