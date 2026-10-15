from pathlib import Path
from collections import Counter
import pickle,json,os,runpy,sys
root=Path('/mnt/data/vbm138_review/gen138'); os.chdir(root);sys.path.insert(0,str(root))
src=(root/'vbm_tafel_v138.py').read_text()
src=src.replace("rng=random.Random(1420); toks=[]; key='y'; NJ=[0,0]","rng=random.Random(1420); toks=[]; key='y'; NJ=[0,0]\nAUDIT={'key_checks':[],'seam_before':[],'seam_after':[], 'event_pairs':[]}")
src=src.replace("    k0=('s',cell[1],cell[2]) if fl else cell", "    k0=('s',cell[1],cell[2]) if fl else cell\n    AUDIT['key_checks'].append((i,key,k0[0],v,Kc,fl))")
src=src.replace("    if not ok(tok):\n        fixed=False", "    AUDIT['seam_before'].append((i,tok,ok(tok)))\n    if not ok(tok):\n        fixed=False")
src=src.replace("    a=_at2(tok)\n    if len(a)>=4", "    AUDIT['seam_after'].append((i,tok,ok(tok)))\n    AUDIT['event_pairs'].append((i,v2, tr, nk, k0[0], extra))\n    a=_at2(tok)\n    if len(a)>=4")
src=src.replace('lines=gen.wrap(toks)',"AUDIT['prewrap']=list(toks)\nlines=gen.wrap(toks)")
src=src.replace('lines=L3',"AUDIT['post_m']=list(L3)\nlines=L3")
src=src.replace("Path('GENERAT_v138_kurzertext.txt').write_text", "AUDIT['post_marker']=list(out)\nPath('AUDIT_gen138.txt').write_text")
src+='\nimport pickle\nwith open("/mnt/data/vbm138_review/stage2_raw.pkl", "wb") as fh: pickle.dump(AUDIT, fh)\n'
(root/'vbm_tafel_v138_audit.py').write_text(src)
d=runpy.run_path('vbm_tafel_v138_audit.py')
a=d['AUDIT'];A=d['_at2']; ok=d['ok']; pairs=d['pair_v']
measure=lambda toks: (len(toks),sum(not ok(t) for t in toks))
counts={
 'key_transition_n':len(a['key_checks']),
 'key_mismatches':sum(exp!=actual for _,exp,actual,*_ in a['key_checks']),
 'key_mismatches_examples':[(idx,e,act,v,k,fl) for idx,e,act,v,k,fl in a['key_checks'] if e!=act][:10],
 'seam_before_n':len(a['seam_before']),
 'seam_before_fails':sum(not good for _,tok,good in a['seam_before']),
 'seam_after_fails':sum(not good for _,tok,good in a['seam_after']),
 'seam_after_examples':[tok for _,tok,good in a['seam_after'] if not good][:15],
 'prewrap_bad':measure(a['prewrap']),
 'post_m_bad':measure([t for s in a['post_m'] for t in s.split()]),
 'post_marker_bad':measure([t for s in a['post_marker'] for t in s.split()]),
 'post_marker_bytes_match':(root/'AUDIT_gen138.txt').read_bytes()==(root/'GENERAT_v138_kurzertext_REFERENZ.txt').read_bytes(),
 'event_missed_pair':sum((tr,nk) not in pairs or pairs[(tr,nk)]!=v2 for _,v2,tr,nk,key,extra in a['event_pairs']),
}
# same source vowel but same visible pair; map class for all pairs
# Count mean total recoding loss from exact mapping by using first output key from RANK vs chosen nk
out=Path('/mnt/data/vbm138_review/stage2.pkl')
with out.with_suffix('.pkl.tmp').open('wb') as fp:pickle.dump(counts,fp)
out.with_suffix('.pkl.tmp').replace(out)
print(json.dumps(counts,indent=2,ensure_ascii=False))
