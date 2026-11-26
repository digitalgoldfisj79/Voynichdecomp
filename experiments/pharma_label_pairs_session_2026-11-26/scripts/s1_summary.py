import pickle, csv
from collections import Counter, defaultdict
J=pickle.load(open('ckpt/s1_join_m300_r800.pkl','rb'))
conc=list(csv.DictReader(open('data/concordance_upload.csv')))
pharma=[r for r in J if r['page'] not in ('f101r','f90r_HERBAL','OFFPAGE')]
print('objects on labelled pharma pages', len(pharma))
print('conf', Counter(r['conf'] for r in pharma)); print('span', sum(r['span'] for r in pharma), 'ambig', sum(r['ambig'] for r in pharma))
print('NONE by page', Counter(r['page'] for r in pharma if r['conf']=='NONE'))
frag_objs=defaultdict(list)
for r in pharma:
    if r['fragment'] is not None: frag_objs[r['fragment']].append(r)
frags_all=sorted({r['fragment_num'] for r in conc if r['fragment_num']},key=int)
lab_frag={r['fragment_num'] for r in conc if r['locus_type']=='Lf' and r['fragment_num']}
boxed={r['fragment_num'] for r in conc if r['locus_type']=='Lf' and r['fragment_num'] and r['box_x']}
print('fragments total',len(frags_all),'labelled',len(lab_frag),'with a label box',len(boxed))
linked={f for f in frag_objs if '|' not in f}
print('labelled+boxed fragments with >=1 object',len(linked & boxed),' without object:',len(boxed-linked))
print('fragments claimed by >1 object',sum(1 for f,v in frag_objs.items() if len(v)>1))
print('multi-claims by class combo', Counter(tuple(sorted(Counter(o['obj_class'] for o in v).items())) for f,v in frag_objs.items() if len(v)>1).most_common(12))
