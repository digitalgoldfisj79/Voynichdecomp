import csv, pickle
from collections import Counter
J={r['obj_id']:r for r in pickle.load(open('ckpt/s1_join_m300_r800.pkl','rb'))}
G={}
for r in csv.DictReader(open('out/s1_eyecheck_grades.csv')): G[r['obj_id']]=r
print('graded objects',len(G),'dupes?',sum(1 for _ in csv.DictReader(open('out/s1_eyecheck_grades.csv')))-len(G))
# pre-set sample: linked to first concordance row of its page, or flagged
minrow={}
for r in J.values():
    if r['vnu_row']: minrow[r['page']]=min(minrow.get(r['page'],99),r['vnu_row'])
def insample(r): return (r['vnu_row'] and r['vnu_row']==minrow.get(r['page'])) or r['span'] or r['ambig'] or r['conf'] in ('LOW','NONE')
S=[o for o,r in J.items() if insample(r) and r['page'] not in ('f90r_HERBAL','OFFPAGE')]
print('sample size',len(S),'graded in sample',sum(1 for o in S if o in G))
ungraded=[o for o in S if o not in G]; print('sample objects not graded:',ungraded)
def tally(ids,lab):
    c=Counter(G[o]['grade'] for o in ids if o in G)
    linked=[o for o in ids if o in G and J[o]['conf']!='NONE']
    cl=Counter(G[o]['grade'] for o in linked)
    w,rt=cl['WRONG'],cl['RIGHT']
    print(lab,dict(c),'| linked-only RIGHT',rt,'WRONG',w,'UNCLEAR',cl['UNCLEAR'],'error rate W/(R+W)=%.3f'%(w/(rt+w) if rt+w else float('nan')),
          ' worst case (UNCLEAR as WRONG) %.3f'%((w+cl['UNCLEAR'])/(rt+w+cl['UNCLEAR'])))
tally(S,'SAMPLE')
tally(list(G),'ALL GRADED')
for conf in ('HIGH','MED','LOW'):
    ids=[o for o in G if J[o]['conf']==conf]; c=Counter(G[o]['grade'] for o in ids); print(conf,dict(c))
