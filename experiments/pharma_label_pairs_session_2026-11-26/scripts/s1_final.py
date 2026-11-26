import csv, pickle, hashlib, json
from collections import defaultdict, Counter
J=pickle.load(open('ckpt/s1_join_m300_r800.pkl','rb'))
G={r['obj_id']:r for r in csv.DictReader(open('out/s1_eyecheck_grades.csv'))}
conc=list(csv.DictReader(open('data/concordance_upload.csv')))
meth={}
for c in conc:
    if c['fragment_num']: meth.setdefault(c['fragment_num'],set()).add(c['assign_method'])
rows=[]
for r in J:
    g=G.get(r['obj_id'],{}); grade=g.get('grade','UNGRADED'); tf=g.get('true_fragment','')
    geom=r['fragment']
    if grade=='RIGHT': fin,src=geom,'geom_eye_confirmed'
    elif grade=='WRONG' and tf: fin,src=tf,'eye_corrected'
    elif grade in ('MISSED','NONE_UNBOXED') and tf: fin,src=tf,'eye_added'
    elif grade=='UNGRADED' and r['conf']=='HIGH': fin,src=geom,'geom_high_unchecked'
    else: fin,src=None,'none'
    rows.append(dict(obj_id=r['obj_id'],folio_seq=r['folio_seq'],obj_index=r['obj_index'],obj_class=r['obj_class'],page=r['page'],
        yale_image_id=r['yale_image_id'],yale_x=r['ox'],yale_y=r['oy'],yale_w=r['ow'],yale_h=r['oh'],
        geom_fragment=geom,geom_zl_locus=r['zl_locus'],geom_vnu_row=r['vnu_row'],dist_px=r['dist'],gap_px=r['gap'],conf=r['conf'],
        span=r['span'],ambig=r['ambig'],eye_grade=grade,eye_fragment=tf or None,eye_note=g.get('note') or None,
        fragment_final=fin,final_source=src,label_method=';'.join(sorted(meth.get(str(fin),[]))) if fin else None))
keys=list(rows[0].keys())
with open('out/stage1_object_fragment_join_v01.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
pickle.dump(rows,open('ckpt/s1_final.pkl','wb'))
h=hashlib.sha256(open('out/stage1_object_fragment_join_v01.csv','rb').read()).hexdigest()
print('rows',len(rows),'sha256',h)
print('final_source',Counter(r['final_source'] for r in rows))
# fragment-level
fr=defaultdict(list)
for r in rows:
    if r['fragment_final'] and '|' not in str(r['fragment_final']): fr[r['fragment_final']].append(r)
allf=sorted({c['fragment_num'] for c in conc if c['fragment_num']},key=int)
lab={c['fragment_num'] for c in conc if c['locus_type']=='Lf' and c['fragment_num']}
print('fragments with >=1 final object',len(fr),'of',len(allf),'; labelled fragments with object',len(set(fr)&lab),'of',len(lab))
print('labelled fragments with NO object:',sorted(lab-set(fr),key=int))
print('parts per fragment',Counter('+'.join(sorted(set(o['obj_class'] for o in v))) for v in fr.values()))
print('multi-object fragments',sum(1 for v in fr.values() if len(v)>1))
# disagreements: eye WRONG where the geometric link's label was assigned by order_gate/by_eye/panel_count
dis=[r for r in rows if r['eye_grade']=='WRONG']
for r in dis:
    gm=';'.join(sorted(meth.get(str(r['geom_fragment']),[]))) if r['geom_fragment'] else ''
    r['geom_label_method']=gm
print('WRONG links by method of the label they hit:',Counter(r['geom_label_method'] for r in dis))
with open('out/stage1_disagreements.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['obj_id','page','obj_class','geom_fragment','geom_label_method','eye_fragment','eye_fragment_label_method','note'])
    for r in dis: w.writerow([r['obj_id'],r['page'],r['obj_class'],r['geom_fragment'],r['geom_label_method'],r['eye_fragment'],';'.join(sorted(meth.get(str(r['eye_fragment']),[]))),r['eye_note']])
json.dump({k:[o['obj_id'] for o in v] for k,v in fr.items()},open('out/stage1_fragment_objects.json','w'),indent=0)
