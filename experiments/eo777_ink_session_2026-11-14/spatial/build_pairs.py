"""Compute and cache eo777 vmsspatial page-pair overlay scores. Usage: golfenv python build_pairs.py <cutoff> <out.pkl>"""
import sys, pickle
from vmsspatial import compare
cut=int(sys.argv[1]); c=compare.Comparison(); r=c.rank(pool='all',cutoff=cut,scope='pages')
keep=['a','b','dice_current','dice_recorded','spatial','x_dice','edit_all','js_word','js_joined']
rows=[{k:x[k] for k in keep}|{'node_a':x['pages_a'][0]['node'],'node_b':x['pages_b'][0]['node'],'hand_a':x['pages_a'][0]['hand'],'hand_b':x['pages_b'][0]['hand'],'sec_a':x['pages_a'][0]['section'],'sec_b':x['pages_b'][0]['section'],'slot_a':x['pages_a'][0]['slot'],'slot_b':x['pages_b'][0]['slot']} for x in r]
pickle.dump(dict(rows=rows,meta=dict(r.meta)),open(sys.argv[2],'wb')); print(len(rows),'pairs saved')
