from pathlib import Path
import hashlib, json, re, sys
root=Path(__file__).resolve().parent
md=(root/'manuscript_v0_4_1.md').read_text(encoding='utf-8')
checks=[]
def check(name, cond, detail=''):
    checks.append({'name':name,'pass':bool(cond),'detail':detail})
required={
 'title':'Recovery Is Not Recognition: A Recoverability-First Framework for Cipher-Like and Structured Historical Texts',
 'generator_section':'# 4. Structural and generative baseline',
 'generator_scope':'Twenty-two models generated complete token sequences',
 'generator_score':'passed 59 of 84 metrics, with seed scores from 56 to 61',
 'split_half':'manuscript split-halves passed 81 of 84 metrics',
 'capacity':'maximum state-space capacity of $\\log_2 30 = 4.9069$ bits',
 'complete_redactions':'Redactions 1 and 4 are complete',
 'incomplete_r2':'twenty-nine scored entries 14/15, giving 0.9991 bits',
 'incomplete_r3':'twenty-five scored entries 12/13, giving 0.9988 bits',
 'edge_clause':'Edge (2014) independently describes variable letter-number assignments',
 'checksum':'1a90e584399aa3627dc28588d0691265b2829b0191696a194d59733479d580f7',
 'onomancy_firewall':'not as a candidate explanation of the Voynich Manuscript',
 'stage_decisions':'`STAGE1_FAIL` and `STAGE2_SURFACE_FAIL`',
 'voynich_sealed':'No Voynich distance matrix was produced',
 'conclusion_boundary':'The paper does not determine whether the Voynich Manuscript contains',
}
for k,v in required.items(): check(k,v in md,v)
for bad in [
 'The complete calculation therefore yields one bit',
 'in four redactions, all thirty residues map onto exactly two outcomes',
 'twenty-three non-message generators',
 'Voynich Manuscript uses onomancy',
 'all compression methods fail',
 'decipherment of the Voynich',
 'This shift - from compatibility',
]: check('forbidden:'+bad,bad.lower() not in md.lower(),bad)
headings=re.findall(r'^(#{1,2})\s+(.+)$',md,re.M)
check('single_abstract',sum(1 for _,h in headings if h=='Abstract')==1)
check('single_conclusion',sum(1 for _,h in headings if h=='16. Conclusion')==1)
check('single_generator_section',sum(1 for _,h in headings if h.startswith('4. Structural and generative baseline'))==1)
check('duplicate_numbered_headings',len([h for _,h in headings if re.match(r'\d+(?:\.\d+)?\.',h)])==len(set(h for _,h in headings if re.match(r'\d+(?:\.\d+)?\.',h))))
check('reference_order_barron_benedetto',md.index('Barron, A. R.') < md.index('Benedetto, D.'))
check('reference_order_chu_cilibrasi',md.index('Chu, C.') < md.index('Cilibrasi, R.'))
artifacts={}
for fn in ['manuscript_v0_4_1.md','cryptologia_recovery_not_recognition_v0_4_1.docx','cryptologia_recovery_not_recognition_v0_4_1.tex','cryptologia_recovery_not_recognition_v0_4_1.pdf','cryptologia_recovery_not_recognition_v0_4_1_word.pdf']:
    p=root/fn
    if p.exists(): artifacts[fn]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
report={'status':'PASS' if all(c['pass'] for c in checks) else 'FAIL','checks':checks,'word_count':len(re.findall(r"\b[\w'-]+\b",md)),'heading_count':len(headings),'artifacts':artifacts}
(root/'VERIFY_RESULT_v0_4_1.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
sys.exit(0 if report['status']=='PASS' else 1)
