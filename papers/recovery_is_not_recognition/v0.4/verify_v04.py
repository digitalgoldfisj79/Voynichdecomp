from pathlib import Path
import hashlib, json, re, sys

root = Path(__file__).resolve().parent
md = (root / 'manuscript_v0_4.md').read_text()
checks = []

def check(name, cond, detail=''):
    checks.append({'name': name, 'pass': bool(cond), 'detail': detail})

required_strings = {
    'title': 'Recovery Is Not Recognition: A Recoverability-First Framework for Cipher-Like and Structured Historical Texts',
    'inverse_disclaimer': 'These categories are not assumed to exhaust historical practice',
    'onomancy_firewall': 'not as a candidate explanation of the Voynich Manuscript',
    'onomancy_count': 'in four redactions, all thirty residues map onto exactly two outcomes',
    'compression_stage1': 'macro accuracy 0.9922',
    'compression_shuffle': 'macro accuracy 1.000',
    'compression_stage2_zlib': 'macro accuracy 0.3798',
    'compression_stage2_bz2': 'macro accuracy 0.2821',
    'matched_null': 'matched-null false-positive rate 1.000',
    'stage_decisions': '`STAGE1_FAIL` and `STAGE2_SURFACE_FAIL`',
    'voynich_sealed': 'No Voynich distance matrix was produced',
    'conclusion_boundary': 'The paper does not determine whether the Voynich Manuscript contains',
}
for key, value in required_strings.items():
    check(key, value in md, value)

for forbidden in [
    'Voynich Manuscript uses onomancy',
    'proves that the Voynich',
    'all compression methods fail',
    'onomancy is a cipher',
    'decipherment of the Voynich',
]:
    check('forbidden:' + forbidden, forbidden.lower() not in md.lower(), forbidden)

headings = re.findall(r'^(#{1,2})\s+(.+)$', md, re.M)
check('single_abstract', sum(1 for _, h in headings if h == 'Abstract') == 1)
check('single_conclusion', sum(1 for _, h in headings if h == '15. Conclusion') == 1)
check('compression_section', sum(1 for _, h in headings if h.startswith('9. Compression proximity')) == 1)
check('onomancy_section', sum(1 for _, h in headings if h.startswith('2.3 Lossy symbolic')) == 1)
numbered = [h for _, h in headings if re.match(r'\d+(?:\.\d+)?\.', h)]
check('duplicate_numbered_headings', len(numbered) == len(set(numbered)))

for ref in ['Benedetto, D.', 'Chardonnens, L. S.', 'Cilibrasi, R.', 'Edge, J.', 'Juste, D.']:
    check('reference:' + ref, ref in md)

artifacts = {}
for filename in [
    'manuscript_v0_4.md',
    'cryptologia_recovery_not_recognition_v0_4.docx',
    'cryptologia_recovery_not_recognition_v0_4.tex',
    'cryptologia_recovery_not_recognition_v0_4.pdf',
    'cryptologia_recovery_not_recognition_v0_4_word.pdf',
]:
    path = root / filename
    if path.exists():
        artifacts[filename] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes': path.stat().st_size,
        }

report = {
    'status': 'PASS' if all(item['pass'] for item in checks) else 'FAIL',
    'checks': checks,
    'word_count': len(re.findall(r"\b[\w'-]+\b", md)),
    'heading_count': len(headings),
    'artifacts': artifacts,
}
(root / 'VERIFY_RESULT_v0_4.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if report['status'] == 'PASS' else 1)
