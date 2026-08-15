from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

CALIB_FOLIOS = ['f10r','f10v','f11r','f11v','f13r','f13v','f14r','f14v','f15r','f15v','f16r','f16v','f17r','f17v','f18r','f18v','f19r','f19v','f1v','f20r','f20v','f21r','f21v','f22r','f22v','f23r','f23v','f24r','f24v','f25r','f25v','f26r','f26v','f27r','f27v','f28r','f28v','f29r','f29v','f2r','f2v','f30r','f30v','f31r','f31v','f32r','f32v','f33r','f33v','f34r','f34v','f35r','f35v','f36r','f36v','f37r','f37v','f38r','f38v','f39r','f39v','f3r','f3v','f40r','f40v','f41r','f42r','f42v','f43r','f43v','f44r','f44v','f45r','f45v','f46r','f46v','f47r','f47v','f48r','f48v','f49r','f49v','f4r','f4v','f50r','f50v','f51r','f51v','f52r','f52v','f53r','f53v','f54r','f54v','f55r','f55v','f56r','f56v','f5r','f6r','f6v','f7r','f7v','f8r','f8v','f9r','f9v']
EXPECTED_WORD_KEYSET_SHA256 = 'c494eb695691e899d6e1dc648f9f7d7ec4afe49141a8890f9c1c40638b6a3f84'
EXPECTED_PAIR_SHA256 = '7f29bb7fe782130ddffe3d7809ce024e04a7eb01fa5c4194440d3be18cea3ed4'
EXPECTED_ENCODER_SHA256 = '54ef0612e623fa1755a488cdb975263c93f77c034085b3fa11eff21b62ba52b0'
EXPECTED_WORDS = 9620
EXPECTED_FOLIOS = 107
RATIO_GRID = [0.55, 0.50, 0.45, 0.40]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-root', type=Path, required=True)
    ap.add_argument('--pipeline-root', type=Path, required=True)
    ap.add_argument('--stageb-script', type=Path, required=True)
    ap.add_argument('--encoder', type=Path, required=True)
    ap.add_argument('--pair-b64', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(args.pipeline_root))
    from vdino3 import cfg, sources, register, crop

    # Freeze word extraction; suppress only downstream component proposals.
    crop.connected_components = lambda *a, **k: []
    crop.eva_soft_partition = lambda *a, **k: []
    cfg.CACHE_DIR = str(args.work / 'cache')
    Path(cfg.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    manifest = args.manifest_root / 'results/corpus_crop_manifest.jsonl'
    expected = []
    folset = set(CALIB_FOLIOS)
    with manifest.open('r', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r.get('kind') == 'word' and r.get('view') == 'norm' and str(r.get('folio')) in folset:
                expected.append({
                    'id': str(r['id']), 'folio': str(r['folio']),
                    'word_index': int(r['word_index']), 'word': str(r.get('word', ''))
                })
    E = (pd.DataFrame(expected)
         .sort_values(['folio', 'word_index', 'id'], kind='stable')
         .drop_duplicates(['folio', 'word_index'], keep='first').reset_index(drop=True))
    keytext = ''.join(f'{f}|{int(i)}\n' for f, i in sorted(zip(E.folio, E.word_index)))
    keyhash = hashlib.sha256(keytext.encode()).hexdigest()
    if len(E) != EXPECTED_WORDS or E.folio.nunique() != EXPECTED_FOLIOS or keyhash != EXPECTED_WORD_KEYSET_SHA256:
        raise RuntimeError(f'population gate failed rows={len(E)} folios={E.folio.nunique()} hash={keyhash}')
    byfolio = {fol: g.copy() for fol, g in E.groupby('folio')}

    man = sources.yale_manifest()
    canvases = sources.yale_canvases(man)
    outroot = args.work / 'stageb_data'
    (outroot / 'results').mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest, outroot / 'results/corpus_crop_manifest.jsonl')

    def assess(folio: str):
        reg, allc = register.register_folio(folio, canvases, 6)
        exact_label = str(reg.canvas_label).strip() == folio[1:]
        H = np.asarray(reg.H_deriv, dtype=float)
        plausible = False
        plausible_error = None
        try:
            tgt, _, _ = register._fetch_yale_derivative(reg.service_id, cfg.REG_DERIVATIVE_PX)
            plausible = bool(register._canvas_plausible(H, cfg.FRAME_LEGACY, tgt.shape))
        except Exception as e:
            plausible_error = repr(e)
        geometry = bool(
            exact_label and
            int(reg.inliers) >= int(cfg.REG_MIN_INLIERS) and
            float(reg.median_reproj_px) <= float(cfg.REG_MAX_MEDIAN_REPROJ_PX) and
            plausible
        )
        flags = {'legacy_055': bool(reg.passed), 'ratio_free_geometry': geometry}
        for t in RATIO_GRID:
            flags[f'ratio_{t:.2f}'] = bool(geometry and float(reg.inlier_ratio) >= t)
        rec = {
            'folio': folio, 'canvas_label': reg.canvas_label, 'service_id': reg.service_id,
            'matches': int(reg.matches), 'inliers': int(reg.inliers),
            'inlier_ratio': float(reg.inlier_ratio),
            'median_reproj_px': float(reg.median_reproj_px),
            'p95_reproj_px': float(reg.p95_reproj_px), 'H_deriv': reg.H_deriv,
            'deriv_scale': float(reg.deriv_scale), 'legacy_passed': bool(reg.passed),
            'legacy_reason': str(reg.reason), 'exact_canvas_label': exact_label,
            'canvas_plausible_recheck': plausible, 'plausibility_error': plausible_error,
            'rule_pass': flags, 'candidate_n': len(allc),
        }
        return reg, rec

    registrations = {}
    selected = {}
    errors = []
    completed = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut = {ex.submit(assess, fol): fol for fol in sorted(byfolio)}
        for f in cf.as_completed(fut):
            fol = fut[f]
            try:
                reg, rec = f.result()
                registrations[fol] = rec
                selected[fol] = reg
            except Exception as e:
                errors.append({'folio': fol, 'error': repr(e), 'trace': traceback.format_exc()[-4000:]})
            completed += 1
            if completed % 10 == 0 or errors:
                print('REG_PROGRESS', completed, '/', len(byfolio), 'errors', len(errors), flush=True)
    if errors:
        raise RuntimeError(f'registration assessment errors: {errors[:3]}')

    rule_names = ['legacy_055', 'ratio_free_geometry'] + [f'ratio_{t:.2f}' for t in RATIO_GRID]
    completeness = {
        rule: {
            'folios_pass': sum(bool(registrations[f]['rule_pass'][rule]) for f in registrations),
            'folios_fail': sorted(f for f in registrations if not registrations[f]['rule_pass'][rule]),
        } for rule in rule_names
    }

    realization = [{
        'folio': f, 'service_id': registrations[f]['service_id'],
        'H_deriv': registrations[f]['H_deriv']
    } for f in sorted(registrations)]
    realization_sha = stable_hash(realization)
    complete_rules = [r for r in rule_names if completeness[r]['folios_pass'] == EXPECTED_FOLIOS]

    audit = {
        'schema': 'u6-stageb-registration-sensitivity-v0.3',
        'target_opened': False, 'true_retention_read': False,
        'primary_v02_status': 'FAIL_RECONSTRUCTION__ABSTAIN_UNRESOLVED',
        'only_changed_variable': 'registration admission; principal rule removes inlier-ratio criterion only',
        'population': {'words': len(E), 'folios': E.folio.nunique(), 'word_keyset_sha256': keyhash},
        'pipeline_contract': {
            'REG_MIN_INLIERS': int(cfg.REG_MIN_INLIERS),
            'REG_MIN_INLIER_RATIO_inherited': float(cfg.REG_MIN_INLIER_RATIO),
            'REG_MAX_MEDIAN_REPROJ_PX': float(cfg.REG_MAX_MEDIAN_REPROJ_PX),
            'REG_DERIVATIVE_PX': int(cfg.REG_DERIVATIVE_PX), 'max_candidates': 6,
            'ratio_grid': RATIO_GRID,
        },
        'completeness': completeness, 'complete_rules': complete_rules,
        'registration_realization_sha256': realization_sha,
        'registrations': registrations,
    }
    audit_path = args.work / 'U6_STAGEB_REGISTRATION_SENSITIVITY_AUDIT_v0_3.json'
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding='utf-8')
    print('SENSITIVITY_COMPLETENESS', json.dumps(completeness, sort_keys=True), flush=True)
    print('REGISTRATION_REALIZATION_SHA256', realization_sha, flush=True)

    if completeness['ratio_free_geometry']['folios_pass'] != EXPECTED_FOLIOS:
        print('FORMAL_SENSITIVITY_STOP RATIO_FREE_INCOMPLETE', flush=True)
        return 20

    # Exact crop reconstruction under the frozen selected homographies. Admission no longer changes them.
    crop_errors = []
    def extract_one(folio: str):
        reg = selected[folio]
        g = byfolio[folio]
        boxes = sources.parse_runtime_boxes(folio)
        mapped = register.transform_boxes(reg, boxes)
        md = {int(x['index']): x for x in mapped}
        miss = [int(x) for x in g.word_index if int(x) not in md]
        if miss:
            return {'folio': folio, 'error': 'mapped_index_missing', 'indices': miss[:20], 'n': len(miss)}
        info = json.loads(sources.fetch(reg.service_id + '/info.json', '.json'))
        full_wh = (int(info['width']), int(info['height']))
        fd = outroot / 'reextract' / folio
        fd.mkdir(parents=True, exist_ok=True)
        w = crop.ProposalWriter(str(fd))
        for wi in g.word_index:
            w.add_word(folio, reg.service_id, full_wh, md[int(wi)])
        w.flush()
        got = []
        with (fd / 'crop_manifest.jsonl').open('r', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                if r.get('kind') == 'word' and r.get('view') == 'norm':
                    got.append(r)
        G = pd.DataFrame(got)
        if len(G) != len(g):
            return {'folio': folio, 'error': 'generated_word_count', 'expected': len(g), 'got': len(G)}
        gm = {(str(r.folio), int(r.word_index)): str(r.id) for r in G.itertuples()}
        mism = []
        for r in g.itertuples():
            k = (str(r.folio), int(r.word_index))
            x = gm.get(k)
            if x != str(r.id):
                mism.append({'key': k, 'expected': str(r.id), 'got': x})
        if mism:
            return {'folio': folio, 'error': 'crop_id_mismatch', 'n': len(mism), 'examples': mism[:10]}
        return None

    completed = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut = {ex.submit(extract_one, fol): fol for fol in sorted(byfolio)}
        for f in cf.as_completed(fut):
            fol = fut[f]
            try:
                err = f.result()
                if err:
                    crop_errors.append(err)
            except Exception as e:
                crop_errors.append({'folio': fol, 'error': repr(e), 'trace': traceback.format_exc()[-4000:]})
            completed += 1
            if completed % 10 == 0 or crop_errors:
                print('CROP_PROGRESS', completed, '/', len(byfolio), 'errors', len(crop_errors), flush=True)
    if crop_errors:
        raise RuntimeError(f'crop reconstruction errors: {crop_errors[:3]}')

    norm = list((outroot / 'reextract').rglob('*_norm.png'))
    if len(norm) != EXPECTED_WORDS:
        raise RuntimeError(f'norm crop count gate failed {len(norm)} != {EXPECTED_WORDS}')

    pair = args.work / 'U6_STAGEB_EVENT_SKELETON.csv'
    pair.write_bytes(zlib.decompress(base64.b64decode(args.pair_b64.read_text().strip())))
    if sha256(pair) != EXPECTED_PAIR_SHA256:
        raise RuntimeError(f'pair hash gate failed {sha256(pair)}')
    if sha256(args.encoder) != EXPECTED_ENCODER_SHA256:
        raise RuntimeError(f'encoder hash gate failed {sha256(args.encoder)}')

    stageout = args.work / 'stageb_out'
    cmd = [sys.executable, str(args.stageb_script), '--data', str(outroot), '--encoder', str(args.encoder),
           '--pair-skeleton', str(pair), '--out', str(stageout)]
    print('STAGEB_CALIBRATION_START target_opened=false true_retention_read=false', flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout[-20000:], flush=True)
    if p.returncode != 0:
        print('STAGEB_RETURN_CODE', p.returncode, flush=True)
        return p.returncode

    result_path = stageout / 'U6_STAGEB_RESULT.json'
    result = json.loads(result_path.read_text(encoding='utf-8'))
    combined = {
        'schema': 'u6-stageb-registration-sensitivity-result-v0.3',
        'target_opened': False, 'true_retention_read': False,
        'primary_v02_status': 'FAIL_RECONSTRUCTION__ABSTAIN_UNRESOLVED',
        'sensitivity_registration_audit': audit,
        'stageb_v02_frozen_calibration': result,
        'formal_sensitivity_verdict': (
            'PASS_SECONDARY_SENSITIVITY__TARGET_REMAINS_SEALED'
            if result.get('formal_verdict') == 'PASS_VTPS_CALIBRATION'
            else 'FAIL_CALIBRATION__ABSTAIN_UNRESOLVED'
        ),
    }
    combined_path = args.work / 'U6_STAGEB_REGISTRATION_SENSITIVITY_RESULT_v0_3.json'
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True), encoding='utf-8')
    print('U6_V03_FINAL_RESULT_JSON_BEGIN', flush=True)
    print(json.dumps(combined, sort_keys=True), flush=True)
    print('U6_V03_FINAL_RESULT_JSON_END', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
