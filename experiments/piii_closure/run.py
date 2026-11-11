#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import random
import re
import shutil
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from piii_core import (
    CRITICAL, SEED, bootstrap_sd, features, grouped_auc, policy_auc,
    random_label_auc, sha256, write_csv,
)

REPS = 8
SUPPORTED = 'abcdefghiklmnopqrstuxyzw'
ARMS = ['iid10', 'iid24', 'iid131', 'cycle131', 'line_fixed', 'sticky24', 'permute10']
PRIMARY = ['iid10', 'iid24', 'iid131', 'cycle131', 'line_fixed']
SOURCES = {
    'melanchthon': 'R code/Historical_ciphers/original/melanchthon_confession.txt',
    'secreta_lat': 'Corpora/Historical_texts/Secreta_Secretorum_LAT',
    'picatrix': 'Corpora/Historical_texts/Picatrix',
    'rettorica': 'Corpora/Historical_texts/Rettorica',
}


def load_vms(repository, transcriber):
    data = json.loads((repository / 'voynich_transcriptions_slim.json').read_text())
    sections = json.loads((repository / 'voynich_section_map.json').read_text())['mapping']
    rows = []
    folios = sorted(
        data['pages'],
        key=lambda x: (int(re.match(r'f(\d+)', x).group(1)) if re.match(r'f(\d+)', x) else 999, x),
    )
    for folio in folios:
        if sections.get(folio) == 'text-only':
            continue
        line_numbers = sorted(
            data['pages'][folio],
            key=lambda x: int(x) if str(x).isdigit() else 9999,
        )
        for line_number in line_numbers:
            text = data['pages'][folio][line_number].get('t', {}).get(transcriber, '')
            tokens = [token.lower() for token in text.split()]
            tokens = [token for token in tokens if re.fullmatch('[a-z]+', token)]
            if len(tokens) >= 2:
                rows.append({'f': folio, 's': sections.get(folio, 'unknown'), 't': tokens})
    return rows


def normalise_plaintext(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode()
    text = text.lower().replace('j', 'i').replace('v', 'u')
    return ''.join(character for character in text if character in SUPPORTED)


def load_table(path):
    rows = list(csv.reader(path.open(encoding='utf-8-sig')))
    header = [cell.strip() for cell in rows[0]]
    table = [[cell.strip().replace(' ', '') for cell in row] for row in rows[1:] if row]
    assert header == list(SUPPORTED)
    assert len(table) >= 100 and all(len(row) == 24 for row in table)
    return header, table


def encode(template, plaintext, header, table, arm, seed):
    rng = random.Random(seed)
    position = 0
    column = {character: i for i, character in enumerate(header)}
    active_table = [row[:] for row in table]
    if arm == 'permute10':
        for i, row in enumerate(active_table):
            permutation = list(range(24))
            random.Random(seed + 99173 + i).shuffle(permutation)
            active_table[i] = [row[j] for j in permutation]

    output = []
    for source_row in template:
        characters = plaintext[position:position + len(source_row['t'])]
        position += len(source_row['t'])
        words = []
        fixed_row = rng.randrange(len(active_table))
        sticky_state = rng.randrange(24)
        for j, character in enumerate(characters):
            if arm in ('iid10', 'permute10'):
                row = rng.randrange(10)
            elif arm == 'iid24':
                row = rng.randrange(24)
            elif arm == 'iid131':
                row = rng.randrange(len(active_table))
            elif arm == 'cycle131':
                row = j % len(active_table)
            elif arm == 'line_fixed':
                row = fixed_row
            elif arm == 'sticky24':
                if j and rng.random() > 0.75:
                    sticky_state = rng.choice([x for x in range(24) if x != sticky_state])
                row = sticky_state
            else:
                raise KeyError(arm)
            words.append(active_table[row][column[character]])
        output.append({'f': source_row['f'], 's': source_row['s'], 't': words})
    return output


def shuffled_vms(vms, mode, seed):
    rng = random.Random(seed)
    output = []
    for row in vms:
        tokens = row['t'][:]
        if mode == 'word':
            rng.shuffle(tokens)
        elif mode == 'character':
            changed = []
            for token in tokens:
                characters = list(token)
                rng.shuffle(characters)
                changed.append(''.join(characters))
            tokens = changed
        else:
            raise KeyError(mode)
        output.append({'f': row['f'], 's': row['s'], 't': tokens})
    return output


def main():
    here = Path(__file__).resolve().parent
    root = Path('/tmp/piii_closure')
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    work = root / 'work'
    work.mkdir()
    hermes = work / 'hermes'
    repository = here.parents[1]

    subprocess.run(
        ['git', 'clone', '--depth', '1', 'https://github.com/hermesj/R_Voynich_Stats.git', str(hermes)],
        check=True,
    )
    shutil.copy2(here / 'PROTOCOL.md', root / 'PROTOCOL.md')
    shutil.copy2(here / 'piii_core.py', root / 'piii_core.py')
    shutil.copy2(here / 'run.py', root / 'run.py')
    (root / 'FREEZE.txt').write_text(f"protocol_sha256={sha256(root / 'PROTOCOL.md')}\n")

    header, table = load_table(
        hermes / 'R code/Historical_ciphers/keys/ciphers_poly_III_in_order.csv'
    )
    vms = load_vms(repository, 'ZLZI')
    sensitivity_vms = {name: load_vms(repository, name) for name in ['ZLZB', 'TTIA']}
    required = sum(len(row['t']) for row in vms)
    plaintexts = {
        name: normalise_plaintext((hermes / path).read_text(errors='ignore'))
        for name, path in SOURCES.items()
    }
    plaintexts = {name: text for name, text in plaintexts.items() if len(text) >= required}
    assert len(plaintexts) == 4

    provenance = {
        'seed': SEED,
        'replicates_per_source_policy': REPS,
        'table_rows': len(table),
        'table_columns': len(header),
        'vms_tokens': required,
        'vms_lines': len(vms),
        'vms_folios': len(set(row['f'] for row in vms)),
        'plaintext_lengths': {name: len(text) for name, text in plaintexts.items()},
        'hermes_commit': subprocess.check_output(
            ['git', '-C', str(hermes), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        'voynichdecomp_commit': subprocess.check_output(
            ['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        'table_sha256': sha256(
            hermes / 'R code/Historical_ciphers/keys/ciphers_poly_III_in_order.csv'
        ),
        'protocol_sha256': sha256(root / 'PROTOCOL.md'),
    }
    (root / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2))

    vms_features = features([row['t'] for row in vms], graph_cap=1000)
    vms_bootstrap_sd = bootstrap_sd(vms, repetitions=200)
    corpora, c2st_rows, metric_rows = [], [], []
    source_names = sorted(plaintexts)

    for arm_index, arm in enumerate(ARMS):
        for source_index, source in enumerate(source_names):
            for replicate in range(REPS):
                seed = SEED + 100000 * arm_index + 1000 * source_index + replicate
                corpus = encode(vms, plaintexts[source], header, table, arm, seed)
                corpora.append((arm, source, replicate, corpus))
                c2st_rows.append({
                    'arm': arm,
                    'source': source,
                    'replicate': replicate,
                    'seed': seed,
                    'auc': grouped_auc(vms, corpus),
                })
                metric_rows.append({
                    'arm': arm,
                    'source': source,
                    'replicate': replicate,
                    **features([row['t'] for row in corpus], graph_cap=1000),
                })

    write_csv(root / 'C2ST_REPLICATES.csv', c2st_rows)
    write_csv(root / 'GLOBAL_METRICS.csv', metric_rows)

    negative = [random_label_auc(vms, SEED + i) for i in range(20)]
    word_shuffle = [
        grouped_auc(vms, shuffled_vms(vms, 'word', SEED + i)) for i in range(10)
    ]
    character_shuffle = [
        grouped_auc(vms, shuffled_vms(vms, 'character', SEED + i)) for i in range(10)
    ]
    policy_macro_auc, policy_accuracy = policy_auc(corpora)
    calibration = {
        'negative_random_label_auc': negative,
        'negative_median': float(np.median(negative)),
        'word_shuffle_auc': word_shuffle,
        'word_shuffle_median': float(np.median(word_shuffle)),
        'character_shuffle_auc': character_shuffle,
        'character_shuffle_median': float(np.median(character_shuffle)),
        'policy_macro_auc': policy_macro_auc,
        'policy_accuracy': policy_accuracy,
    }
    calibration['pass'] = bool(
        calibration['negative_median'] <= 0.60
        and calibration['word_shuffle_median'] >= 0.80
        and calibration['character_shuffle_median'] >= 0.80
        and policy_macro_auc >= 0.80
    )
    (root / 'CALIBRATION.json').write_text(json.dumps(calibration, indent=2))

    discrepancy_rows, arm_summary = [], {}
    for arm in ARMS:
        aucs = [row['auc'] for row in c2st_rows if row['arm'] == arm]
        arm_metrics = [row for row in metric_rows if row['arm'] == arm]
        absolute_z = []
        for metric in CRITICAL:
            candidate_mean = float(np.mean([row[metric] for row in arm_metrics]))
            sd = vms_bootstrap_sd[metric]
            z = (candidate_mean - vms_features[metric]) / sd if sd > 1e-12 else float('inf')
            absolute_z.append(abs(z))
            discrepancy_rows.append({
                'arm': arm,
                'metric': metric,
                'vms': vms_features[metric],
                'candidate_mean': candidate_mean,
                'vms_quire_bootstrap_sd': sd,
                'z': z,
            })
        arm_summary[arm] = {
            'median_auc': float(np.median(aucs)),
            'auc_95_interval': [
                float(np.quantile(aucs, 0.025)), float(np.quantile(aucs, 0.975))
            ],
            'median_absolute_z': float(np.median(absolute_z)),
            'maximum_absolute_z': float(max(absolute_z)),
        }
    write_csv(root / 'STANDARDISED_DISCREPANCIES.csv', discrepancy_rows)

    sensitivity_rows = []
    for transcriber, alternate in sensitivity_vms.items():
        for arm in PRIMARY:
            for replicate in range(3):
                seed = (
                    SEED + 900000 + 10000 * ['ZLZB', 'TTIA'].index(transcriber)
                    + 100 * PRIMARY.index(arm) + replicate
                )
                corpus = encode(
                    alternate, plaintexts['melanchthon'], header, table, arm, seed
                )
                sensitivity_rows.append({
                    'transcriber': transcriber,
                    'arm': arm,
                    'replicate': replicate,
                    'auc': grouped_auc(alternate, corpus),
                })
    write_csv(root / 'TRANSCRIPTION_SENSITIVITY.csv', sensitivity_rows)
    sensitivity_summary = defaultdict(list)
    for row in sensitivity_rows:
        sensitivity_summary[(row['transcriber'], row['arm'])].append(row['auc'])

    compatible = []
    for arm in PRIMARY:
        summary = arm_summary[arm]
        passes = bool(
            summary['median_auc'] <= 0.60
            and summary['median_absolute_z'] <= 2.0
            and summary['maximum_absolute_z'] <= 4.0
            and all(
                np.median(sensitivity_summary[(transcriber, arm)]) <= 0.65
                for transcriber in ['ZLZB', 'TTIA']
            )
        )
        summary['surface_compatible'] = passes
        if passes:
            compatible.append(arm)

    if not calibration['pass']:
        verdict = 'UNRESOLVED_CALIBRATION_FAILED'
    elif compatible:
        verdict = 'PASS_' + ','.join(compatible)
    else:
        verdict = 'FAIL_TESTED_PIII_POLICIES'

    final = {
        'status': 'COMPLETE',
        'formal_verdict': verdict,
        'calibration_pass': calibration['pass'],
        'surface_compatible_arms': compatible,
        'arms': arm_summary,
        'scope': 'Exact external Polygraphia III table under five fixed policies with imposed Voynich line layout.',
        'non_exclusions': [
            'Voynich-fitted systematic nomenclator',
            'changing scribe-specific codebooks',
            'post-encryption surface realiser',
            'alternative segmentation or sparse payload',
        ],
    }
    (root / 'FINAL_RESULT.json').write_text(json.dumps(final, indent=2))

    report = [
        '# PIII-CLOSURE results', '', f'**Formal verdict: {verdict}**', '',
        f"Protocol SHA-256: `{provenance['protocol_sha256']}`", '',
        '## Calibration', '',
        f"- Random-label median AUC: {calibration['negative_median']:.3f} (required ≤0.60)",
        f"- Word-shuffle median AUC: {calibration['word_shuffle_median']:.3f} (required ≥0.80)",
        f"- Character-shuffle median AUC: {calibration['character_shuffle_median']:.3f} (required ≥0.80)",
        f"- Policy macro AUC: {policy_macro_auc:.3f} (required ≥0.80)", '',
        '## Primary results', '',
        '| Policy | Median C2ST AUC | Median |z| | Max |z| | Compatible |',
        '|---|---:|---:|---:|:---:|',
    ]
    for arm in PRIMARY:
        summary = arm_summary[arm]
        report.append(
            f"| {arm} | {summary['median_auc']:.3f} | "
            f"{summary['median_absolute_z']:.2f} | "
            f"{summary['maximum_absolute_z']:.2f} | "
            f"{summary['surface_compatible']} |"
        )
    report.extend([
        '',
        'A FAIL is bounded to the unchanged PIII table and policies tested. The generation was given the actual Voynich line-length template, so it cannot explain line lengths and cannot blame a failure on wrapping. No Voynich token dictionary or Davis hand labels were used.',
        '',
        'Direct exact-token MDL is not reported because a cross-alphabet token mapping was not fitted; fitting one would create a different, much more flexible hypothesis.',
    ])
    (root / 'EXECUTIVE_SUMMARY.md').write_text('\n'.join(report) + '\n')
    (root / 'QA.json').write_text(json.dumps({
        'protocol_frozen': True,
        'no_davis_labels': True,
        'no_vms_table_fit': True,
        'approximate_quire_groups': True,
        'number_of_simulations': len(corpora),
        'all_runs_seeded': True,
    }, indent=2))

    shutil.rmtree(work)
    checksums = []
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'SHA256SUMS.txt':
            checksums.append(f"{sha256(path)}  {path.relative_to(root)}")
    (root / 'SHA256SUMS.txt').write_text('\n'.join(checksums) + '\n')
    zip_path = shutil.make_archive(
        '/tmp/PIII_CLOSURE_2026-07-14', 'zip', '/tmp', 'piii_closure'
    )
    print(json.dumps({
        'formal_verdict': verdict,
        'calibration': calibration,
        'arms': arm_summary,
        'zip_path': zip_path,
    }, indent=2))


if __name__ == '__main__':
    main()
