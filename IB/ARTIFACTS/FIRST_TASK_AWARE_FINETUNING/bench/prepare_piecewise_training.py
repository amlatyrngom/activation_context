"""Freeze a balanced existing-bank subset for matched merged/piecewise QA training."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    with source.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != '34c36167c381e3be22a76692f929373fff272eaefcb6ec14515b39fda63683e3':
        raise ValueError('Expected the frozen original 27B 16k training material')
    material = json.loads(source.read_text())
    learning = set(material['learning_ids'])
    groups = collections.defaultdict(list)
    for row in material['train']:
        groups[row['source_family']].append(row)
    selected, counts = [], {}
    for family, rows in sorted(groups.items()):
        assert len(rows) % 4 == 0
        count = len(rows) // 4
        required = [row for row in rows if row['example']['example_id'] in learning]
        available = [row for row in rows if row['example']['example_id'] not in learning]
        random.Random('20260922:piecewise:' + family).shuffle(available)
        assert len(required) <= count
        selected.extend(required + available[:count-len(required)])
        counts[family] = count
    ids = [row['example']['example_id'] for row in selected]
    held_ids = [row['example']['example_id'] for row in material['held']]
    assert len(ids) == len(set(ids)) == 4000
    assert learning <= set(ids) and not set(ids) & set(held_ids)
    assert len(held_ids) == len(set(held_ids)) == 400
    assert sum(map(len, material['held_cohorts'].values())) == 400
    material['train'] = selected
    material['whole_run_panels'] = True
    material['planned_epochs'] = 1
    parent_recipe = material.pop('recipe')
    material['recipe'] = {
        'source_sha256': digest,
        'source_shares': parent_recipe['source_shares'],
        'selection': '4000 existing training questions: exact quarter of each source family, retain original32 learning IDs, seeded shuffle for remaining choices.',
        'scope': 'Identical questions, order, retrieval, targets and total rows for merged/piecewise;8k evidence soft budget, up to16 passages. Fresh QA training from reconstructionP2.',
        'confirmation': 'No reserved confirmation is read or used by this preparation or comparison.',
    }
    output.mkdir(parents=True)
    path = output/'material.json'
    path.write_text(json.dumps(material, ensure_ascii=False))
    with path.open('rb') as handle:
        material_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    summary = {
        'source_sha256': digest, 'material_sha256': material_digest,
        'train_questions': len(ids), 'training_epochs': 1,
        'training_by_family': counts, 'held_questions': len(held_ids),
        'held_cohorts': {key: len(value) for key, value in material['held_cohorts'].items()},
        'learning_questions': len(learning), 'retrieval_questions': len(material['retrieval']),
        'selection_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        'initialization': 'Original P2 paired reconstruction checkpoint; fresh QA optimizer and schedule.',
        'evidence_tokens': 8192, 'max_passages': 16, 'compression_ratio': 1/16,
        'layout': 'Merged control versus one independently query-conditioned part per passage; same total compressed rows.',
        'limits': 'Soft total evidence budget; preserve whole required sources. Passage lengths vary; this is not fixed-size token chunking.',
    }
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'PRESERVE').write_text('Frozen paired piecewise/merged training material and shared development selection.\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
