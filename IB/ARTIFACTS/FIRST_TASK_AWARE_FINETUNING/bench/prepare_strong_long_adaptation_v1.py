"""Freeze 8k questions from the already frozen fresh10k; no new synthesis or held selection."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

SOURCE_SHA = '4b67c74e365dc46093efa2d0e89bb2770b530e572d88e29035b07702b51d2585'
COUNTS = {'code':2400, 'math_arxiv':960, 'math_textbook':640, 'financial':400,
          'web':1600, 'technical_web':400, 'documents':640, 'legal':480, 'office':480}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def choose(material):
    groups = defaultdict(list)
    for row in material['train']:
        groups[row['source_family']].append(row)
    if set(groups) != set(COUNTS):
        raise ValueError('Expected original nine-family fresh10k mix')
    learning = set(material['learning_ids'])
    chosen = set()
    for family, count in COUNTS.items():
        rows = groups[family]
        mandatory = [r for r in rows if r['example']['example_id'] in learning]
        available = sorted((r for r in rows if r['example']['example_id'] not in learning),
            key=lambda r: hashlib.sha256(('strong-long-v1:'+r['example']['example_id']).encode()).hexdigest())
        if len(mandatory) > count or len(rows) < count:
            raise ValueError('Insufficient family capacity: '+family)
        chosen.update(r['example']['example_id'] for r in mandatory+available[:count-len(mandatory)])
    rows = [r for r in material['train'] if r['example']['example_id'] in chosen]
    ids = [r['example']['example_id'] for r in rows]
    held = {r['example']['example_id'] for r in material['held']}
    if len(ids) != 8000 or len(set(ids)) != 8000 or not learning <= set(ids) or set(ids) & held:
        raise ValueError('Training selection is duplicate, incomplete, or overlaps development IDs')
    return rows


def main(args):
    if sha(args.source) != SOURCE_SHA:
        raise ValueError('Input is not the frozen fresh10k material')
    if args.output.exists():
        raise FileExistsError(args.output)
    with args.source.open() as handle:
        material = json.load(handle)
    if len(material['train']) != 10000 or len(material['held']) != 400:
        raise ValueError('Expected fresh10000 plus original400 development')
    preserved = {key:digest(material[key]) for key in ('held','held_cohorts','retrieval','learning_ids')}
    material['train'] = choose(material)
    material['planned_epochs'] = 1
    material['whole_run_panels'] = True
    material['recipe'] = {**material['recipe'], 'adaptation': {
        'parent_material_sha256':SOURCE_SHA, 'selection':'Exact80% family quotas; retain32 learning IDs; immutable-ID hash selects remainder; preserve source order.',
        'scope':'Fresh8k questions from the frozen continuation10k, shared original400 development/retrieval. No confirmation read.',
        'initialization':'Preselected final32k-question task-aware paired weights; fresh Adam and adaptation schedule.'}}
    if any(digest(material[key]) != value for key,value in preserved.items()):
        raise AssertionError('Development, retrieval, or learning selection changed')
    args.output.mkdir(parents=True)
    path = args.output/'material.json'
    with path.open('x') as handle:
        json.dump(material, handle, ensure_ascii=False)
    summary = {'status':'complete', 'source_sha256':SOURCE_SHA, 'material_sha256':sha(path),
        'train_questions':8000, 'held_questions':400, 'learning_questions':len(material['learning_ids']),
        'retrieval_questions':len(material['retrieval']), 'training_by_family':COUNTS,
        'training_ids_sha256':digest([r['example']['example_id'] for r in material['train']]),
        'preserved':preserved, 'reserved_confirmation_read':False}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (args.output/'PRESERVE').write_text('Frozen fresh8k stronger-initialization long-context material.\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
