"""Freeze nested QA subsets, shared evidence retrieval and evaluation for data-versus-epochs runs."""
import argparse
import collections
import hashlib
import json
import random
import statistics
from pathlib import Path

from select_bank import quotas, read_batch, training_row

SEED = 20260923


def request_id(row):
    return row['example']['study_source']['request_id']


def interleave(groups):
    return [row for _, _, row in sorted(
        ((i / len(rows), family, row) for family, rows in groups.items() for i, row in enumerate(rows)),
        key=lambda value: (value[0], value[1]))]


def length_balanced_order(rows, family):
    """Interleave shuffled source-length quartiles and answer-length halves for nested prefixes."""
    rows = sorted(rows, key=lambda row: (row.get('source_tokens') or 0, row['example']['example_id']))
    bins = collections.defaultdict(list)
    for i, row in enumerate(rows):
        bins[min(3, 4 * i // max(1, len(rows)))].append(row)
    strata = {}
    for quartile, values in bins.items():
        values.sort(key=lambda row: (row.get('answer_tokens') or 0, row['example']['example_id']))
        for half in (0, 1):
            selected = values[len(values) * half // 2:len(values) * (half + 1) // 2]
            random.Random(f'{SEED}:{family}:{quartile}:{half}').shuffle(selected)
            if selected:
                strata[f'{quartile}:{half}'] = selected
    return interleave(strata)


def reserve(groups, wanted, excluded, label):
    """Selecting a reporting question reserves its whole synthesis request from training."""
    chosen = {}
    for family, count in wanted.items():
        rows = [row for row in groups[family] if request_id(row) not in excluded]
        random.Random(f'{SEED}:{label}:{family}').shuffle(rows)
        selected = rows[:count]
        if len(selected) != count:
            raise ValueError(f'Insufficient {family} questions for {label}')
        chosen[family] = selected
        excluded.update(request_id(row) for row in selected)
    return interleave(chosen)


def summarize(rows):
    out = {'questions': len(rows), 'source_requests': len({request_id(row) for row in rows}),
           'by_family': dict(collections.Counter(row['source_family'] for row in rows))}
    for key in ('source_tokens', 'answer_tokens'):
        values = sorted(row[key] for row in rows if row.get(key) is not None)
        out[key] = {'known': len(values), 'sum': sum(values), 'mean': statistics.mean(values),
                    'median': statistics.median(values), 'p95': values[int(.95 * (len(values) - 1))]}
    return out


def prepare(bank, output, max_items=8000, reporting_items=200):
    bank, output = Path(bank), Path(output)
    if output.exists():
        raise FileExistsError('Frozen output already exists: ' + str(output))
    raw = (bank / 'manifest.json').read_bytes()
    manifest = json.loads(raw)
    shares = manifest['source_shares']
    groups = collections.defaultdict(list)
    seen = set()
    for entry in manifest['batches']:
        path = bank / entry['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Published batch hash mismatch: ' + str(path))
        for row in read_batch(path):
            item = training_row(row)
            ident = item['example']['example_id']
            if ident not in seen:
                groups[item['source_family']].append(item)
                seen.add(ident)
    reserved = set()
    held = reserve(groups, quotas(reporting_items, shares), reserved, 'development')
    confirmation = reserve(groups, quotas(reporting_items, shares), reserved, 'confirmation')
    ordered = {family: length_balanced_order([row for row in rows if request_id(row) not in reserved], family)
               for family, rows in groups.items()}
    sizes = (max_items // 4, max_items // 2, max_items)
    selections = {}
    for size in sizes:
        wanted = quotas(size, shares)
        if any(len(ordered[family]) < count for family, count in wanted.items()):
            raise ValueError('Wait for more balanced published data before freezing the study')
        selections[size] = interleave({family: ordered[family][:count] for family, count in wanted.items()})
    learning = random.Random(SEED + 7).sample(selections[sizes[0]], min(32, sizes[0]))
    learning_ids = [row['example']['example_id'] for row in learning]
    retrieval = selections[max_items] + held
    recipe = {'seed': SEED, 'bank_manifest_sha256': hashlib.sha256(raw).hexdigest(),
              'input_batches': manifest['batches'], 'source_shares': shares,
              'development': summarize(held), 'confirmation': summarize(confirmation),
              'training': {str(size): summarize(rows) for size, rows in selections.items()},
              'learning_ids': learning_ids, 'retrieval_questions': len(retrieval),
              'selection': 'Nested family/length-balanced subsets; common learning questions and evidence retrieval; synthesis-request holdout'}
    output.mkdir(parents=True)
    (output / 'PRESERVE').write_text('Frozen data-budget comparison material and reserved confirmation questions.\n')
    for size, rows in selections.items():
        data = {'train': rows, 'held': held, 'retrieval': retrieval, 'learning_ids': learning_ids,
                'whole_run_panels': True, 'recipe': recipe, 'planned_epochs': max_items // size}
        (output / f'material_{size}.json').write_text(json.dumps(data, ensure_ascii=False))
    (output / 'confirmation.json').write_text(json.dumps({'held': confirmation, 'recipe': recipe}, ensure_ascii=False))
    (output / 'summary.json').write_text(json.dumps(recipe, indent=2))
    print(json.dumps({k: v for k, v in recipe.items() if k not in ('input_batches', 'learning_ids')}, indent=2))
    return recipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.bank, args.output)
