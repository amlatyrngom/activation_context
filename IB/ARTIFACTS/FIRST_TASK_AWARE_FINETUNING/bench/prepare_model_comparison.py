"""Freeze matched 9B/27B-generated training sets and two shared evaluation cohorts."""
import argparse
import collections
import errno
import hashlib
import json
import random
import time
from pathlib import Path

from prepare_data_budget import interleave, length_balanced_order, request_id, reserve, summarize
from select_bank import quotas, read_batch, training_row

SEED = 20260925


def question(row):
    return row['example']['query'].strip().casefold()


def load_bank(path, protected, protected_requests=None):
    path = Path(path)
    # The live manifest is atomically replaced; another NFS client can briefly retain its old handle.
    for attempt in range(5):
        try:
            data = (path / 'manifest.json').read_bytes()
            break
        except OSError as error:
            if error.errno != errno.ESTALE or attempt == 4:
                raise
            time.sleep(.2 * 2**attempt)
    manifest = json.loads(data)
    groups, seen = collections.defaultdict(list), set(protected)
    invalid_controls = collections.Counter()
    for entry in manifest['batches']:
        batch = path / entry['file']
        if hashlib.sha256(batch.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Changed or incomplete published batch: ' + str(batch))
        for raw in read_batch(batch):
            row = training_row(raw)
            # Record the request before filtering its protected question away.
            if protected_requests is not None and question(row) in protected:
                protected_requests.add(request_id(row))
            # JSON-valid but visibly corrupted math (e.g. unescaped LaTeX \beta or \frac).
            # Keep the portable originals intact; never repair a gold answer heuristically.
            fields = [row['example']['query'], *row['example']['gold_answers']]
            if any(ord(char) < 32 and char not in '\n\t' for field in fields for char in field):
                invalid_controls[row['source_family']] += 1
                continue
            key = question(row)
            if key not in seen:
                groups[row['source_family']].append(row)
                seen.add(key)
    return groups, {'path': str(path), 'manifest_sha256': hashlib.sha256(data).hexdigest(),
                    'source_shares': manifest['source_shares'], 'input_batches': manifest['batches'],
                    'generator_recipes': manifest.get('recipes', {}),
                    'excluded_corrupted_qa_by_family': dict(invalid_controls),
                    'format_rule': 'Before every split, exclude questions/answers containing ASCII controls except newline/tab; source text and portable bank records remain unchanged.'}


def prepare(bank_9b, bank_27b, output, items=16000, development_per_cohort=200,
            confirmation_per_cohort=200, reserved_material=()):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Frozen output already exists: ' + str(output))
    if items < 1 or development_per_cohort < 1 or confirmation_per_cohort < 0:
        raise ValueError('Positive train/development counts and nonnegative confirmation count required')
    protected = set()
    for path in reserved_material:
        data = json.loads(Path(path).read_text())
        if 'held' in data:
            protected.update(question(row) for row in data['held'])
        else:
            protected.update(q.strip().casefold() for q in data['questions'])
    banks = {name: load_bank(path, protected) for name, path in [('9b', bank_9b), ('27b', bank_27b)]}
    shares = banks['9b'][1]['source_shares']
    if shares != banks['27b'][1]['source_shares']:
        raise ValueError('Both banks must use the same source-family mix')
    reserved_requests, reserved_text = set(), set(protected)
    cohorts = {'development': {}, 'confirmation': {}}
    for split, count in [('development', development_per_cohort), ('confirmation', confirmation_per_cohort)]:
        for name, (groups, _) in banks.items():
            available = {family: [row for row in rows if question(row) not in reserved_text]
                         for family, rows in groups.items()}
            selected = reserve(available, quotas(count, shares), reserved_requests, f'{SEED}:{name}:{split}') if count else []
            cohorts[split][name] = selected
            reserved_text.update(question(row) for row in selected)
    training = {}
    for name, (groups, _) in banks.items():
        ordered = {family: length_balanced_order([row for row in rows
                   if request_id(row) not in reserved_requests and question(row) not in reserved_text], f'{name}:{family}')
                   for family, rows in groups.items()}
        wanted = quotas(items, shares)
        missing = {family: {'needed': count, 'available': len(ordered.get(family, []))}
                   for family, count in wanted.items() if len(ordered.get(family, [])) < count}
        if missing:
            raise ValueError(f'Wait for more {name} balanced releases: ' + json.dumps(missing))
        training[name] = interleave({family: ordered[family][:count] for family, count in wanted.items()})
    held = interleave(cohorts['development'])
    confirmation = interleave(cohorts['confirmation']) if confirmation_per_cohort else []
    # Both arms see exactly the same raw evidence pool, including the same distractors.
    retrieval, seen_ids = [], set()
    for row in training['9b'] + training['27b'] + held:
        ident = row['example']['example_id']
        if ident not in seen_ids:
            retrieval.append(row)
            seen_ids.add(ident)
    cohort_ids = {split: {name: [row['example']['example_id'] for row in rows] for name, rows in groups.items()}
                  for split, groups in cohorts.items()}
    ids = {'train': {name: [row['example']['example_id'] for row in rows] for name, rows in training.items()}, **cohort_ids}
    digest = hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    recipe = {'seed': SEED, 'selection_sha256': digest, 'source_shares': shares,
              'external_reserved_questions': len(protected),
              'banks': {name: meta for name, (_, meta) in banks.items()},
              'training': {name: summarize(rows) for name, rows in training.items()},
              'development': {name: summarize(rows) for name, rows in cohorts['development'].items()},
              'confirmation': {name: summarize(rows) for name, rows in cohorts['confirmation'].items()} if confirmation else {},
              'cohort_ids': cohort_ids, 'retrieval_questions': len(retrieval), 'planned_epochs': 1,
              'learning_selection': 'Up to32 fixed questions from the first half of each domain-interleaved training pool; shared with a possible half-pool/two-pass control.',
              'selection': 'Matched source quotas with source/answer-length strata; whole synthesis requests reserved; exact held questions excluded from both training arms.',
              'comparison': 'Compare training arms within each fixed generator cohort. Across-cohort literal NLL also reflects different questions and gold wording; decoded exactness is not semantic correctness.'}
    output.mkdir(parents=True)
    (output / 'PRESERVE').write_text('Frozen 9B/27B comparison material and reserved evaluation questions.\n')
    for name, rows in training.items():
        # Match the runner's ordering before its item limit, so a half-pool control shares this panel.
        domain_rows = collections.defaultdict(list)
        for row in rows:
            domain_rows[row['domain']].append(row)
        learning_pool = interleave(domain_rows)[:max(1, len(rows)//2)]
        learning = random.Random(f'{SEED}:{name}:learning').sample(learning_pool, min(32, len(learning_pool)))
        material = {'train': rows, 'held': held, 'retrieval': retrieval, 'whole_run_panels': True,
                    'held_cohorts': cohort_ids['development'], 'planned_epochs': 1,
                    'learning_ids': [row['example']['example_id'] for row in learning], 'recipe': recipe}
        (output / f'material_{name}_{items}.json').write_text(json.dumps(material, ensure_ascii=False))
    (output / 'confirmation.json').write_text(json.dumps({'held': confirmation,
        'held_cohorts': cohort_ids['confirmation'], 'recipe': recipe}, ensure_ascii=False))
    (output / 'reserved_questions.json').write_text(json.dumps({'questions': sorted(reserved_text)}, ensure_ascii=False))
    (output / 'summary.json').write_text(json.dumps(recipe, indent=2))
    print(json.dumps({key: value for key, value in recipe.items() if key not in ('banks', 'cohort_ids')}, indent=2))
    return recipe


def extend(bank_9b, bank_27b, output, original, prior_material, items=32000,
           confirmation_per_cohort=200, reserved_material=()):
    """Add questions while retaining the original reporting, learning and retrieval inputs."""
    output, original = Path(output), Path(original)
    if output.exists():
        raise FileExistsError('Frozen output already exists: '+str(output))
    if not prior_material:
        raise ValueError('Extension requires the other original training material for fair confirmation')
    base_bytes = original.read_bytes()
    base = json.loads(base_bytes)
    if items <= len(base['train']) or confirmation_per_cohort < 1:
        raise ValueError('Extension needs more training questions and fresh confirmation')
    old_materials = [base, *[json.loads(Path(p).read_bytes()) for p in prior_material]]
    old_train = [row for material in old_materials for row in material['train']]
    old_train_questions = {question(row) for row in old_train}
    old_train_requests = {request_id(row) for row in old_train}
    protected, protected_requests = set(), set()
    for material in old_materials:
        protected.update(question(row) for row in material['held'])
        protected_requests.update(request_id(row) for row in material['held'])
    for path in reserved_material:
        data = json.loads(Path(path).read_bytes())
        if 'held' in data:
            protected.update(question(row) for row in data['held'])
            protected_requests.update(request_id(row) for row in data['held'])
        else:
            protected.update(q.strip().casefold() for q in data['questions'])
    banks = {name: load_bank(path, protected, protected_requests)
             for name, path in [('9b', bank_9b), ('27b', bank_27b)]}
    shares = banks['27b'][1]['source_shares']
    if shares != banks['9b'][1]['source_shares'] or shares != base['recipe']['source_shares']:
        raise ValueError('Source-family mix changed')
    base_questions = {question(row) for row in base['train']}
    base_requests = {request_id(row) for row in base['train']}
    if len(base_questions) != len(base['train']) or base_questions & protected or base_requests & protected_requests:
        raise ValueError('Original training pool conflicts with reservations or has repeated questions')

    # Old training requests are excluded only from fresh confirmation, not from
    # additional training. reserve() mutates its exclusion set.
    confirmation, reserved_text = {}, protected | old_train_questions
    confirmation_excluded = protected_requests | old_train_requests
    for name, (groups, _) in banks.items():
        eligible = {family: [row for row in rows if question(row) not in reserved_text]
                    for family, rows in groups.items()}
        selected = reserve(eligible, quotas(confirmation_per_cohort, shares),
                           confirmation_excluded, f'{SEED}:extension:{name}:confirmation')
        confirmation[name] = selected
        reserved_text.update(question(row) for row in selected)
    fresh_requests = {request_id(row) for rows in confirmation.values() for row in rows}
    fresh_questions = {question(row) for rows in confirmation.values() for row in rows}
    training_excluded = protected_requests | fresh_requests
    excluded_questions = protected | fresh_questions | base_questions
    wanted = quotas(items-len(base['train']), shares)
    groups = banks['27b'][0]
    ordered = {family: length_balanced_order([row for row in rows
               if request_id(row) not in training_excluded and question(row) not in excluded_questions],
               f'27b:extension:{family}') for family, rows in groups.items()}
    missing = {family: {'needed': count, 'available': len(ordered.get(family, []))}
               for family, count in wanted.items() if len(ordered.get(family, [])) < count}
    if missing:
        raise ValueError('Wait for more eligible 27B releases: '+json.dumps(missing))
    old_by_family = collections.defaultdict(list)
    for row in base['train']:
        old_by_family[row['source_family']].append(row)
    additions = {family: ordered[family][:count] for family, count in wanted.items()}
    train = interleave({family: interleave({'old': old_by_family[family], 'new': additions[family]})
                        for family in wanted})
    assert len(train) == items and len({question(row) for row in train}) == items
    assert len({row['example']['example_id'] for row in train}) == items
    assert dict(collections.Counter(row['source_family'] for row in train)) == quotas(items, shares)
    by_id = {row['example']['example_id']: row for row in train}
    assert all(by_id[row['example']['example_id']] == row for row in base['train'])
    assert set(base['learning_ids']) <= set(by_id)
    assert not {request_id(row) for row in train} & training_excluded
    assert not fresh_requests & old_train_requests
    assert not fresh_questions & old_train_questions
    cohort_ids = {name: [row['example']['example_id'] for row in rows]
                  for name, rows in confirmation.items()}
    digest = hashlib.sha256(json.dumps({'train': list(by_id), 'confirmation': cohort_ids},
                                      sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    recipe = {'seed': SEED, 'selection_sha256': digest, 'source_shares': shares,
              'original_material': str(original), 'original_material_sha256': hashlib.sha256(base_bytes).hexdigest(),
              'original_training_questions': len(base['train']), 'added_training_questions': items-len(base['train']),
              'banks': {name: meta for name, (_, meta) in banks.items()},
              'training': summarize(train), 'confirmation': {name: summarize(rows) for name, rows in confirmation.items()},
              'preserved_fields': ['held', 'held_cohorts', 'learning_ids', 'retrieval'],
              'prior_training_requests_excluded_from_confirmation': len(old_train_requests),
              'reserved_request_count': len(training_excluded), 'planned_epochs': 1,
              'selection': 'Original training rows retained unchanged; balanced eligible additions. Whole synthesis requests reserved. Fixed development, learning and retrieval inputs.'}
    material = {**base, 'train': train, 'recipe': recipe, 'whole_run_panels': True, 'planned_epochs': 1}
    for field in recipe['preserved_fields']:
        assert material[field] == base[field]
    # Publish the directory only after all material and provenance files exist.
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'.', dir=output.parent) as temporary:
        staging = Path(temporary)/'ready'; staging.mkdir()
        (staging/'PRESERVE').write_text('Frozen 32k extension, fixed reporting inputs and fresh reserved confirmation.\n')
        (staging/f'material_27b_{items}.json').write_text(json.dumps(material, ensure_ascii=False))
        (staging/'confirmation.json').write_text(json.dumps({'held': interleave(confirmation),
            'held_cohorts': cohort_ids, 'recipe': recipe}, ensure_ascii=False))
        (staging/'reserved_questions.json').write_text(json.dumps({'questions': sorted(protected | fresh_questions)}, ensure_ascii=False))
        (staging/'summary.json').write_text(json.dumps(recipe, indent=2))
        staging.rename(output)
    print(json.dumps({key: value for key, value in recipe.items() if key != 'banks'}, indent=2))
    return recipe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank-9b', type=Path, required=True)
    parser.add_argument('--bank-27b', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--items', type=int, default=16000)
    parser.add_argument('--development-per-cohort', type=int, default=200)
    parser.add_argument('--confirmation-per-cohort', type=int, default=200)
    parser.add_argument('--reserved-material', type=Path, action='append', default=[])
    parser.add_argument('--extend-from', type=Path)
    parser.add_argument('--prior-material', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.extend_from:
        extend(args.bank_9b, args.bank_27b, args.output, args.extend_from, args.prior_material,
               args.items, args.confirmation_per_cohort, args.reserved_material)
    else:
        prepare(args.bank_9b, args.bank_27b, args.output, args.items, args.development_per_cohort,
                args.confirmation_per_cohort, args.reserved_material)
