"""Freeze a mixed selection from published plain QA records, independent of generation caches."""
import argparse, collections, hashlib, json, random
from pathlib import Path

SHARES = {'code': .30, 'numeric': .25, 'web': .25, 'documents': .20}  # Legacy banks.

def quotas(items, shares):
    if items < 0 or not shares or any(v <= 0 for v in shares.values()) or abs(sum(shares.values()) - 1) > 1e-8:
        raise ValueError('Expected a nonnegative count and positive shares summing to one')
    result = {family: int(items * share) for family, share in shares.items()}
    order = sorted(shares, key=lambda family: -(items * shares[family] - result[family]))
    for family in order[:items - sum(result.values())]:
        result[family] += 1
    return result

def read_batch(path):
    """Read portable JSONL, original {'rows': ...} batches, or plain JSON arrays."""
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text())
        yield from data['rows'] if isinstance(data, dict) else data

def family_of(row):
    return row.get('source_family', row['domain'])

def training_row(row):
    if 'example' in row:
        return row
    source, units = row['source'], row['source']['units']
    example = {'example_id': row['id'], 'query': row['question'], 'gold_answers': [row['answer']],
               'positive_doc_ids': row.get('supporting_unit_ids', [u['id'] for u in units]),
               'study_source': {'text': source['text'],
                   'units': [{'doc_id': u['id'], 'chunk_ids': [u['id'] + ':0'], 'text': u['text']} for u in units],
                   'mode': source.get('mode', 'document'),
                   'request_id': row.get('provenance', {}).get('source_request_id', row['id'])}}
    return {'domain': row['domain'], 'source_family': family_of(row), 'example': example,
            'original_doc_ids': [u.get('document_id', u['id']) for u in units],
            'source_tokens': row.get('source_tokens'), 'answer_tokens': row.get('answer_tokens')}

def select(bank, items, seed):
    bank = Path(bank)
    manifest_bytes = (bank / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    shares = manifest.get('source_shares') or manifest.get('recipe', {}).get('source_shares') or SHARES
    wanted = quotas(items, shares)
    groups = collections.defaultdict(list)
    inputs, seen = [], set()
    for entry in manifest['batches']:
        data = (bank / entry['file']).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry['sha256']:
            raise ValueError('Incomplete or changed batch: ' + entry['file'])
        inputs.append({'file': entry['file'], 'sha256': digest})
        for row in read_batch(bank / entry['file']):
            family = family_of(row)
            normalized = training_row(row)
            ident = normalized['example']['example_id']
            if ident in seen:
                continue
            seen.add(ident)
            groups[family].append(normalized)
    missing = {family: {'needed': n, 'available': len(groups[family])}
               for family, n in wanted.items() if len(groups[family]) < n}
    if missing:
        raise ValueError('Insufficient published questions for the bank mix: ' + json.dumps(missing))
    rng = random.Random(seed)
    selected = []
    for family, count in wanted.items():
        rng.shuffle(groups[family])
        selected.extend((i / max(1, count), family, row) for i, row in enumerate(groups[family][:count]))
    rows = [row for _, _, row in sorted(selected, key=lambda value: (value[0], value[1]))]
    ids = [row['example']['example_id'] for row in rows]
    selection_hash = hashlib.sha256(json.dumps(ids, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    return rows, {'bank_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'batch_files': [e['file'] for e in inputs],
                  'input_batches': inputs, 'selection_sha256': selection_hash, 'seed': seed, 'requested_items': items,
                  'source_shares': shares, 'selected_by_family': wanted,
                  'selected_by_domain': dict(collections.Counter(row['domain'] for row in rows)),
                  'balanced_available': min(int(len(groups[d]) / share) for d, share in shares.items()),
                  'bank_recipes': manifest.get('recipes', {'legacy': manifest.get('recipe')})}

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bank', type=Path, required=True)
    p.add_argument('--items', type=int, required=True)
    p.add_argument('--seed', type=int, default=20260922)
    p.add_argument('--held-material', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows, recipe = select(a.bank, a.items, a.seed)
    held = json.loads(a.held_material.read_text())['held']
    held_questions = {r['example']['query'].strip().casefold() for r in held}
    if any(r['example']['query'].strip().casefold() in held_questions for r in rows):
        raise ValueError('Selection contains a reserved held question')
    material = {'train': rows, 'held': held, 'recipe': recipe}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as handle:
        json.dump(material, handle, ensure_ascii=False)
    print(json.dumps(recipe, indent=2))
