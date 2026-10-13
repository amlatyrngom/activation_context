"""Freeze 128 mixed portable-bank cases for timing, with no semantic labels or audit."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import generate_bank_27b as bank
from select_bank import quotas, read_batch
from run_checker_ab_27b_v1 import BOOLEAN_PROMPT


def main(args):
    assert bank.CHECKER_PROMPT == BOOLEAN_PROMPT
    manifest_bytes = (args.bank / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest['batches'][:5]
    if len(entries) != 5:
        raise ValueError('Need the first five complete balanced releases')
    groups = collections.defaultdict(list)
    for entry in entries:
        path = args.bank / entry['file']
        if bank.fingerprint(path) != entry['sha256']:
            raise ValueError('Incomplete or changed release: ' + entry['file'])
        for row in read_batch(path):
            groups[row['source_family']].append((row, entry))
    wanted = quotas(128, manifest['source_shares'])
    seed = 'checker-prefill-27b-v1-20260921'
    selected = []
    for family, count in wanted.items():
        candidates = sorted(groups[family], key=lambda item: hashlib.sha256((seed + ':' + item[0]['id']).encode()).hexdigest())
        seen_requests = set()
        for row, entry in candidates:
            request = row['provenance']['source_request_id']
            if request in seen_requests:
                continue
            seen_requests.add(request)
            selected.append((len(seen_requests) / count, family, row, entry))
            if len(seen_requests) == count:
                break
        if len(seen_requests) != count:
            raise ValueError('Insufficient distinct source requests: ' + family)
    cases = []
    for _, family, row, entry in sorted(selected, key=lambda item: (item[0], item[1])):
        cases.append({'id': row['id'], 'family': family, 'question': row['question'], 'answer': row['answer'],
            'evidence': row['source']['text'], 'source_units': row['source']['units'],
            'source_request_id': row['provenance']['source_request_id'],
            'source_receipt': {'file': entry['file'], 'sha256': entry['sha256']}})
    assert len(cases) == len({case['id'] for case in cases}) == 128
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'cases_128.jsonl'
    content = ''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases)
    if path.exists() and path.read_text() != content:
        raise ValueError('Cannot replace frozen throughput cases')
    if not path.exists():
        with path.open('x') as handle:
            handle.write(content)
    receipt = {'cases': 128, 'by_family': wanted, 'seed': seed,
        'selection': 'First five balanced releases; per-family seeded question-ID hash; distinct source requests; proportionate interleaving.',
        'bank_manifest_sha256': hashlib.sha256(manifest_bytes).hexdigest(), 'bank_updated_utc': manifest['updated_utc'],
        'input_batches': [{'file': entry['file'], 'sha256': entry['sha256']} for entry in entries],
        'cases_sha256': bank.fingerprint(path), 'semantic_labels': False,
        'purpose': 'Uncached checker throughput and decision agreement only; accepted bank membership is not a correctness label.'}
    bank.write(args.output / 'cases_manifest.json', receipt)
    (args.output / 'PRESERVE').write_text('Frozen 128-case uncached checker throughput probe; diagnostic only.\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
