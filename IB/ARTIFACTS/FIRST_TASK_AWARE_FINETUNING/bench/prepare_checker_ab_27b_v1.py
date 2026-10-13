"""Freeze a small checker comparison from already reviewed exact source/question pairs."""
import argparse
import collections
import json
from pathlib import Path

from generate_bank import fingerprint, write
from select_bank import read_batch


def main(args):
    findings = json.loads(args.findings.read_text())
    rows = {row['id']: row for row in read_batch(args.sample)}
    cases = []
    for finding in findings['findings']:
        row = rows[finding['id']]
        verdict = finding['verdict']
        expected = True if verdict == 'supported' else False if verdict.startswith('definite_') else None
        cases.append({'id': row['id'], 'family': row['source_family'], 'origin': 'fixed_27b_sample',
            'question': row['question'], 'answer': row['answer'], 'evidence': row['source']['text'],
            'source_units': row['source']['units'], 'expected_supported': expected,
            'manual_classification': verdict, 'manual_reason': finding['evidence_and_reason'],
            'source_receipt': {key: finding[key] for key in ('file', 'file_sha256', 'line', 'source_request_id')}})
    old_findings = json.loads(args.old_findings.read_text())
    old_rows = {row['item_id']: row for row in json.loads(args.old_rows.read_text())}
    old_ids = [entry['item_id'] for entry in old_findings['confirmed_defects']]
    old_ids.append(old_findings['additional_reference_issue']['item_id'])
    old_notes = {row['item_id']: row['analysis'] for row in old_findings['rows']}
    for ident in old_ids:
        row, note = old_rows[ident], old_notes[ident]
        missing_definition = ident == old_findings['additional_reference_issue']['item_id']
        cases.append({'id': ident, 'family': row['domain'], 'origin': 'earlier_12_example_report',
            'question': row['question'], 'answer': row['gold'], 'evidence': row['required_source']['text'],
            'source_units': row['required_source']['units'], 'expected_supported': False,
            'manual_classification': 'insufficient_supplied_definition' if missing_definition else 'definite_incorrect',
            'manual_reason': ('The supplied alias does not define the template parameter meanings, so the capacity '
                'answer is unsupported from this evidence alone. No external definition is given to either checker.'
                if missing_definition else note['finding']),
            'source_receipt': {'file': str(args.old_rows), 'file_sha256': fingerprint(args.old_rows),
                               'original_doc_ids': row['original_doc_ids']}})
    if len(cases) != 23 or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Expected 23 distinct reviewed cases')
    # Preserve the predeclared mixed sample order; old regression cases come last.
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'cases_23.jsonl'
    content = ''.join(json.dumps(case, ensure_ascii=False) + '\n' for case in cases)
    if path.exists() and path.read_text() != content:
        raise ValueError('Cannot replace a frozen case set')
    if not path.exists():
        with path.open('x') as handle:
            handle.write(content)
    manifest = {'cases': len(cases), 'expected': dict(collections.Counter(
        'supported' if case['expected_supported'] is True else 'unsupported' if case['expected_supported'] is False else 'unscored_ambiguous'
        for case in cases)), 'case_file_sha256': fingerprint(path),
        'inputs': {str(path): fingerprint(path) for path in (args.findings, args.sample, args.old_findings, args.old_rows)},
        'rules': 'Exact supplied source, question and gold unchanged. Manual expectations/reasons never go into model chats. '
                 'Three qualification/source-quality cases unscored. Missing-definition case requires rejection for insufficient evidence, '
                 'without giving the checker any external Breakpad definition.'}
    write(args.output / 'cases_manifest.json', manifest)
    (args.output / 'PRESERVE').write_text('Frozen 23-case oracle checker comparison; diagnostic data, not training material.\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--findings', type=Path, required=True)
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--old-findings', type=Path, required=True)
    parser.add_argument('--old-rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
