"""Freeze 75 long parents, balanced across five domains and three length bands."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
import generate_long_qa_whole_parent_v1 as worker


def main(args):
    groups = collections.defaultdict(list)
    for packet in worker.read_batch(args.sources):
        groups[(packet['domain'], packet['target_tokens'])].append(packet)
    if len(groups) != 15 or any(len(group) < 5 for group in groups.values()):
        raise ValueError('Expected five domains × three length bands, at least five parents each')
    for group in groups.values():
        group.sort(key=lambda packet: hashlib.sha256(f'{worker.SEED}:{packet["source_id"]}'.encode()).hexdigest())
    selected = [groups[key][index] for index in range(5) for key in sorted(groups)]
    if len({packet['split_group_id'] for packet in selected}) != len(selected):
        raise ValueError('Keep development/confirmation splits possible by distinct parent')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    lengths = [worker.prompt_tokens(tokenizer, worker.generator_chat(packet)) for packet in selected]
    if any(length < packet['source_tokens'] for packet, length in zip(selected, lengths)):
        raise ValueError('Rendered prompt must contain at least the raw parent tokens')
    if max(lengths) + worker.MAX_OUTPUT_TOKENS > worker.MAX_MODEL_TOKENS:
        raise ValueError('Whole-parent prompt exceeds the fixed model budget')
    baseline = json.loads((args.baseline_bank / 'manifest.json').read_text())
    selected_ids = {packet['source_id'] for packet in selected}
    if not selected_ids.issubset(set(baseline['processed_source_ids'])):
        raise ValueError('Selected parents were not all attempted by the original worker')
    old_rows = [row for entry in baseline['batches'] for row in worker.read_batch(args.baseline_bank / entry['file'])]
    paired = [row for row in old_rows if row['provenance']['source_request_id'] in selected_ids]
    label_rows = [row for row in old_rows if worker.TEMPORARY_REFERENCE.search(row['question']) or worker.TEMPORARY_REFERENCE.search(row['answer'])]
    manifest = {
        'schema_version': 1, 'source_file': str(args.sources), 'source_sha256': worker.bank.fingerprint(args.sources),
        'selection': 'Five distinct parents per domain/length band, ordered by seeded content hash, then interleaved',
        'seed': worker.SEED, 'parents': len(selected),
        'by_domain_length': {f'{domain}:{band}': 5 for domain, band in sorted(groups)},
        'generator_model': worker.MODEL, 'tokenizer_directory': str(args.tokenizer),
        'prompt_tokens_min': min(lengths), 'prompt_tokens_max': max(lengths), 'prompt_tokens_sum': sum(lengths),
        'prompt_plus_output_max': max(lengths) + worker.MAX_OUTPUT_TOKENS,
        'max_model_len': worker.MAX_MODEL_TOKENS, 'complete_parents': sum(p['complete_parent'] for p in selected),
        'parent_group_ids': [p['split_group_id'] for p in selected],
        'baseline': {'bank': str(args.baseline_bank), 'attempted_parents': len(baseline['processed_source_ids']),
            'accepted_questions': len(old_rows), 'temporary_or_deictic_reference_questions': len(label_rows),
            'selected_pilot_original_accepted_questions': len(paired),
            'selected_pilot_original_self_contained_by_regex': sum(not worker.TEMPORARY_REFERENCE.search(r['question'] + '\n' + r['answer']) for r in paired),
            'note': 'Original rejected outputs were not retained. Baseline answers are not included in synthesis inputs.'}}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'parents_75.jsonl'
    temporary = args.output / 'parents_75.jsonl.partial'
    if target.exists():
        if list(worker.read_batch(target)) != selected:
            raise ValueError('Cannot replace an existing immutable pilot with different parents')
    else:
        with temporary.open('x') as handle:
            for packet in selected:
                handle.write(json.dumps(packet, ensure_ascii=False) + '\n')
        temporary.rename(target)
    manifest['pilot_sha256'] = worker.bank.fingerprint(target)
    worker.bank.write(args.output / 'manifest.json', manifest)
    (args.output / 'PRESERVE').write_text('Frozen whole-parent QA pilot inputs and validation. No generated questions in this input file.\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'parent_group_ids'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--baseline-bank', type=Path, required=True)
    main(parser.parse_args())
