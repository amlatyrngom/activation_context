"""CPU census of the actual strong8k and gentle4k epoch, without model weights."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

SEED = 20260921
MATERIAL_SHA = '9a9bb6c1a8e96931983eb606a8b793ae5867ba75b07635fbf4fc213319c9fdad'
BUILDER_SHA = 'ee791342b4b21ca22d3be810e34ce1c6c3337cf3921d60bf6b8cfa8cde983054'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def packed_parts(parts):
    """Exact frozen _encode_requests: packing tests source length BEFORE adding rows."""
    result, batch, longest = [], [], 0
    for source, rows in sorted(parts, key=lambda part: part[0]):
        if batch and max(longest, source) * (len(batch) + 1) > 65536:
            result.append(batch); batch = []; longest = 0
        batch.append((source, rows)); longest = max(longest, source)
    if batch:
        result.append(batch)
    return [{'parts': len(batch), 'content_lengths': [p[0] for p in batch], 'rows': [p[1] for p in batch],
             'padded_positions': len(batch) * max(n + rows for n, rows in batch),
             'valid_positions': sum(n + rows for n, rows in batch)} for batch in result]


def epoch_geometry(ids, by_id):
    order = list(range(len(ids))); random.Random(SEED + 1).shuffle(order)
    rng = random.Random(SEED); batches = []
    for start in range(0, len(order), 8):
        logical_ids = [ids[i] for i in order[start:start+8]]
        sorted_ids = sorted(logical_ids, key=lambda ident: by_id[ident]['reader_length'])
        micros = []
        for offset in range(0, len(sorted_ids), 2):
            group = sorted_ids[offset:offset+2]
            selected = [ident for ident in group if rng.random() < .25]
            parts = [part for ident in group for part in by_id[ident]['parts']]
            encoder = packed_parts(parts)
            reader_positions = len(group) * max(by_id[x]['reader_length'] for x in group)
            penalty_positions = len(selected) * max((by_id[x]['penalty_length'] for x in selected), default=0)
            micros.append({'item_ids': group, 'penalty_ids': selected,
                'reader_lengths': [by_id[x]['reader_length'] for x in group],
                'reader_padded_positions': reader_positions,
                'reader_checkpointed_at4096': reader_positions >= 4096,
                'penalty_lengths': [by_id[x]['penalty_length'] for x in selected],
                'penalty_padded_positions': penalty_positions,
                'penalty_checkpointed_at4096': penalty_positions >= 4096,
                'compressed_rows': sum(by_id[x]['compressed_rows'] for x in group),
                'encoder_batches': encoder,
                'encoder_padded_positions': sum(x['padded_positions'] for x in encoder),
                'encoder_max_padded_batch': max(x['padded_positions'] for x in encoder)})
        batches.append({'logical_update': start//8+1, 'questions_seen_before': start,
            'logical_item_ids': logical_ids, 'microbatches': micros})
    return batches


def main(args):
    began = time.monotonic()
    if args.output.exists(): raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if sha(args.material) != MATERIAL_SHA: raise ValueError('Wrong frozen strong8k material')
    import run_qa_strong_long_v1 as bench
    if sha(bench.__file__) != BUILDER_SHA: raise ValueError('Frozen builder changed')
    import run_qa_gentle42k_long_v1 as gentle
    functions = ('harness', 'material_items', 'allocate_passage_rows', 'answer_positions')
    def source_functions(module):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name in functions}
    if source_functions(bench) != source_functions(gentle): raise ValueError('Pool input builders differ')
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as train_module
    import activation.ac_model.ac_model_utils as utils_module
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    source = {name: sha(module.__file__) for name, module in [('ac_model', ac_module), ('trainer', train_module), ('utils', utils_module)]}
    design = json.loads(args.design.read_text())
    if source != design['source_sha256']: raise ValueError('Census requires the frozen source')
    material = json.loads(args.material.read_text())
    strong_ids = json.loads((args.reports/'strong32k_piecewise_8k_8k/material_ids.json').read_text())['train']
    gentle_ids = json.loads((args.reports/'gentle42k_final_lr_familiar4k_8k/material_ids.json').read_text())['train']
    rows = {row['example']['example_id']: row for row in material['train']}
    if len(rows) != 8000 or set(strong_ids) != set(rows) or len(gentle_ids) != 4000 or not set(gentle_ids) <= set(rows):
        raise ValueError('Actual saved pool membership changed')
    # Exact saved training order; per-item passage shuffle is ID-seeded, retrieval is unchanged.
    material['train'] = [rows[ident] for ident in strong_ids]
    material['held'] = material['train'][:1]  # nonempty builder placeholder, never scored
    print('Verified frozen material/source and8000+4000 actual training IDs; building tokenized inputs.', flush=True)
    h = bench.harness()
    study, items, unused = bench.material_items(h, material, passage_parts=True,
        evidence_tokens=8192, max_passages=16, ratio=1/16)
    del unused
    ac = study.ac_model
    trainer = ActivationContextTrainer(h, ActivationContextTrainingConfig(**design['training_config']))
    metadata = {}
    for i, item in enumerate(items):
        ex = trainer.build_example(ac, item)
        if ex.penalty is None: raise ValueError('Every source question must carry a penalty candidate')
        parts = list(zip(item.info['part_source_tokens'], item.info['part_rows']))
        if len(ex.part_requests) != len(parts) or sum(b-a for a,b in ex.spans) != item.info['compressed_rows']:
            raise ValueError('Actual resolved request/span rows differ from builder')
        metadata[item.item_id] = {'reader_length': len(ex.student_ids), 'penalty_length': len(ex.penalty.student_ids),
            'compressed_rows': item.info['compressed_rows'], 'parts': parts,
            'item_sha256': digest(asdict(item)), 'student_ids_sha256': digest(ex.student_ids),
            'penalty_ids_sha256': digest(ex.penalty.student_ids)}
        if i % 1000 == 0: print(f'Prepared actual history lengths: {i}/{len(items)}', flush=True)
    pools = {'strong8k': epoch_geometry(strong_ids, metadata), 'gentle4k': epoch_geometry(gentle_ids, metadata)}
    all_batches = [(pool, row) for pool, rows in pools.items() for row in rows]
    def metric(row, kind):
        if kind == 'reader': return max(m['reader_padded_positions'] if not m['reader_checkpointed_at4096'] else 0 for m in row['microbatches'])
        if kind == 'encoder': return max(m['encoder_padded_positions'] for m in row['microbatches'])
        if kind == 'penalty': return max(m['penalty_padded_positions'] if not m['penalty_checkpointed_at4096'] else 0 for m in row['microbatches'])
        raise ValueError(kind)
    selected = {}
    for reason in ('reader', 'encoder', 'penalty'):
        pool, row = max(all_batches, key=lambda pair: (metric(pair[1], reason),
            max(m['compressed_rows'] for m in pair[1]['microbatches']), pair[0], -pair[1]['logical_update']))
        key = f'{pool}_{row["logical_update"]:04d}'
        if key not in selected: selected[key] = {'pool': pool, **row, 'selection_reasons': [],
            'item_geometry': {ident: metadata[ident] for ident in row['logical_item_ids']},
            'items': [asdict(items[strong_ids.index(ident)]) for ident in row['logical_item_ids']]}
        selected[key]['selection_reasons'].append(reason)
    summary = {}
    for name, batches in pools.items():
        micros = [m for row in batches for m in row['microbatches']]
        summary[name] = {'questions': len(batches)*8, 'logical_batches': len(batches), 'microbatches': len(micros),
            'reader_padded_positions_max': max(m['reader_padded_positions'] for m in micros),
            'reader_checkpointed_at4096': sum(m['reader_checkpointed_at4096'] for m in micros),
            'penalty_padded_positions_max': max(m['penalty_padded_positions'] for m in micros),
            'penalty_checkpointed_at4096': sum(m['penalty_checkpointed_at4096'] for m in micros),
            'encoder_padded_positions_max': max(m['encoder_padded_positions'] for m in micros),
            'encoder_max_padded_batch': max(m['encoder_max_padded_batch'] for m in micros),
            'compressed_rows_per_microbatch_max': max(m['compressed_rows'] for m in micros)}
    identity = {'material_sha256': MATERIAL_SHA, 'builder_sha256': BUILDER_SHA, 'gentle_builder_sha256': sha(gentle.__file__),
        'source_sha256': source, 'strong_design_sha256': sha(args.design),
        'pool_order_sha256': {'strong8k': digest(strong_ids), 'gentle4k': digest(gentle_ids)},
        'selection_rule': 'Union of maximum uncheckpointed reader positions, concurrent encoder padded positions, and actual uncheckpointed penalty positions; ties prefer more compressed rows then pool/name/earlier update.',
        'seed': SEED, 'epoch': 1, 'logical_batch': 8, 'microbatch': 2,
        'encoder_budget': 65536, 'encoder_checkpoint_threshold': 0, 'reader_candidate_threshold': 4096,
        'evidence_tokens': 8192, 'max_passages': 16, 'ratio': 1/16}
    write(args.output/'all_batches.json', {'identity': identity, 'pools': pools, 'item_geometry': metadata})
    write(args.output/'selected_cases.json', {'identity': identity, 'cases': selected})
    result = {'status': 'complete', 'identity': identity, 'pool_summary': summary,
        'selected': {key: {k:v for k,v in row.items() if k != 'items'} for key,row in selected.items()},
        'selected_cases_sha256': sha(args.output/'selected_cases.json'), 'all_batches_sha256': sha(args.output/'all_batches.json'),
        'elapsed_seconds': time.monotonic()-began, 'weights_loaded': False, 'confirmation_read': False,
        'limits': 'CPU geometry ranks memory risk, not actual peak allocation; real GPU candidate memory check still pending.'}
    write(args.output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('selected','identity')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('material', 'design', 'reports', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
