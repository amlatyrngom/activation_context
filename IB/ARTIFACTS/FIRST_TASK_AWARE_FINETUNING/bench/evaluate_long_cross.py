"""Frozen paired-checkpoint cross-evaluation: control900/final2k/final32k × 2k/32k evidence."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

MATERIAL_SHA = 'f937470fde3f231aac98e7fe4f209488108699b329ac00f222a919e3a0a4bc4e'
BENCH_SHA = '25663ab1991a8bc8da3e809674e17d6615267e7409c02f0a04149f5a14f7c4c4'
CONTEXTS = {'2k': (2048, 4), '32k': (32768, 64)}
CHECKPOINTS = ('control900', 'long_adapt_2k', 'long_adapt_32k')
CELLS = tuple(f'{checkpoint}__{context}' for checkpoint in CHECKPOINTS for context in CONTEXTS)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def paired_checkpoint(name, reports, initial_pair, material_sha=MATERIAL_SHA):
    """Require a published final pair, never silently substitute a selected or partial checkpoint."""
    initial_bytes = Path(initial_pair).read_bytes()
    initial = json.loads(initial_bytes)
    if initial.get('update') != 900:
        raise ValueError('Expected the frozen control900 initialization receipt')
    if name == 'control900':
        pair = dict(initial)
        provenance = {'initial_pair_sha256': hashlib.sha256(initial_bytes).hexdigest()}
    else:
        folder = Path(reports)/name
        result_bytes = (folder/'result.json').read_bytes()
        result = json.loads(result_bytes)
        identity = json.loads((folder/'run_identity.json').read_text())['identity']
        config = json.loads((folder/'training_config.json').read_text())
        budget, passages = CONTEXTS['2k' if name == 'long_adapt_2k' else '32k']
        if result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 80:
            raise ValueError(f'{name}: final80-update run is not complete')
        expected = {'material_sha256': material_sha, 'initial_pair_sha256': hashlib.sha256(initial_bytes).hexdigest(),
                    'planned_updates': 80, 'evidence_tokens': budget, 'max_passages': passages,
                    'ratio': 1/16, 'passage_parts': False}
        if any(identity.get(key) != value for key, value in expected.items()) or config.get('micro_batch_examples') != 1:
            raise ValueError(f'{name}: completed run does not match the frozen adaptation design')
        pair = {'ac_checkpoint': result['checkpoint'], 'reader_checkpoint': result['target_lora_checkpoint'], 'update': 80}
        ac = Path(pair['ac_checkpoint']).resolve()
        marker = ac.parent/'completed_epoch.json'
        marker_bytes = marker.read_bytes(); published = json.loads(marker_bytes)
        run_root = ac.parents[2]
        if (run_root/published['ac_checkpoint']).resolve() != ac or (run_root/published['target_lora_checkpoint']).resolve() != Path(pair['reader_checkpoint']).resolve():
            raise ValueError(f'{name}: final result and published checkpoint pair disagree')
        if published.get('epoch_number') != 1:
            raise ValueError(f'{name}: expected the completed first adaptation epoch')
        provenance = {'result_sha256': hashlib.sha256(result_bytes).hexdigest(),
                      'completed_marker_sha256': hashlib.sha256(marker_bytes).hexdigest()}
    ac, reader = (Path(pair[key]).resolve() for key in ('ac_checkpoint', 'reader_checkpoint'))
    required = [ac/'ac_modules.pt', ac/'side_lora/adapter_config.json', reader/'adapter_config.json']
    for directory in (ac/'side_lora', reader):
        weights = [path for path in (directory/'adapter_model.safetensors', directory/'adapter_model.bin') if path.is_file()]
        if not weights:
            raise FileNotFoundError('Missing adapter weights: '+str(directory))
        required.extend(weights)
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError('Incomplete or empty paired checkpoint')
    provenance['files'] = {str(path): {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in required}
    return pair, provenance


def answer_scores(summary, items, originals):
    if len(summary['per_item']) != len(items) or len(items) != len(originals):
        raise ValueError('Evaluation changed item count')
    rows = []
    for measured, item, original in zip(summary['per_item'], items, originals):
        if measured['item_id'] != item.item_id or item.assistant_token_ids != original.assistant_token_ids:
            raise ValueError('Reference IDs/order or assistant targets changed')
        positions = original.info['answer_positions']
        rows.append({'item_id': original.item_id, 'view_item_id': item.item_id,
                     'answer_nll': -sum(measured['target_logp'][position] for position in positions)/len(positions),
                     'answer_tokens': len(positions), 'student_tokens': measured['student_tokens']})
    return {'answer_nll': statistics.mean(row['answer_nll'] for row in rows), 'per_item': rows}


def paired_delta(measured, reference):
    indexed = {row['item_id']: row['answer_nll'] for row in reference['per_item']}
    if set(indexed) != {row['item_id'] for row in measured['per_item']}:
        raise ValueError('Delta compares different questions')
    rows = [{'item_id': row['item_id'], 'delta_nll': row['answer_nll']-indexed[row['item_id']]} for row in measured['per_item']]
    return {'mean_delta_nll': statistics.mean(row['delta_nll'] for row in rows),
            'questions_improved': sum(row['delta_nll'] < 0 for row in rows), 'per_item': rows}


def summarize(output):
    """Publish only measured cells; negative deltas mean lower NLL on the same input condition."""
    cells = {}
    for cell in CELLS:
        path = Path(output)/cell/'result.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('status') == 'complete':
                cells[cell] = record
    deltas = {}
    for context in CONTEXTS:
        same = [record for cell, record in cells.items() if cell.endswith('__'+context)]
        if len({record['input_fingerprint'] for record in same}) > 1:
            raise ValueError('Cells do not share identical inputs at context '+context)
        baseline = cells.get('control900__'+context)
        for checkpoint in CHECKPOINTS[1:]:
            measured = cells.get(checkpoint+'__'+context)
            if measured and baseline:
                deltas[checkpoint+'__'+context+'_vs_initial'] = {view: paired_delta(measured['views'][view], baseline['views'][view]) for view in ('AC', 'full context', 'no context')}
        short, long = cells.get('long_adapt_2k__'+context), cells.get('long_adapt_32k__'+context)
        if short and long:
            deltas['long_vs_short_adaptation__'+context] = {view: paired_delta(long['views'][view], short['views'][view]) for view in ('AC', 'full context', 'no context')}
    summary = {'status': 'complete' if len(cells) == len(CELLS) else 'partial', 'completed_cells': list(cells),
               'expected_cells': list(CELLS), 'scores': {cell: {view: values['answer_nll'] for view, values in record['views'].items()} for cell, record in cells.items()},
               'deltas': deltas, 'interpretation': 'Negative deltas improve literal gold-answer NLL on the same inputs.2k measures short-context retention;32k measures long-context robustness. Current reader matched across views within each checkpoint. No fresh-data or semantic-correctness claim.'}
    write(Path(output)/'summary.json', summary)
    return summary


def run(args):
    checkpoint_name, context = args.cell.split('__')
    if hashlib.sha256(args.material.read_bytes()).hexdigest() != MATERIAL_SHA or hashlib.sha256(args.bench.read_bytes()).hexdigest() != BENCH_SHA:
        raise ValueError('Frozen material or builder snapshot changed')
    material = json.loads(args.material.read_text())
    if hashlib.sha256(args.initial_pair.read_bytes()).hexdigest() != material['recipe']['initial_pair_sha256']:
        raise ValueError('Initialization receipt differs from the frozen material recipe')
    if (len(material['train']), len(material['held']), len(material['retrieval'])) != (640, 100, 4100):
        raise ValueError('Wrong frozen data shape')
    pair, provenance = paired_checkpoint(checkpoint_name, args.reports, args.initial_pair, MATERIAL_SHA)
    identity = {'cell': args.cell, 'material_sha256': MATERIAL_SHA, 'bench_sha256': BENCH_SHA,
                'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checkpoint': pair,
                'checkpoint_provenance': provenance, 'microbatch': 1, 'decode_samples': args.decode_samples}
    folder = args.output/args.cell
    final = folder/'result.json'
    if final.exists():
        previous = json.loads(final.read_text())
        if previous.get('status') != 'complete' or previous['identity'] != identity:
            raise ValueError('Existing completed output has a different identity; choose another output directory')
        print('CROSS_EVAL_ALREADY_COMPLETE', args.cell, flush=True)
        summarize(args.output)
        return
    partial = folder/'partial.json'
    record = json.loads(partial.read_text()) if partial.exists() else {'identity': identity, 'status': 'running', 'views': {}, 'samples': {}, 'attempts': []}
    if record['identity'] != identity:
        raise ValueError('Partial evaluation identity differs; refusing to mix results')
    spec = importlib.util.spec_from_file_location('long_cross_bench', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress, _TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    from activation.common.reporting import HtmlReporter
    from activation.dataset.scoring import exact_match, token_f1
    reporter = HtmlReporter(str(folder), 'Long-context cross-evaluation — '+args.cell,
        'Frozen paired weights;100 fixed questions. No training. NLL is teacher-forced answer likelihood, not answer correctness.')
    def phase(name, **values):
        reporter.status.clear(); reporter.set_status(phase=name, cell=args.cell, **values); reporter.render(force=True)
    started = time.monotonic(); phase('loading frozen paired checkpoint')
    harness = bench.harness(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = harness.module_manager.get_ac_model(bench.AC); ac.prepare(); harness.module_manager.ensure_lora(bench.READER)
    groups = collections.defaultdict(list)
    for row in material['held']:
        groups[row['domain']].append(row)
    held_rows = [row for _, _, row in sorted((i/len(rows), domain, row) for domain, rows in groups.items() for i, row in enumerate(rows))]
    phase('building fixed '+context+' evidence')
    budget, passages = CONTEXTS[context]
    study, _, held = bench.material_items(harness, {'train': material['train'][:1], 'held': held_rows, 'retrieval': material['retrieval']},
        evidence_tokens=budget, max_passages=passages, ratio=1/16)
    input_fingerprint = digest([(item.item_id, item.ac_messages, item.in_context_messages, item.assistant_token_ids) for item in held])
    if record.get('input_fingerprint', input_fingerprint) != input_fingerprint:
        raise ValueError('Rebuilt input differs from the partial evaluation')
    record['input_fingerprint'] = input_fingerprint
    record['input_rows'] = [{'item_id': item.item_id, 'question': item.info['question'], 'gold': item.info['answer'],
        **{key: item.info[key] for key in ('encoded_source_tokens', 'compressed_rows', 'evidence_source_tokens', 'passages')}} for item in held]
    record['row_budget'] = {key: {'total': sum(item.info[key] for item in held), 'min': min(item.info[key] for item in held),
        'max': max(item.info[key] for item in held), 'mean': statistics.mean(item.info[key] for item in held)}
        for key in ('encoded_source_tokens', 'compressed_rows', 'evidence_source_tokens')}
    record['runtime_ac_version'] = ac.version
    trainer = ActivationContextTrainer(harness, ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0,
        micro_batch_examples=1, max_example_tokens=72000, completion_samples=0, completion_max_new_tokens=160, seed=bench.SEED))
    variables = trainer._autotuning_variables(ac)
    trainer._training_deadline = time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    views = {'AC': held, 'full context': study.transform_for_eval_reference(held, 'full_context'),
             'no context': study.transform_for_eval_reference(held, 'no_context')}
    progress = ActivationContextTrainingProgress(epoch=0, epochs=0, epoch_number=0, step=0, steps=0,
        global_step=pair['update'], epoch_elapsed_s=0., elapsed_s=0.)
    with configure_fla_runtime(variables['fla_config']), torch.no_grad():
        for view, items in views.items():
            if view in record['views']:
                continue
            phase('scoring '+view, questions=100)
            torch.cuda.synchronize(ac.device); torch.cuda.reset_peak_memory_stats(ac.device); began = time.monotonic()
            scored = answer_scores(trainer._evaluate(ac, items, ac.device), items, held)
            torch.cuda.synchronize(ac.device)
            record['views'][view] = {**scored, 'seconds': time.monotonic()-began,
                'peak_allocated_gib': torch.cuda.max_memory_allocated(ac.device)/2**30}
            write(partial, record)
        for view, items in views.items():
            samples = record['samples'].setdefault(view, [])
            if trainer._training_deadline-time.monotonic() < 45:
                break
            for index in range(len(samples), args.decode_samples):
                if trainer._training_deadline-time.monotonic() < 30:
                    break
                phase('decoding '+view, completed=index, requested=args.decode_samples)
                try:
                    sample = trainer._sample_completions(ac, [items[index]], progress)[0]
                except _TrainingTimeLimit:
                    break
                _, calls = study.dialect.parse(sample['student'])
                answer = next((str(call['arguments'].get('answer', '')) for call in calls if call['name'] == 'submit_answer'), None)
                if answer is None:
                    try:
                        name, arguments = study.dialect.preferred_format.parse(sample['student'], None)
                        answer = str(arguments.get('answer', '')) if name == 'submit_answer' else ''
                    except ValueError:
                        answer = ''
                gold = held[index].info['answer']
                samples.append({'item_id': held[index].item_id, 'sample': sample, 'extracted_answer': answer,
                    'exact_match': exact_match(answer, [gold]), 'token_f1': token_f1(answer, [gold])})
                write(partial, record)
    if trainer.optimizers or trainer.updates_done:
        raise AssertionError('An evaluation unexpectedly created optimizer state')
    record['attempts'].append({'process_id': os.getpid(), 'seconds': time.monotonic()-started})
    record.update(status='complete', optimizer_updates=0, baseline_policy='Control900 is rescored at both budgets; no historical panel values reused.',
                  decode_counts={view: len(samples) for view, samples in record['samples'].items()})
    write(final, record)
    reporter.initialize_table('metrics', 'Answer NLL', 'Same100 questions; no optimizer updates.', ['view', 'answer NLL', 'seconds', 'peak allocated GiB'])
    for view, values in record['views'].items():
        reporter.add_data_point('metrics', {'view': view, 'answer NLL': values['answer_nll'], 'seconds': round(values['seconds'], 1),
                                          'peak allocated GiB': round(values['peak_allocated_gib'], 2)})
    phase('complete', optimizer_updates=0); reporter.finish()
    summarize(args.output)
    print('LONG_CROSS_EVAL_COMPLETE', args.cell, record['views']['AC']['answer_nll'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--initial-pair', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell', choices=CELLS, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--decode-samples', type=int, default=0, choices=range(13))
    run(parser.parse_args())
