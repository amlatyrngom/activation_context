"""No-training screen of passage structure and row capacity at frozen control900 weights."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

VARIANTS = {'merged_16': (False, 1/16), 'parts_16': (True, 1/16),
            'merged_10': (False, 1/10), 'merged_30': (False, 1/30)}
MATERIAL_SHA = 'f937470fde3f231aac98e7fe4f209488108699b329ac00f222a919e3a0a4bc4e'
BENCH_SHA = '25663ab1991a8bc8da3e809674e17d6615267e7409c02f0a04149f5a14f7c4c4'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def build(bench, harness, material, variant, evidence_tokens=2048, max_passages=4):
    # Match the deployed runner's domain interleave. Its BM25 helper requires a nonempty training list;
    # build one unused training row while retaining all4100 retrieval records, and score only held questions.
    groups = collections.defaultdict(list)
    for row in material['held']:
        groups[row['domain']].append(row)
    held_rows = [row for _, _, row in sorted((i/len(rows), domain, row) for domain, rows in groups.items() for i, row in enumerate(rows))]
    probe = {'train': material['train'][:1], 'held': held_rows, 'retrieval': material['retrieval']}
    parts, ratio = VARIANTS[variant]
    study, _, held = bench.material_items(harness, probe, passage_parts=parts, ratio=ratio, evidence_tokens=evidence_tokens, max_passages=max_passages)
    rows = [{'item_id': item.item_id, **{key: item.info[key] for key in
             ('encoded_source_tokens', 'compressed_rows', 'evidence_source_tokens', 'part_source_tokens', 'part_rows', 'passages')}} for item in held]
    common = digest([(item.item_id, item.in_context_messages, item.assistant_token_ids) for item in held])
    budget = {}
    for key in ('encoded_source_tokens', 'compressed_rows', 'evidence_source_tokens'):
        values = sorted(row[key] for row in rows)
        budget[key] = {'total': sum(values), 'min': values[0], 'mean': statistics.mean(values),
                       'median': statistics.median(values), 'p95': values[int(.95*(len(values)-1))], 'max': values[-1]}
    budget['rows_per_encoder_token'] = budget['compressed_rows']['total'] / budget['encoded_source_tokens']['total']
    budget['rows_per_evidence_token'] = budget['compressed_rows']['total'] / budget['evidence_source_tokens']['total']
    return study, held, {'common_inputs_sha256': common, 'row_budget': budget, 'input_rows': rows}


def answer_scores(summary, view_items, original_items):
    assert len(summary['per_item']) == len(view_items) == len(original_items)
    rows = []
    for measured, view, original in zip(summary['per_item'], view_items, original_items):
        assert measured['item_id'] == view.item_id
        assert view.assistant_token_ids == original.assistant_token_ids
        positions = original.info['answer_positions']
        rows.append({'item_id': original.item_id, 'view_item_id': view.item_id,
                     'answer_nll': -sum(measured['target_logp'][p] for p in positions)/len(positions),
                     'answer_tokens': len(positions), 'student_tokens': measured['student_tokens']})
    return {'answer_nll': statistics.mean(row['answer_nll'] for row in rows), 'per_item': rows}


def main(args):
    if args.cpu_check:
        sys.modules.setdefault('fla', None)
    assert hashlib.sha256(args.bench.read_bytes()).hexdigest() == BENCH_SHA, 'Screen expects its pinned deployed-v2 snapshot'
    assert hashlib.sha256(args.material.read_bytes()).hexdigest() == MATERIAL_SHA, 'Frozen adaptation material changed'
    material = json.loads(args.material.read_text())
    pair_bytes = args.initial_pair.read_bytes()
    pair = json.loads(pair_bytes)
    assert len(material['train']) == 640 and len(material['held']) == 100 and len(material['retrieval']) == 4100
    assert pair['update'] == 900 and all('update_00900' in pair[key] for key in ('ac_checkpoint', 'reader_checkpoint'))
    spec = importlib.util.spec_from_file_location('qa_screen_bench', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    if args.cpu_check:
        harness = bench.harness()
        prepared = {}
        for variant in args.cpu_variants:
            print('CHECKING_INPUTS', variant, flush=True)
            _, held, meta = build(bench, harness, material, variant, args.evidence_tokens, args.max_passages)
            prepared[variant] = meta
        baseline = prepared['merged_16']
        assert all(meta['common_inputs_sha256'] == baseline['common_inputs_sha256'] for meta in prepared.values())
        if 'parts_16' in prepared:
            assert [row['compressed_rows'] for row in prepared['parts_16']['input_rows']] == [row['compressed_rows'] for row in baseline['input_rows']]
        bench.write(args.output/'cpu_inputs.json', {'status': 'passed', 'variants': prepared, 'bench_sha256': BENCH_SHA, 'material_sha256': MATERIAL_SHA})
        print('SCREEN_CPU_INPUTS_PASSED', json.dumps({k: v['row_budget'] for k, v in prepared.items()}), flush=True)
        return

    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress, _TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    from activation.common.reporting import HtmlReporter
    from activation.dataset.scoring import exact_match, token_f1

    folder = args.output/args.variant
    if (folder/'result.json').exists():
        raise FileExistsError('Screen result already exists; choose a new output folder')
    folder.mkdir(parents=True, exist_ok=True)
    reporter = HtmlReporter(str(folder), 'Frozen 4,000 questions × 1.8 passes — '+args.variant,
        f'No additional training. About {args.evidence_tokens/1024:g}k evidence tokens, up to {args.max_passages-1} distractors. '
        'The same100 development questions, evidence and gold targets; answer-only NLL is teacher-forced, not answer correctness.')
    def phase(name, **values):
        reporter.status.clear(); reporter.set_status(phase=name, variant=args.variant, **values); reporter.render(force=True)
    started = time.monotonic()
    phase('loading frozen paired checkpoint')
    harness = bench.harness(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = harness.module_manager.get_ac_model(bench.AC); ac.prepare(); harness.module_manager.ensure_lora(bench.READER)
    loaded_seconds = time.monotonic()-started
    phase('preparing100 fixed inputs')
    prepared_start = time.monotonic()
    study, held, meta = build(bench, harness, material, args.variant, args.evidence_tokens, args.max_passages)
    preparation_seconds = time.monotonic()-prepared_start
    identity = {'material_sha256': MATERIAL_SHA, 'bench_sha256': BENCH_SHA, 'initial_pair_sha256': hashlib.sha256(pair_bytes).hexdigest(),
                'common_inputs_sha256': meta['common_inputs_sha256'], 'evidence_tokens': args.evidence_tokens,
                'max_passages': args.max_passages, 'microbatch': args.micro_batch_examples}
    baseline = None
    if args.variant != 'merged_16':
        baseline = json.loads((args.output/'merged_16/result.json').read_text())
        assert baseline['status'] == 'complete' and baseline['identity'] == identity
        assert baseline['runtime_ac_version'] == ac.version, 'The supposedly frozen checkpoint changed between variants'
        if args.variant == 'parts_16':
            assert [row['compressed_rows'] for row in meta['input_rows']] == [row['compressed_rows'] for row in baseline['input_rows']]
    config = ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0,
        micro_batch_examples=args.micro_batch_examples, max_example_tokens=72000, completion_samples=0, completion_max_new_tokens=160, seed=bench.SEED)
    trainer = ActivationContextTrainer(harness, config)
    variables = trainer._autotuning_variables(ac)
    trainer._training_deadline = time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    result = {'status': 'running', 'variant': args.variant, 'identity': identity, 'initial_pair': pair,
              'runtime_ac_version': ac.version, 'optimizer_updates': 0, 'process_id': os.getpid(), 'microbatch': args.micro_batch_examples,
              'ratio': VARIANTS[args.variant][1], 'passage_parts': VARIANTS[args.variant][0], **meta,
              'loading_seconds': loaded_seconds, 'preparation_seconds': preparation_seconds,
              'scope': 'Zero-training transfer screen; unfavorable results do not rule out adaptation.1/10 adds nominal60% rows; inspect realized counts.',
              'views': {}, 'samples': {}, 'decode_note': 'Same first12 domain-interleaved held items, greedy,160-token cap; optional and budget-bounded.'}
    progress = ActivationContextTrainingProgress(epoch=0, epochs=0, epoch_number=0, step=0, steps=0,
        global_step=900, epoch_elapsed_s=0., elapsed_s=0.)
    full = study.transform_for_eval_reference(held, 'full_context')
    noctx = study.transform_for_eval_reference(held, 'no_context')
    views = {'AC': held, **({'full context': full, 'no context': noctx} if baseline is None else {})}
    with configure_fla_runtime(variables['fla_config']), torch.no_grad():
        for name, items in views.items():
            phase('scoring '+name, questions=len(items))
            torch.cuda.synchronize(ac.device); torch.cuda.reset_peak_memory_stats(ac.device); began = time.monotonic()
            summary = trainer._evaluate(ac, items, ac.device)
            torch.cuda.synchronize(ac.device)
            scored = answer_scores(summary, items, held)
            result['views'][name] = {**scored, 'seconds': time.monotonic()-began,
                                    'peak_allocated_gib': torch.cuda.max_memory_allocated(ac.device)/2**30}
            bench.write(folder/'partial.json', result)
        if baseline is not None:
            result['shared_references'] = {'file': '../merged_16/result.json',
                'full_context_answer_nll': baseline['views']['full context']['answer_nll'],
                'no_context_answer_nll': baseline['views']['no context']['answer_nll']}
        # Mandatory scores are already durable. Decoding is a small optional qualitative sample.
        for name, items in views.items():
            if not args.decode_samples or trainer._training_deadline-time.monotonic() < 45:
                break
            phase('decoding '+name, questions=min(args.decode_samples, len(items)))
            samples = []
            for original, item in zip(held[:args.decode_samples], items[:args.decode_samples]):
                if trainer._training_deadline-time.monotonic() < 30:
                    break
                try:
                    sample = trainer._sample_completions(ac, [item], progress)[0]
                except _TrainingTimeLimit:
                    break
                _, calls = study.dialect.parse(sample['student'])
                answer = next((str(call['arguments'].get('answer', '')) for call in calls if call['name'] == 'submit_answer'), None)
                if answer is None:
                    try:
                        name_parsed, arguments = study.dialect.preferred_format.parse(sample['student'], None)
                        answer = str(arguments.get('answer', '')) if name_parsed == 'submit_answer' else ''
                    except ValueError:
                        answer = ''
                gold = original.info['answer']
                samples.append({'item_id': original.item_id, 'question': original.info['question'], 'gold': gold,
                    'sample': sample, 'extracted_answer': answer, 'exact_match': exact_match(answer, [gold]), 'token_f1': token_f1(answer, [gold])})
                result['samples'][name] = samples
                bench.write(folder/'partial.json', result)
    control_rows = (baseline or result)['row_budget']['compressed_rows']['total']
    control_inputs = (baseline or result)['row_budget']['encoded_source_tokens']['total']
    result['additional_rows_vs_merged_16'] = meta['row_budget']['compressed_rows']['total']-control_rows
    result['realized_row_multiplier_vs_merged_16'] = meta['row_budget']['compressed_rows']['total']/control_rows
    result['additional_encoder_tokens_vs_merged_16'] = meta['row_budget']['encoded_source_tokens']['total']-control_inputs
    if baseline:
        control_scores = {row['item_id']: row['answer_nll'] for row in baseline['views']['AC']['per_item']}
        differences = [row['answer_nll']-control_scores[row['item_id']] for row in result['views']['AC']['per_item']]
        result['paired_delta_vs_merged_16'] = {'mean': statistics.mean(differences), 'median': statistics.median(differences),
                                               'questions_improved': sum(value < 0 for value in differences)}
    assert not trainer.optimizers and not trainer.updates_done
    result['decode_counts'] = {name: len(samples) for name, samples in result['samples'].items()}
    result['status'] = 'complete'; result['elapsed_seconds'] = time.monotonic()-started
    bench.write(folder/'result.json', result)
    reporter.set_text('results', 'Screen result', json.dumps({k: v for k, v in result.items() if k not in ('input_rows', 'samples', 'views')}, indent=2))
    reporter.initialize_table('metrics', 'Answer NLL at unchanged weights', 'Mean per question; lower is better. No optimizer updates.', ['view', 'answer NLL', 'seconds', 'peak allocated GiB'])
    for name, values in result['views'].items():
        reporter.add_data_point('metrics', {'view': name, 'answer NLL': values['answer_nll'], 'seconds': round(values['seconds'], 2), 'peak allocated GiB': round(values['peak_allocated_gib'], 2)})
    phase('complete', optimizer_updates=0); reporter.finish()
    print('PASSAGE_SCREEN_COMPLETE', args.variant, result['views']['AC']['answer_nll'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--initial-pair', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, default='merged_16')
    parser.add_argument('--seconds', type=int, default=300)
    parser.add_argument('--decode-samples', type=int, default=0, choices=range(13))
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--cpu-variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--evidence-tokens', type=int, default=2048)
    parser.add_argument('--max-passages', type=int, default=4)
    parser.add_argument('--micro-batch-examples', type=int, choices=(1,2), default=2)
    main(parser.parse_args())
