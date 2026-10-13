"""One confirmation use: the three fixed final data-budget pairs, never checkpoint selection."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

from evaluate_long_cross import answer_scores, digest, paired_delta, write

BENCH_SHA = 'b75901827e2c2bbffa385e50eea82ab99276ffb9fb742b83464e1b67df4edcb6'
CONFIRMATION_SHA = '701cb2608379a65d40353ec448d932f7a78e531b461b9f382f33e6490caf83f5'
ARMS = {
    'data_2k_4ep': (2000, 4, 'd498d1ac0aa6dd2b71306fffbdfa9202f90262c16c6950056285a83240eac315'),
    'data_4k_2ep': (4000, 2, 'd787a1535f8f73f3813c6470a3b6c288a5bd529c88cd1816b5129b25156544af'),
    'data_8k_1ep': (8000, 1, 'ef803a5c23a618af5d8ce58015e478bb29b5eb677a437b9edf29cc6cc6a7d2f9'),
}
VIEWS = ('AC', 'full context', 'no context')


def final_pair(arm, reports):
    """Read only the final completed pair, with the published marker and original study identity."""
    count, epochs, material_sha = ARMS[arm]
    folder = Path(reports)/arm
    raw = (folder/'result.json').read_bytes(); result = json.loads(raw)
    identity = json.loads((folder/'run_identity.json').read_text())['identity']
    config = json.loads((folder/'training_config.json').read_text())
    expected = {'arm': 'joint', 'epochs': epochs, 'planned_updates': 1000, 'steps_per_epoch': count//8,
                'material_sha256': material_sha, 'evidence_tokens': 2048, 'max_passages': 4,
                'ratio': 1/16, 'passage_parts': False, 'initial_pair_sha256': None}
    expected_config = {'micro_batch_examples': 2, 'examples_per_update': 8, 'num_epochs': epochs,
                       'learning_rate_ac': 5e-5, 'learning_rate_target_lora': 2e-5, 'schedule_total_updates': 1000}
    if result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 1000:
        raise ValueError(arm+': final1000-update run is not complete')
    if any(identity.get(key) != value for key, value in expected.items()) or any(config.get(key) != value for key, value in expected_config.items()):
        raise ValueError(arm+': run identity/config differs from the frozen data-budget study')
    ac, reader = Path(result['checkpoint']).resolve(), Path(result['target_lora_checkpoint']).resolve()
    marker_raw = (ac.parent/'completed_epoch.json').read_bytes(); marker = json.loads(marker_raw)
    root = ac.parents[2]
    if marker.get('epoch_number') != epochs or (root/marker['ac_checkpoint']).resolve() != ac or (root/marker['target_lora_checkpoint']).resolve() != reader:
        raise ValueError(arm+': final result and published pair disagree')
    required = [ac/'ac_modules.pt', ac/'side_lora/adapter_config.json', reader/'adapter_config.json']
    for directory in (ac/'side_lora', reader):
        weights = [directory/name for name in ('adapter_model.safetensors', 'adapter_model.bin') if (directory/name).is_file()]
        if not weights:
            raise FileNotFoundError('Missing adapter weights: '+str(directory))
        required.extend(weights)
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError('Incomplete or empty final paired weights')
    return {'ac_checkpoint': str(ac), 'reader_checkpoint': str(reader), 'update': 1000,
            'training_questions': count, 'training_epochs': epochs,
            'result_sha256': hashlib.sha256(raw).hexdigest(), 'marker_sha256': hashlib.sha256(marker_raw).hexdigest(),
            'files': {str(path): {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in required}}


def summarize(output):
    records = {}
    for arm in ARMS:
        path = Path(output)/arm/'result.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('status') == 'complete':
                records[arm] = record
    if len({record['input_fingerprint'] for record in records.values()}) > 1:
        raise ValueError('Confirmation arms do not use the same exact inputs')
    if len({digest(record['identity']['evaluation_design']) for record in records.values()}) > 1:
        raise ValueError('Confirmation arms belong to different frozen evaluations')
    deltas = {}
    for measured, reference in [('data_4k_2ep', 'data_2k_4ep'), ('data_8k_1ep', 'data_2k_4ep'), ('data_8k_1ep', 'data_4k_2ep')]:
        if measured in records and reference in records:
            deltas[measured+'_vs_'+reference] = {view: paired_delta(records[measured]['views'][view], records[reference]['views'][view]) for view in VIEWS}
    summary = {'status': 'complete' if len(records) == len(ARMS) else 'partial', 'confirmation_used': True,
               'completed_arms': list(records), 'expected_arms': list(ARMS),
               'scores': {arm: {view: values['answer_nll'] for view, values in record['views'].items()} for arm, record in records.items()},
               'deltas': deltas,
               'interpretation': 'Same reserved200 questions, same evidence, fixed final1000-update pairs. Negative deltas improve literal gold-answer NLL. Full/no-context use each pair\'s current reader. This confirmation cohort is now used; do not select checkpoints or tune on it. The later27B study has a separate confirmation cohort.'}
    write(Path(output)/'summary.json', summary)
    return summary


def run(args):
    if hashlib.sha256(args.bench.read_bytes()).hexdigest() != BENCH_SHA:
        raise ValueError('Pinned input builder changed')
    common_path = args.material_dir/'material_2000.json'; confirmation_path = args.material_dir/'confirmation.json'
    if hashlib.sha256(common_path.read_bytes()).hexdigest() != ARMS['data_2k_4ep'][2] or hashlib.sha256(confirmation_path.read_bytes()).hexdigest() != CONFIRMATION_SHA:
        raise ValueError('Frozen retrieval material or confirmation questions changed')
    # Freeze all three final pairs before exposing any confirmation result.
    pairs = {arm: final_pair(arm, args.reports) for arm in ARMS}
    pair = pairs[args.arm]
    design = {'confirmation_sha256': CONFIRMATION_SHA, 'retrieval_material_sha256': ARMS['data_2k_4ep'][2],
              'builder_sha256': BENCH_SHA, 'final_pairs': pairs, 'microbatch': 2, 'ratio': 1/16,
              'evidence_tokens': 2048, 'max_passages': 4,
              'evaluator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'metric_helper_sha256': hashlib.sha256(Path(__file__).with_name('evaluate_long_cross.py').read_bytes()).hexdigest()}
    identity = {'arm': args.arm, 'evaluation_design': design, 'decode_samples': args.decode_samples}
    folder = args.output/args.arm; final = folder/'result.json'; partial = folder/'partial.json'
    if final.exists():
        previous = json.loads(final.read_text())
        if previous.get('status') != 'complete' or previous['identity'] != identity:
            raise ValueError('Completed confirmation identity changed; refusing a second result')
        summarize(args.output); print('CONFIRMATION_ALREADY_COMPLETE', args.arm, flush=True); return
    record = json.loads(partial.read_text()) if partial.exists() else {'identity': identity, 'status': 'running', 'views': {}, 'samples': {}}
    if record['identity'] != identity:
        raise ValueError('Partial confirmation identity changed')
    used_path = args.output/'confirmation_used.json'
    if used_path.exists() and json.loads(used_path.read_text())['evaluation_design'] != design:
        raise ValueError('Confirmation already belongs to a different final-pair comparison')
    material = json.loads(common_path.read_text()); confirmation = json.loads(confirmation_path.read_text())
    if len(material['retrieval']) != 8200 or len(confirmation['held']) != 200:
        raise ValueError('Expected exactly8200 retrieval records and200 reserved questions')
    groups = collections.defaultdict(list)
    for row in confirmation['held']:
        groups[row['domain']].append(row)
    held_rows = [row for _, _, row in sorted((i/len(rows), domain, row) for domain, rows in groups.items() for i, row in enumerate(rows))]
    spec = importlib.util.spec_from_file_location('confirmation_bench', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress, _TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    from activation.common.reporting import HtmlReporter
    from activation.dataset.scoring import exact_match, token_f1
    reporter = HtmlReporter(str(folder), 'Reserved confirmation — '+args.arm,
        'Fixed final1000-update pair;200 previously reserved questions. No training or checkpoint selection. NLL is literal gold-answer loss.')
    def phase(name, **values):
        reporter.status.clear(); reporter.set_status(phase=name, arm=args.arm, **values); reporter.render(force=True)
    started = time.monotonic(); phase('loading final paired checkpoint')
    harness = bench.harness(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = harness.module_manager.get_ac_model(bench.AC); ac.prepare(); harness.module_manager.ensure_lora(bench.READER)
    phase('building fixed confirmation evidence', questions=200, retrieval_records=8200)
    study, _, held = bench.material_items(harness, {'train': material['train'][:1], 'held': held_rows, 'retrieval': material['retrieval']},
        evidence_tokens=2048, max_passages=4, ratio=1/16)
    fingerprint = digest([(item.item_id, item.ac_messages, item.in_context_messages, item.assistant_token_ids) for item in held])
    if record.get('input_fingerprint', fingerprint) != fingerprint:
        raise ValueError('Rebuilt confirmation inputs changed')
    record['input_fingerprint'] = fingerprint
    record['input_rows'] = [{'item_id': item.item_id, 'question': item.info['question'], 'gold': item.info['answer'],
        **{key: item.info[key] for key in ('encoded_source_tokens', 'compressed_rows', 'evidence_source_tokens', 'passages')}} for item in held]
    record['row_budget'] = {key: {'total': sum(item.info[key] for item in held), 'min': min(item.info[key] for item in held),
        'max': max(item.info[key] for item in held), 'mean': statistics.mean(item.info[key] for item in held)}
        for key in ('encoded_source_tokens', 'compressed_rows', 'evidence_source_tokens')}
    trainer = ActivationContextTrainer(harness, ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0,
        micro_batch_examples=2, max_example_tokens=72000, completion_samples=0, completion_max_new_tokens=160, seed=bench.SEED))
    variables = trainer._autotuning_variables(ac); trainer._training_deadline = time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    views = {'AC': held, 'full context': study.transform_for_eval_reference(held, 'full_context'),
             'no context': study.transform_for_eval_reference(held, 'no_context')}
    progress = ActivationContextTrainingProgress(epoch=0, epochs=0, epoch_number=0, step=0, steps=0,
        global_step=1000, epoch_elapsed_s=0., elapsed_s=0.)
    if not used_path.exists():
        write(used_path, {'confirmation_used': True, 'first_evaluation_started_unix': time.time(), 'evaluation_design': design,
            'policy': 'One fixed-final comparison only. This cohort is no longer untouched. No checkpoint selection or subsequent tuning on these questions.'})
    write(partial, record)
    with configure_fla_runtime(variables['fla_config']), torch.no_grad():
        for view, items in views.items():
            if view in record['views']:
                continue
            phase('scoring '+view, questions=200)
            torch.cuda.synchronize(ac.device); torch.cuda.reset_peak_memory_stats(ac.device); began = time.monotonic()
            values = answer_scores(trainer._evaluate(ac, items, ac.device), items, held)
            torch.cuda.synchronize(ac.device)
            record['views'][view] = {**values, 'seconds': time.monotonic()-began,
                'peak_allocated_gib': torch.cuda.max_memory_allocated(ac.device)/2**30}
            write(partial, record)
        for view, items in views.items():
            samples = record['samples'].setdefault(view, [])
            for index in range(len(samples), args.decode_samples):
                if trainer._training_deadline-time.monotonic() < 30:
                    break
                phase('optional fixed decodes: '+view, completed=index, requested=args.decode_samples)
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
        raise AssertionError('Confirmation evaluation created optimizer state')
    record.update(status='complete', confirmation_used=True, optimizer_updates=0, total_seconds=time.monotonic()-started,
                  decode_counts={view: len(samples) for view, samples in record['samples'].items()})
    write(final, record)
    reporter.initialize_table('metrics', 'Confirmation answer NLL', 'Fixed final pair; same200 questions across all arms.', ['view', 'answer NLL', 'seconds', 'peak GiB'])
    for view, values in record['views'].items():
        reporter.add_data_point('metrics', {'view': view, 'answer NLL': values['answer_nll'], 'seconds': round(values['seconds'], 1), 'peak GiB': round(values['peak_allocated_gib'], 2)})
    phase('complete', confirmation_used=True, optimizer_updates=0); reporter.finish(); summarize(args.output)
    print('DATA_CONFIRMATION_COMPLETE', args.arm, record['views']['AC']['answer_nll'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--material-dir', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seconds', type=int, default=420)
    parser.add_argument('--decode-samples', type=int, default=12, choices=range(13))
    run(parser.parse_args())
