"""One fixed-final comparison of the five 16k-question systems on reserved QA."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

from evaluate_long_cross import answer_scores, digest, paired_delta, write

ARMS = {
    'oracle_9b_16k_r16': ('9b', 1/16, '2B'),
    'oracle_9b_16k_r8': ('9b', 1/8, '2B'),
    'oracle_27b_16k_r16': ('27b', 1/16, '2B'),
    'oracle_27b_16k_r8': ('27b', 1/8, '2B'),
    'encoder4b_oracle27b_16k_r16': ('27b', 1/16, '4B'),
}
VIEWS = ('AC', 'full context', 'no context')
COMPARISONS = (
    ('oracle_9b_16k_r8', 'oracle_9b_16k_r16'),
    ('oracle_27b_16k_r8', 'oracle_27b_16k_r16'),
    ('oracle_27b_16k_r16', 'oracle_9b_16k_r16'),
    ('oracle_27b_16k_r8', 'oracle_9b_16k_r8'),
    ('encoder4b_oracle27b_16k_r16', 'oracle_27b_16k_r16'),
)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cohort_ids(confirmation):
    groups = confirmation['held_cohorts']
    ids = [row['example']['example_id'] for row in confirmation['held']]
    members = [ident for values in groups.values() for ident in values]
    if (set(groups) != {'9b', '27b'} or any(len(values) != 200 for values in groups.values())
            or len(ids) != 400 or len(set(ids)) != 400 or len(set(members)) != 400
            or set(ids) != set(members)):
        raise ValueError('Confirmation cohorts must partition 400 unique questions into 200+200')
    return groups


def split_scores(scored, cohorts):
    indexed = {row['item_id']: row for row in scored['per_item']}
    if len(indexed) != 400 or set(indexed) != {ident for values in cohorts.values() for ident in values}:
        raise ValueError('Scored questions differ from the reserved cohorts')
    return {name: {'answer_nll': statistics.mean(indexed[ident]['answer_nll'] for ident in ids),
                   'per_item': [indexed[ident] for ident in ids]} for name, ids in cohorts.items()}


def shared_fingerprint(held, full, noctx):
    if not len(held) == len(full) == len(noctx):
        raise ValueError('Reference transformation changed the number of questions')
    return digest([(original.item_id, contextual.ac_messages, absent.ac_messages,
                    original.tools, original.assistant_token_ids, original.info['answer_positions'])
                   for original, contextual, absent in zip(held, full, noctx)])


def final_pair(arm, reports, material_hashes):
    bank, ratio, encoder = ARMS[arm]
    folder = Path(reports)/arm
    result_path = folder/'result.json'
    result = json.loads(result_path.read_text())
    identity = json.loads((folder/'run_identity.json').read_text())['identity']
    config = json.loads((folder/'training_config.json').read_text())
    expected = {'arm': 'joint', 'epochs': 1, 'planned_updates': 2000, 'steps_per_epoch': 2000,
                'material_sha256': material_hashes[bank], 'evidence_tokens': 2048, 'max_passages': 4,
                'ratio': ratio, 'passage_parts': False, 'initial_pair_sha256': None}
    expected_config = {'loss_kind': 'sft', 'micro_batch_examples': 2, 'examples_per_update': 8,
                       'num_epochs': 1, 'learning_rate_ac': 5e-5, 'learning_rate_target_lora': 2e-5,
                       'schedule_total_updates': 2000, 'warmup_updates': 50,
                       'no_context_penalty_weight': 1, 'no_context_penalty_fraction': .25}
    if result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 2000:
        raise ValueError(arm+': fixed 16,000-question final pair is not complete')
    if any(identity.get(k) != v for k, v in expected.items()) or any(config.get(k) != v for k, v in expected_config.items()):
        raise ValueError(arm+': final training identity/config differs from the approved comparison')
    if encoder == '4B':
        initialization = identity.get('encoder_initialization', {})
        if (initialization.get('encoder_base') != 'Qwen/Qwen3.5-4B'
                or initialization.get('reader_base') != 'Qwen/Qwen3.5-9B'
                or initialization.get('checkpoint_label') != 'P4 epoch1'):
            raise ValueError('4B arm lacks the original P4 paired initialization identity')
    ac, reader = Path(result['checkpoint']).resolve(), Path(result['target_lora_checkpoint']).resolve()
    marker_path = ac.parent/'completed_epoch.json'
    marker = json.loads(marker_path.read_text()); run_root = ac.parents[2]
    if (marker.get('epoch_number') != 1 or (run_root/marker['ac_checkpoint']).resolve() != ac
            or (run_root/marker['target_lora_checkpoint']).resolve() != reader):
        raise ValueError(arm+': published epoch marker does not match the final AC/reader pair')
    required = [ac/'ac_modules.pt']
    for directory, base in ((ac/'side_lora', 'Qwen/Qwen3.5-'+encoder), (reader, 'Qwen/Qwen3.5-9B')):
        adapter = directory/'adapter_config.json'
        values = json.loads(adapter.read_text())
        if values.get('base_model_name_or_path') != base or values.get('r') != 128:
            raise ValueError(arm+': adapter base identity or rank differs')
        weights = [directory/name for name in ('adapter_model.safetensors', 'adapter_model.bin') if (directory/name).is_file()]
        if not weights:
            raise FileNotFoundError('Missing final adapter weights: '+str(directory))
        required.extend([adapter, *weights])
    if any(not path.is_file() or path.stat().st_size == 0 for path in required):
        raise ValueError(arm+': final pair contains missing or empty weights')
    return {'ac_checkpoint': str(ac), 'reader_checkpoint': str(reader),
            'result_sha256': file_sha(result_path), 'marker_sha256': file_sha(marker_path),
            'files': {str(path): {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in required}}


def summarize(output):
    records = {}
    for arm in ARMS:
        path = Path(output)/arm/'result.json'
        if path.exists():
            record = json.loads(path.read_text())
            if record.get('status') == 'complete':
                records[arm] = record
    if len({r['shared_input_fingerprint'] for r in records.values()}) > 1:
        raise ValueError('Confirmation arms have different full/no-context evidence or targets')
    if len({digest(r['identity']['evaluation_design']) for r in records.values()}) > 1:
        raise ValueError('Confirmation arms belong to different fixed-final comparisons')
    deltas = {a+'_vs_'+b: {cohort: {view: paired_delta(records[a]['cohorts'][cohort][view], records[b]['cohorts'][cohort][view])
                                 for view in VIEWS} for cohort in ('9b', '27b')}
              for a, b in COMPARISONS if a in records and b in records}
    result = {'status': 'complete' if len(records) == len(ARMS) else 'partial', 'confirmation_used': True,
              'completed_arms': list(records), 'expected_arms': list(ARMS),
              'scores': {arm: {cohort: {view: values['answer_nll'] for view, values in views.items()}
                               for cohort, views in record['cohorts'].items()} for arm, record in records.items()},
              'deltas': deltas,
              'interpretation': 'Fixed final 16,000 questions ×1 epoch; separate 200-question generator cohorts. Compare systems within a cohort. Full/no-context use each final paired reader. P4 changes both encoder base and pretrained pair; not isolated encoder capacity. No checkpoint selection or further tuning on these confirmation questions.'}
    write(Path(output)/'summary.json', result)
    return result


def run(args):
    frozen = json.loads(args.design.read_text())
    paths = {'builder': args.bench, 'harness_4b': args.harness_4b,
             'material_9b': args.material_dir/'material_9b_16000.json',
             'material_27b': args.material_dir/'material_27b_16000.json',
             'confirmation': args.material_dir/'confirmation.json'}
    if any(file_sha(path) != frozen['sha256'][name] for name, path in paths.items()):
        raise ValueError('Frozen material or evaluation builder changed')
    hashes = {bank: frozen['sha256']['material_'+bank] for bank in ('9b', '27b')}
    # All five fixed finals must exist before any confirmation is exposed.
    pairs = {arm: final_pair(arm, args.reports, hashes) for arm in ARMS}
    design = {'frozen_design_sha256': file_sha(args.design), 'final_pairs': pairs,
              'evaluator_sha256': file_sha(__file__), 'metric_helper_sha256': file_sha(Path(__file__).with_name('evaluate_long_cross.py'))}
    identity = {'arm': args.arm, 'evaluation_design': design}
    folder = args.output/args.arm; final = folder/'result.json'; partial = folder/'partial.json'
    if final.exists():
        previous = json.loads(final.read_text())
        if previous.get('status') != 'complete' or previous['identity'] != identity:
            raise ValueError('Completed confirmation identity changed')
        summarize(args.output); return
    record = json.loads(partial.read_text()) if partial.exists() else {'identity': identity, 'status': 'running', 'views': {}, 'cohorts': {}}
    if record['identity'] != identity:
        raise ValueError('Partial confirmation identity changed')
    used_path = args.output/'confirmation_used.json'
    if used_path.exists() and json.loads(used_path.read_text())['evaluation_design'] != design:
        raise ValueError('Confirmation has already been used with a different final-pair comparison')
    material = json.loads(paths['material_9b'].read_text())
    confirmation = json.loads(paths['confirmation'].read_text()); cohorts = cohort_ids(confirmation)
    spec = importlib.util.spec_from_file_location('confirmation_builder', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    bank, ratio, encoder = ARMS[args.arm]
    harness_factory = bench.harness
    if encoder == '4B':
        spec4 = importlib.util.spec_from_file_location('confirmation_harness4b', args.harness_4b)
        module4 = importlib.util.module_from_spec(spec4); spec4.loader.exec_module(module4)
        harness_factory = module4.harness
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import configure_fla_runtime
    from activation.common.reporting import HtmlReporter
    reporter = HtmlReporter(str(folder), 'Reserved confirmation — '+args.arm,
        '16,000 unique questions ×1 epoch; fixed final pair. Separate 200-question 9B/27B cohorts; no training or checkpoint selection.')
    def phase(name, **values):
        reporter.status.clear(); reporter.set_status(phase=name, **values); reporter.render(force=True)
    started = time.monotonic(); phase('loading fixed final pair')
    pair = pairs[args.arm]
    harness = harness_factory(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = harness.module_manager.get_ac_model(bench.AC); ac.prepare(); harness.module_manager.ensure_lora(bench.READER)
    phase('building fixed confirmation evidence', questions=400)
    study, _, held = bench.material_items(harness, {'train': material['train'][:1], 'held': confirmation['held'], 'retrieval': material['retrieval']},
                                        evidence_tokens=2048, max_passages=4, ratio=ratio)
    views = {'AC': held, 'full context': study.transform_for_eval_reference(held, 'full_context'),
             'no context': study.transform_for_eval_reference(held, 'no_context')}
    common = shared_fingerprint(held, views['full context'], views['no context'])
    ac_fingerprint = digest([(item.item_id, item.ac_messages) for item in held])
    if record.get('shared_input_fingerprint', common) != common or record.get('ac_input_fingerprint', ac_fingerprint) != ac_fingerprint:
        raise ValueError('Rebuilt inputs differ from the partial confirmation')
    for arm in ARMS:
        path = args.output/arm/'result.json'
        if not path.exists():
            path = args.output/arm/'partial.json'
        if path.exists() and json.loads(path.read_text()).get('shared_input_fingerprint') != common:
            raise ValueError('Cross-arm shared evidence differs; stop before scoring')
    record.update(shared_input_fingerprint=common, ac_input_fingerprint=ac_fingerprint,
                  ratio=ratio, encoder=encoder, source_bank=bank,
                  input_rows=[{'item_id': item.item_id, 'question': item.info['question'], 'gold': item.info['answer'],
                               **{k: item.info[k] for k in ('encoded_source_tokens', 'compressed_rows', 'evidence_source_tokens', 'passages')}} for item in held])
    trainer = ActivationContextTrainer(harness, ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0,
        micro_batch_examples=2, max_example_tokens=72000, completion_samples=0, seed=bench.SEED))
    variables = trainer._autotuning_variables(ac); trainer._training_deadline = time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    if not used_path.exists():
        write(used_path, {'confirmation_used': True, 'first_evaluation_started_unix': time.time(), 'evaluation_design': design})
    write(partial, record)
    with configure_fla_runtime(variables['fla_config']), torch.no_grad():
        for view, items in views.items():
            if view in record['views']:
                continue
            phase('scoring '+view, questions=400)
            torch.cuda.synchronize(ac.device); torch.cuda.reset_peak_memory_stats(ac.device); began = time.monotonic()
            scored = answer_scores(trainer._evaluate(ac, items, ac.device), items, held)
            torch.cuda.synchronize(ac.device)
            record['views'][view] = {**scored, 'seconds': time.monotonic()-began,
                                     'peak_allocated_gib': torch.cuda.max_memory_allocated(ac.device)/2**30}
            for cohort, values in split_scores(scored, cohorts).items():
                record['cohorts'].setdefault(cohort, {})[view] = values
            write(partial, record)
    if trainer.optimizers or trainer.updates_done:
        raise AssertionError('Confirmation unexpectedly created optimizer state')
    record.update(status='complete', confirmation_used=True, optimizer_updates=0, total_seconds=time.monotonic()-started)
    write(final, record)
    reporter.initialize_table('metrics', 'Reserved confirmation answer NLL', 'Compare arms within each cohort; each row averages 200 questions.', ['cohort', 'view', 'answer NLL'])
    for cohort, values in record['cohorts'].items():
        for view, score in values.items():
            reporter.add_data_point('metrics', {'cohort': cohort.upper(), 'view': view, 'answer NLL': score['answer_nll']})
    phase('complete', confirmation_used=True); reporter.finish(); summarize(args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--design', type=Path, required=True)
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--harness-4b', type=Path, required=True)
    parser.add_argument('--material-dir', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    run(parser.parse_args())
