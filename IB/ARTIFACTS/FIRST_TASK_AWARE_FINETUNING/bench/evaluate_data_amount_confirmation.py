"""Fixed final confirmation: 32k unique QA x 1 epoch versus original 16k x 2."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

from evaluate_data_amount_cohort import ARMS, final_pair, validate_design
from evaluate_model_confirmation import cohort_ids, shared_fingerprint, split_scores, file_sha, VIEWS
from evaluate_long_cross import answer_scores, digest, paired_delta, write

COMPARISONS = (('oracle_27b_32k', 'oracle_27b_16k_2ep'),)


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
              'interpretation': 'Fixed final 32,000 unique questions ×1 epoch versus 16,000 ×2; equal question presentations, same chosen recipe and original P2. Separate 200-question generator cohorts. Full/no-context use each final paired reader. Actual token and time budgets can differ. No checkpoint selection or further tuning on these confirmation questions.'}
    write(Path(output)/'summary.json', result)
    return result


def run(args):
    frozen = validate_design(args.evaluation_design)
    paths = {'builder': args.bench, 'retrieval_material': args.material_dir/'material_9b_16000.json',
             'confirmation': args.confirmation}
    if any(file_sha(path) != frozen['sha256'][name] for name, path in paths.items()):
        raise ValueError('Frozen data-amount evaluation inputs or builder changed')
    # Both complete final pairs are required before opening fresh confirmation questions.
    pairs = {arm: final_pair(arm, args.reports, design=frozen, design_path=args.evaluation_design) for arm in ARMS}
    design = {'data_amount_design_sha256': file_sha(args.evaluation_design), 'final_pairs': pairs,
              'evaluator_sha256': file_sha(__file__), 'helper_sha256': frozen['helper_sha256']}
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
    material = json.loads(paths['retrieval_material'].read_text())
    confirmation = json.loads(paths['confirmation'].read_text()); cohorts = cohort_ids(confirmation)
    spec = importlib.util.spec_from_file_location('confirmation_builder', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    spec = frozen['arms'][args.arm]
    bank, ratio, encoder = '27b', spec['ratio'], '2B'
    harness_factory = bench.harness
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import configure_fla_runtime
    from activation.common.reporting import HtmlReporter
    import activation.ac_model.ac_model_training as implementation
    if file_sha(implementation.__file__) != frozen['trainer_sha256']:
        raise ValueError('Pinned training/evaluation implementation changed')
    reporter = HtmlReporter(str(folder), 'Reserved confirmation — '+args.arm,
        f"{spec['questions']:,} unique questions ×{spec['epochs']} epochs; fixed final pair. Separate 200-question 9B/27B cohorts; no training or checkpoint selection.")
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-design', 'bench', 'material-dir', 'confirmation', 'reports', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    run(parser.parse_args())
