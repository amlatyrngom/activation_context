"""Decode the exact held trajectory cohort with an initial or explicitly saved pair."""
import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

from run_qa import AC, NAME, READER, harness, write
from run_trajectory_tasks import build_items, digest, cohort_metrics

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main(args):
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress, _TrainingTimeLimit
    from activation.ac_model.ac_model_study import ActivationContextStudyGenerator
    from activation.autotuners import configure_fla_runtime
    from activation.common.reporting import HtmlReporter

    started = time.monotonic()
    os.environ['ACTIVATION_SYNC_ROOT'] = str(args.output.resolve())
    selection = json.loads((args.run / 'selection.json').read_text())
    run_identity = json.loads((args.run / 'identity.json').read_text())
    pair = json.loads(args.pair.read_text())
    task = selection['request']['task']
    assert digest(selection) == run_identity['selection']
    assert len(selection['held']) == 100
    if args.condition == 'initial':
        assert digest(pair) == run_identity['initial_pair'], 'Initial controls require the original pair'
    if args.match_selection:
        other = json.loads(args.match_selection.read_text())
        assert selection['held'] == other['held'], 'Cross-host held selections differ'
    ac_path, reader_path = Path(pair['ac_checkpoint']), Path(pair['reader_checkpoint'])
    identity = {'condition': args.condition, 'label': args.label, 'task': task,
                'held_selection': digest(selection['held']), 'run_identity': run_identity,
                'pair_files': {name: file_hash(path) for name, path in {
                    'heads': ac_path / 'ac_modules.pt',
                    'encoder': ac_path / 'side_lora/adapter_model.safetensors',
                    'reader': reader_path / 'adapter_model.safetensors',
                    'encoder_config': ac_path / 'side_lora/adapter_config.json',
                    'reader_config': reader_path / 'adapter_config.json'}.items()},
                'evaluator': file_hash(__file__), 'max_new_tokens': args.max_new_tokens,
                'generation': 'Greedy, before the first assistant target; original chat template settings',
                'reader': {'initial': 'Immutable initial reader', 'trained': 'Saved trained reader',
                           'candidate': 'Alternative frozen initializer'}[args.condition]}
    report = args.output / 'report'
    result_path = report / 'result.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        assert result['identity'] == identity, 'Cohort output belongs to another checkpoint or recipe'
        if result['status'] == 'complete':
            return 0
    else:
        result = {'identity': identity, 'status': 'running', 'rows': [], 'views': {}}
    r = HtmlReporter(str(report), args.label,
                     'The complete fixed held cohort; offline generations, not live task success.',
                     page_rewrite_seconds=math.inf)
    r.set_text('scope', 'Comparison scope',
               'All 100 held items in their original order. '
               + ('All views use the alternative initializer; these do not replace the main run\'s frozen controls. '
                  if args.condition == 'candidate' else
                  'Full/no-history controls use the initial pair. A trained-reader no-history view is a separate ablation. ')
               + 'No items are removed for answer quality.')
    r.set_trajectories('examples', 'Responses', 'Greedy next assistant turn; the stated token cap applies.', [])
    r.set_status(phase='loading paired checkpoint'); r.render(force=True)
    h = harness(ac_path, reader_path)
    ac = h.module_manager.get_ac_model(AC)
    study = ActivationContextStudyGenerator(h, AC)
    tokenizer_identity = {'vocabulary': digest(study.tokenizer.get_vocab()),
                          'chat_template': digest(study.tokenizer.chat_template),
                          'special_tokens': digest(study.tokenizer.special_tokens_map)}
    assert tokenizer_identity == json.loads((args.run / 'tokenizer_identity.json').read_text())
    held = build_items(study, args.material, selection['held'], task, run_identity['ratio'])
    expected_ids = [entry['choice']['id' if task == 'trajectory_qa' else 'window_id'] for entry in selection['held']]
    assert [item.item_id for item in held] == expected_ids
    config = replace(ActivationContextTrainingConfig(**json.loads((args.run / 'training_config.json').read_text())),
                     loss_kind='sft', no_context_penalty_weight=0, micro_batch_examples=1,
                     completion_max_new_tokens=args.max_new_tokens, resume_optimizer=False, resume_skip_steps=0)
    trainer = ActivationContextTrainer(h, config)
    trainer._training_deadline = started + args.seconds - 90
    progress = ActivationContextTrainingProgress(0, 1, ac.epoch_number, 0, 0, 0, 0, 0)
    if not result['rows']:
        result['rows'] = [{'item_id': item.item_id, 'root_id': item.info['root_id'],
                          'source': item.info['source'], 'history_tokens': item.info['source_tokens'],
                          'question': item.info.get('question', 'Continue the original task after the cut.'),
                          'gold': study.tokenizer.decode(item.assistant_token_ids[0], skip_special_tokens=False),
                          'views': {}} for item in held]
        write(report / 'cohort_inputs.json', [{'item_id': item.item_id,
               'history': item.in_context_messages[:item.in_context_start], 'tools': item.tools,
               'info': item.info} for item in held])
    assert [row['item_id'] for row in result['rows']] == expected_ids
    views = ({'frozen full history': study.transform_for_eval_reference(held, 'full_context'),
              'frozen no history': study.transform_for_eval_reference(held, 'no_context'),
              'initial AC': held} if args.condition == 'initial' else
             {'trained AC': held, 'trained reader without history': study.transform_for_eval_reference(held, 'no_context')})
    if args.condition == 'candidate':
        views = {'candidate full history': study.transform_for_eval_reference(held, 'full_context'),
                 'candidate no history': study.transform_for_eval_reference(held, 'no_context'),
                 'candidate AC': held}

    def publish(phase, name=None, completed=None):
        r.status.clear()
        r.set_status(phase=phase, view=name or '', completed=f'{completed}/100' if completed is not None else '',
                     token_cap=args.max_new_tokens, elapsed_minutes=f'{(time.monotonic()-started)/60:.1f}')
        r.widgets['examples']['items'] = [
            {'title': row['item_id'], 'state': 'finished' if all('response' in row['views'].get(view, {}) for view in views) else 'running',
             'stats': {'source': row['source'], 'history tokens': row['history_tokens'],
                       **{view + ' NLL': round(data['gold_nll'], 4) for view, data in row['views'].items()}},
             'prompt': row['question'], 'steps': [{'role': 'Recorded target', 'turn': 0, 'content': row['gold'], 'calls': [], 'results': []}] +
             [{'role': view, 'turn': 0, 'content': data['response'], 'calls': [], 'results': []}
              for view, data in row['views'].items() if 'response' in data]}
            for row in result['rows']]
        write(result_path, result)
        r.render(force=True)
        print(json.dumps({'phase': phase, 'view': name, 'completed': completed}), flush=True)

    variables = trainer._autotuning_variables(ac)
    h.module_manager.ensure_lora(READER)
    ac.prepare()
    try:
        with configure_fla_runtime(variables['fla_config']), torch.no_grad():
            for name, items in views.items():
                if STOP:
                    raise _TrainingTimeLimit()
                if name not in result['views']:
                    publish('scoring fixed cohort', name)
                    summary = trainer.eval(AC, items, release=False)
                    for original, item, row, score in zip(held, items, result['rows'], summary['per_item']):
                        assert item.item_id == score['item_id']
                        positions = original.info['answer_positions'] if task == 'trajectory_qa' else range(len(score['target_logp']))
                        row['views'][name] = {'gold_nll': -sum(score['target_logp'][i] for i in positions) / len(positions)}
                    result['views'][name] = {'cohorts': cohort_metrics(summary, held, task), 'complete': False}
                    publish('scored fixed cohort', name)
                for index, (item, row) in enumerate(zip(items, result['rows'])):
                    if 'response' in row['views'][name]:
                        continue
                    if STOP:
                        raise _TrainingTimeLimit()
                    sample = trainer._sample_completions(ac, [item], progress)[0]
                    row['views'][name]['response'] = sample['student']
                    publish('decoding fixed cohort', name, index + 1)
                result['views'][name]['complete'] = True
                publish('view complete', name, 100)
    except _TrainingTimeLimit:
        result['status'] = 'paused for continuation'
        publish('paused for continuation')
        return 75
    result['status'] = 'complete'
    publish('complete')
    r.finish()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--condition', choices=['initial', 'trained', 'candidate'], required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--match-selection', type=Path)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--seconds', type=int, default=6300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    raise SystemExit(main(args))
