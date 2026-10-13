"""Private real-workload execution tuning; no quality training or checkpoint promotion."""
import argparse
import json
import math
import os
import pickle
from pathlib import Path
import time


def main(args):
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import tune_ac_execution
    from activation.common.reporting import HtmlReporter
    from run_qa import AC, harness, write
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['ACTIVATION_SYNC_ROOT'] = str(args.output.resolve())
    report = HtmlReporter(str(args.output / 'report'), 'AC execution performance',
        'Real QA histories · identical objective and logical eight-question update · disposable probes',
        page_rewrite_seconds=math.inf)
    report.set_status(phase='preparing representative workload', hardware=args.label)
    report.set_text('scope', 'What this measures',
        'Forward/backward, gradient clipping, Adam and moment transfers. Includes the longest distinct histories in the fixture. '
        'Reference preparation, reporting and startup are measured separately. No quality/convergence claim.')
    report.set_text('policy', 'Fastest validated execution policy', 'Measurements pending.')
    columns = ['Reader batch', 'Questions/backward', 'Shared encoding', 'Encoder checkpoint from',
        'Reader checkpoint from', 'Adam moments', 'Questions/hour', '8k training hours', 'Peak GPU GiB', 'Result']
    report.initialize_table('candidates', 'Measured execution choices',
        'Same questions and loss. Projected hours exclude preparation, reports, checkpoints and queueing. '
        'The cases include a demanding tail; this is not a full-population speed estimate.', columns)
    report.set_text('receipt', 'Validation details', 'Native baseline, repeat and both gradient groups must pass before selection.')
    report.render(force=True)
    started = time.monotonic()
    pair = json.loads(args.initial_pair.read_text())
    h = harness(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = h.module_manager.get_ac_model(AC)
    config = ActivationContextTrainingConfig(loss_kind='sft', learning_rate_ac=5e-5,
        learning_rate_target_lora=2e-5, examples_per_update=8, num_epochs=1,
        logits_chunk_tokens=64, gradient_checkpointing_min_tokens=0,
        no_context_penalty_weight=1, no_context_penalty_fraction=.25,
        completion_samples=0, seed=20260924, schedule_total_updates=1000)
    trainer = ActivationContextTrainer(h, config)
    with args.items.open('rb') as stream:
        items = pickle.load(stream)
    examples = [trainer.build_example(ac, item) for item in items]
    write(args.output / 'workload.json', [{'id': e.item.item_id, 'source_tokens': e.item.info.get('source_tokens'),
        'reader_tokens': len(e.student_ids), 'target_tokens': len(e.target_ids), 'weight': e.item.weight} for e in examples])
    report.set_status(phase='measuring baseline, repeat and execution policies', hardware=args.label,
                      questions=len(examples), budget_seconds=args.seconds)
    report.render(force=True)
    def progress(record):
        report.status.clear()
        report.set_status(phase=record['phase'].replace('_', ' '), hardware=args.label,
            measured_cases=f"{record['completed_cases']}/{record['total_cases']} planned checks",
            elapsed_minutes=round(record['elapsed_seconds'] / 60, 1))
        if record.get('policy'):
            report.set_text('policy', 'Policy currently being measured', json.dumps(record['policy'], indent=2))
        report.render(force=True)
    try:
        result = tune_ac_execution(h, AC, examples, config, seconds=args.seconds, progress_callback=progress)
        result['total_seconds_including_setup'] = time.monotonic() - started
        write(args.output / 'result.json', result)
        report.set_text('policy', 'Fastest validated execution policy', json.dumps(result['policy'], indent=2))
        rows = []
        for candidate in result['receipt']['candidates']:
            policy = candidate['policy']; pace = candidate.get('seconds_per_question')
            rows.append(dict(zip(columns, [policy['reader_batch'], policy['graph_examples'], policy['reuse_encodings'],
                policy['side_checkpoint_tokens'], policy['target_checkpoint_tokens'], policy['optimizer_moments'],
                round(3600 / pace) if pace else None, round(8000 * pace / 3600, 2) if pace else None,
                round(max((c['peak_memory_bytes'] for c in candidate['cases'] + candidate.get('capacity_cases', [])), default=0) / 2**30, 2),
                'Passed' if candidate['passed'] else candidate.get('error_type', 'Numerical or memory gate failed')])))
        report.widgets['candidates']['rows'] = rows
        report.set_text('receipt', 'Validation details',
            'Each passing choice matches a native singleton baseline and its repeat for loss and both gradient groups. '
            'All cases use the same penalty choices and real Adam state. The full receipt is in ../result.json. '
            'Zero checkpoint threshold means checkpoint every sequence; a larger value skips shorter sequences.')
        report.set_status(phase='complete', hardware=args.label,
            elapsed_minutes=round((time.monotonic() - started) / 60, 1))
        print(json.dumps(result, default=str), flush=True)
    except BaseException as error:
        report.set_status(phase='probe stopped', error=str(error), hardware=args.label)
        raise
    finally:
        report.render(force=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--items', type=Path, required=True)
    parser.add_argument('--initial-pair', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--seconds', type=float, default=1200)
    main(parser.parse_args())
