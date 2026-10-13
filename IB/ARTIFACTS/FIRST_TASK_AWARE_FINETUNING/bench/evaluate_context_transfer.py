"""Fixed development100, matched 2k/32k evidence; final trained pairs, NLL only."""
import argparse
import math
from pathlib import Path
import statistics
import time

from evaluate_model_comparison_cohort import load_inputs, checkpoint, load_module
from evaluate_model_confirmation import shared_fingerprint
from evaluate_qualitative_cohort import read, write, digest as file_sha
from evaluate_long_cross import answer_scores, digest

CONDITIONS = ('oracle_27b_16k_r16', 'oracle_27b_16k_r8', 'encoder4b_oracle27b_16k_r16')
CONTEXTS = {'2k': (2048, 4), '32k': (32768, 64)}
HELPERS = ('evaluate_model_comparison_cohort.py', 'evaluate_model_confirmation.py',
           'evaluate_qualitative_cohort.py', 'evaluate_long_cross.py')


def input_rows(trainer, ac, items):
    """Compare reported budgets with the actual resolved encoder requests and reader spans."""
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    rows = []
    for item in items:
        example = trainer.build_example(ac, item)
        if len(example.part_requests) != 1 or len(example.spans) != 1:
            raise ValueError('Expected the trained merged-passage layout')
        request = example.part_requests[0]
        length = len(tokenize_with_parts(ac.side.tokenizer, request.messages, [], ac.pad_id, tools=request.tools)[0])
        resolved = ac.num_view_rows(length, request.compression_ratio)
        span = example.spans[0][1]-example.spans[0][0]
        uncapped = max(ac.config.min_view_rows, math.ceil(length*request.compression_ratio))
        if (length != item.info['encoded_source_tokens'] or resolved != item.info['compressed_rows']
                or span != resolved or request.compression_ratio != item.info['ratio']):
            raise ValueError('Builder budget and resolved encoder/reader inputs disagree: '+item.item_id)
        rows.append({'item_id': item.item_id, 'encoder_tokens': length, 'compressed_rows': resolved,
            'uncapped_rows': uncapped, 'row_cap_applied': uncapped > ac.config.max_view_rows,
            'rows_per_encoder_token': resolved/length, 'evidence_tokens': item.info['evidence_source_tokens'],
            'passages': item.info['passages'], 'reader_tokens': len(example.student_ids),
            'answer_tokens': len(item.info['answer_positions'])})
    return rows


def summarize_rows(rows):
    keys = ('encoder_tokens', 'compressed_rows', 'evidence_tokens', 'passages', 'reader_tokens')
    return {**{key: {'min': min(r[key] for r in rows), 'mean': statistics.mean(r[key] for r in rows),
                    'max': max(r[key] for r in rows), 'total': sum(r[key] for r in rows)} for key in keys},
            'capped_items': sum(r['row_cap_applied'] for r in rows),
            'aggregate_rows_per_encoder_token': sum(r['compressed_rows'] for r in rows)/sum(r['encoder_tokens'] for r in rows)}


def main(args):
    started = time.monotonic()
    cell = args.condition+'__'+args.context
    output = args.output/cell/'result.json'
    status = args.output/cell/'status.json'
    def progress(phase, **extra):
        value = {'cell': cell, 'phase': phase, 'elapsed_this_attempt_seconds': round(time.monotonic()-started, 1), **extra}
        write(status, value); print(value, flush=True)
    progress('checking frozen development selection and final checkpoint')
    material, selection = load_inputs(args)
    pair, encoder, ratio, update, hashes = checkpoint(args, selection)
    budget, passages = CONTEXTS[args.context]
    identity = {'condition': args.condition, 'context': args.context, 'encoder': encoder, 'ratio': ratio,
        'training_unique_questions': 16000, 'training_epochs': 1, 'final_update': update,
        'paired_checkpoint': pair, 'checkpoint_sha256': hashes,
        'selection_sha256': file_sha(args.selection), 'pinned_sha256': selection['pinned_sha256'],
        'evaluator_sha256': file_sha(__file__),
        'helper_sha256': {name: file_sha(Path(__file__).with_name(name)) for name in HELPERS},
        'microbatch': 1, 'evidence_budget_tokens': budget, 'max_passages': passages,
        'layout': 'merged', 'views': 'AC and the same final paired reader with full/no context',
        'scope': 'Fixed development100:50 per generator. Added-distractor robustness; required source stays whole. No training, decodes, checkpoint selection or confirmation access.',
        'size_comparison': 'P4 and P2 use different pretrained AC/reader pairs; not isolated encoder capacity.'}
    if output.exists():
        result = read(output)
        if result['identity'] != identity:
            raise ValueError('Existing context-transfer evaluation identity changed')
        if result['status'] == 'complete':
            progress('already complete'); return
    else:
        result = {'status': 'running', 'identity': identity, 'views': {}}
    bench = load_module('context_transfer_builder', args.bench)
    factory = bench.harness if encoder == '2B' else load_module('context_transfer_harness4b', args.harness_4b).harness
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import _TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model_training as implementation
    if file_sha(implementation.__file__) != selection['trainer_sha256']:
        raise ValueError('Frozen trainer bytes changed')
    progress('loading final paired checkpoint')
    harness = factory(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = harness.module_manager.get_ac_model(bench.AC); ac.prepare()
    harness.module_manager.ensure_lora(bench.READER)
    selected = set(selection['selected_ids'])
    selected_rows = [r for r in material['held'] if r['example']['example_id'] in selected]
    progress('building fixed100 evidence', evidence_budget_tokens=budget)
    study, _, held = bench.material_items(harness, {'train': material['train'][:1], 'held': selected_rows,
        'retrieval': material['retrieval']}, evidence_tokens=budget, max_passages=passages, ratio=ratio)
    if [item.item_id for item in held] != selection['selected_ids']:
        raise ValueError('Selected question order changed')
    for item, row in zip(held, selected_rows):
        if any(unit['text'] not in item.info['passages_text'] for unit in row['example']['study_source']['units']):
            raise ValueError('Required source was not retained whole')
    full = study.transform_for_eval_reference(held, 'full_context')
    noctx = study.transform_for_eval_reference(held, 'no_context')
    common = shared_fingerprint(held, full, noctx)
    ac_hash = digest([(item.item_id, item.ac_messages) for item in held])
    if result.get('shared_input_fingerprint', common) != common or result.get('ac_input_fingerprint', ac_hash) != ac_hash:
        raise ValueError('Rebuilt context-transfer inputs changed')
    for condition in CONDITIONS:
        path = args.output/(condition+'__'+args.context)/'result.json'
        if path.exists():
            other = read(path)
            if other.get('shared_input_fingerprint', common) != common or other['identity']['selection_sha256'] != identity['selection_sha256']:
                raise ValueError('Conditions differ in evidence or selected questions')
    trainer = ActivationContextTrainer(harness, ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0,
        micro_batch_examples=1, max_example_tokens=72000, completion_samples=0, seed=bench.SEED))
    rows = input_rows(trainer, ac, held)
    cohorts = {name: [ident for ident in ids if ident in selected] for name, ids in material['held_cohorts'].items()}
    result.update(shared_input_fingerprint=common, ac_input_fingerprint=ac_hash, input_rows=rows,
        row_budget=summarize_rows(rows), max_view_rows=ac.config.max_view_rows, selected_ids=selection['selected_ids'],
        evidence=[{'item_id': item.item_id, 'question': item.info['question'], 'gold': item.info['answer'],
                   'passages_text': item.info['passages_text']} for item in held])
    variables = trainer._autotuning_variables(ac)
    trainer._training_deadline = time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    write(output, result)
    try:
        with configure_fla_runtime(variables['fla_config']), torch.no_grad():
            for name, items in {'AC': held, 'full context': full, 'no context': noctx}.items():
                if name in result['views']:
                    continue
                progress('scoring '+name, questions=100)
                torch.cuda.synchronize(ac.device); torch.cuda.reset_peak_memory_stats(ac.device); began = time.monotonic()
                scored = answer_scores(trainer._evaluate(ac, items, ac.device), items, held)
                torch.cuda.synchronize(ac.device)
                by_id = {r['item_id']: r for r in scored['per_item']}
                result['views'][name] = {**scored,
                    'cohorts': {bank: {'answer_nll': statistics.mean(by_id[ident]['answer_nll'] for ident in ids),
                                      'questions': len(ids)} for bank, ids in cohorts.items()},
                    'seconds': time.monotonic()-began, 'peak_allocated_gib': torch.cuda.max_memory_allocated(ac.device)/2**30}
                write(output, result)
    except _TrainingTimeLimit:
        result['status'] = 'partial'; write(output, result)
        progress('time limit; resume with the same invocation'); return
    if trainer.optimizers or trainer.updates_done:
        raise AssertionError('Context-transfer evaluation unexpectedly trained')
    result.update(status='complete', optimizer_updates=0, elapsed_this_attempt_seconds=round(time.monotonic()-started,1))
    write(output,result); progress('complete', questions=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'bench', 'harness-4b', 'material-dir', 'reports', 'selection', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--context', choices=CONTEXTS, required=True)
    parser.add_argument('--seconds', type=int, default=1500)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write(args.output/(args.condition+'__'+args.context)/'status.json', {'phase': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
