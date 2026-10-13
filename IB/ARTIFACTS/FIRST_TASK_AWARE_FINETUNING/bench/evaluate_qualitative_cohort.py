"""Score/decode all100 old development items with frozen P2 or a saved AC pair.

One process per checkpoint identity. P2 produces both baseline views; control500
and control900 each produce only their AC view. NLL uses original all100 batch2
partners; free decoding starts before any answer token and is greedy/max160.
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def parsed_answer(study, text):
    try:
        _, calls = study.dialect.parse(text)
        for call in calls:
            if call['name'] == 'submit_answer':
                return str(call['arguments'].get('answer', '')), 'submit_answer'
        name, arguments = study.dialect.preferred_format.parse(text, None)
        if name == 'submit_answer':
            return str(arguments.get('answer', '')), 'submit_answer'
    except (ValueError, KeyError, TypeError):
        pass
    match = re.search(r'<parameter=answer>\s*(.*?)\s*</parameter>', text, re.S)
    return (match.group(1), 'xml_recovered') if match else (text, 'unparsed_raw_response')


def main(args):
    started = time.monotonic()
    output = args.output_dir / (args.condition + '.json')
    status = args.output_dir / (args.condition + '_status.json')
    def progress(phase, **extra):
        record = {'phase': phase, 'condition': args.condition,
                  'elapsed_seconds': round(time.monotonic() - started, 1), **extra}
        write(status, record)
        print(json.dumps(record), flush=True)
    progress('checking frozen identities')
    spec = importlib.util.spec_from_file_location('original_hill_bench', args.bench)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    frozen = args.condition == 'frozen_p2'
    update = 0 if frozen else int(args.condition.removeprefix('control_'))
    ac_path = args.p2_root / 'AC_MODELS/rag_poc/epoch_001' if frozen else args.checkpoint_root / 'selected' / f'update_{update:05d}' / 'ac'
    reader_path = args.p2_root / 'LORAS/rag_reader/epoch_001' if frozen else args.checkpoint_root / 'selected' / f'update_{update:05d}' / 'reader'
    panel = read(args.run_dir / f'panel_{update:05d}.json')
    recipe = json.loads(next(w['text'] for w in read(args.run_dir / 'report_data.json')['widgets'] if w['name'] == 'recipe'))
    material = read(args.material)
    ids = read(args.run_dir / 'material_ids.json')
    material_sha = digest(args.material)
    assert material_sha == recipe['material_sha256'], 'Frozen material changed'
    assert len(ids['held']) == len(set(ids['held'])) == 100
    assert panel['progress']['global_step'] == update
    for split in ('train', 'held'):
        by_id = {row['example']['example_id']: row for row in material[split]}
        material[split] = [by_id[ident] for ident in ids[split]]
    files = {'ac_modules': ac_path / 'ac_modules.pt',
             'encoder_lora': ac_path / 'side_lora/adapter_model.safetensors',
             'reader_lora': reader_path / 'adapter_model.safetensors'}
    identity = {'condition': args.condition, 'update': update,
        'reader_identity': 'Frozen P2 epoch1 reader at start of QA training' if frozen else f'Paired trained reader at control update{update}',
        'ac_checkpoint': str(ac_path), 'reader_checkpoint': str(reader_path),
        'checkpoint_sha256': {name: digest(path) for name, path in files.items()},
        'material_sha256': material_sha, 'material_ids_sha256': digest(args.run_dir / 'material_ids.json'),
        'bench_sha256': digest(args.bench), 'evaluator_sha256': digest(__file__),
        'cohort': 'All100 original hill-control development items, in original material_ids order; no selection by answer quality',
        'nll': 'Mean gold-answer token NLL; every token is conditioned on earlier gold tokens. Tool syntax excluded.',
        'decoding': 'Greedy free decoding from the assistant prefix before any answer token; enable_thinking=False; max_new_tokens=160; one request at a time as original sampler.',
        'nll_microbatch': 2, 'nll_batching': 'Original complete100-item list; trainer sorts by sequence length and retains the original batch partners.'}
    if output.exists():
        result = read(output)
        assert result['identity'] == identity, 'Existing output belongs to another evaluator/input/checkpoint identity'
        if result['status'] == 'complete':
            progress('already complete', items=100)
            return
    else:
        result = {'status': 'running', 'identity': identity, 'views': {}, 'rows': []}
    progress('loading fixed checkpoint', reader=identity['reader_identity'])
    h = bench.harness(ac_path, reader_path)
    ac = h.module_manager.get_ac_model(bench.AC)
    ac.prepare()
    h.module_manager.ensure_lora(bench.READER)
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
    from activation.autotuners import configure_fla_runtime
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = replace(ActivationContextTrainingConfig(**read(args.run_dir / 'training_config.json')),
                     micro_batch_examples=2, no_context_penalty_weight=0, completion_max_new_tokens=160)
    trainer = ActivationContextTrainer(h, config)
    variables = trainer._autotuning_variables(ac)
    trainer._training_deadline = time.monotonic() + args.seconds
    progress('building exact original evidence and distractors')
    study, _, held = bench.material_items(h, material)
    assert [item.item_id for item in held] == ids['held']
    for item in held:
        item.info['chat_template_kwargs'] = {'enable_thinking': False}
    view_items = {'frozen_full_context': study.transform_for_eval_reference(held, 'full_context'),
                  'frozen_no_context': study.transform_for_eval_reference(held, 'no_context')} if frozen else {args.condition: held}
    source_rows = {row['example']['example_id']: row for row in material['held']}
    if not result['rows']:
        for item in held:
            result['rows'].append({'item_id': item.item_id, 'domain': item.kind,
                'question': item.info['question'], 'gold': item.info['answer'],
                'required_source': source_rows[item.item_id]['example']['study_source'],
                'full_evidence': item.info['passages_text'], 'views': {}})
    assert [row['item_id'] for row in result['rows']] == ids['held']
    checkpoint_progress = ActivationContextTrainingProgress(**panel['progress'])
    with configure_fla_runtime(variables['fla_config']), torch.no_grad():
        for name, items in view_items.items():
            if name not in result['views']:
                progress('scoring all100 with original batch2', view=name)
                summary = trainer._evaluate(ac, items, ac.device)
                old_name = 'held full context' if name == 'frozen_full_context' else 'held no context' if name == 'frozen_no_context' else 'held AC'
                old_rows = {row['item_id']: row for row in panel['summaries'][old_name]['per_item']}
                differences = []
                for original, item, row, record in zip(held, items, summary['per_item'], result['rows']):
                    assert item.item_id == row['item_id']
                    positions = original.info['answer_positions']
                    nll = -sum(row['target_logp'][i] for i in positions) / len(positions)
                    previous = old_rows[row['item_id']]
                    old_nll = -sum(previous['target_logp'][i] for i in positions) / len(positions)
                    differences.append(nll - old_nll)
                    record['views'][name] = {'answer_nll': nll, 'original_panel_answer_nll': old_nll,
                        'gold_token_scores': [{'text': study.tokenizer.decode([original.assistant_token_ids[0][i]], skip_special_tokens=False),
                                               'nll': -row['target_logp'][i]} for i in positions]}
                result['views'][name] = {'reader_identity': identity['reader_identity'],
                    'mean_answer_nll': sum(row['views'][name]['answer_nll'] for row in result['rows']) / 100,
                    'original_panel_max_abs_nll_difference': max(map(abs, differences)),
                    'original_panel_mean_nll_difference': sum(differences) / 100,
                    'complete': False}
                write(output, result)
            for index, (item, record) in enumerate(zip(items, result['rows'])):
                if 'response' in record['views'][name]:
                    continue
                progress('free decoding', view=name, completed=index, total=100)
                sample = trainer._sample_completions(ac, [item], checkpoint_progress)[0]
                answer, parsing = parsed_answer(study, sample['student'])
                record['views'][name].update(response=sample['student'], answer=answer, parsing=parsing,
                    generated_tokens=len(study.tokenizer.encode(sample['student'], add_special_tokens=False)))
                write(output, result)
            result['views'][name]['complete'] = True
            write(output, result)
    result['status'] = 'complete'
    result['elapsed_seconds'] = round(time.monotonic() - started, 1)
    write(output, result)
    progress('complete', items=100, views=list(view_items))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--material', type=Path, required=True)
    parser.add_argument('--p2-root', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--condition', choices=['frozen_p2', 'control_500', 'control_900'], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write(args.output_dir / (args.condition + '_status.json'),
              {'phase': 'failed', 'condition': args.condition, 'error': f'{type(error).__name__}: {error}'})
        raise
