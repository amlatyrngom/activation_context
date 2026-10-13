"""Read-only reconstruction retention on the original P2 held passages."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    from activation.common.reporting import _write_atomic
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, json.dumps(value, indent=2, allow_nan=False))


def summary(rows):
    tokens = sum(row['positions'] for row in rows)
    if not tokens:
        return None
    return {
        'passages': len(rows), 'target_tokens': tokens,
        'token_weighted_nll': sum(row['gold_nll'] * row['positions'] for row in rows) / tokens,
        'passage_mean_nll': statistics.mean(row['gold_nll'] for row in rows),
        'token_accuracy': sum(row['token_accuracy'] * row['positions'] for row in rows) / tokens,
    }


def report(folder, manifest, phase, condition, completed=0):
    from activation.common.reporting import HtmlReporter
    r = HtmlReporter(str(folder), 'Reconstruction retained after QA training',
        'The original 420 held passages, 60 per source, at 1/16 compression. All paired checkpoints '
        'are remeasured on this L40S. No training or checkpoint changes.', page_rewrite_seconds=float('inf'))
    r.initialize_table('quality', 'Pure reconstruction',
        'NLL and next-token accuracy are token-weighted. Completed checkpoints use all 420 passages; '
        'partial checkpoint metrics remain blank. Accuracy uses the gold preceding tokens. '
        'Scored assistant tokens include the chat end token.',
        ['checkpoint', 'status', 'passages', 'NLL', 'NLL increase', 'token accuracy', 'accuracy change'])
    r.initialize_table('sources', 'Results by source', 'All 60 held passages per source.',
        ['checkpoint', 'source', 'NLL', 'token accuracy'])
    r.set_text('scope', 'What this measures',
        'Original reconstruction prompt and original held source text; no QA question, answer prompt, '
        'distractors or synthesis. Encoder, readouts and reader travel as their trained pair. '
        'This measures retention in the complete QA-tuned system, not the encoder alone. '
        'Four fixed short passages also receive greedy reconstructions, capped at 320 tokens. '
        'The original historical P2 score is context only; all deltas use the fresh P2 measurement.')
    r.set_trajectories('samples', 'Actual greedy reconstructions',
        'Same four short passages for every checkpoint. Inspect omissions and changes relative to the source.', [])
    baseline_path = folder / 'p2_reconstruction' / 'result.json'
    baseline = read(baseline_path).get('summary') if baseline_path.exists() else None
    trajectories = []
    for name, cell in manifest['conditions'].items():
        path = folder / name / 'result.json'
        result = read(path) if path.exists() else {}
        measured = result.get('summary')
        r.add_data_point('quality', {
            'checkpoint': cell['label'], 'status': result.get('status', 'pending'),
            'passages': f"{len(result.get('per_item', []))}/420",
            'NLL': f"{measured['token_weighted_nll']:.5f}" if measured else '',
            'NLL increase': f"{measured['token_weighted_nll'] - baseline['token_weighted_nll']:+.5f}" if measured and baseline else '',
            'token accuracy': f"{100 * measured['token_accuracy']:.2f}%" if measured else '',
            'accuracy change': f"{100 * (measured['token_accuracy'] - baseline['token_accuracy']):+.2f} pp" if measured and baseline else '',
        })
        for source, measured_source in result.get('by_source', {}).items():
            r.add_data_point('sources', {'checkpoint': cell['label'], 'source': source,
                'NLL': f"{measured_source['token_weighted_nll']:.5f}",
                'token accuracy': f"{100 * measured_source['token_accuracy']:.2f}%"})
        for sample in result.get('samples', []):
            trajectories.append({'title': cell['label'] + ' · ' + sample['source'], 'state': 'finished',
                'stats': f"{sample['generated_tokens']} tokens; " + ('EOS' if sample['ended_at_eos'] else 'token cap'),
                'prompt': sample['passage'], 'file': f'{name}/result.json',
                'steps': [{'role': 'assistant', 'turn': 1, 'content': sample['student'], 'calls': [], 'results': []}]})
    r.set_trajectories('samples', 'Actual greedy reconstructions',
        'Same four short passages for every checkpoint; source above and reconstruction below.', trajectories)
    r.set_status(phase=phase, checkpoint=manifest['conditions'][condition]['label'], passages=f'{completed}/420')
    if (folder / 'report.tressoir.html').exists():
        r.last_page_widgets = list(r.widgets)
        r.last_page_time = time.time()
    r.render(force=True)


def main(args):
    began = time.monotonic()
    manifest = read(args.manifest)
    cell = manifest['conditions'][args.condition]
    folder = args.output / args.condition
    output = folder / 'result.json'
    def progress(phase, completed=0):
        value = {'phase': phase, 'condition': args.condition, 'completed_passages': completed,
                 'total_passages': 420, 'elapsed_seconds': round(time.monotonic() - began, 1)}
        write(folder / 'status.json', value)
        report(args.output, manifest, phase, args.condition, completed)
        print(json.dumps(value), flush=True)

    if sha(__file__) != manifest['worker_sha256'] or sha(args.passages) != manifest['passages_sha256']:
        raise ValueError('Frozen reconstruction evaluator or held material changed')
    for name, expected in manifest['helper_sha256'].items():
        if Path(name).name != name or sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('Pinned decoding helper changed: ' + name)
    passages = read(args.passages)['passages']
    if Counter(p['source'] for p in passages) != Counter(dict.fromkeys(manifest['sources'], 60)):
        raise ValueError('Expected the original 420 non-wiki held passages')
    for path, expected in cell.get('publication_sha256', {}).items():
        if sha(path) != expected:
            raise ValueError('Selected final publication changed: ' + path)
    pair = cell['pair']
    ac_path, reader_path = Path(pair['ac_checkpoint']), Path(pair['reader_checkpoint'])
    if ac_path.name != 'epoch_001' or reader_path.name != 'epoch_001':
        raise ValueError('Only preselected completed epoch pairs are evaluated')
    marker_path = ac_path.parent / 'completed_epoch.json'
    marker = read(marker_path)
    root = ac_path.parents[2]
    if (marker['epoch_number'] != 1 or (root / marker['ac_checkpoint']).resolve() != ac_path.resolve()
            or (root / marker['target_lora_checkpoint']).resolve() != reader_path.resolve()):
        raise ValueError('Completed checkpoint marker does not bind this pair')
    payloads = {'ac_modules': sha(ac_path / 'ac_modules.pt'), 'completed_marker': sha(marker_path)}
    for label, directory, base in [('encoder', ac_path / 'side_lora', 'Qwen/Qwen3.5-2B'),
                                   ('reader', reader_path, 'Qwen/Qwen3.5-9B')]:
        config = read(directory / 'adapter_config.json')
        if config['r'] != 128 or config['base_model_name_or_path'] != base:
            raise ValueError('Wrong paired adapter: ' + label)
        weights = [directory / name for name in ('adapter_model.safetensors', 'adapter_model.bin')
                   if (directory / name).is_file()]
        if len(weights) != 1:
            raise ValueError('Missing or ambiguous adapter weights')
        payloads[label + '_weights'] = sha(weights[0])
        payloads[label + '_config'] = sha(directory / 'adapter_config.json')
    if cell.get('payload_sha256') and any(payloads[k] != v for k, v in cell['payload_sha256'].items()):
        raise ValueError('Original archived P2 payload identity changed')
    if cell.get('training_result'):
        training = read(cell['training_result'])
        if (training.get('stop_reason') != 'complete'
                or training.get('completed_global_updates') != cell['completed_stage_updates']
                or Path(training['checkpoint']).resolve() != ac_path.resolve()
                or Path(training['target_lora_checkpoint']).resolve() != reader_path.resolve()):
            raise ValueError('Expected this completed QA stage and its own final pair')
    progress('loading paired checkpoint')
    import torch
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.ac_model import ActivationContextModelConfig, ActivationContextStudyGenerator
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    import activation.ac_model.ac_model_study as study_module
    source = {key: sha(module.__file__) for key, module in [('ac_model', ac_module),
        ('trainer', training_module), ('utils', utils_module), ('study', study_module)]}
    if source != manifest['source_sha256']:
        raise ValueError('Runtime source differs from frozen QA source')
    if torch.cuda.get_device_name() != 'NVIDIA L40S':
        raise ValueError('The matched comparison is reserved for L40S')
    torch.set_num_threads(4)
    torch.manual_seed(20260915)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs={
        'qwen4b': ModelConfig('qwen4b', 'Qwen/Qwen3.5-9B'),
        'side_base': ModelConfig('side_base', 'Qwen/Qwen3.5-2B')}))
    h.module_manager.register_lora('rag_reader', 'qwen4b', rank=128, dropout=0, checkpoint_path=str(reader_path))
    h.module_manager.register_ac_model(ActivationContextModelConfig(
        'rag_poc', 'side_base', 'rag_side', 'qwen4b', 'rag_reader', min_view_rows=2,
        default_compression_ratio=1/16, checkpoint_path=str(ac_path),
        intervention_frequency=-1, input_pooling_stride=0, mixer_layers=2))
    ac = h.module_manager.get_ac_model('rag_poc')
    ac.prepare()
    peft = h.module_manager.ensure_lora('rag_reader')
    study = ActivationContextStudyGenerator(h, 'rag_poc')
    items = study.generate_reconstruction_samples(passages, ratio=1/16, variant='reconstruct', seed=20260916)
    if len(items) != 420:
        raise ValueError('Reconstruction generation dropped passages')
    items = [replace(item, kind=item.dataset_id) for item in items]
    items.sort(key=lambda item: item.info['passage_tokens'] > 288)
    trainer = ActivationContextTrainer(h, ActivationContextTrainingConfig(
        loss_kind='sft', no_context_penalty_weight=0, micro_batch_examples=2,
        logits_chunk_tokens=256, max_example_tokens=16384, gradient_checkpointing_min_tokens=0,
        completion_samples=4, completion_max_new_tokens=320, seed=20260915))
    variables = trainer._autotuning_variables(ac)
    input_rows = []
    for item in items:
        example = trainer.build_example(ac, item)
        if len(example.part_requests) != 1 or len(example.spans) != 1:
            raise ValueError('Reconstruction must use a single flat source passage')
        request = example.part_requests[0]
        n = len(tokenize_with_parts(ac.side.tokenizer, request.messages, [], ac.pad_id, tools=request.tools)[0])
        count = example.spans[0][1] - example.spans[0][0]
        if request.compression_ratio != 1/16 or count != ac.num_view_rows(n, 1/16):
            raise ValueError('Reconstruction compression changed')
        input_rows.append({'item_id': item.item_id, 'encoder_tokens': n, 'rows': count,
                           'target_tokens': len(item.assistant_token_ids[0])})
    identity = {'manifest_sha256': sha(args.manifest), 'source_sha256': source, 'pair': pair,
        'payload_sha256': payloads,
        'model_revisions': {key: getattr(model.model.config, '_commit_hash', None)
            for key, model in [('encoder', ac.side), ('reader', ac.target)]},
        'inputs_sha256': digest([asdict(item) for item in items]), 'row_budget_sha256': digest(input_rows),
        'runtime': {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'fla': variables['provenance']['fla'], 'logits_chunk_tokens': variables['logits_chunk_tokens'],
            'tf32': torch.backends.cuda.matmul.allow_tf32,
            'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction},
        'optimizer_updates': 0, 'training_config': asdict(trainer.config)}
    if args.condition != 'p2_reconstruction':
        baseline = read(args.output / 'p2_reconstruction' / 'result.json')['identity']
        for key in ('model_revisions', 'inputs_sha256', 'row_budget_sha256', 'runtime', 'training_config'):
            if identity[key] != baseline[key]:
                raise ValueError('Matched reconstruction condition changed: ' + key)
    if output.exists():
        result = read(output)
        if result['identity'] != identity:
            raise ValueError('Partial reconstruction evaluation identity changed')
        if result['status'] == 'complete':
            return
    else:
        result = {'identity': identity, 'status': 'scoring', 'per_item': [], 'samples': [], 'row_budgets': input_rows}
    if [r['item_id'] for r in result['per_item']] != [item.item_id for item in items[:len(result['per_item'])]]:
        raise ValueError('Partial reconstruction row order changed')
    torch.cuda.reset_peak_memory_stats(ac.device)
    with configure_fla_runtime(variables['fla_config']):
        for offset in range(len(result['per_item']), len(items), 20):
            progress('scoring reconstruction', offset)
            chunk = trainer._evaluate(ac, items[offset:offset+20], ac.device)
            if chunk['dropped_too_long'] or chunk['items'] != len(items[offset:offset+20]):
                raise ValueError('Reconstruction evaluation dropped passages')
            result['per_item'].extend(chunk['per_item'])
            write(output, result)
        result.update(summary=summary(result['per_item']), status='decoding',
            by_source={source: summary([row for row in result['per_item'] if row['kind'] == source])
                for source in manifest['sources']})
        write(output, result)
        raw_eos = peft.generation_config.eos_token_id
        eos = [raw_eos] if isinstance(raw_eos, int) else list(raw_eos or [])
        eot = ac.target.model_config.model_description.eot_token
        if eot is None:
            raise ValueError('Missing reader end-of-turn token')
        eos.append(ac.target.tokenizer.convert_tokens_to_ids(eot))
        from evaluate_model_comparison_cohort import sample_with_token_ids
        progress('decoding four fixed short passages', 420)
        selected = [next(item for item in items if item.kind == source and item.info['passage_tokens'] <= 288)
                    for source in ('swe', 'code', 'docs', 'math')]
        stamp = ActivationContextTrainingProgress(0, 0, 0, 0, 0, 0, 0, time.monotonic()-began)
        done = {sample['item_id'] for sample in result['samples']}
        for item in selected:
            if item.item_id in done:
                continue
            sample, captured = sample_with_token_ids(trainer, ac, item, stamp, eos)
            text = item.in_context_messages[-1]['content']
            result['samples'].append({**sample, **captured, 'source': item.kind, 'passage': text,
                                      'exact_text_match': sample['student'] == text})
            write(output, result)
            progress('decoding four fixed short passages', 420)
    result.update(status='complete', elapsed_seconds=time.monotonic()-began,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(ac.device)/2**30)
    write(output, result)
    progress('checkpoint complete', 420)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--passages', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition', required=True)
    main(parser.parse_args())
