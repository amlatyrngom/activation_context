"""Bounded train-only P2 batching proof; subprocesses never save model weights.

The source snapshot stays untouched. Only this worker's method binding changes.
CPU: --cpu-check. GPU: --material ... --output ... --p2-root ... --bench ...
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time

AC_SHA = 'aaeb3c68225d3092a024d44552b84480dc37cc8319d68c2ab1701fce70980bfd'
TRAINER_SHA = '38ca9885bfe35e47432352aa716381e53c18aa57748428eedf5f5f6b7fc6b5aa'
UTILS_SHA = '1b464ee4e8cafc6e7582c1c04e27bc95432cb8616b90f5bd90ad8b6c6fd5096c'
BUILDER_SHA = '524b54df83fa197f2737214166b5bbd57bdd16a4295320a9ce1920159b296527'
MATERIAL_SHA = '9eff31670c2206dd9bb1eeda841461c324e75068ea2b5afaacd7f74ed49c05a5'
PATCH_SHA = 'abfee2f20b6152671c74fbde6f2af1f275e06abeef051d513c9c1c75efa4b51e'
SEED = 20260921
CONDITIONS = {'pieces_original_65536': (True, False, 65536),
              'pieces_optimized_65536': (True, True, 65536),
              'pieces_optimized_16384': (True, True, 16384),
              'merged_optimized_65536': (False, True, 65536),
              'merged_optimized_16384': (False, True, 16384)}
MASK_BLOCK = '''    attention_mask = None
    if batch_size > 1:
        attention_mask = torch.zeros(batch_size, max_seq, device=device, dtype=torch.long)
        for row, length in enumerate(seq_lengths):
            attention_mask[row, :length] = 1
'''


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''): h.update(data)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')
    temporary.replace(path)


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def methods(module):
    original = module.ActivationContextModel._encode_prepared_batch
    source = textwrap.dedent(inspect.getsource(original))
    assert source.count(MASK_BLOCK) == 1
    modified = source.replace(MASK_BLOCK, '    attention_mask = None\n')
    old = 'and max_seq >= getattr(self, "_side_gradient_checkpointing_min_tokens", 0))'
    assert modified.count(old) == 1
    modified = modified.replace(old, 'and max_seq * batch_size >= getattr(self, "_side_gradient_checkpointing_min_tokens", 0))')
    scope = dict(module.__dict__)
    exec(compile(modified, '<piecewise_batching_v1>', 'exec'), scope)
    return original, scope['_encode_prepared_batch']


@contextmanager
def method_scope(ac, method, geometry):
    """Record actual prepared batches; all temporary instance bindings restored."""
    import types
    sentinel = object()
    previous = ac.__dict__.get('_encode_prepared_batch', sentinel)
    def measured(self, items, indexes, results, device):
        content = [len(item[0]) for item in items]
        rows = [int(item[3]) for item in items]
        if self.modules.pooling is not None:
            raise ValueError('This proof is specifically for the no-content-pooling P2 path')
        seq = [n+r for n, r in zip(content, rows)]
        geometry.append({'batch': len(items), 'content_lengths': content, 'rows': rows,
                         'sequence_lengths': seq, 'real_tokens': sum(seq),
                         'padded_tokens': len(items)*max(seq), 'grad_enabled': __import__('torch').is_grad_enabled()})
        return method(self, items, indexes, results, device)
    ac._encode_prepared_batch = types.MethodType(measured, ac)
    try: yield
    finally:
        if previous is sentinel: del ac._encode_prepared_batch
        else: ac._encode_prepared_batch = previous


def parameters(h, ac):
    named = {}
    for prefix, module in [('ac', ac.modules), ('side', ac.side.model), ('reader', ac.target.model)]:
        for name, p in module.named_parameters(): named.setdefault(id(p), prefix+'.'+name)
    groups = [('ac', ac.trainable_parameters()), ('reader', h.module_manager.lora_parameters(ac.config.target_model_lora_name))]
    return [(group+':'+named.get(id(p), str(i)), p) for group, values in groups for i, p in enumerate(values)]


def compare_tensors(left, right, *, rtol, atol):
    import torch
    if left.shape != right.shape: raise ValueError('Tensor shape changed')
    if not torch.isfinite(left).all() or not torch.isfinite(right).all(): raise ValueError('Nonfinite tensor')
    difference = (left.float()-right.float())
    base = float(torch.linalg.vector_norm(left.float()))
    error = float(torch.linalg.vector_norm(difference))
    # An absolute RMS floor makes zero/nearly-zero heads interpretable.
    rms_error = error/max(1, left.numel())**.5
    rms_base = base/max(1, left.numel())**.5
    return {'relative_l2': error/max(base, 1e-30), 'rms_error': rms_error,
            'max_abs': float(difference.abs().max()) if left.numel() else 0.,
            'passed': rms_error <= atol+rtol*rms_base}


def proof(h, ac, trainer, examples, original, optimized, *, cpu=False):
    import torch
    from activation.ac_model.ac_model_training import disable_dropout
    params = parameters(h, ac)
    for base in dict.fromkeys([ac.side.model, ac.target.model]): base.train(); disable_dropout(base)
    ac.set_mode('training'); ac._training_gradient_checkpointing = True
    ac._side_gradient_checkpointing_min_tokens = 0
    trainer._target_gradient_checkpointing_min_tokens = 0
    trainer._training_deadline = None
    trainer._store_teacher_targets(ac, [x.penalty for x in examples], ac.device, lora=None)
    # Reference computation must not leave either base in evaluation mode.
    for base in dict.fromkeys([ac.side.model, ac.target.model]): base.train(); disable_dropout(base)
    rtol, atol = ((2e-5, 2e-7) if cpu else (0.03, 2e-6))
    reference = None; comparisons = {}; outputs_by_path = {}; geometries = {}
    original_encode = ac.encode_batch
    previous_instance = ac.__dict__.get('encode_batch'); had_instance = 'encode_batch' in ac.__dict__
    try:
        for label, method, budget in [('original', original, 65536), ('optimized', optimized, 65536),
                                      ('optimized_16384', optimized, 16384)]:
            ac.config.encode_batch_max_tokens = budget
            for _, p in params: p.grad = None
            torch.manual_seed(SEED); trainer._penalty_rng.seed(SEED)
            captured = []
            def capture(*args, **kwargs):
                values = original_encode(*args, **kwargs)
                for value in values:
                    captured.append(value.input_embeds.detach().float().cpu().clone())
                    captured.extend(value.layer_inputs[k].detach().float().cpu().clone() for k in sorted(value.layer_inputs))
                return values
            ac.encode_batch = capture
            geometry = []
            with method_scope(ac, method, geometry):
                outcomes = trainer._batch_loss(ac, examples, ac.device)
                assert all(row['penalty'] is not None for row in outcomes), 'Proof must exercise reader-only no-context penalty'
                loss = sum(row['objective']+row['penalty'] for row in outcomes)/len(outcomes)
                loss.backward()
            geometries[label] = geometry
            scalar = float(loss.detach()); outputs_by_path[label] = captured
            gradients = [(name, None if p.grad is None else p.grad.detach().float().cpu().clone()) for name, p in params]
            if reference is None:
                reference = (scalar, gradients)
            else:
                assert len(reference[1]) == len(gradients)
                for (name_a, a), (name_b, b) in zip(reference[1], gradients):
                    assert name_a == name_b and (a is None) == (b is None), name_a
                    comparisons[label+':'+name_a] = {'both_absent': True, 'passed': True} if a is None else compare_tensors(a, b, rtol=rtol, atol=atol)
                loss_tolerance = 1e-6 if cpu else 1e-3
                comparisons[label+':loss'] = {'original': reference[0], 'optimized': scalar,
                    'absolute_difference': abs(reference[0]-scalar), 'absolute_tolerance': loss_tolerance,
                    'passed': abs(reference[0]-scalar) <= loss_tolerance}
            del gradients, loss, outcomes
    finally:
        if had_instance: ac.encode_batch = previous_instance
        else: del ac.encode_batch
        for _, p in params: p.grad = None
    a = outputs_by_path['original']; outputs = {}
    for label in ('optimized', 'optimized_16384'):
        b = outputs_by_path[label]
        assert len(a) == len(b) and len(a) > len(examples), 'Deep AC outputs must be compared'
        outputs[label] = [compare_tensors(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b)]
    for group in ('ac:', 'reader:'):
        assert any(name.startswith(group) and tensor is not None and bool(tensor.abs().any()) for name, tensor in reference[1]), group
    del reference, outputs_by_path
    return {'passed': all(row['passed'] for row in comparisons.values()) and all(row['passed'] for rows in outputs.values() for row in rows),
            'precision_policy': {'relative_rms_tolerance': rtol, 'absolute_rms_floor': atol,
                'meaning': 'Numerical equivalence screen, not a claim of bitwise equality.'},
            'all_trainable_parameter_gradients': comparisons, 'input_rows_and_all_deep_deltas': outputs,
            'geometry': geometries, 'side_and_reader_checkpointing_forced_on': True,
            'no_context_penalty_fraction_for_proof': 1.0}


def gate_proof(ac, examples, original, optimized, *, cpu=False):
    """Two real short requests: old gate off, new whole-batch gate on."""
    import torch
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    requests = [request for example in examples for request in example.part_requests]
    sized = []
    for request in requests:
        n = len(tokenize_with_parts(ac.side.tokenizer, request.messages, [], ac.pad_id, tools=request.tools)[0])
        rows = ac.num_view_rows(n, request.compression_ratio)
        sized.append((n+rows, request))
    sized.sort(key=lambda x: x[0]); chosen = sized[:2]
    if len(chosen) < 2: raise ValueError('Need two genuine training passage requests')
    ac.config.encode_batch_max_tokens = 65536
    ac._side_gradient_checkpointing_min_tokens = max(x[0] for x in chosen)+1
    originals = ac.side.decoder_forward; calls = []
    def tracked(*args, **kwargs):
        calls.append(bool(kwargs['gradient_checkpointing']))
        return originals(*args, **kwargs)
    had = 'decoder_forward' in ac.side.__dict__; previous = ac.side.__dict__.get('decoder_forward')
    params = ac.trainable_parameters(); reference = None; result = {}
    ac.side.decoder_forward = tracked
    try:
        for label, method in [('original', original), ('optimized', optimized)]:
            calls.clear()
            for p in params: p.grad = None
            geometry = []
            with method_scope(ac, method, geometry):
                values = ac.encode_batch([x[1] for x in chosen], with_layers=True)
                tensors = [tensor for value in values for tensor in [value.input_embeds, *value.layer_inputs.values()]]
                loss = sum(t.float().square().mean() for t in tensors)
                loss.backward()
            grads = [None if p.grad is None else p.grad.detach().float().cpu().clone() for p in params]
            result[label] = {'checkpoint_flags': list(calls), 'geometry': geometry, 'loss': float(loss.detach())}
            if reference is None: reference = grads
            else:
                assert all((a is None) == (b is None) for a, b in zip(reference, grads))
                checks = [compare_tensors(a, b, rtol=2e-5 if cpu else .03, atol=2e-7 if cpu else 2e-6)
                          for a, b in zip(reference, grads) if a is not None]
                result['gradient_checks'] = checks
            del values, tensors, grads, loss
    finally:
        if had: ac.side.decoder_forward = previous
        else: del ac.side.decoder_forward
        for p in params: p.grad = None
    result['threshold'] = ac._side_gradient_checkpointing_min_tokens
    result['passed'] = result['original']['checkpoint_flags'] == [False] and result['optimized']['checkpoint_flags'] == [True] and all(x['passed'] for x in result['gradient_checks'])
    return result


def cpu_check():
    import torch
    import activation.ac_model.ac_model as module
    from activation.tests.test_interventions import tiny_hybrid, build, trainer_and_examples, open_heads, open_attention
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder); model = tiny_hybrid(root/'tiny')
        h, ac = build(root/'state', model, frequency=-1, input_pooling_stride=0)
        open_heads(ac); open_attention(ac)
        trainer, _, examples, _ = trainer_and_examples(h, ac, micro_batch_examples=2, completion_samples=0)
        original, optimized = methods(module)
        value = proof(h, ac, trainer, examples[:2], original, optimized, cpu=True)
        edge = gate_proof(ac, examples, original, optimized, cpu=True)
        print(json.dumps({'proof': value, 'checkpoint_gate': edge}, allow_nan=False), flush=True)
        assert value['passed'] and edge['passed']


def gpu_worker(args):
    import torch
    import activation.ac_model.ac_model as module
    import activation.ac_model.ac_model_training as training
    import activation.ac_model.ac_model_utils as utils
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import configure_fla_runtime
    for path, expected in [(module.__file__, AC_SHA), (training.__file__, TRAINER_SHA), (utils.__file__, UTILS_SHA),
                           (args.bench, BUILDER_SHA), (args.material, MATERIAL_SHA),
                           (Path(__file__).with_name('piecewise_batching_v1.patch'), PATCH_SHA)]:
        if digest(path) != expected: raise ValueError('Pinned source/material changed: '+str(path))
    output = args.output/(args.worker+'.json')
    if output.exists(): raise FileExistsError(output)
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location('piecewise_probe_builder', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    source = args.p2_root
    ac_path = source/'AC_MODELS/rag_poc/epoch_001'; reader_path = source/'LORAS/rag_reader/epoch_001'
    files = [ac_path/'ac_modules.pt', ac_path/'side_lora/adapter_config.json', reader_path/'adapter_config.json']
    for directory in (ac_path/'side_lora', reader_path):
        weights = [directory/name for name in ('adapter_model.safetensors','adapter_model.bin') if (directory/name).is_file()]
        if len(weights) != 1: raise ValueError('Expected exactly one P2 adapter payload')
        files += weights
    identities = {str(path.relative_to(source)): digest(path) for path in files}
    print(json.dumps({'phase': 'loading fresh P2', 'condition': args.worker}), flush=True)
    h = bench.harness(ac_path, reader_path); ac = h.module_manager.get_ac_model(bench.AC)
    ac.prepare(); h.module_manager.ensure_lora(bench.READER); ac.epoch_number = 0
    device = ac.device  # train() deliberately releases both bases back to CPU.
    material = json.loads(args.material.read_text())
    assert len(material['train']) == 4000 and len(material['held']) == 400 and len(material['retrieval']) == 32398
    # The frozen material is family-grouped. Span required-source lengths instead
    # of accidentally benchmarking only the first family; no loss-based selection.
    ordered = sorted(material['train'], key=lambda row: (int(row['source_tokens']), row['example']['example_id']))
    quantile_indexes = [round(i*(len(ordered)-1)/7) for i in range(8)]
    selected = [ordered[i] for i in quantile_indexes]
    selection = [{'item_id': row['example']['example_id'], 'source_family': row['source_family'],
                  'required_source_tokens': int(row['source_tokens']), 'quantile_index': i}
                 for i, row in zip(quantile_indexes, selected)]
    material['train'] = selected; material['held'] = selected[:1]  # Builder requires nonempty lists; this training-only placeholder is never evaluated.
    is_proof = args.worker == 'proof'
    pieces, changed, budget = (True, False, 65536) if is_proof else CONDITIONS[args.worker]
    print(json.dumps({'phase': 'building eight frozen training inputs', 'condition': args.worker}), flush=True)
    _, items, held = bench.material_items(h, material, passage_parts=pieces, evidence_tokens=8192, max_passages=16, ratio=1/16)
    assert len(held) == 1 and held[0].item_id == items[0].item_id and len(items) == 8
    del held
    del material, selected, ordered; gc.collect()
    ac.config.encode_batch_max_tokens = budget
    receipt = {'condition': args.worker, 'material_sha256': MATERIAL_SHA, 'builder_sha256': BUILDER_SHA,
        'source': {'ac_model': AC_SHA, 'trainer': TRAINER_SHA, 'utils': UTILS_SHA, 'patch': PATCH_SHA},
        'P2_payload_sha256': identities, 'device': torch.cuda.get_device_name(ac.device),
        'runtime': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                    'side_model': ac.side.model_config.model_id, 'reader_model': ac.target.model_config.model_id,
                    'side_revision': getattr(ac.side.model.config, '_commit_hash', None),
                    'reader_revision': getattr(ac.target.model.config, '_commit_hash', None),
                    'side_dtype': str(next(ac.side.model.parameters()).dtype),
                    'reader_dtype': str(next(ac.target.model.parameters()).dtype)},
        'inputs': [{'item_id': item.item_id, 'kind': item.kind, 'row_budget': {key: item.info[key] for key in
                ('passages','evidence_source_tokens','encoded_source_tokens','compressed_rows','part_source_tokens','part_rows')},
                'target_sha256': object_hash(item.assistant_token_ids), 'full_sha256': object_hash(item.in_context_messages)} for item in items],
        'selection': selection,
        'selection_rule': 'Eight evenly spaced order statistics, including endpoints, after sorting TRAIN ONLY by (required source_tokens, immutable example_id). This covers length extremes; not a random quality sample.',
        'passage_parts': pieces, 'encode_batch_max_tokens': budget,
        'training_updates_write_model_checkpoints': False, 'status': 'running'}
    write(output, receipt)
    config = ActivationContextTrainingConfig(loss_kind='sft', learning_rate_ac=5e-5, learning_rate_target_lora=2e-5,
        examples_per_update=8, micro_batch_examples=2, num_epochs=4, warmup_updates=0, schedule_total_updates=4,
        shuffle_examples=False, no_context_penalty_weight=1., no_context_penalty_fraction=1. if is_proof else .25,
        checkpoint_every_epoch=False, checkpoint_every_updates=None, completion_samples=0,
        gradient_checkpointing=True, gradient_checkpointing_min_tokens=1700, seed=SEED)
    trainer = ActivationContextTrainer(h, config); variables = trainer._autotuning_variables(ac)
    original, optimized = methods(module)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.cuda.reset_peak_memory_stats(ac.device)
    with configure_fla_runtime(variables['fla_config']):
        if is_proof:
            examples = [trainer.build_example(ac, item) for item in items]
            # Same two mixed-length reader examples, all their AC parts and every trainable parameter.
            receipt['equivalence'] = proof(h, ac, trainer, examples[:2], original, optimized)
            write(output, receipt)
            receipt['checkpoint_gate'] = gate_proof(ac, examples, original, optimized)
            receipt['status'] = 'passed' if receipt['equivalence']['passed'] and receipt['checkpoint_gate']['passed'] else 'failed'
        else:
            geometry = []; ac.profile_phases = False
            with method_scope(ac, optimized if changed else original, geometry):
                torch.cuda.synchronize(device); before = time.monotonic()
                stats = trainer.train(bench.AC, items, reporting_data=[])
                torch.cuda.synchronize(device)
            assert stats.steps == 4 and stats.checkpoint_path is None and stats.target_lora_checkpoint_path is None
            assert any(stats.penalty_counts), 'Timing must include actual no-context penalty updates'
            receipt.update(status='complete', train_call_seconds=time.monotonic()-before,
                first_update_seconds=stats.step_seconds[0], steady_update_seconds=stats.step_seconds[1:],
                steady_median_seconds=statistics.median(stats.step_seconds[1:]),
                loss=stats.loss, penalty_counts=stats.penalty_counts, grad_norm_ac=stats.grad_norm_ac,
                grad_norm_reader=stats.grad_norm_reader, geometry=geometry, autotuning=stats.autotuning,
                padded_decoder_tokens=sum(x['padded_tokens'] for x in geometry), real_decoder_tokens=sum(x['real_tokens'] for x in geometry),
                timing_scope='One cache/kernel warmup update then three measured updates; same eight questions repeated, fresh optimizer and P2 in each process; no panels.')
    receipt.update(elapsed_seconds=time.monotonic()-started, peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30)
    write(output, receipt)
    if receipt['status'] == 'failed': raise RuntimeError('Numerical equivalence screen failed; do not adopt patch')


def controller(args):
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output/'summary.json'
    if result_path.exists(): raise FileExistsError(result_path)
    summary = {'status': 'running', 'script_sha256': digest(__file__), 'conditions': {},
        'material_sha256': MATERIAL_SHA, 'deadline_seconds': args.seconds,
        'limits': 'Train-only eight-question length-quantile timing selection; numerical tolerances and exact batch geometry retained. Does not establish training quality or average speed over the full source-length distribution.'}
    start = time.monotonic(); write(result_path, summary)
    for label in ['proof', *CONDITIONS]:
        remaining = args.seconds-(time.monotonic()-start)
        if remaining < 90:
            summary['conditions'][label] = {'status': 'not_run_budget'}; continue
        env = dict(os.environ); env['ACTIVATION_SYNC_ROOT'] = str(args.workspace/label)
        state = Path(env['ACTIVATION_SYNC_ROOT']); state.mkdir(parents=True, exist_ok=True)
        if args.autotune_root:
            link = state/'AUTOTUNE'
            if not link.exists(): link.symlink_to(args.autotune_root, target_is_directory=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', label, '--material', str(args.material),
               '--output', str(args.output), '--p2-root', str(args.p2_root), '--bench', str(args.bench)]
        with (args.output/(label+'.log')).open('x') as log:
            try:
                completed = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=max(30,remaining-20))
                status = {'status': 'complete' if completed.returncode == 0 else 'failed', 'returncode': completed.returncode}
            except subprocess.TimeoutExpired:
                status = {'status': 'timeout'}
        summary['conditions'][label] = status
        print(json.dumps({'condition': label, **status}), flush=True); write(result_path, summary)
        if label == 'proof' and status['status'] != 'complete':
            summary['status'] = 'proof_failed'; write(result_path, summary); return 1
    results = [json.loads((args.output/(label+'.json')).read_text()) for label, status in summary['conditions'].items()
               if status['status'] == 'complete' and label != 'proof']
    if results:
        identity = [(r['item_id'], r['full_sha256'], r['target_sha256'], r['row_budget']['compressed_rows']) for r in results[0]['inputs']]
        for result in results:
            assert [(r['item_id'], r['full_sha256'], r['target_sha256'], r['row_budget']['compressed_rows']) for r in result['inputs']] == identity
        summary['identical_full_evidence_targets_and_total_rows'] = True
        summary['timing'] = {r['condition']: {k:r[k] for k in ('steady_median_seconds','first_update_seconds','peak_memory_gib','padded_decoder_tokens','real_decoder_tokens')} for r in results}
    summary['status'] = 'complete' if all(x['status'] == 'complete' for x in summary['conditions'].values()) else 'partial'
    summary['elapsed_seconds'] = time.monotonic()-start; write(result_path, summary)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-check', action='store_true')
    parser.add_argument('--worker', choices=['proof', *CONDITIONS])
    parser.add_argument('--material', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--p2-root', type=Path, default=Path(os.environ.get('QA_P2_ROOT','/missing/P2')))
    parser.add_argument('--bench', type=Path, default=Path(__file__).with_name('run_qa_comparison_v3.py'))
    parser.add_argument('--autotune-root', type=Path)
    parser.add_argument('--seconds', type=int, default=1080)
    args = parser.parse_args()
    if args.cpu_check: cpu_check()
    elif args.worker: gpu_worker(args)
    else:
        if not all((args.material, args.output, args.workspace)): parser.error('--material, --output, --workspace required')
        sys.exit(controller(args))
