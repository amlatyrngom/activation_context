"""Completed-piecewise encoder budget proof, then ordinary-kernel timings; no saves."""
import argparse
from dataclasses import replace
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time

import probe_piecewise_checkpoint_v1 as previous
import evaluate_piecewise_cohort_v3 as paired

ARM = 'piecewise_27b_4k_8k'
BUDGETS = (65536, 32768, 16384)
CASES = ('failing', 'quantiles')
MEMORY_SHA = '721f78cdf844d530cf23eb05bfe466971019785828601e7fdf0b14fa6aaded4f'
PREVIOUS_SHA = '12df00c8fcac924cf7a11525006c626d8ef572b6fdefa809468124512fba2139'
PAIRED_SHA = 'ddd905e0d28a952cf7a1965e2527355f5dfe2e4bfea59b1c31acb5e3e799cd30'
SEED = previous.SEED
write, sha, read = previous.write, previous.digest, paired.read


def select_cases(material, memory):
    by_id = {r['example']['example_id']: r for r in material['train']}
    if len(by_id) != 4000:
        raise ValueError('Expected exact4000 unique training questions')
    selected = {}
    for case in CASES:
        ids = [r['item_id'] for r in memory['cases']['pieces_'+case]['inputs']]
        if len(ids) != 8 or len(set(ids)) != 8 or not set(ids).issubset(by_id):
            raise ValueError('Recorded eight-question case changed')
        selected[case] = [by_id[ident] for ident in ids]
    return selected


def penalty_ids(examples, fraction, offset, updates):
    rng = random.Random(SEED)
    for _ in range(offset): rng.random()
    order = sorted(examples, key=lambda x: len(x.student_ids))
    return [[ex.item.item_id for ex in order if ex.penalty is not None and rng.random() < fraction]
            for _ in range(updates)]


def optimizer_guard(state, groups, validate):
    if state.get('updates_done') != 500 or state.get('epoch_number') != 1:
        raise ValueError('Expected completed500-update epoch1 optimizer')
    optimizer = state['optimizer']; validate(optimizer, groups)
    values = optimizer['state']
    if len(values) != 615 or any(not {'step', 'exp_avg', 'exp_avg_sq'}.issubset(v) for v in values.values()):
        raise ValueError('Final optimizer moments are missing/incomplete')
    steps = [float(v['step']) for v in values.values()]
    if any(step != 500 for step in steps):
        raise ValueError('Final Adam parameter clocks differ from500')
    return {'states': len(values), 'parameter_step_min': min(steps), 'parameter_step_max': max(steps),
            'updates_done': 500, 'epoch_number': 1, 'group_sizes': [len(g['params']) for g in optimizer['param_groups']]}


def prove_case(h, ac, trainer, examples, case, *, budgets=BUDGETS, cpu=False, flush=lambda _: None):
    import torch
    from activation.agent_training.agent_training_utils import disable_dropout
    from activation.harness.hf_utils import decoder_layers
    params = previous.parameters(h, ac); original_config = ac.config
    ac.set_mode('training'); ac._training_gradient_checkpointing = True
    ac._side_gradient_checkpointing_min_tokens = 0; trainer._target_gradient_checkpointing_min_tokens = 0
    trainer._training_deadline = None
    trainer._store_teacher_targets(ac, [ex.penalty for ex in examples], ac.device, lora=None)
    for model in dict.fromkeys([ac.side.model, ac.target.model]): model.train(); disable_dropout(model)
    assert all(loaded.model.training and all(layer.training for layer in decoder_layers(loaded.model))
               for loaded in (ac.side, ac.target))
    offset = 120 if case == 'failing' else 0
    wanted_penalties = penalty_ids(examples, trainer.config.no_context_penalty_fraction, offset, 1)[0]
    rtol, atol, loss_tol = (2e-5, 2e-7, 1e-6) if cpu else (.03, 2e-6, .001)
    metadata = [(name, p._version, p.requires_grad) for name, p in params]
    reference = None; records = {}
    original_encode = ac.encode_batch; sentinel = object(); saved_encode = ac.__dict__.get('encode_batch', sentinel)
    variants = [('baseline', budgets[0]), ('unchanged_repeat', budgets[0]), *[(str(b), b) for b in budgets[1:]]]
    try:
        for label, budget in variants:
            ac.config = replace(original_config, encode_batch_max_tokens=budget)
            for _, p in params: p.grad = None
            trainer._penalty_rng.seed(SEED)
            for _ in range(offset): trainer._penalty_rng.random()
            torch.manual_seed(SEED); outputs = []; trace = []; selected = []; loss_value = 0.
            calls = {'side_encoder': 0, 'reader': 0}; before_backward = {}; handles = []
            def capture(*args, **kwargs):
                values = original_encode(*args, **kwargs)
                for value in values:
                    outputs.extend(t.detach().float().cpu().clone() for t in
                        [value.input_embeds, *[value.layer_inputs[k] for k in sorted(value.layer_inputs)]])
                return values
            ac.encode_batch = capture
            for name, loaded in [('side_encoder', ac.side), ('reader', ac.target)]:
                def count_call(module, args, _name=name): calls[_name] += 1
                handles.append(decoder_layers(loaded.model)[0].register_forward_pre_hook(count_call))
            ordered = sorted(examples, key=lambda x: len(x.student_ids))
            try:
                with previous.trace_calls(ac, trace):
                    for start in range(0, len(ordered), 2):
                        group = ordered[start:start+2]
                        outcomes = trainer._batch_loss(ac, group, ac.device)
                        selected.extend(ex.item.item_id for ex, value in zip(group, outcomes) if value['penalty'] is not None)
                        loss = sum((value['objective']+(value['penalty'] if value['penalty'] is not None else 0.))*ex.item.weight/len(ordered)
                                   for ex, value in zip(group, outcomes))
                        loss_value += float(loss.detach()); before = dict(calls); loss.backward()
                        if not all(calls[k] > before[k] for k in calls): raise AssertionError('Checkpoint recomputation missing')
                        before_backward = before
                        del loss, outcomes
            finally:
                for handle in handles: handle.remove()
            if selected != wanted_penalties: raise AssertionError('Penalty selection changed')
            if [(name, p._version, p.requires_grad) for name, p in params] != metadata:
                raise AssertionError('Proof mutated parameter version/trainability')
            side = [v for v in trace if v['kind'] == 'side']
            if not side or any(not v['checkpoint'] or v['mask_is_none'] != (v['shape'][0] == 1) for v in side):
                raise AssertionError('Original side mask/checkpoint policy changed')
            grads = [(name, None if p.grad is None else p.grad.detach().float().cpu().clone()) for name, p in params]
            record = {'budget': budget, 'loss': loss_value, 'selected_penalty_ids': selected, 'trace': trace,
                      'first_layer_calls': calls, 'last_microbatch_calls_before_backward': before_backward,
                      'parameter_metadata_unchanged': True}
            if reference is None:
                reference = (loss_value, outputs, grads)
                for prefix in ('ac:', 'reader:'):
                    assert any(name.startswith(prefix) and g is not None and bool(g.abs().any()) for name, g in grads)
            else:
                checks = {}
                if len(reference[1]) != len(outputs) or len(reference[2]) != len(grads): raise AssertionError('Output/parameter count changed')
                for (name, left), (other, right) in zip(reference[2], grads):
                    if name != other or (left is None) != (right is None): raise AssertionError('Gradient ownership changed: '+name)
                    checks[name] = {'passed': True, 'both_absent': True} if left is None else previous.compare_tensors(left, right, rtol=rtol, atol=atol)
                output_checks = [previous.compare_tensors(a, b, rtol=rtol, atol=atol) for a, b in zip(reference[1], outputs)]
                record.update(all_trainable_gradients=checks, all_ac_outputs=output_checks,
                    loss_abs_difference=abs(loss_value-reference[0]),
                    passed=abs(loss_value-reference[0]) <= loss_tol and all(v['passed'] for v in checks.values())
                        and all(v['passed'] for v in output_checks))
            records[label] = record; flush(records)
            del grads, outputs
    finally:
        if saved_encode is sentinel: del ac.encode_batch
        else: ac.encode_batch = saved_encode
        ac.config = original_config
        for _, p in params: p.grad = None
    return {'passed': all(v['passed'] for k, v in records.items() if k != 'baseline'), 'comparisons': records,
        'tolerances': {'relative_rms': rtol, 'absolute_rms_floor': atol, 'joint_loss_abs': loss_tol},
        'selected_penalty_ids': wanted_penalties, 'parameter_count': len(params),
        'scope': 'Original masks, threshold0; completed trained weights; no optimizer update; deterministic diagnostic only.'}


def worker(args):
    import torch
    if args.worker == 'proof':
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8' or torch.cuda.is_initialized():
            raise ValueError('Set deterministic workspace before CUDA initialization')
        torch.use_deterministic_algorithms(True, warn_only=False)
    else:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') or torch.are_deterministic_algorithms_enabled():
            raise ValueError('Ordinary timing must retain production determinism flags')
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import configure_fla_runtime
    design = read(args.design)
    if design.get('version') != 2 or design.get('status') != 'frozen': raise ValueError('Expected frozen V2 design')
    if sha(args.material) != design['material_sha256'] or sha(args.bench) != design['builder_sha256']:
        raise ValueError('Frozen material/builder changed')
    actual_source = {key: sha(module.__file__) for key, module in
        [('ac_model', ac_module), ('trainer', training_module), ('utils', utils_module)]}
    if actual_source != design['source_sha256'] or actual_source != paired.SOURCE: raise ValueError('Frozen source changed')
    material = read(args.material)
    ids = {split: [r['example']['example_id'] for r in previous.ordered_rows(material[split])] for split in ('train', 'held')}
    pair, publication = paired.paired_final(ARM, args.reports, args.runs_root, design, ids)
    payload = paired.payload_hashes(pair); optimizer_path = Path(pair['ac_checkpoint'])/'optimizer.pt'
    if not optimizer_path.is_file() or optimizer_path.stat().st_size == 0: raise ValueError('Missing completed optimizer')
    optimizer_sha = sha(optimizer_path)
    if sha(args.memory_receipt) != MEMORY_SHA: raise ValueError('Original case receipt changed')
    memory = read(args.memory_receipt); cases = select_cases(material, memory)
    selected_case = None if args.worker == 'proof' else args.worker.rsplit('_', 1)[0]
    budget = 65536 if selected_case is None else int(args.worker.rsplit('_', 1)[1])
    if selected_case is not None and (selected_case not in CASES or budget not in BUDGETS): raise ValueError('Unknown timing condition')
    output = args.output/(args.worker+'.json')
    if output.exists(): raise FileExistsError('Do not overwrite prior measurements: '+str(output))
    receipt = {'status': 'running', 'condition': args.worker, 'script_sha256': sha(__file__),
        'design_sha256': sha(args.design), 'material_sha256': sha(args.material), 'builder_sha256': sha(args.bench),
        'source_sha256': actual_source, 'pair': pair, 'publication': publication, 'payload_sha256': payload,
        'optimizer_sha256': optimizer_sha, 'memory_case_receipt_sha256': MEMORY_SHA,
        'device': torch.cuda.get_device_name(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'source_masks': 'original', 'gradient_checkpointing_min_tokens': 0, 'checkpoint_saves': False,
        'initialization': 'Own completed piecewise V2 paired500 final; timings inherit the actual final Adam states.',
        'cases': {}, 'trace': []}
    write(output, receipt); began = time.monotonic()
    spec = importlib.util.spec_from_file_location('budget_builder', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    h = bench.harness(pair['ac_checkpoint'], pair['reader_checkpoint'], encode_batch_max_tokens=budget)
    ac = h.module_manager.get_ac_model(bench.AC); ac.prepare(); h.module_manager.ensure_lora(bench.READER)
    state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
    receipt['optimizer_guard'] = optimizer_guard(state, [ac.trainable_parameters(), h.module_manager.lora_parameters(bench.READER)], bench.validate_optimizer_layout)
    if ac.epoch_number != 1: raise ValueError('Loaded AC epoch differs from completed optimizer')
    del state; gc.collect()
    config = ActivationContextTrainingConfig(**read(args.reports/ARM/'training_config.json'))
    config = replace(config, num_epochs=4, shuffle_examples=False, resume_optimizer=True, resume_skip_steps=0,
        checkpoint_every_epoch=False, checkpoint_every_updates=None, completion_samples=0, max_duration_seconds=None,
        gradient_checkpointing_min_tokens=0)
    trainer = ActivationContextTrainer(h, config); variables = trainer._autotuning_variables(ac)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    receipt['production_flags'] = {'tf32_matmul': True, 'tf32_cudnn': True, 'autotuning': variables['provenance']}
    with configure_fla_runtime(variables['fla_config']):
        for case in CASES if selected_case is None else (selected_case,):
            subset = cases[case]
            # Build each old eight-row case separately: its original local passage-shuffle stream restarts at SEED.
            _, items, placeholder = bench.material_items(h, {**material, 'train': subset, 'held': subset[:1]},
                passage_parts=True, evidence_tokens=8192, max_passages=16, ratio=1/16)
            del placeholder
            fingerprints = [{'item_id': item.item_id, 'full_sha256': previous.object_hash(item.in_context_messages),
                'target_sha256': previous.object_hash(item.assistant_token_ids), 'compressed_rows': item.info['compressed_rows'],
                'encoder_tokens': item.info['encoded_source_tokens']} for item in items]
            if fingerprints != memory['cases']['pieces_'+case]['inputs']: raise ValueError('Exact recorded training inputs changed')
            examples = [trainer.build_example(ac, item) for item in items]
            receipt['cases'][case] = {'inputs': fingerprints}; write(output, receipt)
            if selected_case is None:
                def flush(records):
                    receipt['cases'][case]['comparisons'] = records; write(output, receipt)
                proof = prove_case(h, ac, trainer, examples, case, flush=flush)
                receipt['cases'][case] = {**receipt['cases'][case], **proof}; write(output, receipt)
                if not proof['passed']:
                    receipt['status'] = 'failed'; write(output, receipt)
                    raise RuntimeError('Strict budget equivalence failed; no timing/adoption')
            else:
                offset = 120 if case == 'failing' else 0
                for _ in range(offset): trainer._penalty_rng.random()
                wanted = penalty_ids(examples, config.no_context_penalty_fraction, offset, 4)
                observed = []; old_batch = trainer._batch_loss
                def record_penalties(model, batch, device, **kwargs):
                    values = old_batch(model, batch, device, **kwargs)
                    observed.append([ex.item.item_id for ex, value in zip(batch, values) if value['penalty'] is not None])
                    return values
                trainer._batch_loss = record_penalties
                try:
                    with previous.trace_calls(ac, receipt['trace']):
                        stats = trainer.train(bench.AC, items, reporting_data=[])
                finally: del trainer._batch_loss
                actual = [sum(observed[i:i+4], []) for i in range(0, len(observed), 4)]
                if actual != wanted: raise AssertionError('Warmup/steady selected penalty question IDs changed')
                if (stats.steps != 4 or stats.checkpoint_path is not None or stats.target_lora_checkpoint_path is not None
                        or trainer.updates_done[bench.AC] != 504 or stats.resumed_from != str(optimizer_path)):
                    raise AssertionError('Timing did not inherit exact optimizer or attempted checkpoint saving')
                receipt.update(step_seconds=stats.step_seconds, steady_median_seconds=statistics.median(stats.step_seconds[1:]),
                    peak_memory_gib=stats.peak_memory_bytes/2**30, penalty_counts=stats.penalty_counts,
                    selected_penalty_ids=actual, expected_penalty_ids=wanted, warmup_updates=1, measured_updates=3,
                    loss=stats.loss, grad_norm_ac=stats.grad_norm_ac, grad_norm_reader=stats.grad_norm_reader,
                    learning_rates=stats.learning_rates, inherited_from=stats.resumed_from,
                    completed_runtime_updates=trainer.updates_done[bench.AC])
                prepared_batches = [row for row in receipt['trace'] if row['kind'] == 'prepared']
                useful = sum(sum(row['sequence_lengths']) for row in prepared_batches)
                padded = sum(row['padded_tokens'] for row in prepared_batches)
                receipt['encoding_summary'] = {'actual_side_encoder_forward_batches': len(prepared_batches),
                    'logical_updates_including_warmup': 4, 'useful_side_encoder_positions': useful,
                    'padded_side_encoder_positions': padded, 'padding_fraction': 1-useful/padded,
                    'max_padded_side_encoder_batch': max(row['padded_tokens'] for row in prepared_batches),
                    'note': 'Forward batches; checkpoint backward recomputation is enabled but not counted by this method-level counter.'}
                if any(not all(math.isclose(a, b, rel_tol=1e-12, abs_tol=0) for a, b in zip(lr, [5e-6, 2e-6]))
                       for lr in stats.learning_rates): raise AssertionError('Expected parent terminal learning rates')
    if paired.payload_hashes(pair) != payload or sha(optimizer_path) != optimizer_sha:
        raise AssertionError('Parent paired weights/optimizer were mutated')
    receipt.update(status='complete', elapsed_seconds=time.monotonic()-began, parent_payloads_rehashed_unchanged=True)
    write(output, receipt)


def controller(args):
    args.output.mkdir(parents=True, exist_ok=True); summary_path = args.output/'summary.json'
    if summary_path.exists(): raise FileExistsError(summary_path)
    started = time.monotonic(); summary = {'status': 'running', 'script_sha256': sha(__file__), 'conditions': {}, 'seconds_budget': args.seconds}
    write(summary_path, summary)
    for condition in ['proof', *[f'{case}_{budget}' for case in CASES for budget in BUDGETS]]:
        remaining = args.seconds-(time.monotonic()-started)
        if remaining < 120:
            summary['status'] = 'partial_budget'; write(summary_path, summary); return 1
        workspace = args.workspace/condition; workspace.mkdir(parents=True, exist_ok=True)
        if (workspace/'AC_MODELS').exists(): raise FileExistsError('Use a fresh private measurement workspace')
        if not (workspace/'AUTOTUNE').exists(): (workspace/'AUTOTUNE').symlink_to(args.autotune_root, target_is_directory=True)
        env = dict(os.environ); env['ACTIVATION_SYNC_ROOT'] = str(workspace)
        if condition == 'proof': env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        else: env.pop('CUBLAS_WORKSPACE_CONFIG', None)
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', condition]
        for name in ('material','bench','design','memory_receipt','reports','runs_root','output'):
            command += ['--'+name.replace('_','-'), str(getattr(args,name))]
        with (args.output/(condition+'.log')).open('x') as log:
            try:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=remaining-30)
                record = {'returncode': completed.returncode, 'status': 'complete' if completed.returncode == 0 else 'failed'}
            except subprocess.TimeoutExpired: record = {'status': 'timeout'}
        output = args.output/(condition+'.json')
        if output.exists(): record['worker_status'] = read(output)['status']
        summary['conditions'][condition] = record; write(summary_path, summary)
        print(json.dumps({condition: record}), flush=True)
        if record['status'] != 'complete' or record.get('worker_status') != 'complete':
            summary['status'] = 'proof_failed' if condition == 'proof' else 'timing_failed'; write(summary_path, summary); return 1
    proof = read(args.output/'proof.json')
    for case in CASES:
        records = [read(args.output/f'{case}_{budget}.json') for budget in BUDGETS]
        for key in ('payload_sha256','optimizer_sha256','selected_penalty_ids','learning_rates'):
            if any(record[key] != records[0][key] for record in records[1:]): raise AssertionError('Timing variants differ: '+key)
        if any(record[key] != proof[key] for record in records for key in ('payload_sha256','optimizer_sha256')):
            summary['status'] = 'identity_failed'; write(summary_path, summary)
            raise AssertionError('Timing initialization differs from the proved final pair')
    summary.update(status='complete', elapsed_seconds=time.monotonic()-started)
    write(summary_path, summary); return 0


def cpu_check():
    import torch
    from activation.tests.test_interventions import build, tiny_hybrid, trainer_and_examples, open_heads, open_attention
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True, warn_only=False)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        h, ac = build(root/'state', tiny_hybrid(root/'reader'), frequency=-1, input_pooling_stride=0,
                      side_dir=tiny_hybrid(root/'side', hidden=16, seed=112))
        open_heads(ac); open_attention(ac)
        trainer, _, examples, _ = trainer_and_examples(h, ac, micro_batch_examples=2, completion_samples=0)
        from activation.ac_model.ac_model_utils import tokenize_with_parts
        lengths = [len(tokenize_with_parts(ac.side.tokenizer, r.messages, [], ac.pad_id, tools=r.tools)[0]) for ex in examples for r in ex.part_requests]
        record = prove_case(h, ac, trainer, examples, 'quantiles', budgets=(65536, max(lengths), 1), cpu=True)
        assert record['passed']; assert record['comparisons']['baseline']['trace'] != record['comparisons'][str(max(lengths))]['trace']
        print(json.dumps({'status':'pass','equivalence':record,'scope':'Real tinyCPU models, original masks, changed batch geometry; no GPU or source edit.'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check', action='store_true'); parser.add_argument('--worker')
    for name in ('material','bench','design','memory-receipt','reports','runs-root','output','workspace','autotune-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--seconds', type=int, default=2520); args = parser.parse_args()
    if (sha(previous.__file__) != PREVIOUS_SHA or sha(paired.__file__) != PAIRED_SHA
            or sha(Path(__file__).with_name('probe_piecewise_original_v1.py')) != previous.HELPER_SHA):
        raise ValueError('Pinned existing probe/final guard changed')
    if args.cpu_check: cpu_check()
    elif args.worker: worker(args)
    else: sys.exit(controller(args))
