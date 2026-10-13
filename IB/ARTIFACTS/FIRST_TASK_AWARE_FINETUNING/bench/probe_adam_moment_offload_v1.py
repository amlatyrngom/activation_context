"""Private, fail-closed moment-placement probe; never a training continuation entrypoint."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace, asdict
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

SEED = 20260921
MOMENTS = ('exp_avg', 'exp_avg_sq')
PROOFS = ('resident', 'resident_repeat', 'offloaded')
FITS = ('fit_first', 'fit_worst')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str)+'\n')
    temporary.replace(path)


def tensor_record(tensor):
    import torch
    # CPU copy and hashing only: no dtype conversion or GPU reduction.
    cpu = tensor.detach().cpu().contiguous()
    if (cpu.is_floating_point() or cpu.is_complex()) and not bool(torch.isfinite(cpu).all()):
        raise AssertionError('Non-finite tensor in exact numerical proof')
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype), 'sha256': hashlib.sha256(raw).hexdigest()}


def optimizer_record(optimizer):
    import torch
    saved = optimizer.state_dict()
    return {'param_groups': saved['param_groups'], 'state': {str(key): {
        name: tensor_record(value) if torch.is_tensor(value) else value
        for name, value in state.items()} for key, state in saved['state'].items()}}


def parameter_record(optimizer, *, gradients=False):
    return {f'{group_index}:{index}': None if (value := (p.grad if gradients else p)) is None else tensor_record(value)
            for group_index, group in enumerate(optimizer.param_groups) for index, p in enumerate(group['params'])}


def clocks(optimizer):
    return {str(index): {'value': float(state['step']), 'dtype': str(state['step'].dtype),
                        'device': str(state['step'].device)}
            for index, state in optimizer.state_dict()['state'].items()}


def check_optimizer_progress(optimizer, expected_clock, *, require_gradients=False, expected_state_ids=None):
    """All615 inherited states must advance; all630 group positions retain their identity."""
    saved = optimizer.state_dict()
    grouped_ids = [str(ident) for group in saved['param_groups'] for ident in group['params']]
    state_ids = sorted(str(ident) for ident in saved['state'])
    active_ids = sorted(str(ident) for group, packed in zip(optimizer.param_groups, saved['param_groups'])
                        for parameter, ident in zip(group['params'], packed['params']) if parameter.grad is not None)
    if len(grouped_ids) != 630 or len(set(grouped_ids)) != 630 or len(state_ids) != 615:
        raise AssertionError('Expected630 unique group positions,615 populated states and15 inactive positions')
    if not set(state_ids) <= set(grouped_ids):
        raise AssertionError('Optimizer state contains a foreign parameter ID')
    if expected_state_ids is not None and state_ids != expected_state_ids:
        raise AssertionError('Inherited populated parameter IDs changed')
    if any(float(state['step']) != expected_clock for state in saved['state'].values()):
        raise AssertionError('Every inherited Adam parameter clock must equal '+str(expected_clock))
    if require_gradients and active_ids != state_ids:
        raise AssertionError('Active gradient IDs must match exactly the615 inherited populated state IDs')
    return {'group_parameter_ids':grouped_ids, 'populated_state_ids':state_ids, 'active_gradient_ids':active_ids,
            'inactive_group_ids':sorted(set(grouped_ids)-set(state_ids)), 'expected_clock':expected_clock}


class MomentOffload:
    """Only the enrolled optimizer; original AdamW.step, step tensors, and groups stay intact."""
    def __init__(self, optimizer, events=None):
        self.optimizer = optimizer
        self.events = events if events is not None else []
        self.handles = [optimizer.register_step_pre_hook(self._before),
                        optimizer.register_step_post_hook(self._after)]

    def move(self, to_cpu):
        import torch
        started = time.monotonic(); total = 0
        for parameter, state in self.optimizer.state.items():
            if not state:
                continue
            if set(state) != {'step', *MOMENTS}:
                raise ValueError('Only populated non-AMSGrad Adam state is supported')
            clock = state['step']; clock_device = clock.device; clock_dtype = clock.dtype; clock_value = float(clock)
            target = torch.device('cpu') if to_cpu else parameter.device
            for key in MOMENTS:
                value = state[key]
                if value.shape != parameter.shape:
                    raise ValueError('Moment/parameter shape mismatch')
                total += value.numel()*value.element_size()
                if value.device != target:
                    # No pinned memory, streams, async copies, casts, or step-tensor moves.
                    state[key] = value.to(device=target, non_blocking=False)
            if state['step'] is not clock or clock.device != clock_device or clock.dtype != clock_dtype or float(clock) != clock_value:
                raise AssertionError('Moment placement changed Adam step tensor')
        if any(p.device.type == 'cuda' for g in self.optimizer.param_groups for p in g['params']):
            torch.cuda.synchronize()
        self.events.append({'action': 'park' if to_cpu else 'restore', 'bytes': total,
                            'seconds': time.monotonic()-started})

    def park(self):
        self.move(True)

    def _before(self, optimizer, args, kwargs):
        assert optimizer is self.optimizer
        self.move(False)

    def _after(self, optimizer, args, kwargs):
        assert optimizer is self.optimizer
        self.park()

    def close(self):
        # Also called after a failed original step (PyTorch post-hooks do not run then).
        try:
            self.park()
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()


def batch_order(rows):
    queues = defaultdict(list)
    for row in rows:
        queues[row['domain']].append(row)
    ordered = [row for _, _, row in sorted(((i/len(values), domain, row)
        for domain, values in queues.items() for i, row in enumerate(values)), key=lambda x:(x[0], x[1]))]
    order = list(range(len(ordered))); random.Random(SEED+1).shuffle(order)
    return [ordered[i] for i in order]


def input_record(item):
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    return {'item_id': item.item_id, 'full_context_sha256': digest(item.in_context_messages),
            'ac_messages_sha256': digest(item.ac_messages), 'gold_ids_sha256': digest(item.assistant_token_ids),
            'info': {key: item.info.get(key) for key in ('encoded_source_tokens','compressed_rows','part_source_tokens','part_rows','passages')}}


def memory():
    import torch
    torch.cuda.synchronize()
    return {'allocated_gib': torch.cuda.memory_allocated()/2**30,
            'reserved_gib': torch.cuda.memory_reserved()/2**30,
            'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'free_gib': torch.cuda.mem_get_info()[0]/2**30}


def payload_paths(pair):
    ac, reader = Path(pair['ac_checkpoint']), Path(pair['reader_checkpoint'])
    return {'ac_modules': ac/'ac_modules.pt', 'encoder_config': ac/'side_lora/adapter_config.json',
        'encoder_weights': ac/'side_lora/adapter_model.safetensors', 'reader_config': reader/'adapter_config.json',
        'reader_weights': reader/'adapter_model.safetensors', 'optimizer': ac/'optimizer.pt'}


def validate_inputs(args):
    manifest = read(args.manifest); design = read(args.design); pair = read(args.pair)
    for key, path in [('design', args.design), ('pair', args.pair), ('material', args.material), ('subset', args.subset)]:
        if sha(path) != manifest['input_sha256'][key]:
            raise ValueError('Pinned input changed: '+key)
    for name, expected in manifest['helper_sha256'].items():
        if sha(Path(__file__).parent/name) != expected:
            raise ValueError('Pinned helper changed: '+name)
    if sha(__file__) != manifest['worker_sha256']:
        raise ValueError('Probe worker changed')
    payload = {name: sha(path) for name, path in payload_paths(pair).items()}
    if payload != {**design['initial_payload_sha256'], 'optimizer': manifest['optimizer_sha256']}:
        raise ValueError('42k paired weights/actual Adam bytes differ')
    return manifest, design, pair, payload


@contextmanager
def instrument(trainer, training_module, receipt, *, offload, proof, penalty_offset=0):
    """Observe the unchanged train loop; only state placement is changed for the enrolled optimizer."""
    import torch
    original_persistent = training_module.persistent_optimizer
    original_state_to = training_module.optimizer_state_to
    original_clip = torch.nn.utils.clip_grad_norm_
    original_head = training_module.head_logits
    original_tune = trainer._autotuning_variables
    original_loss = trainer._example_loss
    original_sample = trainer._sample_completions
    original_evaluate = trainer._evaluate
    holder = {}; hooks = []; raw_gradients = []; clipped_gradients = []
    receipt.update(steps=[], loss_calls=[], logits_calls=[], placements=[], transfer_events=[], tuning_calls=0, decode_calls=0)

    def persistent(*args, **kwargs):
        optimizer = original_persistent(*args, **kwargs)
        if 'optimizer' in holder:
            raise AssertionError('Only one optimizer instance is permitted')
        holder['optimizer'] = optimizer
        if offload:
            holder['offloader'] = MomentOffload(optimizer, receipt['transfer_events'])
        def before_step(opt, args, kwargs):
            holder['step_began'] = time.monotonic()
            holder['gradient_state_guard'] = check_optimizer_progress(opt,5250+len(receipt['steps']),
                require_gradients=True,expected_state_ids=receipt['initial_parameter_ids']['populated_state_ids'])
            receipt['placements'].append({'phase': 'before_original_adam', **memory()})
            if proof:
                clipped_gradients.append(parameter_record(opt, gradients=True))
        def after_step(opt, args, kwargs):
            # Enrolled offloader post-hook runs first, returning moments to CPU.
            row = {'seconds_original_step_including_post_offload': time.monotonic()-holder['step_began'],
                   'clocks': clocks(opt), 'memory': memory(),
                   'gradient_state_guard':holder['gradient_state_guard'],
                   'post_step_clock_guard':check_optimizer_progress(opt,5251+len(receipt['steps']),require_gradients=True,
                       expected_state_ids=receipt['initial_parameter_ids']['populated_state_ids'])}
            if proof:
                row.update(parameters=parameter_record(opt), optimizer=optimizer_record(opt))
            receipt['steps'].append(row)
            write(holder['output'], receipt)
        hooks.extend([optimizer.register_step_pre_hook(before_step), optimizer.register_step_post_hook(after_step)])
        return optimizer

    def state_to(optimizer, device):
        result = original_state_to(optimizer, device)
        if optimizer is holder.get('optimizer') and torch.device(device).type == 'cuda':
            if holder.get('restored'):
                raise AssertionError('Unexpected second resident-state restore/autotuning point')
            if len(optimizer.state) != 615 or {float(s['step']) for s in optimizer.state.values()} != {5250.0}:
                raise AssertionError('Must inherit all615 actual Adam5250 states before the first baseline')
            holder['restored'] = True
            receipt['initial_clocks'] = clocks(optimizer)
            receipt['initial_parameter_ids'] = check_optimizer_progress(optimizer,5250)
            receipt['placements'].append({'phase': 'after_exact_adam_restore', **memory()})
            if offload:
                holder['offloader'].park()
            receipt['placements'].append({'phase': 'before_initial_baseline', **memory()})
            receipt['initial_moment_devices'] = sorted({str(state[k].device) for state in optimizer.state.values() for k in MOMENTS})
        return result

    def tune(model):
        if 'optimizer' in holder:
            raise AssertionError('Autotuning must occur at original pre-optimizer point')
        receipt['tuning_calls'] += 1
        if receipt['tuning_calls'] != 1:
            raise AssertionError('No retuning after moment offload')
        value = original_tune(model)
        receipt['math_policy'] = {key: value[key] for key in ('logits_chunk_tokens', 'fla_config',
            'side_gradient_checkpointing_min_tokens', 'target_gradient_checkpointing_min_tokens')}
        receipt['autotuning_provenance'] = value['provenance']
        expected = getattr(trainer, '_probe_expected_math_policy', None)
        if expected is not None and receipt['math_policy'] != expected:
            raise AssertionError('Ordinary32k policy differs from proved same-host policy; stop before baseline/update')
        return value

    def head(states, head, fp32=False):
        # Actual calls, including checkpoint recomputations, teacher targets and each evaluation view.
        receipt['logits_calls'].append({'states_shape': list(states.shape), 'dtype': str(states.dtype),
            'fp32': fp32, 'grad_enabled': torch.is_grad_enabled(),
            'resolved_chunk_tokens': trainer._logits_chunk_tokens})
        return original_head(states, head, fp32)

    def loss(model, example, device, **kwargs):
        if torch.is_grad_enabled() and not holder.get('offset_applied'):
            for _ in range(penalty_offset): trainer._penalty_rng.random()
            holder['offset_applied'] = True
        if holder.get('restored') and offload:
            if any(state[k].device.type != 'cpu' for state in holder['optimizer'].state.values() for k in MOMENTS):
                raise AssertionError('Moments must be CPU resident throughout baseline/forward/backward')
        value = original_loss(model, example, device, **kwargs)
        receipt['loss_calls'].append({'item_id': example.item.item_id, 'grad_enabled': torch.is_grad_enabled(),
            'objective': float(value[0].detach()), 'target_logp': tensor_record(value[3]),
            'math_flags': {'tf32_matmul': torch.backends.cuda.matmul.allow_tf32, 'tf32_cudnn': torch.backends.cudnn.allow_tf32,
                           'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction},
            'penalty': float(trainer._penalty_term.detach()) if torch.is_grad_enabled() and trainer._penalty_term is not None else None})
        return value

    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        if proof:
            raw_gradients.append(parameter_record(holder['optimizer'], gradients=True))
        receipt['placements'].append({'phase': 'after_backward_before_clip', **memory()})
        return original_clip(parameters, *args, **kwargs)

    def samples(model, items, progress):
        if items:
            raise AssertionError('Probe forbids decoding; completion_samples must be zero')
        return original_sample(model, items, progress)

    def evaluate(*args, **kwargs):
        receipt['placements'].append({'phase': 'evaluation_start', **memory()})
        result = original_evaluate(*args, **kwargs)
        receipt['placements'].append({'phase': 'evaluation_end', **memory()})
        return result

    training_module.persistent_optimizer = persistent; training_module.optimizer_state_to = state_to
    training_module.head_logits = head; torch.nn.utils.clip_grad_norm_ = clip
    trainer._autotuning_variables = tune; trainer._example_loss = loss
    trainer._sample_completions = samples; trainer._evaluate = evaluate
    try:
        yield holder
    finally:
        try:
            if 'offloader' in holder:
                holder['offloader'].close()
        finally:
            for hook in hooks: hook.remove()
            training_module.persistent_optimizer = original_persistent; training_module.optimizer_state_to = original_state_to
            training_module.head_logits = original_head; torch.nn.utils.clip_grad_norm_ = original_clip
            for key in ('_autotuning_variables','_example_loss','_sample_completions','_evaluate'):
                trainer.__dict__.pop(key, None)
            receipt['raw_gradients'] = raw_gradients; receipt['clipped_gradients'] = clipped_gradients


def compare_proofs(records):
    keys = ('inputs','reporting_inputs','math_policy','loss_calls','logits_calls','steps',
            'raw_gradients','clipped_gradients','initial_clocks','initial_parameter_ids','final_optimizer','save_reload_exact')
    def scientific(record, key):
        if key == 'steps':
            return [{k:v for k,v in row.items() if k not in ('memory','seconds_original_step_including_post_offload')}
                    for row in record[key]]
        return record[key]
    base = records[0]
    comparisons = {record['condition']: {key: scientific(base,key) == scientific(record,key) for key in keys}
                   for record in records[1:]}
    return {'passed': all(all(row.values()) for row in comparisons.values()), 'exact_checks': comparisons,
            'scope': 'Same-GPU deterministic native2k, micro1, checkpoint0, original first16 training questions, two real Adam updates. Not32k/cross-device equivalence.'}


def worker(args):
    import torch
    proof = args.worker in PROOFS; offload = args.worker != 'resident' and args.worker != 'resident_repeat'
    if proof:
        if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8' or torch.cuda.is_initialized():
            raise ValueError('Deterministic setup must precede CUDA initialization')
        torch.use_deterministic_algorithms(True, warn_only=False)
    elif os.environ.get('CUBLAS_WORKSPACE_CONFIG') or torch.are_deterministic_algorithms_enabled():
        raise ValueError('Fit must retain ordinary production numerical flags')
    import run_qa_pi42k_merged32k_v1 as bench
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training
    import activation.ac_model.ac_model_utils as utils
    import activation.agent_training.agent_training_utils as optim_utils
    manifest, design, pair, payload = validate_inputs(args)
    actual = {key: sha(module.__file__) for key,module in [('ac_model',ac_module),('trainer',training),('utils',utils)]}
    if actual != design['source_sha256'] or sha(optim_utils.__file__) != manifest['optimizer_helper_sha256']:
        raise ValueError('Frozen scientific source differs')
    output = args.output/(args.worker+'.json')
    if output.exists(): raise FileExistsError(output)
    receipt = {'status':'running','condition':args.worker,'input_sha256':manifest['input_sha256'],
        'payload_sha256':payload,'worker_sha256':sha(__file__),'source_sha256':actual,
        'device':torch.cuda.get_device_name(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'torch_version':torch.__version__,'decode_calls':0}
    write(output,receipt); began = time.monotonic()
    try:
        material = bench.selected_material(read(args.material),read(args.subset))
        ordered = batch_order(material['train'])
        if proof: material['train'] = ordered[:16]
        else: material['train'] = ordered
        h = bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=65536)
        ac = h.module_manager.get_ac_model(bench.AC); ac.prepare(); h.module_manager.ensure_lora(bench.READER)
        receipt['inherited_optimizer_guard'] = bench.check_inherited_optimizer(Path(pair['ac_checkpoint']),
            [ac.trainable_parameters(),h.module_manager.lora_parameters(bench.READER)])
        ac.epoch_number = 0
        study, train, held = bench.material_items(h,material,passage_parts=False,
            evidence_tokens=2048 if proof else 32768,max_passages=4 if proof else 64,ratio=1/16)
        # Strict original first16 order; fit batches preserve the original shuffled logical8 boundaries.
        batch_index = 0
        if args.worker == 'fit_worst':
            # Maximum single-example live geometry: encoder+compressed reader+gold. Micro1 means no padded batch.
            geometries = [item.info['encoded_source_tokens']+item.info['compressed_rows']+len(item.assistant_token_ids[0]) for item in train]
            index = max(range(len(train)),key=lambda i:(geometries[i],i)); batch_index = index//8
            receipt['worst_geometry_selection'] = {'definition':'max encoder tokens + compressed rows + gold tokens; original logical8 containing that item',
                'maximum_item_index':index,'proxy':geometries[index], 'caveat':'A geometry proxy, not an exhaustive proof of all960 examples.'}
        if not proof: train = train[batch_index*8:(batch_index+1)*8]
        config = training.ActivationContextTrainingConfig(**design['training_config'])
        config = replace(config,shuffle_examples=False,checkpoint_every_epoch=False,checkpoint_every_updates=None,
                         max_duration_seconds=None,reporting_interval=1.0)
        trainer = training.ActivationContextTrainer(h,config)
        full_candidates = study.transform_for_eval_reference(held,'full_context')
        held_index = max(range(len(held)),key=lambda i:len(trainer.build_example(ac,full_candidates[i]).student_ids))
        reporting = [held[held_index]]; full = [full_candidates[held_index]]
        none = study.transform_for_eval_reference(reporting,'no_context')
        receipt.update(training_config=asdict(config),selected_batch_index=batch_index,inputs=[input_record(x) for x in train],
            reporting_inputs=[input_record(x) for x in reporting],
            diagnostic_overrides='Original shuffled batch supplied directly; shuffle disabled only to avoid reshuffling selected diagnostic subset. No model/checkpoint writes; one final reporting boundary.',
            production_flags={'tf32_matmul':True,'tf32_cudnn':True,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction})
        if not proof:
            prior = read(args.output/'summary.json')
            if not prior.get('exact_proof',{}).get('passed'):
                raise AssertionError('32k fit requires completed exact resident-repeat/offload proof')
            trainer._probe_expected_math_policy = read(args.output/'resident.json')['math_policy']
        def callback(progress, summaries):
            # Existing _evaluate under the same active runtime; never call public evaluate (which retunes).
            receipt.setdefault('reader_controls',[]).append({'global_step':progress.global_step,
                'full':trainer._evaluate(ac,full,ac.device),'none':trainer._evaluate(ac,none,ac.device)})
        with instrument(trainer,training,receipt,offload=offload,proof=proof,penalty_offset=batch_index*8) as holder:
            holder['output'] = output
            stats = trainer.train(bench.AC,train,reporting_data=reporting,reporting_callback=callback)
            optimizer = holder['optimizer']
            expected = 2 if proof else 1
            if stats.steps != expected or stats.stop_reason != 'complete' or stats.checkpoint_path is not None or stats.target_lora_checkpoint_path is not None:
                raise AssertionError('Probe must complete exact diagnostic steps without model checkpoint publication')
            if trainer.updates_done[bench.AC] != expected or len(optimizer.state) != 615:
                raise AssertionError('Stage clock/state count differs')
            receipt['final_optimizer'] = optimizer_record(optimizer)
            receipt['final_clocks'] = clocks(optimizer)
            receipt['final_clock_guard'] = check_optimizer_progress(optimizer,5250+expected,
                expected_state_ids=receipt['initial_parameter_ids']['populated_state_ids'])
            # Real CPU serialization plus load_state_dict; normalize only device placement for exact tensor checks.
            checkpoint = args.output/(args.worker+'.optimizer_roundtrip.pt')
            torch.save({'optimizer':optimizer.state_dict(),'updates_done':expected,'epoch_number':1},checkpoint)
            saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
            optimizer.load_state_dict(saved['optimizer']); optim_utils.optimizer_state_to(optimizer,'cpu')
            receipt['save_reload_exact'] = (optimizer_record(optimizer)==receipt['final_optimizer'] and saved['updates_done']==expected and saved['epoch_number']==1)
            checkpoint.unlink()
            if not receipt['save_reload_exact']: raise AssertionError('Save/reload changed optimizer values/groups')
            receipt.update(step_seconds=stats.step_seconds,loss=stats.loss,grad_norm=stats.grad_norm,
                grad_norm_ac=stats.grad_norm_ac,grad_norm_reader=stats.grad_norm_reader,
                learning_rates=stats.learning_rates,penalty_counts=stats.penalty_counts,
                peak_memory_gib=stats.peak_memory_bytes/2**30,resumed_from=stats.resumed_from)
        if {name:sha(path) for name,path in payload_paths(pair).items()} != payload:
            raise AssertionError('Frozen parent payload was modified')
        receipt.update(status='complete',elapsed_seconds=time.monotonic()-began,parent_payload_unchanged=True)
        write(output,receipt)
    except BaseException as error:
        receipt.update(status='failed',error=repr(error),elapsed_seconds=time.monotonic()-began)
        write(output,receipt); raise


def controller(args):
    args.output.mkdir(parents=True,exist_ok=True)
    path = args.output/'summary.json'
    if path.exists(): raise FileExistsError(path)
    summary = {'status':'running','scope':'Private probe only; no production adoption','conditions':{},'seconds_budget':args.seconds}
    write(path,summary); began = time.monotonic()
    for condition in (*PROOFS,*FITS):
        remaining = args.seconds-(time.monotonic()-began)
        if remaining < 120:
            summary['status']='partial_budget';write(path,summary);return 1
        workspace = args.workspace/condition
        workspace.mkdir(parents=True,exist_ok=False)
        (workspace/'AUTOTUNE').symlink_to(args.autotune_root,target_is_directory=True)
        env = dict(os.environ);env['ACTIVATION_SYNC_ROOT']=str(workspace)
        if condition in PROOFS: env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        else: env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',condition]
        for name in ('manifest','design','pair','material','subset','output'):
            command += ['--'+name,str(getattr(args,name))]
        with (args.output/(condition+'.log')).open('x') as log:
            try:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining-30)
                record={'returncode':result.returncode}
            except subprocess.TimeoutExpired:record={'timeout':True}
        result_path=args.output/(condition+'.json')
        result=read(result_path) if result_path.exists() else{}
        record['status']=result.get('status','missing');summary['conditions'][condition]=record;write(path,summary)
        if record.get('returncode')!=0 or record['status']!='complete':
            summary['status']='failed';write(path,summary);return 1
        if condition=='offloaded':
            summary['exact_proof']=compare_proofs([read(args.output/(name+'.json')) for name in PROOFS])
            write(path,summary)
            if not summary['exact_proof']['passed']:
                summary['status']='numerical_gate_failed';write(path,summary);return 1
        if condition in FITS and result['math_policy']!=read(args.output/'resident.json')['math_policy']:
            summary['status']='math_policy_changed';write(path,summary);return 1
    summary.update(status='complete',elapsed_seconds=time.monotonic()-began,
        interpretation='Exact equality only for tested native2k deterministic case.32k workers test production-flag fit/time, not32k resident equivalence or whole960 fit.')
    write(path,summary);return 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',choices=(*PROOFS,*FITS))
    for name in ('manifest','design','pair','material','subset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workspace',type=Path);parser.add_argument('--autotune-root',type=Path)
    parser.add_argument('--seconds',type=int,default=2520)
    args=parser.parse_args()
    return worker(args) if args.worker else controller(args)

if __name__=='__main__':
    raise SystemExit(main())
