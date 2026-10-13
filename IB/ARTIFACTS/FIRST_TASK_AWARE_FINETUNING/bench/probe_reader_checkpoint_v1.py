"""Isolated reader checkpoint threshold0→4096; encoder stays0, packing stays65536.

Uses the existing strong32k paired4000 loader and one recorded real8k batch.
Strict deterministic proof precedes ordinary production-kernel timing processes.
No product/source edits, parent saves, checkpoint selection or new training arm.
"""
import argparse
from dataclasses import replace
import json,os,statistics,subprocess,sys,tempfile,time
from pathlib import Path
import probe_strong_piecewise_budget_v1 as base

BASE_SHA='377e022e868dce0ae64fed25e639e886276189b67acfc1832003982758faf381'
previous=base.previous
SEED=base.SEED
write,sha,read=base.write,base.sha,base.read
penalty_ids=base.penalty_ids


def prove_reader_case(h, ac, trainer, examples, case, *, budgets=(65536,), cpu=False, flush=lambda _: None):
    import torch
    from activation.agent_training.agent_training_utils import disable_dropout
    from activation.harness.hf_utils import decoder_layers
    if budgets != (65536,):raise ValueError('Packing must remain65536')
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
    variants = [('baseline',0),('unchanged_repeat',0),('reader4096',4096)]
    try:
        for label, reader_threshold in variants:
            ac.config = replace(original_config, encode_batch_max_tokens=65536)
            trainer._target_gradient_checkpointing_min_tokens=reader_threshold
            for _, p in params: p.grad = None
            trainer._penalty_rng.seed(SEED)
            for _ in range(offset): trainer._penalty_rng.random()
            torch.manual_seed(SEED); outputs = []; trace = []; selected = []; loss_value = 0.; recomputations=[]
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
                        trace_start=len(trace)
                        outcomes = trainer._batch_loss(ac, group, ac.device)
                        outputs.extend(value['target_logp'].detach().float().cpu().clone() for value in outcomes)
                        selected.extend(ex.item.item_id for ex, value in zip(group, outcomes) if value['penalty'] is not None)
                        loss = sum((value['objective']+(value['penalty'] if value['penalty'] is not None else 0.))*ex.item.weight/len(ordered)
                                   for ex, value in zip(group, outcomes))
                        loss_value += float(loss.detach()); before = dict(calls); loss.backward()
                        active=[row for row in trace[trace_start:] if row['kind'] in ('side','reader') and row['grad_enabled']]
                        expected={'side_encoder':sum(row['checkpoint'] for row in active if row['kind']=='side'),
                                  'reader':sum(row['checkpoint'] for row in active if row['kind']=='reader')}
                        actual={key:calls[key]-before[key] for key in calls}
                        if actual!=expected or not actual['side_encoder']:raise AssertionError('Checkpoint recomputation differs from observed gates')
                        recomputations.append({'expected':expected,'actual':actual})
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
            reader=[row for row in trace if row['kind']=='reader' and row['grad_enabled']]
            if any(row['checkpoint']!=(row['shape'][0]*row['shape'][1]>=reader_threshold) for row in reader):raise AssertionError('Reader whole-batch threshold not applied')
            if label=='reader4096' and any(row['checkpoint'] for row in reader):raise AssertionError('Candidate must disable every reader recomputation in this saved case')
            grads = [(name, None if p.grad is None else p.grad.detach().float().cpu().clone()) for name, p in params]
            record = {'budget':65536,'side_checkpoint_threshold':0,'reader_checkpoint_threshold':reader_threshold,'recomputations':recomputations,'loss': loss_value, 'selected_penalty_ids': selected, 'trace': trace,
                      'first_layer_calls': calls, 'last_microbatch_calls_before_backward': before_backward,
                      'parameter_metadata_unchanged': True}
            if not cpu and sum(value is not None for _,value in grads)!=615:raise AssertionError('Expected all615 active trainable gradients')
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
                record.update(all_trainable_gradients=checks, all_ac_outputs_and_gold_logps=output_checks,
                    loss_abs_difference=abs(loss_value-reference[0]),
                    passed=abs(loss_value-reference[0]) <= loss_tol and all(v['passed'] for v in checks.values())
                        and all(v['passed'] for v in output_checks))
            records[label] = record; flush(records)
            del grads, outputs
    finally:
        if saved_encode is sentinel: del ac.encode_batch
        else: ac.encode_batch = saved_encode
        ac.config = original_config
        trainer._target_gradient_checkpointing_min_tokens=0
        for _, p in params: p.grad = None
    return {'passed': all(v['passed'] for k, v in records.items() if k != 'baseline'), 'comparisons': records,
        'tolerances': {'relative_rms': rtol, 'absolute_rms_floor': atol, 'joint_loss_abs': loss_tol},
        'selected_penalty_ids': wanted_penalties, 'parameter_count': len(params),
        'scope':'Original65536 masks/precision. Side threshold0 both; reader0 versus4096. AC outputs, reader gold log-probs, joint loss and all gradients; deterministic proof only. No updates.'}


def worker(args):
    import torch
    condition=args.worker
    if condition not in ('proof','reader0','reader4096'):raise ValueError('Unknown bounded condition')
    threshold=4096 if condition=='reader4096' else 0
    # The reused proof loader enables deterministic execution before importing
    # model modules (whose GPU capability checks can initialize CUDA).
    original=None
    if condition!='proof':
        from activation.ac_model import ActivationContextTrainer
        original=ActivationContextTrainer._autotuning_variables
    def variables(self,ac):
        values=original(self,ac)
        if values['side_gradient_checkpointing_min_tokens']!=0 or values['target_gradient_checkpointing_min_tokens']!=0:
            raise ValueError('Expected original both-zero policy before private target override')
        return {**values,'target_gradient_checkpointing_min_tokens':threshold,
            'provenance':{**values['provenance'],'private_reader_checkpoint_probe':{
                'side_min_tokens':0,'target_min_tokens':threshold,'threshold_unit':'reader B×max padded sequence; side max sequence',
                'script_sha256':sha(__file__)}}}
    old_cases,old_budgets,old_proof=base.CASES,base.BUDGETS,base.prove_case
    base.CASES=('failing',);base.BUDGETS=(65536,);base.prove_case=prove_reader_case
    if original is not None:ActivationContextTrainer._autotuning_variables=variables
    args.output.mkdir(parents=True,exist_ok=True)
    policy={'condition':condition,'wrapper_sha256':sha(__file__),'loader_sha256':BASE_SHA,'side_threshold':0,
        'reader_threshold':threshold if condition!='proof' else [0,0,4096],
        'encoder_budget':65536,'scope':'one recorded eight-question case; proof then separate ordinary timing',
        'checkpoint_saves':False,'parent_updates':4000,'timing_final_updates':4004,'warmup_updates':1,'steady_updates':3}
    write(args.output/'policy.json',policy)
    args.worker='proof' if condition=='proof' else 'failing_65536'
    output=args.output/(args.worker+'.json')
    try:
        base.worker(args)
        receipt=read(output)
        receipt['reader_checkpoint_policy']=policy
        if condition!='proof':
            active=[row for row in receipt['trace'] if row['kind'] in ('side','reader') and row['grad_enabled']]
            if not active or any(not row['checkpoint'] for row in active if row['kind']=='side'):
                raise AssertionError('Timing disabled encoder checkpointing')
            reader=[row for row in active if row['kind']=='reader']
            if any(row['checkpoint']!=(row['shape'][0]*row['shape'][1]>=threshold) for row in reader):
                raise AssertionError('Timing target checkpoint gate differs')
            if threshold and any(row['checkpoint'] for row in reader):
                raise AssertionError('Candidate timing must disable every reader recomputation')
            receipt['reader_forward_summary']={'grad_enabled_calls':len(reader),'checkpointed_calls':sum(row['checkpoint'] for row in reader),
                'padded_positions':sum(row['shape'][0]*row['shape'][1] for row in reader),
                'checkpointed_padded_positions':sum(row['shape'][0]*row['shape'][1] for row in reader if row['checkpoint'])}
        write(output,receipt)
    except BaseException as error:
        if output.exists():
            receipt=read(output)
            receipt.update(base_worker_status=receipt['status'],status='failed',
                wrapper_error=f'{type(error).__name__}: {error}',reader_checkpoint_policy=policy)
            write(output,receipt)
        raise
    finally:
        if original is not None:ActivationContextTrainer._autotuning_variables=original
        base.CASES,base.BUDGETS,base.prove_case=old_cases,old_budgets,old_proof
        # The reused loader already rehashes successful workers. Preserve that
        # check even when a diagnostic gate raises before its normal finalizer.
        if output.exists():
            receipt=read(output)
            if not receipt.get('parent_payloads_rehashed_unchanged') and 'pair' in receipt:
                unchanged=(base.paired.payload_hashes(receipt['pair'])==receipt['payload_sha256']
                    and sha(Path(receipt['pair']['ac_checkpoint'])/'optimizer.pt')==receipt['optimizer_sha256'])
                receipt['parent_payloads_rehashed_unchanged']=unchanged;write(output,receipt)
                if not unchanged:raise AssertionError('Parent pair or optimizer changed')


def controller(args):
    args.output.mkdir(parents=True,exist_ok=True);summary_path=args.output/'summary.json'
    if summary_path.exists():raise FileExistsError('Preserve existing probe outputs')
    began=time.monotonic();summary={'status':'running','wrapper_sha256':sha(__file__),'conditions':{},'seconds_budget':args.seconds}
    write(summary_path,summary)
    for condition in ('proof','reader0','reader4096'):
        remaining=args.seconds-(time.monotonic()-began)
        if remaining<120:summary['status']='partial_budget';write(summary_path,summary);return 1
        workspace=args.workspace/condition;workspace.mkdir(parents=True,exist_ok=True)
        if (workspace/'AC_MODELS').exists():raise FileExistsError('Requires unused private workspace')
        if not (workspace/'AUTOTUNE').exists():(workspace/'AUTOTUNE').symlink_to(args.autotune_root,target_is_directory=True)
        folder=args.output/condition;folder.mkdir(parents=True,exist_ok=True)
        env=dict(os.environ);env['ACTIVATION_SYNC_ROOT']=str(workspace)
        if condition=='proof':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        else:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',condition,'--output',str(folder)]
        for name in ('material','bench','design','parent_receipt','memory_receipt','reports','runs_root'):
            command+=['--'+name.replace('_','-'),str(getattr(args,name))]
        with (folder/'worker.log').open('x') as log:
            try:
                result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining-30)
                status='complete' if result.returncode==0 else 'failed'
            except subprocess.TimeoutExpired:status='timeout'
        output=folder/('proof.json' if condition=='proof' else 'failing_65536.json')
        receipt=read(output) if output.exists() else {}
        summary['conditions'][condition]={'process_status':status,'worker_status':receipt.get('status'),'output':str(output)}
        write(summary_path,summary)
        if status!='complete' or receipt.get('status')!='complete':
            summary['status']='proof_failed' if condition=='proof' else 'timing_failed';write(summary_path,summary);return 1
        if condition=='proof' and not receipt['cases']['failing']['passed']:
            summary['status']='proof_failed';write(summary_path,summary);return 1
    proof=read(args.output/'proof/proof.json');control=read(args.output/'reader0/failing_65536.json');candidate=read(args.output/'reader4096/failing_65536.json')
    for key in ('payload_sha256','optimizer_sha256','selected_penalty_ids','learning_rates','cases'):
        if control[key]!=candidate[key]:raise AssertionError('Paired timing inputs/state/draws differ: '+key)
    if any(record[key]!=proof[key] for record in (control,candidate) for key in ('payload_sha256','optimizer_sha256')):
        raise AssertionError('Timing weights/state differ from proof')
    def geometry(record):
        return [{key:value for key,value in row.items() if key not in ('allocated_gib_before','checkpoint')}
                for row in record['trace']]
    if geometry(control)!=geometry(candidate):raise AssertionError('Timing shapes/masks/encoder packing changed')
    summary.update(status='complete',elapsed_seconds=time.monotonic()-began,
        control_steady_median_seconds=control['steady_median_seconds'],candidate_steady_median_seconds=candidate['steady_median_seconds'],
        measured_speedup=control['steady_median_seconds']/candidate['steady_median_seconds'],
        control_peak_gib=control['peak_memory_gib'],candidate_peak_gib=candidate['peak_memory_gib'],
        interpretation='One saved production-size logical batch, one warmup+three updates per condition. No training adoption or full-run speed guarantee.')
    write(summary_path,summary);return 0


def cpu_check():
    import torch
    from activation.tests.test_interventions import build,tiny_hybrid,trainer_and_examples,open_heads,open_attention
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True,warn_only=False)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        h,ac=build(root/'state',tiny_hybrid(root/'reader'),frequency=-1,input_pooling_stride=0,
            side_dir=tiny_hybrid(root/'side',hidden=16,seed=112))
        open_heads(ac);open_attention(ac)
        trainer,_,examples,_=trainer_and_examples(h,ac,micro_batch_examples=2,completion_samples=0)
        proof=prove_reader_case(h,ac,trainer,examples,'quantiles',cpu=True)
        if not proof['passed']:raise AssertionError('TinyCPU checkpoint-policy proof failed')
        print(json.dumps({'status':'passed','scope':'Real tinyCPU models with checkpoint recomputation; no GPU or full-model equivalence claim','proof':proof}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=('proof','reader0','reader4096'));parser.add_argument('--cpu-check',action='store_true')
    for name in ('material','bench','design','parent-receipt','memory-receipt','reports','runs-root','output','workspace','autotune-root'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--seconds',type=int,default=840);args=parser.parse_args()
    if (sha(base.__file__)!=BASE_SHA or sha(previous.__file__)!=base.PREVIOUS_SHA or sha(base.paired.__file__)!=base.PAIRED_SHA
            or sha(Path(__file__).with_name('probe_piecewise_original_v1.py'))!=previous.HELPER_SHA):
        raise ValueError('Frozen diagnostic loader/helpers changed')
    if args.cpu_check:cpu_check()
    elif args.worker:worker(args)
    else:sys.exit(controller(args))
