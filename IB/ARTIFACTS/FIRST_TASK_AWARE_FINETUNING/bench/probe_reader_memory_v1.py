"""Candidate4096 only: worst actual8k-context batches, fresh parent process per case.

Reuses the completed exact numerical proof. Each private process performs one
ordinary full update with inherited Adam4000, saves nothing, and rehashes parent.
This is a finite-pool memory fit, not another numerical or timing comparison.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import probe_strong_piecewise_budget_v1 as base
import reader_checkpoint_geometry_v1 as census

BASE_SHA = '377e022e868dce0ae64fed25e639e886276189b67acfc1832003982758faf381'
write, read, sha = base.write, base.read, base.sha


def validate_proof(proof):
    if proof.get('status') != 'complete' or not proof.get('parent_payloads_rehashed_unchanged'):
        raise ValueError('Completed original reader checkpoint proof required')
    case = proof['cases']['failing']
    if case['tolerances'] != {'relative_rms': .03, 'absolute_rms_floor': 2e-6, 'joint_loss_abs': .001}:
        raise ValueError('Numerical acceptance policy changed')
    for label in ('unchanged_repeat', 'reader4096'):
        row = case['comparisons'][label]
        active = [v for v in row['all_trainable_gradients'].values() if not v.get('both_absent')]
        if (not row['passed'] or row['loss_abs_difference'] != 0 or len(active) != 615
                or any(v['max_abs'] != 0 for v in active + row['all_ac_outputs_and_gold_logps'])):
            raise ValueError('Expected the actual exact GPU proof, not an unrun or tolerance-only candidate')
    if (case['comparisons']['reader4096']['side_checkpoint_threshold'] != 0
            or case['comparisons']['reader4096']['reader_checkpoint_threshold'] != 4096):
        raise ValueError('Wrong checkpoint policy')


def worker(args):
    import torch
    if torch.are_deterministic_algorithms_enabled() or os.environ.get('CUBLAS_WORKSPACE_CONFIG'):
        raise ValueError('Memory fit must retain ordinary production execution')
    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError('Original BF16 reduction policy is required')
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as train_module
    import activation.ac_model.ac_model_utils as utils_module
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingItem
    from activation.autotuners import configure_fla_runtime
    import run_qa_strong_long_v1 as bench

    packet = read(args.cases); prior = read(args.proof); validate_proof(prior)
    receipt = read(args.census)
    if (receipt['status'] != 'complete' or sha(args.cases) != receipt['selected_cases_sha256']
            or receipt['identity'] != packet['identity']):
        raise ValueError('Census/case publication mismatch')
    source = {key: sha(module.__file__) for key,module in [('ac_model',ac_module),('trainer',train_module),('utils',utils_module)]}
    if source != packet['identity']['source_sha256'] or source != prior['source_sha256'] or sha(bench.__file__) != packet['identity']['builder_sha256']:
        raise ValueError('Source/builder differs from census and completed numerical proof')
    case = packet['cases'][args.worker]
    pair, publication = base.strong_final(args.reports,args.runs_root,read(args.parent_receipt))
    payload = base.paired.payload_hashes(pair); optimizer_path = Path(pair['ac_checkpoint'])/'optimizer.pt'
    optimizer_sha = sha(optimizer_path)
    if payload != prior['payload_sha256'] or optimizer_sha != prior['optimizer_sha256'] or pair != prior['pair']:
        raise ValueError('Original32k paired weights/Adam differ from completed proof')
    output = args.output/(args.worker+'.json')
    if output.exists(): raise FileExistsError(output)
    record = {'status':'running','case':args.worker,'worker_sha256':sha(__file__),
        'census_sha256':sha(args.census),'cases_sha256':sha(args.cases),'prior_proof_sha256':sha(args.proof),
        'source_sha256':source,'pair':pair,'payload_sha256':payload,'optimizer_sha256':optimizer_sha,
        'publication':publication,'device':torch.cuda.get_device_name(),'trace':[],
        'scope':'One ordinary full update on a CPU-selected real batch, actual Adam4000; no saves. No new numerical comparison or adoption.'}
    write(output,record); original = ActivationContextTrainer._autotuning_variables
    def variables(self,ac):
        values=original(self,ac)
        if values['side_gradient_checkpointing_min_tokens'] != 0 or values['target_gradient_checkpointing_min_tokens'] != 0:
            raise ValueError('Original long-context threshold must be0 for both sides')
        return {**values,'target_gradient_checkpointing_min_tokens':4096,
            'provenance':{**values['provenance'],'private_reader_checkpoint_threshold':4096}}
    ActivationContextTrainer._autotuning_variables=variables
    try:
        h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=65536)
        ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();h.module_manager.ensure_lora(bench.READER)
        state=torch.load(optimizer_path,map_location='cpu',weights_only=False)
        record['optimizer_guard']=base.optimizer_guard(state,[ac.trainable_parameters(),h.module_manager.lora_parameters(bench.READER)],bench.validate_optimizer_layout)
        del state
        config=ActivationContextTrainingConfig(**read(args.reports/base.ARM/'training_config.json'))
        config=replace(config,num_epochs=1,shuffle_examples=False,resume_optimizer=True,resume_skip_steps=0,
            checkpoint_every_epoch=False,checkpoint_every_updates=None,completion_samples=0,max_duration_seconds=None,
            gradient_checkpointing_min_tokens=0)
        trainer=ActivationContextTrainer(h,config)
        items=[ActivationContextTrainingItem(**row) for row in case['items']]
        if [item.item_id for item in items] != case['logical_item_ids']: raise ValueError('Selected logical order changed')
        examples=[trainer.build_example(ac,item) for item in items]
        meta={}
        for item,ex in zip(items,examples):
            expected_item=case['item_geometry'][item.item_id]
            if (census.digest(asdict(item))!=expected_item['item_sha256']
                    or census.digest(ex.student_ids)!=expected_item['student_ids_sha256']
                    or census.digest(ex.penalty.student_ids)!=expected_item['penalty_ids_sha256']):
                raise ValueError('Actual prompt/target/penalty token IDs differ from CPU census')
            meta[item.item_id]={'reader_length':len(ex.student_ids),'penalty_length':len(ex.penalty.student_ids),
                'parts':list(zip(item.info['part_source_tokens'],item.info['part_rows'])),
                'compressed_rows':item.info['compressed_rows']}
        order=sorted(examples,key=lambda ex:len(ex.student_ids))
        actual_geometry=[]
        for offset in range(0,8,2):
            group=order[offset:offset+2]; expected=case['microbatches'][offset//2]
            if [ex.item.item_id for ex in group] != expected['item_ids']: raise ValueError('Reader microbatch order changed')
            if [len(ex.student_ids) for ex in group] != expected['reader_lengths']: raise ValueError('Reader history tokenization changed')
            parts=[part for ex in group for part in meta[ex.item.item_id]['parts']]
            if census.packed_parts(parts) != expected['encoder_batches']: raise ValueError('Encoder packing changed')
            actual_geometry.append(expected)
        for _ in range(case['questions_seen_before']): trainer._penalty_rng.random()
        expected_penalties=[ident for row in case['microbatches'] for ident in row['penalty_ids']]
        selected=[]; original_loss=trainer._batch_loss
        def record_loss(model,group,device,**kwargs):
            outcomes=original_loss(model,group,device,**kwargs)
            selected.extend(ex.item.item_id for ex,value in zip(group,outcomes) if value['penalty'] is not None)
            return outcomes
        trainer._batch_loss=record_loss
        with base.previous.trace_calls(ac,record['trace']):
            stats=trainer.train(bench.AC,items,reporting_data=[])
        if (stats.stop_reason != 'complete' or stats.steps != 1 or stats.resumed_from != str(optimizer_path) or trainer.updates_done[bench.AC] != 4001
                or stats.checkpoint_path is not None or stats.target_lora_checkpoint_path is not None or selected != expected_penalties):
            raise AssertionError('Memory fit changed optimizer/update/penalty protocol')
        if any(abs(a-5e-6)>1e-18 or abs(b-2e-6)>1e-18 for a,b in stats.learning_rates): raise AssertionError('Parent terminal rates changed')
        optimizer=trainer.optimizers[bench.AC]
        if len(optimizer.state) != 615 or {float(value['step']) for value in optimizer.state.values()} != {4001.}:
            raise AssertionError('Actual final Adam clocks do not match one full update')
        active=[row for row in record['trace'] if row.get('grad_enabled')]
        if any(not row['checkpoint'] for row in active if row['kind']=='side'): raise AssertionError('Encoder checkpointing disabled')
        if any(row['checkpoint'] != (row['shape'][0]*row['shape'][1]>=4096) for row in active if row['kind']=='reader'):
            raise AssertionError('Reader checkpoint gate changed')
        prepared=[row for row in record['trace'] if row['kind']=='prepared']
        expected_prepared=[row for micro in case['microbatches'] for row in micro['encoder_batches']]
        observed=[{'parts':row['batch'],'content_lengths':row['content_lengths'],'rows':row['rows'],
            'padded_positions':row['padded_tokens'],'valid_positions':sum(row['sequence_lengths'])} for row in prepared]
        if observed!=expected_prepared: raise AssertionError('Actual encoder forward geometry differs from census')
        record.update(status='complete',actual_geometry=actual_geometry,selected_penalty_ids=selected,
            step_seconds=stats.step_seconds,peak_allocated_gib=stats.peak_memory_bytes/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            total_device_gib=torch.cuda.get_device_properties(0).total_memory/2**30,
            actual_adam_steps_before=4000,actual_adam_steps_after=4001,all615_adam_clocks_checked=True,
            loss=stats.loss,learning_rates=stats.learning_rates,autotuning=stats.autotuning,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            precision={'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
                'tf32_cudnn':torch.backends.cudnn.allow_tf32,
                'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction})
    except BaseException as error:
        record.update(status='oom' if isinstance(error,torch.OutOfMemoryError) else 'failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        ActivationContextTrainer._autotuning_variables=original
        record['parent_payloads_rehashed_unchanged']=(base.paired.payload_hashes(pair)==payload and sha(optimizer_path)==optimizer_sha)
        write(output,record)
        if not record['parent_payloads_rehashed_unchanged']: raise AssertionError('Parent was mutated')


def controller(args):
    args.output.mkdir(parents=True,exist_ok=False);args.workspace.mkdir(parents=True,exist_ok=False)
    packet=read(args.cases);validate_proof(read(args.proof))
    if not 1<=len(packet['cases'])<=3: raise ValueError('Only the selected at-most-three cases')
    began=time.monotonic(); result={'status':'running','cases':{},'seconds_budget':args.seconds}
    write(args.output/'summary.json',result)
    for name in packet['cases']:
        left=args.seconds-(time.monotonic()-began)
        if left<120:
            result.update(status='partial_budget',elapsed_seconds=time.monotonic()-began)
            write(args.output/'summary.json',result);return 1
        work=args.workspace/name;work.mkdir();(work/'AUTOTUNE').symlink_to(args.autotune_root,target_is_directory=True)
        env=dict(os.environ);env['ACTIVATION_SYNC_ROOT']=str(work);env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        command=[sys.executable,__file__,'--worker',name,'--output',str(args.output)]
        for key in ('cases','census','proof','parent_receipt','reports','runs_root'):
            command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        with (args.output/(name+'.log')).open('x') as log:
            try:
                completed=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=left-30)
            except subprocess.TimeoutExpired:
                result['cases'][name]={'status':'timeout'};result['status']='timeout'
                write(args.output/'summary.json',result);return 1
        path=args.output/(name+'.json');record=read(path) if path.exists() else {}
        result['cases'][name]={'exit_code':completed.returncode,'status':record.get('status'),
            'peak_allocated_gib':record.get('peak_allocated_gib'),'step_seconds':record.get('step_seconds')}
        result['status']='failed' if completed.returncode else 'running';write(args.output/'summary.json',result)
        if completed.returncode: return 1  # OOM is a failed candidate fit, never silently switch policy.
    result.update(status='complete',elapsed_seconds=time.monotonic()-began,
        interpretation='Only these actual selected batches fit; no whole-pool safety or speed claim, no production adoption.')
    write(args.output/'summary.json',result);return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cases','census','proof','parent-receipt','reports','runs-root','output','workspace','autotune-root'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--worker');parser.add_argument('--seconds',type=int,default=840)
    args=parser.parse_args()
    if (sha(base.__file__) != BASE_SHA or sha(base.previous.__file__) != base.PREVIOUS_SHA
            or sha(base.paired.__file__) != base.PAIRED_SHA
            or sha(Path(__file__).with_name('probe_piecewise_original_v1.py')) != base.previous.HELPER_SHA):
        raise ValueError('Pinned mature-parent helpers changed')
    if args.worker: worker(args)
    else: sys.exit(controller(args))
