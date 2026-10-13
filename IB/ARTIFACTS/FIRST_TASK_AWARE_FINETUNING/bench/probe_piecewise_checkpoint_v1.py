"""Original-mask checkpoint-only proof and exact failed-batch fit; no model saves.

Every GPU worker starts from P2. The failed logical batch's DATA/order is exact;
its lost fifteen updates and optimizer moments are not reconstructed.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import time
import types

from probe_piecewise_original_v1 import (digest, write, parameters, compare_tensors,
    object_hash, AC_SHA, TRAINER_SHA, UTILS_SHA, MATERIAL_SHA, SEED)

BUILDER_SHA='86231c5ea0464a4de7c5fbb8cb386500ebd3c1a1374ea61fd41ea2af70d7981f'
HELPER_SHA='ebd7796c5b4a3812c0c9f05ab102257a03c6d2a7792fc7efc229ae2be642dcfe'
ORDER_SHA='eb9cbaf282c8698872d6d1891825ca988135642821a3d74a704039edd0f67e1e'
CONDITIONS={'original_failing':(True,'failing',1700,2),
            'pieces_failing':(True,'failing',0,3), 'pieces_quantiles':(True,'quantiles',0,3),
            'merged_failing':(False,'failing',0,3), 'merged_quantiles':(False,'quantiles',0,3)}


def compact_hash(value):
    import hashlib
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def ordered_rows(rows):
    groups=defaultdict(list)
    for row in rows:groups[row['domain']].append(row)
    return [row for _,_,row in sorted(((i/len(values),domain,row) for domain,values in groups.items() for i,row in enumerate(values)),key=lambda x:(x[0],x[1]))]


def select_material(material,failed_report):
    saved=json.loads((failed_report/'material_ids.json').read_text())
    identity=json.loads((failed_report/'run_identity.json').read_text())['identity']
    config=json.loads((failed_report/'training_config.json').read_text())
    if compact_hash(saved)!=ORDER_SHA or identity['material_ids_sha256']!=ORDER_SHA or identity['material_sha256']!=MATERIAL_SHA:
        raise ValueError('Failed run no longer matches the pinned frozen order/material')
    expected={'seed':SEED,'shuffle_examples':True,'num_epochs':1,'examples_per_update':8,'micro_batch_examples':2,
              'gradient_checkpointing_min_tokens':1700,'schedule_total_updates':500,'warmup_updates':50}
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Unexpected failed-run shuffle/batch/checkpoint recipe')
    rows=ordered_rows(material['train'])
    if [r['example']['example_id'] for r in rows]!=saved['train'] or len(rows)!=4000:
        raise ValueError('Exact frozen interleave could not be reproduced')
    order=list(range(4000));random.Random(SEED+1).shuffle(order)
    indexes=order[15*8:16*8]
    ranked=sorted(rows,key=lambda r:(int(r['source_tokens']),r['example']['example_id']))
    quantile_positions=[round(i*(len(ranked)-1)/7) for i in range(8)]
    return {'failing':[rows[i] for i in indexes],'quantiles':[ranked[i] for i in quantile_positions]}, {
        'original_logical_batch':16,'completed_prior_updates':15,'original_epoch':1,'shuffle_seed':SEED+1,
        'original_interleaved_indexes':indexes,'shuffled_index_order_sha256':compact_hash(order),
        'material_ids_sha256':ORDER_SHA,'metadata_sha256':{n:digest(failed_report/n) for n in ('material_ids.json','run_identity.json','training_config.json')},
        'quantile_positions':quantile_positions,'note':'Exact failed batch membership; trainer sorts these eight by reader length before grouping into microbatches of two. All probe weights/moments start fresh from P2.'}


def batch_geometry(examples,ac):
    """The frozen trainer's reader sort and frozen AC greedy content-token budget."""
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    ordered=sorted(examples,key=lambda x:len(x.student_ids));result=[]
    for start in range(0,len(ordered),2):
        group=ordered[start:start+2];parts=[]
        for ex in group:
            for request in ex.part_requests:
                n=len(tokenize_with_parts(ac.side.tokenizer,request.messages,[],ac.pad_id,tools=request.tools)[0])
                rows=ac.num_view_rows(n,request.compression_ratio);parts.append((n,rows))
        parts.sort(key=lambda p:p[0]);batches=[];current=[];longest=0
        for part in parts:
            if current and max(longest,part[0])*(len(current)+1)>65536:
                batches.append(current);current=[];longest=0
            current.append(part);longest=max(longest,part[0])
        if current:batches.append(current)
        result.append({'item_ids':[x.item.item_id for x in group],'reader_lengths':[len(x.student_ids) for x in group],
            'reader_old_checkpoint':max(len(x.student_ids) for x in group)*len(group)>=1700,
            'side_batches':[{'parts':len(b),'content_lengths':[n for n,r in b],'rows':[r for n,r in b],
                'sequence_lengths':[n+r for n,r in b],'padded_tokens':len(b)*max(n+r for n,r in b),
                'old_checkpoint':max(n+r for n,r in b)>=1700,'candidate_checkpoint':True} for b in batches]})
    return result


@contextmanager
def trace_calls(ac,records,flush=lambda:None):
    """Observe the original methods; do not change their arguments or mask policy."""
    import torch
    saved=[];sentinel=object()
    original_prepare=ac._encode_prepared_batch
    def prepare(self,items,indexes,results,device):
        ns=[len(item[0]) for item in items];rs=[int(item[3]) for item in items]
        records.append({'kind':'prepared','batch':len(items),'content_lengths':ns,'rows':rs,
            'sequence_lengths':[n+r for n,r in zip(ns,rs)],'padded_tokens':len(items)*max(n+r for n,r in zip(ns,rs)),
            'threshold':getattr(self,'_side_gradient_checkpointing_min_tokens',None),
            'allocated_gib_before':torch.cuda.memory_allocated(device)/2**30 if device.type=='cuda' else None})
        flush();return original_prepare(items,indexes,results,device)
    saved.append((ac,'_encode_prepared_batch',ac.__dict__.get('_encode_prepared_batch',sentinel)))
    ac._encode_prepared_batch=types.MethodType(prepare,ac)
    for label,model in [('side',ac.side),('reader',ac.target)]:
        original=model.decoder_forward
        def call(*args,_name=label,_original=original,_model=model,**kwargs):
            embeds=args[0] if args else kwargs['inputs_embeds'];mask=args[1] if len(args)>1 else kwargs.get('attention_mask')
            records.append({'kind':_name,'shape':list(embeds.shape),'checkpoint':kwargs.get('gradient_checkpointing'),
                'mask_is_none':mask is None,'mask_shape':None if mask is None else list(mask.shape),
                'grad_enabled':torch.is_grad_enabled(),'base_training':_model.model.training,
                'allocated_gib_before':torch.cuda.memory_allocated(embeds.device)/2**30 if embeds.device.type=='cuda' else None})
            flush();return _original(*args,**kwargs)
        saved.append((model,'decoder_forward',model.__dict__.get('decoder_forward',sentinel)));model.decoder_forward=call
    try:yield
    finally:
        for obj,name,prior in reversed(saved):
            if prior is sentinel:delattr(obj,name)
            else:setattr(obj,name,prior)


def reduced_short_items(items,trainer,ac):
    """Two real independently encoded passage parts; reduction is proof-only, never training data."""
    from activation.common.ac_parts import direct_parts,AC_PART_TYPE
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    candidates=[]
    for item in items:
        ex=trainer.build_example(ac,item)
        for index,request in enumerate(ex.part_requests):
            n=len(tokenize_with_parts(ac.side.tokenizer,request.messages,[],ac.pad_id,tools=request.tools)[0])
            length=n+ac.num_view_rows(n,request.compression_ratio)
            if length<1700:candidates.append((length,item.item_id,index,item))
    chosen=[];used=set()
    for length,ident,index,item in sorted(candidates,key=lambda x:x[:3]):
        if ident in used:continue
        chosen.append((length,index,item));used.add(ident)
        if len(chosen)==2:break
    if len(chosen)!=2:raise ValueError('No two distinct real short-part questions available for checkpoint gate proof')
    small=[];receipt=[]
    for length,index,item in chosen:
        messages=deepcopy(item.ac_messages);cursor=0
        for message in messages:
            if not isinstance(message.get('content'),list):continue
            keep=[]
            for part in message['content']:
                if isinstance(part,dict) and part.get('type')==AC_PART_TYPE:
                    if cursor==index:keep.append(part)
                    cursor+=1
                else:keep.append(part)
            message['content']=keep
        assert len(direct_parts(messages))==1
        small.append(replace(item,ac_messages=messages,item_id=item.item_id+':checkpoint_probe'))
        receipt.append({'original_item_id':item.item_id,'original_part_index':index,'decoder_tokens':length,
            'original_target_sha256':object_hash(item.assistant_token_ids),'modified_ac_messages_sha256':object_hash(messages)})
    return small,receipt


def equivalence(h,ac,trainer,examples,*,cpu=False):
    import torch
    from activation.ac_model.ac_model_training import disable_dropout
    from activation.harness.hf_utils import decoder_layers
    params=parameters(h,ac);ac.set_mode('training');ac._training_gradient_checkpointing=True
    trainer._training_deadline=None
    for base in dict.fromkeys([ac.side.model,ac.target.model]):base.train();disable_dropout(base)
    trainer._store_teacher_targets(ac,[x.penalty for x in examples],ac.device,lora=None)
    modes_after_reference={name:{'base_training':loaded.model.training,
        'all_decoder_layers_training':all(layer.training for layer in decoder_layers(loaded.model))}
        for name,loaded in [('side',ac.side),('reader',ac.target)]}
    assert all(all(flags.values()) for flags in modes_after_reference.values()),'Reference generation changed training mode'
    original_encode=ac.encode_batch;sentinel=object();previous=ac.__dict__.get('encode_batch',sentinel)
    reference=None;records={};rtol,atol=(2e-5,2e-7) if cpu else (.03,2e-6)
    try:
        for label,threshold in [('original',1700),('original_repeat',1700),('checkpoint_all',0)]:
            ac._side_gradient_checkpointing_min_tokens=threshold;trainer._target_gradient_checkpointing_min_tokens=threshold
            for p in dict.fromkeys(p for _,p in params):p.grad=None
            torch.manual_seed(SEED);trainer._penalty_rng.seed(SEED);outputs=[];trace=[]
            def capture(*a,**kw):
                values=original_encode(*a,**kw)
                for value in values:
                    outputs.extend(t.detach().float().cpu().clone() for t in [value.input_embeds,*[value.layer_inputs[k] for k in sorted(value.layer_inputs)]])
                return values
            ac.encode_batch=capture
            calls={'side':0,'reader':0};handles=[]
            for name,loaded in [('side',ac.side),('reader',ac.target)]:
                assert loaded.model.training and all(layer.training for layer in decoder_layers(loaded.model))
                def count_call(module,args,_name=name):calls[_name]+=1
                handles.append(decoder_layers(loaded.model)[0].register_forward_pre_hook(count_call))
            try:
                with trace_calls(ac,trace):
                    outcomes=trainer._batch_loss(ac,examples,ac.device)
                    assert all(x['penalty'] is not None for x in outcomes)
                    loss=sum(x['objective']+x['penalty'] for x in outcomes)/len(outcomes)
                    calls_before_backward=dict(calls);loss.backward()
            finally:
                for handle in handles:handle.remove()
            if label=='checkpoint_all':
                assert all(calls[name]>calls_before_backward[name] for name in calls),'Both decoder layers must actually recompute during backward'
            gradients=[(name,None if p.grad is None else p.grad.detach().float().cpu().clone()) for name,p in params]
            scalar=float(loss.detach());record={'loss':scalar,'trace':trace,
                'first_layer_calls_before_backward':calls_before_backward,'first_layer_calls_after_backward':dict(calls)}
            if reference is None:
                reference=(scalar,outputs,gradients)
                for prefix in ('ac:','reader:'):
                    assert any(name.startswith(prefix) and g is not None and bool(g.abs().any()) for name,g in gradients)
            else:
                assert len(reference[1])==len(outputs) and len(reference[2])==len(gradients)
                checks={}
                for (na,a),(nb,b) in zip(reference[2],gradients):
                    assert na==nb and (a is None)==(b is None),na
                    checks[na]={'passed':True,'both_absent':True} if a is None else compare_tensors(a,b,rtol=rtol,atol=atol)
                output_checks=[compare_tensors(a,b,rtol=rtol,atol=atol) for a,b in zip(reference[1],outputs)]
                tolerance=1e-6 if cpu else 1e-3
                record.update(all_trainable_gradients=checks,all_ac_outputs=output_checks,
                    loss_abs_difference=abs(scalar-reference[0]),loss_abs_tolerance=tolerance,
                    passed=abs(scalar-reference[0])<=tolerance and all(c['passed'] for c in checks.values()) and all(c['passed'] for c in output_checks))
            records[label]=record;del gradients,outcomes,loss,outputs
    finally:
        if previous is sentinel:del ac.encode_batch
        else:ac.encode_batch=previous
        for _,p in params:p.grad=None
    original_side=[r for r in records['original']['trace'] if r['kind']=='side'];new_side=[r for r in records['checkpoint_all']['trace'] if r['kind']=='side']
    assert original_side and all(not r['checkpoint'] for r in original_side)
    assert new_side and all(r['checkpoint'] for r in new_side)
    assert [r['mask_is_none'] for r in original_side]==[r['mask_is_none'] for r in new_side]
    assert all(not r['mask_is_none'] for r in original_side if r['shape'][0]>1),'Original right-padding masks must remain'
    return {'passed':records['original_repeat']['passed'] and records['checkpoint_all']['passed'],
        'training_modes_after_reference':modes_after_reference,
        'comparisons':records,'tolerances':{'relative_rms':rtol,'absolute_rms_floor':atol,'joint_loss_abs':1e-6 if cpu else 1e-3},
        'policy':'Same weights/masks/budget; unchanged criteria, original repeat measures baseline GPU repeatability.'}


def worker(args):
    import torch
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.autotuners import configure_fla_runtime
    import importlib.util
    pins=[(ac_module.__file__,AC_SHA),(training_module.__file__,TRAINER_SHA),(utils_module.__file__,UTILS_SHA),
          (args.material,MATERIAL_SHA),(args.bench,BUILDER_SHA),(Path(__file__).with_name('probe_piecewise_original_v1.py'),HELPER_SHA)]
    for path,expected in pins:
        if digest(path)!=expected:raise ValueError('Pinned source/material/helper changed: '+str(path))
    output=args.output/(args.worker+'.json')
    if output.exists():raise FileExistsError(output)
    began=time.monotonic();material=json.loads(args.material.read_text());selections,selection_receipt=select_material(material,args.failed_report)
    proof=args.worker=='proof';pieces,which,threshold,epochs=(True,'failing',1700,0) if proof else CONDITIONS[args.worker]
    selected=selections[which]
    # Proof may use additional genuine quantile items solely to find short parts.
    if proof:selected=list({r['example']['example_id']:r for r in selections['failing']+selections['quantiles']}.values())
    material['train']=selected;material['held']=selected[:1]
    spec=importlib.util.spec_from_file_location('checkpoint_probe_builder',args.bench);bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    source=args.p2_root;ac_path=source/'AC_MODELS/rag_poc/epoch_001';reader_path=source/'LORAS/rag_reader/epoch_001'
    payloads={'ac_modules':ac_path/'ac_modules.pt'}
    for label,directory in [('encoder',ac_path/'side_lora'),('reader',reader_path)]:
        payloads[label+'_config']=directory/'adapter_config.json'
        candidates=[directory/name for name in ('adapter_model.safetensors','adapter_model.bin') if (directory/name).is_file()]
        if len(candidates)!=1:raise ValueError('Ambiguous or missing P2 adapter payload')
        payloads[label+'_weights']=candidates[0]
    p2_hashes={key:digest(path) for key,path in payloads.items()}
    if p2_hashes!=json.loads((args.bench.parent/'PIECEWISE_TRAINING_DESIGN.json').read_text())['P2_pair_sha256']:
        raise ValueError('Original paired P2 payload changed')
    print(json.dumps({'phase':'loading fresh P2','condition':args.worker}),flush=True)
    h=bench.harness(ac_path,reader_path,encode_batch_max_tokens=65536);ac=h.module_manager.get_ac_model(bench.AC)
    ac.prepare();h.module_manager.ensure_lora(bench.READER);ac.epoch_number=0;device=ac.device
    print(json.dumps({'phase':'building exact training inputs','condition':args.worker}),flush=True)
    _,items,placeholder=bench.material_items(h,material,passage_parts=pieces,evidence_tokens=8192,max_passages=16,ratio=1/16)
    assert len(placeholder)==1;del placeholder,material,selections,selected;gc.collect()
    config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=5e-5,learning_rate_target_lora=2e-5,
        examples_per_update=8,micro_batch_examples=2,num_epochs=max(1,epochs),shuffle_examples=False,
        warmup_updates=50,schedule_total_updates=500,no_context_penalty_weight=1.,no_context_penalty_fraction=1. if proof else .25,
        checkpoint_every_epoch=False,completion_samples=0,gradient_checkpointing=True,gradient_checkpointing_min_tokens=threshold,seed=SEED)
    trainer=ActivationContextTrainer(h,config);variables=trainer._autotuning_variables(ac)
    examples=[trainer.build_example(ac,item) for item in items]
    if not proof and len(items)!=8:raise ValueError('Fit must contain exactly one logical batch')
    receipt={'status':'running','condition':args.worker,'material_sha256':MATERIAL_SHA,'builder_sha256':BUILDER_SHA,
        'source':{'ac_model':AC_SHA,'trainer':TRAINER_SHA,'utils':UTILS_SHA},'helper_sha256':HELPER_SHA,
        'selection':selection_receipt,'device':torch.cuda.get_device_name(device),'threshold':threshold,
        'encode_batch_max_tokens':65536,'passage_parts':pieces,'micro_batch_examples':2,'examples_per_update':8,
        'P2_pair_sha256':p2_hashes,
        'geometry_before_forward':batch_geometry(examples[:8] if proof else examples,ac),'trace':[],
        'inputs':[{'item_id':item.item_id,'full_sha256':object_hash(item.in_context_messages),'target_sha256':object_hash(item.assistant_token_ids),
                   'compressed_rows':item.info['compressed_rows'],'encoder_tokens':item.info['encoded_source_tokens']} for item in items],
        'initialization':'Fresh original P2, fresh optimizer; data batch16 is exact but lost original15 updates/moments are not replayed.',
        'autotuning':{key:variables[key] for key in ('side_gradient_checkpointing_min_tokens','target_gradient_checkpointing_min_tokens')},
        'runtime':{'torch':torch.__version__,'transformers':__import__('transformers').__version__},
        'optimizer_checkpoints_written':False}
    write(output,receipt)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.cuda.reset_peak_memory_stats(device)
    try:
        with configure_fla_runtime(variables['fla_config']):
            if proof:
                reduced,receipt['proof_inputs']=reduced_short_items(items,trainer,ac)
                proof_examples=[trainer.build_example(ac,item) for item in reduced]
                receipt['equivalence']=equivalence(h,ac,trainer,proof_examples)
                receipt['status']='passed' if receipt['equivalence']['passed'] else 'failed'
            else:
                if which=='failing':
                    for _ in range(15*8):trainer._penalty_rng.random()
                receipt['penalty_rng_initial_examples_consumed']=120 if which=='failing' else 0
                with trace_calls(ac,receipt['trace'],lambda:write(output,receipt)):
                    stats=trainer.train(bench.AC,items,reporting_data=[])
                assert stats.steps==epochs and stats.checkpoint_path is None and stats.target_lora_checkpoint_path is None
                assert any(stats.penalty_counts),'Actual no-context penalty must be exercised'
                receipt.update(status='complete',step_seconds=stats.step_seconds,steady_median_seconds=statistics.median(stats.step_seconds[1:]),
                    penalty_counts=stats.penalty_counts,loss=stats.loss,grad_norm_ac=stats.grad_norm_ac,grad_norm_reader=stats.grad_norm_reader,
                    peak_memory_gib=stats.peak_memory_bytes/2**30)
    except torch.OutOfMemoryError as error:
        receipt.update(status='oom',error=str(error),peak_memory_gib=torch.cuda.max_memory_allocated(device)/2**30)
        write(output,receipt);raise
    receipt['elapsed_seconds']=time.monotonic()-began
    receipt.setdefault('peak_memory_gib',torch.cuda.max_memory_allocated(device)/2**30);write(output,receipt)
    if receipt['status']=='failed':raise RuntimeError('Checkpoint-only equivalence failed; no relaxation allowed')


def controller(args):
    args.output.mkdir(parents=True,exist_ok=True);summary_path=args.output/'summary.json'
    if summary_path.exists():raise FileExistsError(summary_path)
    start=time.monotonic();summary={'status':'running','script_sha256':digest(__file__),'conditions':{},'seconds_budget':args.seconds}
    write(summary_path,summary)
    for condition in ['proof',*CONDITIONS]:
        remaining=args.seconds-(time.monotonic()-start)
        if remaining<90:summary['conditions'][condition]={'status':'not_run_budget'};continue
        state=args.workspace/condition;state.mkdir(parents=True,exist_ok=True)
        if args.autotune_root and not (state/'AUTOTUNE').exists():(state/'AUTOTUNE').symlink_to(args.autotune_root,target_is_directory=True)
        env=dict(os.environ);env['ACTIVATION_SYNC_ROOT']=str(state)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker',condition,'--material',str(args.material),'--bench',str(args.bench),
             '--failed-report',str(args.failed_report),'--p2-root',str(args.p2_root),'--output',str(args.output)]
        with (args.output/(condition+'.log')).open('x') as log:
            try:
                run=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(30,remaining-20))
                state_record={'status':'complete' if run.returncode==0 else 'failed','returncode':run.returncode}
            except subprocess.TimeoutExpired:state_record={'status':'timeout'}
        path=args.output/(condition+'.json')
        if path.exists():state_record['worker_status']=json.loads(path.read_text())['status']
        summary['conditions'][condition]=state_record;write(summary_path,summary);print(json.dumps({condition:state_record}),flush=True)
        if condition=='proof' and state_record['status']!='complete':summary['status']='proof_failed';write(summary_path,summary);return 1
    required=['proof','pieces_failing','pieces_quantiles','merged_failing','merged_quantiles']
    summary['status']='complete' if all(summary['conditions'].get(c,{}).get('status')=='complete' for c in required) else 'partial'
    for selection in ('failing','quantiles'):
        paths=[args.output/(layout+'_'+selection+'.json') for layout in ('pieces','merged')]
        if all(p.exists() for p in paths):
            a,b=[json.loads(p.read_text()) for p in paths]
            assert a['inputs']==b['inputs'] or [(x['item_id'],x['full_sha256'],x['target_sha256'],x['compressed_rows']) for x in a['inputs']]==[(x['item_id'],x['full_sha256'],x['target_sha256'],x['compressed_rows']) for x in b['inputs']]
    summary['elapsed_seconds']=time.monotonic()-start;write(summary_path,summary);return 0


def cpu_check():
    import torch
    from activation.tests.test_interventions import tiny_hybrid,build,trainer_and_examples,open_heads,open_attention
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        h,ac=build(root/'state',tiny_hybrid(root/'model'),frequency=-1,input_pooling_stride=0,
                   side_dir=tiny_hybrid(root/'side',hidden=16,seed=112))
        open_heads(ac);open_attention(ac)
        trainer,items,_,_=trainer_and_examples(h,ac,micro_batch_examples=2,completion_samples=0)
        small,metadata=reduced_short_items(items,trainer,ac)
        examples=[trainer.build_example(ac,item) for item in small]
        record=equivalence(h,ac,trainer,examples,cpu=True)
        print(json.dumps({'equivalence':record,'proof_inputs':metadata},allow_nan=False),flush=True)
        assert record['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--worker',choices=['proof',*CONDITIONS])
    for name in ('material','bench','failed-report','p2-root','output','workspace','autotune-root'):p.add_argument('--'+name,type=Path)
    p.add_argument('--seconds',type=int,default=1080);args=p.parse_args()
    if args.cpu_check:cpu_check()
    elif args.worker:worker(args)
    else:
        if any(getattr(args,n) is None for n in ('material','bench','failed_report','p2_root','output','workspace')):p.error('GPU input paths are required')
        sys.exit(controller(args))
