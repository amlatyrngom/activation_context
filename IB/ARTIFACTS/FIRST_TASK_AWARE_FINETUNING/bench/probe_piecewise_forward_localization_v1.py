"""Read-only localization of a failed packing equivalence; never an adoption gate."""
import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import types

import probe_strong_piecewise_budget_v1 as base

BASE_SHA='377e022e868dce0ae64fed25e639e886276189b67acfc1832003982758faf381'
EXPECTED_CONTENT=[109,133,142,144,164,181,201,353,366,409,427,428,444,474,479,761,972,1025,1026,1048,1146,2083,2328,2360]
EXPECTED_ROWS=[4,5,6,6,7,8,10,19,20,23,24,24,25,27,27,45,58,61,62,63,69,128,145,147]
write,read,sha=base.write,base.read,base.sha


def tensor_hash(tensor):
    import torch
    value=tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def metadata_tensor(tensor):
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'sha256':tensor_hash(tensor)}


def localize(h,ac,trainer,examples,case,*,cpu=False,flush=lambda _:None):
    import torch
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    from activation.agent_training.agent_training_utils import disable_dropout
    from activation.harness.hf_utils import decoder_layers
    if case!='failing' and not cpu:raise ValueError('Only the first affected case is authorized')
    ordered=sorted(examples,key=lambda item:len(item.student_ids))
    group=ordered if cpu else ordered[2:4]
    requests=[request for example in group for request in example.part_requests]
    tokens=[tokenize_with_parts(ac.side.tokenizer,r.messages,[],ac.pad_id,tools=r.tools)[0] for r in requests]
    lengths=[len(ids) for ids in tokens]
    rows=[ac.num_view_rows(n,r.compression_ratio) for n,r in zip(lengths,requests)]
    ranked=sorted(range(len(requests)),key=lambda index:(lengths[index],index))
    if not cpu and ([lengths[i] for i in ranked]!=EXPECTED_CONTENT or [rows[i] for i in ranked]!=EXPECTED_ROWS):
        raise ValueError('Exact affected24-passage geometry changed')
    selected=[ranked[0],ranked[-1]]
    if len(set(selected))!=2:raise ValueError('Need distinct shortest/longest requests')
    params=base.previous.parameters(h,ac)
    parameter_state=[(name,p._version,p.requires_grad) for name,p in params]
    ac.set_mode('training');ac._training_gradient_checkpointing=True
    ac._side_gradient_checkpointing_min_tokens=0
    for model in dict.fromkeys([ac.side.model,ac.target.model]):model.train();disable_dropout(model)
    if not torch.is_grad_enabled() or not ac.side.model.training:raise AssertionError('Expected training-mode gradients')
    original_config=ac.config;reference={};reference_meta={};results={}
    # Tiny CPU check changes grouping too; the real GPU comparison remains exactly32768.
    candidate=max(lengths) if cpu else 32768
    variants=[('baseline',65536),('unchanged_repeat',65536),('32768' if not cpu else 'small_cpu',candidate)]
    limits=(2e-5,2e-7) if cpu else (.03,2e-6)
    report={'kind':'forward_localization_only','adoption_allowed':False,
        'scope':'Encoder forward only; no reader forward, backward, optimizer step or save.',
        'item_ids':[example.item.item_id for example in group],
        'selection_rule':'Shortest and longest content token lengths, ties by original request index; no loss-based selection.',
        'selected_request_indexes':selected,'request_count':len(requests),
        'requests':[{'index':i,'token_ids_sha256':base.previous.object_hash(tokens[i]),
            'request_sha256':base.previous.object_hash({'messages':r.messages,'tools':r.tools,'compression':r.compression_ratio}),
            'content_tokens':lengths[i],'rows':rows[i]} for i,r in enumerate(requests)],
        'tolerances':{'relative_rms':limits[0],'absolute_rms_floor':limits[1]},'variants':results}
    report['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,
        'side_layer_type':type(decoder_layers(ac.side.model)[0]).__module__}
    for package in ('transformers','flash-linear-attention','causal-conv1d'):
        try:report['runtime'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:report['runtime'][package]=None
    module=sys.modules[type(decoder_layers(ac.side.model)[0]).__module__]
    report['runtime']['side_layer_module_sha256']=sha(module.__file__)
    def publish():flush(report)
    try:
        for label,budget in variants:
            ac.config=replace(original_config,encode_batch_max_tokens=budget)
            torch.manual_seed(base.SEED)
            active={};handles=[];saved=[];sentinel=object();seen=set()
            current={'budget':budget,'stages':{},'metadata':{},'metadata_changed':[],'batches':[]}
            results[label]=current
            if ac.device.type=='cuda':torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device)
            def remember(obj,name,value):
                saved.append((obj,name,obj.__dict__.get(name,sentinel)));setattr(obj,name,value)
            def record(stage,index,value):
                key=f'{index}:{stage}'
                if key in seen:raise AssertionError('Unexpected repeated forward capture: '+key)
                seen.add(key)
                data=value.detach().float().cpu().clone()
                if label=='baseline':
                    reference[key]=data
                    current['stages'][key]={'shape':list(data.shape),'dtype':str(value.dtype),'sha256':tensor_hash(data)}
                else:
                    if key not in reference:raise AssertionError('Unexpected request/stage identity')
                    comparison=base.previous.compare_tensors(reference[key],data,rtol=limits[0],atol=limits[1])
                    left=reference[key].double();right=data.double()
                    comparison.update(shape=list(data.shape),dtype=str(value.dtype),sha256=tensor_hash(data),
                        baseline_squared_norm=float((left*left).sum()),candidate_squared_norm=float((right*right).sum()),
                        dot_product=float((left*right).sum()),elements=data.numel())
                    current['stages'][key]=comparison
            def batched(stage,value,sequence=False):
                if isinstance(value,(tuple,list)):value=value[0]
                if not isinstance(value,torch.Tensor):raise TypeError('Expected layer tensor: '+stage)
                for row,index in enumerate(active['indexes']):
                    if index in selected:
                        n=lengths[index]+rows[index] if sequence else lengths[index]
                        record(stage,index,value[row,:n])
            original_prepare=ac._encode_prepared_batch
            def prepare(self,items,indexes,outputs,device):
                active.clear();active.update(indexes=list(indexes))
                for item,index in zip(items,indexes):
                    if item[0]!=tokens[index] or item[3]!=rows[index]:raise AssertionError('Request mapping/token/rows changed')
                current['batches'].append({'indexes':list(indexes),'content_tokens':[lengths[i] for i in indexes],
                    'sequence_tokens':[lengths[i]+rows[i] for i in indexes]})
                return original_prepare(items,indexes,outputs,device)
            remember(ac,'_encode_prepared_batch',types.MethodType(prepare,ac))
            for i,module in enumerate(ac.modules.mixer):
                if i==0:
                    def mixer_pre(module,args):
                        x,valid=args
                        for row,index in enumerate(active['indexes']):
                            if not bool(valid[row,:lengths[index]].all()) or bool(valid[row,lengths[index]:].any()):
                                raise AssertionError('Mixer padding validity changed')
                        batched('mixer_input',x)
                    handles.append(module.register_forward_pre_hook(mixer_pre))
                handles.append(module.register_forward_hook(lambda module,args,out,_i=i:batched(f'mixer_{_i:02d}_output',out)))
            original_side=ac.side.decoder_forward
            def side_forward(inputs_embeds,attention_mask,*args,**kwargs):
                if not kwargs.get('gradient_checkpointing') or not torch.is_grad_enabled() or not ac.side.model.training:
                    raise AssertionError('Original grad-enabled checkpoint policy changed')
                if len(active['indexes'])>1 and attention_mask is None:raise AssertionError('Original padding mask missing')
                for row,index in enumerate(active['indexes']):
                    n=lengths[index]+rows[index]
                    if attention_mask is not None and (not bool(attention_mask[row,:n].all()) or bool(attention_mask[row,n:].any())):
                        raise AssertionError('Side-encoder padding mask changed')
                    if index in selected:
                        current['metadata'][f'{index}:side_mask']={'batch_shape':list(inputs_embeds.shape),
                            'mask_none':attention_mask is None,'valid_mask':None if attention_mask is None else metadata_tensor(attention_mask[row,:n]),
                            'checkpoint_enabled':True,'grad_enabled':True,'base_training':True}
                batched('side_input',inputs_embeds,True)
                result=original_side(inputs_embeds,attention_mask,*args,**kwargs)
                batched('side_final_norm',result,True)
                return result
            remember(ac.side,'decoder_forward',side_forward)
            def layer_pre(module,args,kwargs,layer):
                for row,index in enumerate(active['indexes']):
                    if index not in selected:continue
                    n=lengths[index]+rows[index];meta={}
                    position=kwargs.get('position_ids')
                    if position is not None:
                        value=position[0 if position.shape[0]==1 else row,:n]
                        # Frozen decoder text positions should be0..n-1 for this independent passage.
                        expected=torch.arange(n,device=value.device)
                        if value.ndim!=1 or not torch.equal(value,expected):raise AssertionError('Passage position IDs differ')
                        meta['position_ids']=metadata_tensor(value)
                    else:meta['position_ids']=None
                    mask=kwargs.get('attention_mask')
                    if mask is None:meta['mask']=None
                    elif isinstance(mask,torch.Tensor):
                        value=mask[0 if mask.shape[0]==1 else row]
                        if mask.ndim==2:value=value[:n]
                        elif mask.ndim==4:value=value[...,:n,:n]
                        else:raise ValueError('Unrecognized decoder mask rank')
                        meta['mask']={**metadata_tensor(value),'min':float(value.min()),'max':float(value.max())}
                    else:raise TypeError('Unrecognized decoder mask type')
                    key=f'{index}:layer_{layer:02d}'
                    current['metadata'][key]=meta
                    if label=='baseline':reference_meta[key]=meta
                    elif meta!=reference_meta[key]:
                        # Keep None-versus-materialized mask differences visible. A
                        # derived backend representation may change without changing
                        # the validated caller padding mask; diagnosing it is the task.
                        current['metadata_changed'].append(key)
            for i,layer in enumerate(decoder_layers(ac.side.model)):
                handles.append(layer.register_forward_pre_hook(lambda module,args,kwargs,_i=i:layer_pre(module,args,kwargs,_i),with_kwargs=True))
                handles.append(layer.register_forward_hook(lambda module,args,out,_i=i:batched(f'side_layer_{_i:02d}_output',out,True)))
            try:
                outputs=ac.encode_batch(requests,with_layers=True)
                for index in selected:
                    record('output_input_rows',index,outputs[index].input_embeds)
                    for layer,tensor in sorted(outputs[index].layer_inputs.items()):record(f'output_delta_{layer:02d}',index,tensor)
                del outputs
            finally:
                for handle in handles:handle.remove()
                remaining_hook_ids={key for model in (ac.modules,ac.side.model) for module in model.modules()
                    for hook_map in (module._forward_hooks,module._forward_pre_hooks) for key in hook_map}
                if any(handle.id in remaining_hook_ids for handle in handles):
                    raise AssertionError('Diagnostic hook was not removed')
                for obj,name,prior in reversed(saved):
                    if prior is sentinel:delattr(obj,name)
                    else:setattr(obj,name,prior)
            if seen!=set(reference):raise AssertionError('Missing captured request/stage')
            if [(name,p._version,p.requires_grad) for name,p in params]!=parameter_state:raise AssertionError('Parameter state changed')
            if any(p.grad is not None for _,p in params):raise AssertionError('Unexpected backward/gradient accumulation')
            if label!='baseline':
                current['first_nonzero_by_request']={str(index):next((key for key,z in current['stages'].items() if key.startswith(str(index)+':') and z['max_abs']>0),None) for index in selected}
                current['first_gate_failure_by_request']={str(index):next((key for key,z in current['stages'].items() if key.startswith(str(index)+':') and not z['passed']),None) for index in selected}
                current['all_captured_stages_within_original_gate']=all(z['passed'] for z in current['stages'].values())
                if label=='unchanged_repeat' and any(z['max_abs']!=0 for z in current['stages'].values()):
                    raise AssertionError('Unchanged repeat is not exact; localization is not attributable to batching')
            gc.collect()
            if ac.device.type=='cuda':
                torch.cuda.synchronize(ac.device)
                current['peak_allocated_gib']=torch.cuda.max_memory_allocated(ac.device)/2**30
            current['parameter_metadata_unchanged']=True;current['diagnostic_hooks_removed']=True;publish()
    finally:ac.config=original_config
    report['status']='diagnostic_complete'
    report['equivalence_validated']=False
    return report


class DiagnosticComplete(Exception):
    def __init__(self,result):self.result=result


def worker(args):
    original=base.prove_case
    path=args.output/'proof.json'
    def diagnostic(*positional,**kwargs):
        def flush(value):
            receipt=read(path);receipt['forward_localization']=value;write(path,receipt)
        result=localize(*positional,flush=flush)
        raise DiagnosticComplete(result)
    base.prove_case=diagnostic
    try:
        base.worker(args)
        raise AssertionError('Expected diagnostic to stop after its exact affected case')
    except DiagnosticComplete as done:
        receipt=read(path)
        if base.paired.payload_hashes(receipt['pair'])!=receipt['payload_sha256'] or sha(Path(receipt['pair']['ac_checkpoint'])/'optimizer.pt')!=receipt['optimizer_sha256']:
            raise AssertionError('Parent payload changed')
        receipt.update(status='diagnostic_complete',forward_localization=done.result,
            localization_worker_sha256=sha(__file__),base_worker_sha256=BASE_SHA,
            parent_payloads_rehashed_unchanged=True,equivalence_validated=False,optimizer_updates=0,
            reader_forward_calls=0,saved_activation_offload=False)
        write(path,receipt)
    finally:base.prove_case=original


def cpu_check():
    import tempfile
    import torch
    from activation.tests.test_interventions import build,tiny_hybrid,trainer_and_examples,open_heads,open_attention
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        h,ac=build(root/'state',tiny_hybrid(root/'reader'),frequency=-1,input_pooling_stride=0,
            side_dir=tiny_hybrid(root/'side',hidden=16,seed=112))
        open_heads(ac);open_attention(ac)
        trainer,_,examples,_=trainer_and_examples(h,ac,micro_batch_examples=2,completion_samples=0)
        sentinel=object()
        before_prepare=ac.__dict__.get('_encode_prepared_batch',sentinel)
        before_side=ac.side.__dict__.get('decoder_forward',sentinel)
        original_budget=ac.config.encode_batch_max_tokens
        result=localize(h,ac,trainer,examples,'tiny',cpu=True)
        assert original_budget==ac.config.encode_batch_max_tokens
        assert before_prepare is ac.__dict__.get('_encode_prepared_batch',sentinel)
        assert before_side is ac.side.__dict__.get('decoder_forward',sentinel)
        assert all(value['diagnostic_hooks_removed'] for value in result['variants'].values())
        assert result['status']=='diagnostic_complete' and result['equivalence_validated'] is False
        assert len(result['selected_request_indexes'])==2
        assert result['variants']['baseline']['batches']!=result['variants']['small_cpu']['batches']
        print(json.dumps({'status':'pass','result':result}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true')
    for name in ('material','bench','design','parent-receipt','memory-receipt','reports','runs-root','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if (sha(base.__file__)!=BASE_SHA or sha(base.previous.__file__)!=base.PREVIOUS_SHA or sha(base.paired.__file__)!=base.PAIRED_SHA
            or sha(Path(__file__).with_name('probe_piecewise_original_v1.py'))!=base.previous.HELPER_SHA):
        raise ValueError('Pinned loader/proof guards changed')
    if args.cpu_check:cpu_check()
    else:
        args.output.mkdir(parents=True,exist_ok=True);args.worker='proof';worker(args)
