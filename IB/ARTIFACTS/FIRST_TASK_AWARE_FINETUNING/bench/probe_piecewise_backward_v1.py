"""Backward repeatability diagnostic over the pinned checkpoint-only probe loader.

No optimizer, deterministic-mode request, mask change or tolerance change. The
original failed probe/source stay untouched. GPU invocation uses --worker proof.
"""
import argparse
import hashlib
from pathlib import Path

import probe_piecewise_checkpoint_v1 as base

BASE_SHA='12df00c8fcac924cf7a11525006c626d8ef572b6fdefa809468124512fba2139'


def equivalence(h,ac,trainer,examples,*,cpu=False):
    import torch
    from activation.ac_model.ac_model_training import disable_dropout
    from activation.harness.hf_utils import decoder_layers
    params=base.parameters(h,ac)
    ac.set_mode('training');ac._training_gradient_checkpointing=True
    trainer._training_deadline=None
    for model in dict.fromkeys([ac.side.model,ac.target.model]):model.train();disable_dropout(model)
    trainer._store_teacher_targets(ac,[example.penalty for example in examples],ac.device,lora=None)
    modes={name:{'base_training':loaded.model.training,
            'all_decoder_layers_training':all(layer.training for layer in decoder_layers(loaded.model))}
        for name,loaded in [('side',ac.side),('reader',ac.target)]}
    assert all(all(flags.values()) for flags in modes.values())
    def state():
        return {name:{'version':p._version,'requires_grad':p.requires_grad,'shape':list(p.shape),'dtype':str(p.dtype)} for name,p in params}
    initial=state();initial_hash=base.compact_hash(initial);state_checks=[]
    def unchanged(where):
        current=state();differences={name:{'before':initial[name],'after':current[name]} for name in initial if initial[name]!=current[name]}
        if differences:raise AssertionError({'where':where,'parameter_metadata_changed':differences})
        state_checks.append(where)
    def clear():
        for p in dict.fromkeys(p for _,p in params):p.grad=None
        assert all(p.grad is None for _,p in params)
    def gradients():
        return [(name,None if p.grad is None else p.grad.detach().float().cpu().clone()) for name,p in params]
    def scalars(gs):
        return {name:{'parameter_value':float(p.detach()),'parameter_dtype':str(p.dtype),
            'gradient':None if g is None else float(g), 'gradient_dtype':None if p.grad is None else str(p.grad.dtype)}
            for (name,p),(gn,g) in zip(params,gs) if p.ndim==0}
    rtol,atol=(2e-5,2e-7) if cpu else (.03,2e-6)
    loss_tolerance=1e-6 if cpu else 1e-3
    records={};reference=None
    original_encode=ac.encode_batch;sentinel=object();previous=ac.__dict__.get('encode_batch',sentinel)
    def record(label,scalar,outputs,gs,trace,before,after):
        nonlocal reference
        values={'loss':scalar,'raw_scalar_parameters':scalars(gs),'trace':trace,
            'first_layer_calls_before_backward':dict(before),'first_layer_calls_after_backward':dict(after)}
        if reference is None:
            reference=(scalar,outputs,gs)
            for prefix in ('ac:','reader:'):
                assert any(name.startswith(prefix) and g is not None and bool(g.abs().any()) for name,g in gs)
        else:
            assert len(reference[1])==len(outputs) and len(reference[2])==len(gs)
            checks={}
            for (left_name,left),(name,right) in zip(reference[2],gs):
                assert name==left_name and (left is None)==(right is None),name
                checks[name]={'passed':True,'both_absent':True} if left is None else base.compare_tensors(left,right,rtol=rtol,atol=atol)
            output_checks=[base.compare_tensors(a,b,rtol=rtol,atol=atol) for a,b in zip(reference[1],outputs)]
            groups={}
            for prefix in ('ac:ac.','ac:side.','reader:reader.'):
                selected=[v for k,v in checks.items() if k.startswith(prefix) and not v.get('both_absent')]
                groups[prefix]={'present':len(selected),'failed':sum(not v['passed'] for v in selected),
                    'nonexact':sum(v['max_abs']!=0 for v in selected),
                    'max_relative_l2':max((v['relative_l2'] for v in selected),default=0.)}
            values.update(all_trainable_gradients=checks,gradient_groups=groups,all_ac_outputs=output_checks,
                loss_abs_difference=abs(scalar-reference[0]),loss_abs_tolerance=loss_tolerance,
                passed=abs(scalar-reference[0])<=loss_tolerance and all(v['passed'] for v in checks.values()) and all(v['passed'] for v in output_checks))
        records[label]=values
    try:
        for label,threshold in [('original',1700),('fresh_original',1700),('checkpoint_all',0)]:
            ac._side_gradient_checkpointing_min_tokens=threshold;trainer._target_gradient_checkpointing_min_tokens=threshold
            unchanged(label+':before_forward');clear()
            torch.manual_seed(base.SEED);trainer._penalty_rng.seed(base.SEED)
            outputs=[];trace=[]
            def capture(*args,**kwargs):
                values=original_encode(*args,**kwargs)
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
                with base.trace_calls(ac,trace):
                    outcomes=trainer._batch_loss(ac,examples,ac.device)
                    assert all(x['penalty'] is not None for x in outcomes)
                    loss=sum(x['objective']+x['penalty'] for x in outcomes)/len(outcomes)
                    scalar=float(loss.detach());before=dict(calls)
                    unchanged(label+':after_forward')
                    loss.backward(retain_graph=label=='original')
                    unchanged(label+':after_backward')
                    gs=gradients();record(label,scalar,outputs,gs,list(trace),before,calls)
                    if label=='checkpoint_all':
                        assert all(calls[name]>before[name] for name in calls),'Both decoders must recompute'
                    if label=='original':
                        clear();unchanged('retained_original:before_backward')
                        before_second=dict(calls)
                        loss.backward()
                        unchanged('retained_original:after_backward')
                        second=gradients()
                        record('retained_original',scalar,outputs,second,list(trace),before_second,calls)
                        # No new decoder forward was called between these backward traversals.
                        assert calls==before_second,'Original decoder unexpectedly recomputed'
                        del second
            finally:
                for handle in handles:handle.remove()
            del outcomes,loss,gs,outputs
    finally:
        if previous is sentinel:del ac.encode_batch
        else:ac.encode_batch=previous
        clear()
    original_side=[x for x in records['original']['trace'] if x['kind']=='side']
    candidate_side=[x for x in records['checkpoint_all']['trace'] if x['kind']=='side']
    assert original_side and all(not x['checkpoint'] for x in original_side)
    assert candidate_side and all(x['checkpoint'] for x in candidate_side)
    assert [x['mask_is_none'] for x in original_side]==[x['mask_is_none'] for x in candidate_side]
    assert all(not x['mask_is_none'] for x in original_side if x['shape'][0]>1)
    return {'passed':all(records[k]['passed'] for k in ('retained_original','fresh_original','checkpoint_all')),
        'comparisons':records,'training_modes_after_reference':modes,
        'owned_parameter_count':len(initial),'parameter_metadata_sha256':initial_hash,'parameter_state_checks':state_checks,
        'tolerances':{'relative_rms':rtol,'absolute_rms_floor':atol,'joint_loss_abs':loss_tolerance},
        'diagnostic_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'loader_probe_sha256':BASE_SHA,
        'policy':'Retained original graph backward twice, then fresh original and checkpoint-all; ordinary execution, original masks, no optimizer or tolerance changes. Scalar values are signed; gradient tensor comparisons include every owned parameter. A failed unchanged-repeat gate remains failed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu-check',action='store_true')
    p.add_argument('--worker',choices=['proof'],default='proof')
    for name in ('material','bench','failed-report','p2-root','output'):p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()!=BASE_SHA:raise ValueError('Pinned original probe changed')
    base.equivalence=equivalence
    if args.cpu_check:base.cpu_check()
    else:
        if any(getattr(args,n) is None for n in ('material','bench','failed_report','p2_root','output')):p.error('GPU input paths are required')
        args.output.mkdir(parents=True,exist_ok=True);base.worker(args)
