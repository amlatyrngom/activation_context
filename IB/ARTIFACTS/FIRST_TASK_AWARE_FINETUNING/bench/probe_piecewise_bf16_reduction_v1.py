"""One read-only BF16 GEMM-reduction flag trial; no packing adoption or training."""
import argparse
from contextlib import contextmanager
from pathlib import Path

import probe_piecewise_forward_localization_v1 as localized

base = localized.base
write, read, sha = base.write, base.read, base.sha
LOCALIZATION_SHA = 'c1b41225d6dfe51667615442005c80ff41f55f9239d7c18df450d729933194fa'
OPERATOR_RECEIPT_SHA = '46944853e7f360d87afbff165300598cc2bde7a63b58e2e4b17a42a06323a521'


def reduction_flags():
    import torch
    return (torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k)


@contextmanager
def reduction_setting(allowed):
    import torch
    original = reduction_flags()
    try:
        # In installed PyTorch2.13 a bare bool also resets split-K to True.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (allowed, original[1])
        if reduction_flags() != (allowed, original[1]):
            raise AssertionError('Requested reduction-only change was not applied')
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original
        if reduction_flags() != original:
            raise AssertionError('BF16 reduction policy did not restore')


def compare_flag(h, ac, trainer, examples, case, *, cpu=False, flush=lambda _: None):
    import torch
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    flags = reduction_flags()
    if flags[0] is not True:
        raise ValueError('Expected the measured enabled-reduction starting policy')
    tf32 = torch.backends.cuda.matmul.allow_tf32
    result = {'kind':'bf16_reduction_flag_forward_diagnostic', 'equivalence_validated':False,
        'adoption_allowed':False, 'backward_calls':0, 'optimizer_updates':0, 'reader_forward_calls':0,
        'scope':'Same original encoder masks/checkpoint policy; two selected passages; full encoder forward only.',
        'original_flags':{'reduced_precision_reduction':flags[0],'split_k':flags[1],'tf32_matmul':tf32},
        'policies':{}, 'cross_policy_final_outputs':{}}
    saved = ac.__dict__.get('encode_batch', None)
    had_override = 'encode_batch' in ac.__dict__
    original_encode = ac.encode_batch
    final_outputs = {}
    try:
        for label, allowed in [('original_enabled',True),('reduction_disabled',False)]:
            captured = []
            def capture(requests, *args, **kwargs):
                outputs = original_encode(requests,*args,**kwargs)
                lengths = [len(tokenize_with_parts(ac.side.tokenizer,r.messages,[],ac.pad_id,tools=r.tools)[0]) for r in requests]
                ranked = sorted(range(len(requests)),key=lambda i:(lengths[i],i))
                selected = [ranked[0],ranked[-1]]
                row = {}
                for index in selected:
                    row[str(index)+':input_rows'] = outputs[index].input_embeds.detach().float().cpu().clone()
                    for layer,tensor in sorted(outputs[index].layer_inputs.items()):
                        row[f'{index}:delta_{layer:02d}'] = tensor.detach().float().cpu().clone()
                captured.append(row)
                return outputs
            ac.encode_batch = capture
            with reduction_setting(allowed):
                current_flags = reduction_flags()
                def publish(value):
                    result['policies'][label] = {'flags':{'reduced_precision_reduction':current_flags[0],
                        'split_k':current_flags[1],'tf32_matmul':torch.backends.cuda.matmul.allow_tf32},'localization':value}
                    flush(result)
                value = localized.localize(h,ac,trainer,examples,case,cpu=cpu,flush=publish)
                publish(value)
            if len(captured) != 3:
                raise AssertionError('Expected original/repeat/32768 encoder captures only')
            if torch.backends.cuda.matmul.allow_tf32 != tf32 or reduction_flags()!=flags:
                raise AssertionError('An unrelated precision policy changed')
            final_outputs[label] = captured
        baseline = final_outputs['original_enabled'][0]
        rtol,atol = (2e-5,2e-7) if cpu else (.03,2e-6)
        for label,index in [('disabled_65536_vs_original_65536',0),('disabled_32768_vs_original_65536',2)]:
            candidate = final_outputs['reduction_disabled'][index]
            if set(candidate)!=set(baseline):raise AssertionError('Selected request/output identity changed')
            result['cross_policy_final_outputs'][label] = {key:base.previous.compare_tensors(baseline[key],value,rtol=rtol,atol=atol)
                for key,value in candidate.items()}
        left=result['policies']['original_enabled']['localization'];right=result['policies']['reduction_disabled']['localization']
        if (left['requests']!=right['requests'] or left['selected_request_indexes']!=right['selected_request_indexes']):
            raise AssertionError('Flag trial changed requests or selected passages')
        for variant in left['variants']:
            if (left['variants'][variant]['batches']!=right['variants'][variant]['batches']
                    or left['variants'][variant]['metadata']!=right['variants'][variant]['metadata']):
                raise AssertionError('Flag trial changed batch/mask/position/checkpoint metadata')
        result.update(status='diagnostic_complete',precision_flags_restored=True,
            decision_rule='Inspect contraction of mixer/full-attention/final-row differences. No backward or optimizer test; full unchanged-gate proof and production speed remain required.')
        flush(result)
        return result
    finally:
        if had_override:ac.encode_batch=saved
        else:del ac.encode_batch
        if reduction_flags()!=flags or torch.backends.cuda.matmul.allow_tf32!=tf32:
            raise AssertionError('Diagnostic leaked precision settings')


def worker(args):
    if sha(args.operator_receipt)!=OPERATOR_RECEIPT_SHA:
        raise ValueError('Measured operator attribution receipt changed')
    operator=read(args.operator_receipt)
    if operator.get('status')!='diagnostic_complete' or operator.get('equivalence_validated') is not False:
        raise ValueError('Expected the prior read-only attribution, not equivalence')
    original=base.prove_case;path=args.output/'proof.json'
    def diagnostic(*positional,**kwargs):
        receipt=read(path)
        if receipt['payload_sha256']!=operator['payload_sha256'] or receipt['source_sha256']!=operator['source_sha256']:
            raise ValueError('Flag trial parent/source differs from measured projection failure')
        def flush(value):
            receipt=read(path);receipt['bf16_reduction_trial']=value;write(path,receipt)
        result=compare_flag(*positional,flush=flush)
        raise localized.DiagnosticComplete(result)
    base.prove_case=diagnostic
    try:
        base.worker(args)
        raise AssertionError('Expected read-only diagnostic stop')
    except localized.DiagnosticComplete as done:
        receipt=read(path)
        if base.paired.payload_hashes(receipt['pair'])!=receipt['payload_sha256'] or sha(Path(receipt['pair']['ac_checkpoint'])/'optimizer.pt')!=receipt['optimizer_sha256']:
            raise AssertionError('Parent payload changed')
        receipt.update(status='diagnostic_complete',bf16_reduction_trial=done.result,
            worker_sha256=sha(__file__),operator_receipt_sha256=OPERATOR_RECEIPT_SHA,
            parent_payloads_rehashed_unchanged=True,equivalence_validated=False,optimizer_updates=0,reader_forward_calls=0,backward_calls=0)
        write(path,receipt)
    finally:base.prove_case=original


def cpu_check():
    import tempfile
    import torch
    from activation.tests.test_interventions import build,tiny_hybrid,trainer_and_examples,open_heads,open_attention
    import activation.ac_model.ac_model as ac_source
    import activation.ac_model.ac_model_training as trainer_source
    import activation.ac_model.ac_model_utils as utils_source
    source={key:sha(module.__file__) for key,module in [('ac_model',ac_source),('trainer',trainer_source),('utils',utils_source)]}
    if source!=base.paired.SOURCE:raise ValueError('CPU fixture must use frozen source')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    before=reduction_flags()
    try:
        # Enabled reduction + disabled split-K is not a supported PyTorch pair.
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=(False,False)
        try:
            with reduction_setting(False):
                assert reduction_flags()==(False,False)
                raise RuntimeError('intentional restoration check')
        except RuntimeError:pass
        assert reduction_flags()==(False,False)
    finally:torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=before
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        h,ac=build(root/'state',tiny_hybrid(root/'reader'),frequency=-1,input_pooling_stride=0,
            side_dir=tiny_hybrid(root/'side',hidden=16,seed=112))
        open_heads(ac);open_attention(ac)
        trainer,_,examples,_=trainer_and_examples(h,ac,micro_batch_examples=2,completion_samples=0,no_context_penalty_weight=0.)
        had='encode_batch' in ac.__dict__;prior=ac.__dict__.get('encode_batch')
        result=compare_flag(h,ac,trainer,examples,'tiny',cpu=True)
        assert ('encode_batch' in ac.__dict__)==had and ac.__dict__.get('encode_batch') is prior
        assert reduction_flags()==before
        assert all(v['max_abs']==0 for v in result['cross_policy_final_outputs']['disabled_65536_vs_original_65536'].values())
        print(__import__('json').dumps({'status':'pass','source_sha256':source,'split_k_false_and_exception_restoration':True,
            'result':result}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true')
    for name in ('material','bench','design','parent-receipt','memory-receipt','operator-receipt','reports','runs-root','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if (sha(localized.__file__)!=LOCALIZATION_SHA or sha(base.__file__)!=localized.BASE_SHA
            or sha(base.previous.__file__)!=base.PREVIOUS_SHA or sha(base.paired.__file__)!=base.PAIRED_SHA
            or sha(Path(__file__).with_name('probe_piecewise_original_v1.py'))!=base.previous.HELPER_SHA):
        raise ValueError('Frozen dependency changed')
    if args.cpu_check:cpu_check()
    else:
        args.output.mkdir(parents=True,exist_ok=True);args.worker='proof';worker(args)
