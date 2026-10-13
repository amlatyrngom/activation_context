"""Read-only first full-attention attribution; not a packing equivalence gate."""
import argparse
from dataclasses import replace
import gc
import importlib.metadata
from pathlib import Path
import sys
import types

import probe_piecewise_forward_localization_v1 as localized

base = localized.base
write, read, sha = base.write, base.read, base.sha
LOCALIZATION_SHA = 'c1b41225d6dfe51667615442005c80ff41f55f9239d7c18df450d729933194fa'
QWEN_SHA = '67cf849081143a998f0e189c551e4b5f532365a055805570747385e400372abb'


class LayerCaptured(Exception):
    pass


def canonical_mask(mask, row, n, *, causal):
    """Reject changed or unaccounted-for attention semantics before reference replay."""
    import torch
    expected = torch.ones(n, n, dtype=torch.bool).tril()
    if not causal:
        raise ValueError('Reference requires the original causal text attention')
    if mask is None:
        return expected
    if mask.ndim == 2:
        selected = mask[0 if mask.shape[0] == 1 else row].detach().cpu()
        if not bool(selected[:n].bool().all()) or bool(selected[n:].bool().any()):
            raise ValueError('Padding mask does not bind the selected valid prefix')
        return expected
    if mask.ndim != 4:
        raise ValueError('Unsupported native attention mask representation')
    selected = mask[0 if mask.shape[0] == 1 else row, :, :n].detach().cpu()
    if selected.dtype == torch.bool:
        allowed = selected
    else:
        if not bool(((selected == 0) | (selected <= -1e4)).all()):
            raise ValueError('Non-mask attention bias cannot be dropped by the reference')
        allowed = selected == 0
    if allowed.shape[-2] != n or allowed.shape[-1] < n:
        raise ValueError('Unexpected native causal-mask shape')
    if bool(allowed[..., n:].any()) or not torch.equal(allowed[..., :n], expected.expand_as(allowed[..., :n])):
        raise ValueError('Causality or masked padding differs from the reference')
    return expected


def fp32_reference(q, k, v, allowed, scale, device):
    """Detached 113-position reference only; never substituted into the model."""
    import torch
    if q.ndim != 3 or k.shape != v.shape or q.shape[1:] != k.shape[1:]:
        raise ValueError('Expected Q[H,S,D], K/V[KVH,S,D] with one bound valid prefix')
    if q.shape[0] % k.shape[0] or allowed.shape != (q.shape[1], q.shape[1]):
        raise ValueError('Grouped-query head mapping or mask is invalid')
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            q, k, v = (x.to(device=device, dtype=torch.float32) for x in (q, k, v))
            groups = q.shape[0] // k.shape[0]
            k = k.repeat_interleave(groups, dim=0); v = v.repeat_interleave(groups, dim=0)
            scores = (q @ k.transpose(-1, -2)) * scale
            scores = scores.masked_fill(~allowed.to(device), float('-inf'))
            return (scores.softmax(-1) @ v).transpose(0, 1).contiguous().cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def capture_case(h, ac, trainer, examples, case, *, cpu=False, flush=lambda _: None, tensor_path=None):
    import torch
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    from activation.agent_training.agent_training_utils import disable_dropout
    from activation.harness.hf_utils import decoder_layers
    if case != 'failing' and not cpu:
        raise ValueError('Only the exact affected first case is authorized')
    ordered = sorted(examples, key=lambda ex: len(ex.student_ids))
    group = ordered if cpu else ordered[2:4]
    requests = [request for ex in group for request in ex.part_requests]
    tokens = [tokenize_with_parts(ac.side.tokenizer, r.messages, [], ac.pad_id, tools=r.tools)[0] for r in requests]
    lengths = [len(ids) for ids in tokens]
    rows = [ac.num_view_rows(n, r.compression_ratio) for n, r in zip(lengths, requests)]
    ranked = sorted(range(len(requests)), key=lambda i: (lengths[i], i))
    if not cpu and ([lengths[i] for i in ranked] != localized.EXPECTED_CONTENT or [rows[i] for i in ranked] != localized.EXPECTED_ROWS):
        raise ValueError('Exact affected24-passage geometry changed')
    selected = ranked[0]; n = lengths[selected] + rows[selected]
    if not cpu and (selected != 2 or n != 113):
        raise ValueError('Expected original shortest113-position passage')
    layers = decoder_layers(ac.side.model)
    layer_index = next(i for i, layer in enumerate(layers) if getattr(layer, 'block_type', None) == 'full_attention')
    if not cpu and layer_index != 3:
        raise ValueError('Expected first full-attention layer3')
    layer = layers[layer_index]; attention = layer.self_attn
    model_module = sys.modules[type(layer).__module__]
    if sha(model_module.__file__) != QWEN_SHA:
        raise ValueError('Installed Qwen source differs from the recorded localization')
    params = base.previous.parameters(h, ac)
    state = [(name, p._version, p.requires_grad) for name, p in params]
    ac.set_mode('training'); ac._training_gradient_checkpointing = True
    ac._side_gradient_checkpointing_min_tokens = 0
    for model in dict.fromkeys([ac.side.model, ac.target.model]):
        model.train(); disable_dropout(model)
    if not torch.is_grad_enabled():
        raise AssertionError('Expected native grad-enabled checkpoint forward')
    saved_config = ac.config; baseline = None; baseline_identity = None; tensors = {}
    candidate = max(lengths) if cpu else 32768
    result = {'kind': 'first_full_attention_operator_attribution', 'equivalence_validated': False,
        'scope': 'Native encoder forward stops after first full-attention layer; detached small FP32 reference; no reader/backward/update/save.',
        'selected_request_index': selected, 'valid_positions': n, 'layer_index': layer_index,
        'item_ids': [ex.item.item_id for ex in group], 'token_ids_sha256': base.previous.object_hash(tokens[selected]),
        'request_count': len(requests), 'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda,
            'qwen_source_sha256': sha(model_module.__file__), 'attention_implementation': attention.config._attn_implementation},
        'variants': {}}
    for package in ('transformers', 'peft', 'flash-linear-attention'):
        result['runtime'][package] = importlib.metadata.version(package)
    projection_module = sys.modules[type(attention.q_proj).__module__]
    result['runtime']['projection_source_sha256'] = sha(projection_module.__file__)
    try:
        for label, budget in [('baseline', 65536), ('unchanged_repeat', 65536), ('32768', candidate)]:
            ac.config = replace(saved_config, encode_batch_max_tokens=budget)
            torch.manual_seed(base.SEED)
            active = {}; values = {}; identity = {}; handles = []; overrides = []; sentinel = object()
            current = {'budget': budget, 'stages': {}, 'branches': {}, 'native_attention_calls': 0,
                'native_flags': {'tf32_matmul': torch.backends.cuda.matmul.allow_tf32,
                    'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}}
            result['variants'][label] = current
            def remember(obj, name, value):
                overrides.append((obj, name, obj.__dict__.get(name, sentinel))); setattr(obj, name, value)
            def row_index():
                return active['indexes'].index(selected)
            def record(name, tensor):
                if name in values:
                    raise AssertionError('Repeated capture without backward: ' + name)
                values[name] = tensor.detach().cpu().clone()
                current['stages'][name] = localized.metadata_tensor(values[name])
            def batched(name, tensor):
                record(name, tensor[row_index(), :n])
            original_prepare = ac._encode_prepared_batch
            def prepare(self, items, indexes, outputs, device):
                active.clear(); active['indexes'] = list(indexes)
                if selected not in indexes:
                    raise AssertionError('Shortest passage must be in the first encoder batch')
                for item, index in zip(items, indexes):
                    if item[0] != tokens[index] or item[3] != rows[index]:
                        raise AssertionError('Frozen passage mapping changed')
                current['prepared_batch'] = {'indexes': list(indexes), 'content_lengths': [len(item[0]) for item in items],
                    'sequence_lengths': [len(item[0])+item[3] for item in items]}
                return original_prepare(items, indexes, outputs, device)
            remember(ac, '_encode_prepared_batch', types.MethodType(prepare, ac))
            original_side = ac.side.decoder_forward
            def side_forward(embeds, mask, *args, **kwargs):
                if not kwargs.get('gradient_checkpointing') or not torch.is_grad_enabled() or not ac.side.model.training:
                    raise AssertionError('Native checkpoint/train-mode policy changed')
                r = row_index()
                if (mask is None) != (embeds.shape[0] == 1):
                    raise AssertionError('Original singleton-versus-padding policy changed')
                if mask is not None and (not bool(mask[r, :n].all()) or bool(mask[r, n:].any())):
                    raise AssertionError('Selected passage padding changed')
                current['side_shape'] = list(embeds.shape)
                batched('side_input', embeds)
                return original_side(embeds, mask, *args, **kwargs)
            remember(ac.side, 'decoder_forward', side_forward)
            def forbid_reader(*args):
                raise AssertionError('No reader forward is authorized')
            handles.append(decoder_layers(ac.target.model)[0].register_forward_pre_hook(forbid_reader))
            def layer_pre(module, args, kwargs):
                hidden = args[0] if args else kwargs['hidden_states']
                batched('layer_input', hidden)
                position = kwargs.get('position_ids')
                if position is None:
                    raise ValueError('Explicit native text position IDs required')
                pos = position[0 if position.shape[0] == 1 else row_index(), :n]
                if not torch.equal(pos, torch.arange(n, device=pos.device)):
                    raise ValueError('Independent passage position IDs changed')
                identity['position_ids'] = localized.metadata_tensor(pos)
                cos, sin = kwargs['position_embeddings']
                identity['rope_cos'] = localized.metadata_tensor(cos[0 if cos.shape[0] == 1 else row_index(), :n])
                identity['rope_sin'] = localized.metadata_tensor(sin[0 if sin.shape[0] == 1 else row_index(), :n])
            handles.append(layer.register_forward_pre_hook(layer_pre, with_kwargs=True))
            for name, module in [('input_norm', layer.input_layernorm), ('post_attention_norm', layer.post_attention_layernorm), ('mlp', layer.mlp)]:
                handles.append(module.register_forward_hook(lambda mod, args, out, key=name: batched(key, out)))
            projections = [('attn.'+name, getattr(attention, name)) for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
            projections += [('mlp.'+name, getattr(layer.mlp, name)) for name in ('gate_proj', 'up_proj', 'down_proj')]
            for name, projection in projections:
                handles.append(projection.register_forward_hook(lambda mod, args, out, key=name: batched(key+'.combined', out)))
                base_layer = getattr(projection, 'base_layer', projection)
                handles.append(base_layer.register_forward_hook(lambda mod, args, out, key=name: batched(key+'.base', out)))
                branch = {'base_weight_dtype': str(base_layer.weight.dtype), 'active_adapters': list(getattr(projection, 'active_adapters', []))}
                current['branches'][name] = branch
                if getattr(projection, 'disable_adapters', False) or getattr(projection, 'merged', False):
                    raise ValueError('Expected original active, unmerged side LoRA')
                for adapter in branch['active_adapters']:
                    if adapter not in projection.lora_A:
                        continue
                    branch[adapter] = {'A_dtype': str(projection.lora_A[adapter].weight.dtype),
                        'B_dtype': str(projection.lora_B[adapter].weight.dtype), 'scaling': float(projection.scaling[adapter])}
                    for letter in ('A', 'B'):
                        mod = getattr(projection, 'lora_'+letter)[adapter]
                        key = name+'.lora_'+letter+':'+adapter
                        handles.append(mod.register_forward_hook(lambda mod, args, out, key=key: batched(key, out)))
            registry = model_module.ALL_ATTENTION_FUNCTIONS
            def get_interface(implementation, fallback):
                native = registry.get_interface(implementation, fallback)
                def call(module, q, k, v, mask, **kwargs):
                    if module is not attention:
                        return native(module, q, k, v, mask, **kwargs)
                    current['native_attention_calls'] += 1
                    if kwargs.get('dropout', 0) != 0:
                        raise ValueError('Reference cannot ignore attention dropout')
                    allowed = canonical_mask(mask, row_index(), n, causal=kwargs.get('is_causal', module.is_causal))
                    identity['allowed_mask'] = localized.metadata_tensor(allowed)
                    identity['scale'] = float(kwargs.get('scaling', module.scaling))
                    for name, tensor in [('q', q), ('k', k), ('v', v)]:
                        record('attention.'+name, tensor[row_index(), :, :n])
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if not cpu: activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities) as profile:
                        output = native(module, q, k, v, mask, **kwargs)
                    names = sorted({event.name for event in profile.events() if any(term in event.name.lower()
                        for term in ('attention', 'attn', 'flash', 'sdpa', 'cudnn'))})
                    current['native_attention_operator_names'] = names
                    current['native_qkv_shapes'] = [list(t.shape) for t in (q,k,v)]
                    current['native_mask_shape'] = None if mask is None else list(mask.shape)
                    attended = output[0]
                    if attended.ndim != 4 or list(attended.shape[1:3]) != [q.shape[2], q.shape[1]]:
                        raise ValueError('Expected native output [B,S,H,D]')
                    batched('attention.native_output', attended)
                    return output
                return call
            remember(model_module, 'ALL_ATTENTION_FUNCTIONS', types.SimpleNamespace(get_interface=get_interface))
            def layer_post(module, args, output):
                batched('layer_output', output)
                raise LayerCaptured()
            handles.append(layer.register_forward_hook(layer_post))
            try:
                try:
                    ac.encode_batch(requests, with_layers=True)
                    raise AssertionError('Expected stop at captured layer3')
                except LayerCaptured:
                    pass
            finally:
                for handle in handles: handle.remove()
                remaining = {key for model in (ac.modules, ac.side.model, ac.target.model) for mod in model.modules()
                    for hooks in (mod._forward_hooks, mod._forward_pre_hooks) for key in hooks}
                if any(handle.id in remaining for handle in handles):
                    raise AssertionError('A diagnostic hook survived cleanup')
                for obj, name, prior in reversed(overrides):
                    if prior is sentinel: delattr(obj, name)
                    else: setattr(obj, name, prior)
            if current['native_attention_calls'] != 1:
                raise AssertionError('Expected exactly one captured native attention call')
            if [(name,p._version,p.requires_grad) for name,p in params] != state or any(p.grad is not None for _,p in params):
                raise AssertionError('Unexpected parameter/trainability/gradient change')
            if baseline_identity is not None and identity != baseline_identity:
                raise ValueError('Masks/positions/scaling differ; reference comparison is not authorized')
            reference = fp32_reference(values['attention.q'], values['attention.k'], values['attention.v'],
                torch.ones(n,n,dtype=torch.bool).tril(), identity['scale'], ac.device)
            values['attention.fp32_reference'] = reference
            compare = lambda a,b: base.previous.compare_tensors(a.float(),b.float(),rtol=.03,atol=2e-6)
            current['reference_vs_native'] = compare(reference, values['attention.native_output'])
            current['reference_note'] = 'Detached FP32 matmul/softmax, TF32 disabled only here; original output was never replaced; not equivalence.'
            current['identity'] = identity
            if baseline is None:
                baseline = values; baseline_identity = identity
            else:
                if set(values) != set(baseline): raise AssertionError('Captured stage/branch coverage changed')
                current['comparisons_to_baseline'] = {name: compare(baseline[name], value) for name,value in values.items()}
                current['first_native_nonzero'] = next((key for key,value in current['comparisons_to_baseline'].items()
                    if key != 'attention.fp32_reference' and value['max_abs'] != 0), None)
                if label == 'unchanged_repeat' and any(value['max_abs'] != 0 for value in current['comparisons_to_baseline'].values()):
                    raise AssertionError('Unchanged repetition differs; cannot attribute to packing')
            tensors[label] = values
            current['hooks_restored'] = True; current['parameter_metadata_unchanged'] = True
            flush(result); gc.collect()
        if tensor_path is not None:
            if tensor_path.exists(): raise FileExistsError(tensor_path)
            torch.save(tensors, tensor_path)
            result['captured_tensors'] = {'path':str(tensor_path),'sha256':sha(tensor_path)}
        result.update(status='diagnostic_complete', equivalence_validated=False, reader_forward_calls=0, backward_calls=0, optimizer_updates=0)
        return result
    finally:
        ac.config = saved_config


def worker(args):
    original = base.prove_case; path = args.output/'proof.json'
    def diagnostic(*positional, **kwargs):
        def flush(value):
            receipt = read(path); receipt['operator_attribution'] = value; write(path, receipt)
        result = capture_case(*positional, flush=flush, tensor_path=args.output/'operator_tensors.pt')
        raise localized.DiagnosticComplete(result)
    base.prove_case = diagnostic
    try:
        base.worker(args)
        raise AssertionError('Diagnostic should stop before further cases or optimizer work')
    except localized.DiagnosticComplete as done:
        receipt = read(path)
        if base.paired.payload_hashes(receipt['pair']) != receipt['payload_sha256'] or sha(Path(receipt['pair']['ac_checkpoint'])/'optimizer.pt') != receipt['optimizer_sha256']:
            raise AssertionError('Parent payload changed')
        receipt.update(status='diagnostic_complete', operator_attribution=done.result,
            operator_worker_sha256=sha(__file__), localization_worker_sha256=LOCALIZATION_SHA,
            parent_payloads_rehashed_unchanged=True, equivalence_validated=False, reader_forward_calls=0,
            backward_calls=0, optimizer_updates=0, saved_activation_offload=False)
        write(path, receipt)
    finally:
        base.prove_case = original


def cpu_check():
    import tempfile
    import torch
    from activation.tests.test_interventions import build,tiny_hybrid,trainer_and_examples,open_heads,open_attention
    import activation.ac_model.ac_model as ac_source
    import activation.ac_model.ac_model_training as trainer_source
    import activation.ac_model.ac_model_utils as utils_source
    source = {key: sha(module.__file__) for key,module in
        [('ac_model',ac_source),('trainer',trainer_source),('utils',utils_source)]}
    if source != base.paired.SOURCE:
        raise ValueError('CPU check must use the same frozen implementation')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        h,ac=build(root/'state',tiny_hybrid(root/'reader'),frequency=-1,input_pooling_stride=0,
            side_dir=tiny_hybrid(root/'side',hidden=16,seed=112))
        open_heads(ac);open_attention(ac)
        with torch.no_grad():
            for name,parameter in ac.side.model.named_parameters():
                if '.lora_B.' in name: parameter.normal_(std=.01)
        trainer,_,examples,_=trainer_and_examples(h,ac,micro_batch_examples=2,completion_samples=0,no_context_penalty_weight=0.)
        original_budget=ac.config.encode_batch_max_tokens
        result=capture_case(h,ac,trainer,examples,'tiny',cpu=True,tensor_path=root/'tensors.pt')
        assert original_budget==ac.config.encode_batch_max_tokens
        assert result['variants']['baseline']['side_shape'] != result['variants']['32768']['side_shape']
        assert all(row['hooks_restored'] for row in result['variants'].values())
        assert result['status']=='diagnostic_complete' and result['equivalence_validated'] is False
        assert all(row['max_abs']==0 for row in result['variants']['unchanged_repeat']['comparisons_to_baseline'].values())
        assert len(result['variants']['baseline']['stages'])>=25
        cases=0
        for mask in (torch.zeros(1,1,3,3), torch.ones(1,4), torch.ones(1,1,3,3,dtype=torch.bool)):
            try: canonical_mask(mask,0,3,causal=True)
            except ValueError: cases+=1
        assert cases==3
        reference=fp32_reference(torch.randn(4,3,8),torch.randn(2,3,8),torch.randn(2,3,8),torch.ones(3,3,dtype=torch.bool).tril(),.5,torch.device('cpu'))
        assert reference.shape==(3,4,8)
        assert all(p.grad is None for _,p in base.previous.parameters(h,ac))
        assert any('lora_B' in key and tensor.abs().any() for key,tensor in
            torch.load(root/'tensors.pt',weights_only=True)['baseline'].items())
        print(__import__('json').dumps({'status':'pass','negative_masks':cases,'source_sha256':source,'result':result}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--cpu-check',action='store_true')
    for name in ('material','bench','design','parent-receipt','memory-receipt','reports','runs-root','output'):
        parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if (sha(localized.__file__)!=LOCALIZATION_SHA or sha(base.__file__)!=localized.BASE_SHA
            or sha(base.previous.__file__)!=base.PREVIOUS_SHA or sha(base.paired.__file__)!=base.PAIRED_SHA
            or sha(Path(__file__).with_name('probe_piecewise_original_v1.py'))!=base.previous.HELPER_SHA):
        raise ValueError('Frozen dependency changed')
    if args.cpu_check:cpu_check()
    else:
        args.output.mkdir(parents=True,exist_ok=True);args.worker='proof';worker(args)
