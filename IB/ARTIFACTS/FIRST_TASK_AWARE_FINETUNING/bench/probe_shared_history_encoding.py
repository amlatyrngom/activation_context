"""Private GPU gate for differentiable encoding shared by independent QA examples.

One real warm-up Adam update allocates optimizer state. The measured forwards
then hold weights fixed, force all no-history penalties on, and compare every
trainable gradient. This is not a training continuation or a resume proof.
"""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import time

from run_qa import AC, harness, write
from run_trajectory_tasks import build_items, digest


def request_key(request):
    # Exact ratio, including all encoder inputs; no persistent or rollout cache.
    return json.dumps(request._asdict(), sort_keys=True, separators=(',', ':'))


class SharedEncoding:
    def __init__(self, model):
        self.model = model
        self.counts = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def encode_batch(self, requests, *, with_layers=False):
        unique, indexes, positions = [], {}, []
        for request in requests:
            key = request_key(request)
            if key not in indexes:
                indexes[key] = len(unique)
                unique.append(request)
            positions.append(indexes[key])
        outputs = self.model.encode_batch(unique, with_layers=with_layers)
        self.counts.append({'requested': len(requests), 'encoded': len(unique)})
        return [outputs[index] for index in positions]


class ProbeComplete(Exception):
    pass


def gradient_difference(reference, actual):
    import torch
    assert reference.keys() == actual.keys()
    accumulators = defaultdict(lambda: [0., 0., 0., 0.])
    maximum = 0.
    for key, first in reference.items():
        second = actual[key]
        assert (first is None) == (second is None), key
        if first is None:
            continue
        first, second = first.float(), second.float()
        assert torch.isfinite(first).all() and torch.isfinite(second).all(), key
        delta = second - first
        group = key.split(':')[0]
        totals = accumulators[group]
        totals[0] += float(first.square().sum())
        totals[1] += float(second.square().sum())
        totals[2] += float(delta.square().sum())
        totals[3] += float((first * second).sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {'maximum_absolute_difference': maximum, 'optimizer_groups': {
        group: {'reference_norm': math.sqrt(a), 'actual_norm': math.sqrt(b),
                'relative_l2': math.sqrt(c / max(a, 1e-30)),
                'cosine': dot / max(math.sqrt(a * b), 1e-30)}
        for group, (a, b, c, dot) in accumulators.items()}}


def main(args):
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_study import ActivationContextStudyGenerator
    if not torch.cuda.is_available():
        raise RuntimeError('Run on an allocated GPU, never a login node')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'result.json').exists():
        raise FileExistsError('Use a fresh probe directory')
    os.environ['ACTIVATION_SYNC_ROOT'] = str(args.output.resolve())
    selection = json.loads(args.selection.read_text())
    groups = defaultdict(list)
    for entry in selection['train']:
        groups[entry['choice']['window_id']].append(entry)
    groups = sorted((v for v in groups.values() if len(v) >= 4),
                    key=lambda v: v[0]['choice']['source_tokens'])
    assert len(groups) >= 2
    selected = [groups[0][:4], groups[-1][:4]]
    assert selected[-1][0]['choice']['source_tokens'] >= 28000
    entries = [entry for group in selected for entry in group]
    write(args.output / 'selection.json', entries)
    pair = json.loads(args.pair.read_text())
    h = harness(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = h.module_manager.get_ac_model(AC)
    ac.epoch_number = 0
    ac.config.encode_batch_max_tokens = 65536
    study = ActivationContextStudyGenerator(h, AC)
    items = build_items(study, args.material, entries, 'trajectory_qa', 1 / 16)
    config = ActivationContextTrainingConfig(loss_kind='sft', learning_rate_ac=5e-5,
        learning_rate_target_lora=2e-5, examples_per_update=4, micro_batch_examples=1,
        num_epochs=1, warmup_updates=0, schedule_total_updates=2, seed=7,
        teacher_cache_block_examples=8, no_context_penalty_weight=1,
        no_context_penalty_fraction=1, reporting_interval=.5, completion_samples=0,
        logits_chunk_tokens=64, gradient_checkpointing_min_tokens=0,
        checkpoint_every_epoch=False, max_duration_seconds=args.seconds,
        shuffle_examples=False)
    trainer = ActivationContextTrainer(h, config)
    record = {'status': 'running', 'source_selection': digest(selection),
              'pair': pair, 'gpu': torch.cuda.get_device_name(),
              'source_tokens': [group[0]['choice']['source_tokens'] for group in selected],
              'penalty_fraction': 1, 'offload_moments': args.offload_moments,
              'cases': [], 'limit': 'Fixed-weight gradient/performance gate after one warm-up update; no resume claim.'}
    offloader = None

    def callback(progress, summary):
        nonlocal offloader
        optimizer = trainer.optimizers[AC]
        if progress.epoch == 0:
            if args.offload_moments:
                from probe_adam_moment_offload_v1 import MomentOffload
                offloader = MomentOffload(optimizer)
                offloader.park()
            return
        assert progress.global_step == 1
        device = next(ac.target.model.parameters()).device
        examples = [trainer.build_example(ac, item) for item in items]
        trainer._prepare_teacher_block(ac, examples, device)
        for offset, length_label in [(0, 'short'), (4, 'long')]:
            for count in [2, 4]:
                batch = examples[offset:offset + count]
                requests = [request for example in batch for request in example.part_requests]
                assert len(requests) == count and len({request_key(r) for r in requests}) == 1
                captured = []
                for mode in ['duplicate', 'duplicate_repeat', 'shared', 'shared_repeat']:
                    optimizer.zero_grad(set_to_none=True)
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    model = SharedEncoding(ac) if mode.startswith('shared') else ac
                    started = time.monotonic()
                    with torch.enable_grad():
                        results = trainer._batch_loss(model, batch, device)
                        assert all(r['penalty'] is not None for r in results)
                        losses = [r['objective'] + r['penalty'] for r in results]
                        loss = sum(value * (example.item.weight / count)
                                   for value, example in zip(losses, batch))
                        loss.backward()
                    torch.cuda.synchronize()
                    measurement = {'mode': mode, 'seconds': time.monotonic() - started,
                                   'loss': float(loss.detach()),
                                   'peak_memory_bytes': torch.cuda.max_memory_allocated(),
                                   'requests': model.counts if isinstance(model, SharedEncoding) else [{'requested': count, 'encoded': count}]}
                    gradients = {f'{gi}:{pi}': None if p.grad is None else p.grad.detach().cpu().clone()
                                 for gi, group in enumerate(optimizer.param_groups)
                                 for pi, p in enumerate(group['params'])}
                    for gi in range(len(optimizer.param_groups)):
                        values = [g for key, g in gradients.items()
                                  if key.startswith(f'{gi}:') and g is not None]
                        assert values and all(torch.isfinite(g).all() for g in values)
                        assert any(torch.count_nonzero(g) for g in values), f'No gradient in optimizer group {gi}'
                    if captured:
                        measurement['gradient_difference'] = gradient_difference(captured[0][1], gradients)
                        measurement['loss_absolute_difference'] = abs(measurement['loss'] - captured[0][0]['loss'])
                    captured.append((measurement, gradients if not captured else None))
                    del results, losses, loss, gradients
                    print(json.dumps({'length': length_label, 'questions': count, **measurement}), flush=True)
                case = {'length': length_label, 'questions': count,
                        'measurements': [entry[0] for entry in captured]}
                case['passed'] = all(m.get('loss_absolute_difference', 0) <= .01 and
                    all(g['relative_l2'] <= .05 and g['cosine'] >= .995
                        for g in m.get('gradient_difference', {}).get('optimizer_groups', {}).values())
                    for m in case['measurements'])
                record['cases'].append(case)
                write(args.output / 'result.json', record)
                del captured
        optimizer.zero_grad(set_to_none=True)
        record['status'] = 'pass' if all(c['passed'] for c in record['cases']) else 'numerical gate failed'
        write(args.output / 'result.json', record)
        raise ProbeComplete()

    try:
        trainer.train(AC, items, reporting_callback=callback)
    except ProbeComplete:
        pass
    finally:
        if offloader is not None:
            offloader.close()
    assert record['status'] == 'pass', record['status']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ['selection', 'material', 'pair', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=2700)
    parser.add_argument('--offload-moments', action='store_true')
    main(parser.parse_args())
