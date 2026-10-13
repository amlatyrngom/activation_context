"""Fresh-engine 2048/8192 prefill-budget comparison on the same uncached 64-request batches."""
import argparse
import fcntl
import json
import os
import threading
import time
from pathlib import Path

from generate_bank import fingerprint, write
from select_bank import read_batch
from run_checker_ab_27b_v1 import BOOLEAN_PROMPT, MODEL_ID, MODEL_REVISION, SCHEMAS, chat_for, parse_output


def engine_kwargs(prefill_tokens):
    return {'quantization': 'fp8', 'enable_lora': False, 'max_model_len': 8192,
            'gpu_memory_utilization': .85, 'max_num_seqs': 64, 'speculative_config': None,
            'max_num_batched_tokens': prefill_tokens}


def sampling_kwargs():
    return {'max_tokens': 48, 'temperature': 0, 'top_p': .8, 'top_k': 20, 'min_p': 0., 'presence_penalty': 0.}


def aggregate_calls(calls):
    return {'requests': sum(len(call['records']) for call in calls),
            'call_seconds': sum(call['seconds'] for call in calls),
            'prompt_tokens': sum(record['prompt_tokens'] for call in calls for record in call['records']),
            'cached_prompt_tokens': sum(record['cached_prompt_tokens'] for call in calls for record in call['records']),
            'output_tokens': sum(record['output_tokens'] for call in calls for record in call['records']),
            'invalid_outputs': sum(record['parse_error'] is not None for call in calls for record in call['records'])}


def compare(arms):
    recipes = []
    for arm, expected_budget in zip(arms, (2048, 8192)):
        recipe = json.loads(json.dumps(arm['recipe']))
        if recipe['engine_kwargs'].pop('max_num_batched_tokens') != expected_budget:
            raise ValueError('Wrong prefill condition')
        recipes.append(recipe)
    if recipes[0] != recipes[1] or arms[0]['prompt_tokens'] != arms[1]['prompt_tokens']:
        raise ValueError('Inputs, tokenizer counts and all other recipe settings must match')
    a, b = [{(call['pass'], record['case_id']): record for call in arm['steady_calls'] for record in call['records']} for arm in arms]
    if set(a) != set(b) or len(a) != 256:
        raise ValueError('Both engines must finish the identical case/pass pairs')
    valid = [key for key in a if a[key]['parse_error'] is None and b[key]['parse_error'] is None]
    changed = [{'pass': key[0], 'case_id': key[1], 'p2048': a[key]['supported'], 'p8192': b[key]['supported']}
               for key in valid if a[key]['supported'] != b[key]['supported']]
    return {'paired_requests': len(a), 'valid_paired_requests': len(valid),
            'matching_decisions': len(valid) - len(changed), 'changed_decisions': changed,
            'not_semantic_accuracy': True}


def run_arm(args):
    import torch
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.common.reporting import HtmlReporter
    from activation.harness.hf_utils import FREE_DEVICE
    from vllm.sampling_params import StructuredOutputsParams

    cases = list(read_batch(args.cases))
    if len(cases) != 128 or len({case['id'] for case in cases}) != 128:
        raise ValueError('Expected exactly 128 fixed distinct cases')
    if args.model_path.name != MODEL_REVISION or not (args.model_path / 'config.json').is_file():
        raise ValueError('Use the pinned prefetched 27B snapshot')
    if torch.cuda.device_count() != 1:
        raise ValueError('This diagnostic requires exactly one visible GPU')
    folder = Path(os.environ['QA_REPORT_ROOT']) / args.run_name / f'p{args.prefill_tokens}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = folder / 'results.json'
    if path.exists():
        raise FileExistsError('Preserve the existing attempt and choose another run name')
    (folder / 'PRESERVE').write_text('Uncached 27B checker throughput diagnostic; preserve exact records and engine log.\n')
    recipe = {'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_path': str(args.model_path),
        'engine_kwargs': engine_kwargs(args.prefill_tokens), 'sampling_kwargs': sampling_kwargs(),
        'enable_thinking': False, 'checker_prompt': BOOLEAN_PROMPT, 'cases_sha256': fingerprint(args.cases),
        'worker_sha256': fingerprint(Path(__file__)), 'checker_helper_sha256': fingerprint(Path(__file__).with_name('run_checker_ab_27b_v1.py')),
        'batch_requests': 64, 'steady_passes': 2,
        'cache_policy': 'Production cache configuration retained. Reset prefix cache before warm-up and every measured batch; report any within-batch hits.',
        'scope': 'Uncached checker call throughput, not producer end-to-end rate or semantic accuracy.'}
    result = {'status': 'loading model', 'recipe': recipe, 'warmup': None, 'steady_calls': [], 'gpu_memory_samples': []}
    write(path, result)
    reporter = HtmlReporter(str(folder), f'27B checker prefill budget {args.prefill_tokens}',
        'Fixed 128 questions, 64 requests per call, two measured passes. Prefix cache cleared before each measured batch. Throughput only.')
    reporter.initialize_table('calls', 'Steady checker calls', 'Startup and warm-up excluded; prefix clearing excluded from call timing.',
        ['pass', 'batch', 'requests', 'seconds', 'prompt_tokens', 'cached_tokens', 'invalid_outputs'])
    reporter.set_status(phase='loading fresh engine', prefill_token_budget=args.prefill_tokens, completed_requests='0/256')
    reporter.render(force=True)
    started = time.monotonic()
    phase = ['fresh engine load']
    stop_memory = threading.Event()

    def sample_memory():
        while not stop_memory.is_set():
            try:
                free, total = torch.cuda.mem_get_info(0)
                result['gpu_memory_samples'].append({'seconds': time.monotonic()-started, 'phase': phase[0],
                    'free_bytes': free, 'total_bytes': total, 'used_bytes': total-free})
            except Exception as error:
                result['gpu_memory_sample_error'] = str(error)
                return
            stop_memory.wait(.5)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    model = None
    try:
        h = HarnessRuntime(HarnessRuntimeConfig(model_configs={'qa': ModelConfig('qa', str(args.model_path), engine_kwargs=engine_kwargs(args.prefill_tokens))}))
        model = h.loaded_models['qa']
        model.ensure_engine_loaded()
        result['fresh_engine_load_seconds'] = time.monotonic() - started
        replicas = model.vllm_model.replicas
        if len(replicas) != 1:
            raise ValueError('Expected one engine replica')
        config = replicas[0].stats._config
        scheduler = config.scheduler_config
        cache = config.cache_config
        result['effective_engine_config'] = {
            'max_num_batched_tokens': scheduler.max_num_batched_tokens, 'max_num_seqs': scheduler.max_num_seqs,
            'enable_chunked_prefill': getattr(scheduler, 'enable_chunked_prefill', None),
            'enable_prefix_caching': getattr(cache, 'enable_prefix_caching', None),
            'cache_dtype': getattr(cache, 'cache_dtype', None), 'num_gpu_blocks': getattr(cache, 'num_gpu_blocks', None),
            'block_size': getattr(cache, 'block_size', None)}
        if scheduler.max_num_batched_tokens != args.prefill_tokens or scheduler.max_num_seqs != 64:
            raise ValueError('Effective scheduler configuration differs from the requested condition')
        chats = [chat_for(case, 'boolean_only') for case in cases]
        lengths = []
        for chat in chats:
            rendered = model.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            lengths.append(len(model.tokenizer.encode(rendered, add_special_tokens=False)))
        if max(lengths) + 48 > 8192:
            raise ValueError('Input exceeds model context; never truncate the fixed evidence')
        result['prompt_tokens'] = {'min': min(lengths), 'max': max(lengths), 'total': sum(lengths)}

        def call(indices, label, pass_number=None, batch_number=None):
            phase[0] = label
            reporter.status.clear()
            reporter.set_status(phase=label, completed_requests=f'{sum(len(c["records"]) for c in result["steady_calls"])}/256',
                                prefill_token_budget=args.prefill_tokens)
            reporter.render(force=True)
            reset_started = time.monotonic()
            reset_results = [replica.run(replica.engine.reset_prefix_cache()) for replica in replicas]
            if any(reset is False for reset in reset_results):
                raise RuntimeError('Prefix cache reset refused; cannot claim an uncached comparison')
            reset_seconds = time.monotonic() - reset_started
            before = time.monotonic()
            outputs = model.engine_chat_many([chats[index] for index in indices], chat_kwargs={
                'chat_template_kwargs': {'enable_thinking': False},
                'sampling_params': sampling_kwargs() | {'structured_outputs': StructuredOutputsParams(json=SCHEMAS['boolean_only'])}})
            seconds = time.monotonic() - before
            if len(outputs) != len(indices):
                raise RuntimeError('Wrong output count')
            records = []
            for index, output in zip(indices, outputs):
                supported, error, _ = parse_output(output.text, 'boolean_only')
                records.append({'case_id': cases[index]['id'], 'supported': supported, 'parse_error': error,
                    'raw_output': output.text, 'prompt_tokens': output.prompt_token_count,
                    'output_tokens': output.output_token_count, 'cached_prompt_tokens': output.cached_prompt_token_count,
                    'finish_reason': output.finish_reason})
            measured = {'pass': pass_number, 'batch': batch_number, 'seconds': seconds,
                        'prefix_reset_seconds': reset_seconds, 'prefix_reset_results': reset_results, 'records': records}
            if pass_number is None:
                result['warmup'] = measured
            else:
                result['steady_calls'].append(measured)
                reporter.add_data_point('calls', {'pass': pass_number, 'batch': batch_number, 'requests': len(indices),
                    'seconds': round(seconds, 4), 'prompt_tokens': sum(r['prompt_tokens'] for r in records),
                    'cached_tokens': sum(r['cached_prompt_tokens'] for r in records),
                    'invalid_outputs': sum(r['parse_error'] is not None for r in records)})
            result['kv_metrics'] = model.vllm_model.metrics()
            result['status'] = 'running'
            write(path, result)
            reporter.render(force=True)

        call(list(range(64)), 'warm-up: 64 requests')
        for pass_number in (1, 2):
            for batch_number, start in enumerate((0, 64), 1):
                call(list(range(start, start + 64)), 'steady checker calls', pass_number, batch_number)
        result['summary'] = aggregate_calls(result['steady_calls'])
        summary = result['summary']
        summary['requests_per_second'] = summary['requests'] / summary['call_seconds']
        summary['uncached_prompt_tokens_per_second'] = (summary['prompt_tokens']-summary['cached_prompt_tokens']) / summary['call_seconds']
        record_map = {(call['pass'], record['case_id']): record for call in result['steady_calls'] for record in call['records']}
        valid_replays = [case['id'] for case in cases if all(record_map[(pass_number, case['id'])]['parse_error'] is None for pass_number in (1, 2))]
        summary['valid_within_arm_replays'] = len(valid_replays)
        summary['matching_within_arm_replays'] = sum(record_map[(1, ident)]['supported'] == record_map[(2, ident)]['supported'] for ident in valid_replays)
        result['status'] = 'complete'
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        phase[0] = 'shutdown'
        stop_memory.set()
        monitor.join(timeout=2)
        samples = result['gpu_memory_samples']
        result['memory_summary'] = {'minimum_sampled_free_bytes': min((sample['free_bytes'] for sample in samples), default=None),
            'maximum_sampled_used_bytes': max((sample['used_bytes'] for sample in samples), default=None),
            'sample_interval_seconds': .5, 'note': 'Device-wide sampled usage, including engine subprocess; not an exact allocator peak.'}
        stats = [sample for replica in result.get('kv_metrics', []) for sample in replica['samples']]
        result['memory_summary']['maximum_sampled_kv_usage'] = max((sample['kv_usage'] for sample in stats), default=None)
        result['elapsed_before_shutdown_seconds'] = time.monotonic() - started
        write(path, result)
        reporter.status.clear()
        reporter.set_status(phase=result['status'], prefill_token_budget=args.prefill_tokens)
        reporter.finish()
        if model is not None:
            model.engine_to_device(FREE_DEVICE)
    return 0


def summarize(args):
    from activation.common.reporting import HtmlReporter
    folder = Path(os.environ['QA_REPORT_ROOT']) / args.run_name
    arms = [json.loads((folder / f'p{budget}/results.json').read_text()) for budget in (2048, 8192)]
    if any(arm['status'] != 'complete' for arm in arms):
        raise ValueError('Both fresh-engine arms must complete before comparison')
    comparison = compare(arms)
    comparison['arms'] = {str(budget): arm['summary'] for budget, arm in zip((2048, 8192), arms)}
    comparison['uncached_call_speedup_8192_over_2048'] = arms[0]['summary']['call_seconds'] / arms[1]['summary']['call_seconds']
    comparison['scope'] = 'Uncached grounding checker call throughput only; no whole-bank end-to-end throughput or semantic accuracy claim.'
    write(folder / 'comparison.json', comparison)
    reporter = HtmlReporter(str(folder), '27B uncached checker throughput', comparison['scope'])
    columns = ['prefill_budget', 'fresh_load_seconds', 'warmup_seconds', 'steady_seconds', 'requests_per_second',
               'uncached_prompt_tokens_per_second', 'invalid_outputs', 'minimum_sampled_free_GiB', 'maximum_sampled_kv_usage']
    reporter.initialize_table('conditions', 'Fresh engine conditions', 'Same 128 cases, two passes in 64-request batches; prefix cache reset before each call.', columns)
    for budget, arm in zip((2048, 8192), arms):
        memory, summary = arm['memory_summary'], arm['summary']
        reporter.add_data_point('conditions', dict(zip(columns, [budget, round(arm['fresh_engine_load_seconds'], 3),
            round(arm['warmup']['seconds'], 3), round(summary['call_seconds'], 3), round(summary['requests_per_second'], 2),
            round(summary['uncached_prompt_tokens_per_second'], 2), summary['invalid_outputs'],
            None if memory['minimum_sampled_free_bytes'] is None else round(memory['minimum_sampled_free_bytes']/2**30, 2),
            memory['maximum_sampled_kv_usage']])))
    reporter.set_status(phase='comparison complete', unique_cases=128, paired_requests=comparison['paired_requests'],
        valid_decision_agreement=f'{comparison["matching_decisions"]}/{comparison["valid_paired_requests"]}',
        measured_call_speedup=f'{comparison["uncached_call_speedup_8192_over_2048"]:.3f}x')
    reporter.finish()
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--prefill-tokens', type=int, choices=(2048, 8192))
    parser.add_argument('--run-name', default='checker_prefill_27b_v1')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if not args.summarize and (args.model_path is None or args.prefill_tokens is None):
        parser.error('Each fresh-engine arm needs --model-path and --prefill-tokens')
    raise SystemExit(summarize(args) if args.summarize else run_arm(args))
