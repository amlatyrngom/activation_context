"""One-engine, 23-case comparison: current boolean checker versus a brief check before verdict."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from generate_bank import fingerprint, write
from select_bank import read_batch

MODEL_ID = 'Qwen/Qwen3.5-27B'
MODEL_REVISION = 'fc05daec18b0a78c049392ed2e771dde82bdf654'
# Frozen verbatim from the deployed worker, rather than importing a moving production prompt.
BOOLEAN_PROMPT = ('Judge the question and proposed answer using only the supplied evidence. '
    'Check the question\'s factual premises before judging the answer. Reject incorrect premises even when the answer agrees with them. '
    'Recompute numerical differences, their direction and units; trace the exact named code branch rather than borrowing an adjacent branch '
    'or guessing missing dependency definitions. Reject ambiguous, incorrect, or unsupported answers. '
    'Reject any question or answer that depends on temporary numbered source labels such as Source 0 or Source 2: '
    'the question must name the actual entity, document or function, and the answer must stand alone. '
    'supported is true only when both the question\'s premises and the answer are correct without unstated facts. '
    'Return JSON {"supported":true or false}.')
BRIEF_PROMPT = BOOLEAN_PROMPT.rsplit('Return JSON ', 1)[0] + (
    'Return JSON with the check field FIRST, then supported. The check is at most 40 words: '
    'write a brief independent calculation or exact source evidence that tests the factual premise and answer. '
    'Compute from the evidence rather than accepting the question\'s suggested result; '
    'if essential evidence is missing, say what is missing. Then give the supported boolean.')
CONDITIONS = ('boolean_only', 'brief_check')
PROMPTS = dict(zip(CONDITIONS, (BOOLEAN_PROMPT, BRIEF_PROMPT)))
SCHEMAS = {
    'boolean_only': {'type': 'object', 'properties': {'supported': {'type': 'boolean'}},
                     'required': ['supported'], 'additionalProperties': False},
    'brief_check': {'type': 'object', 'properties': {'check': {'type': 'string'}, 'supported': {'type': 'boolean'}},
                    'required': ['check', 'supported'], 'additionalProperties': False}}
OUTPUT_LIMITS = {'boolean_only': 48, 'brief_check': 128}


def chat_for(case, condition):
    # Never pass manual labels, reasons, origins or IDs to the checker.
    return [{'role': 'system', 'content': PROMPTS[condition]},
            {'role': 'user', 'content': case['evidence'] + '\nQuestion: ' + case['question'] + '\nAnswer: ' + case['answer']}]


def parse_output(text, condition):
    try:
        result = json.loads(text)
    except (TypeError, ValueError):
        return None, 'invalid_json', None
    if not isinstance(result, dict) or type(result.get('supported')) is not bool:
        return None, 'missing_boolean', result
    if condition == 'brief_check':
        if not isinstance(result.get('check'), str) or not result['check'].strip():
            return None, 'missing_check', result
        if list(result) != ['check', 'supported']:
            return None, 'check_not_before_verdict', result
    elif list(result) != ['supported']:
        return None, 'unexpected_fields', result
    return result['supported'], None, result


def summary_of(cases, records):
    output = {}
    for condition in CONDITIONS:
        rows = [record for record in records if record['condition'] == condition]
        scored = [record for record in rows if record['expected_supported'] is not None]
        output[condition] = {'completed_cases': len(rows), 'scorable_cases': len(scored),
            'correct_decisions': sum(record['supported'] == record['expected_supported'] for record in scored),
            'false_accepts': sum(record['supported'] is True and record['expected_supported'] is False for record in scored),
            'false_rejects': sum(record['supported'] is False and record['expected_supported'] is True for record in scored),
            'invalid_outputs': sum(record['parse_error'] is not None for record in rows),
            'output_tokens': sum(record['output_tokens'] for record in rows)}
    by_key = {(record['case_id'], record['condition']): record for record in records}
    paired = []
    for case in cases:
        if all((case['id'], condition) in by_key for condition in CONDITIONS):
            decisions = {condition: by_key[(case['id'], condition)]['supported'] for condition in CONDITIONS}
            if decisions['boolean_only'] != decisions['brief_check']:
                paired.append({'case_id': case['id'], 'expected_supported': case['expected_supported'], **decisions})
    output['changed_decisions'] = paired
    return output


def main(args):
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.common.reporting import HtmlReporter
    from vllm.sampling_params import StructuredOutputsParams

    cases = list(read_batch(args.cases))
    if not 20 <= len(cases) <= 24 or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Use one fixed set of 20–24 distinct cases')
    if args.model_path.name != MODEL_REVISION or not (args.model_path / 'config.json').is_file():
        raise ValueError('Use the prefetched pinned 27B snapshot: ' + MODEL_REVISION)
    folder = Path(os.environ['QA_REPORT_ROOT']) / args.run_name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'results.json').exists():
        raise FileExistsError('Immutable attempt exists; inspect it or use a different run name')
    (folder / 'PRESERVE').write_text('Diagnostic checker A/B, not training data. Preserve raw outputs and frozen cases.\n')
    recipe = {'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'model_path': str(args.model_path),
        'quantization': 'dynamic FP8', 'enable_thinking': False, 'temperature': 0, 'presence_penalty': 0.,
        'top_p': .8, 'top_k': 20, 'min_p': 0., 'max_model_len': 8192, 'max_num_seqs': 64,
        'enable_prefix_caching': False, 'output_limits': OUTPUT_LIMITS, 'prompts': PROMPTS,
        'worker_sha256': fingerprint(Path(__file__)), 'cases_sha256': fingerprint(args.cases),
        'ordering': 'Two halves; A then B for first half, B then A for second. Each condition warms up on the same shortest case first. '
                    'Warm-ups excluded from decisions and comparison timing. Prefix caching disabled for both conditions.',
        'limits': 'Small diagnostic set includes known failures; not population accuracy or a model-size comparison.'}
    write(folder / 'recipe.json', recipe)
    write(folder / 'cases.json', cases)
    results = {'status': 'loading model', 'recipe': recipe, 'warmups': [], 'batches': [], 'records': [], 'summary': {}}
    write(folder / 'results.json', results)
    reporter = HtmlReporter(str(folder), '27B checker comparison',
        'Same pinned FP8 model and evidence; boolean-only versus a brief independent check before verdict. Thinking is off. '
        'Three manually ambiguous cases are displayed but excluded from correctness counts.')
    columns = ['condition', 'cases', 'correct', 'false_accepts', 'false_rejects', 'invalid', 'call_seconds', 'output_tokens']
    reporter.initialize_table('summary', 'Checker decisions', 'Timing excludes model startup and warm-up. This diagnostic set includes known failures.', columns)
    reporter.initialize_table('cases', 'Per-case decisions', 'Expected ? means deliberately unscored; raw checks are saved in results.json.',
        ['case', 'family', 'expected', 'boolean', 'brief', 'brief_check'])
    started = time.monotonic()
    reporter.set_status(phase='loading pinned 27B model', cases=len(cases), completed_checks='0/' + str(2*len(cases)))
    reporter.render(force=True)
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs={'qa': ModelConfig('qa', str(args.model_path), engine_kwargs={
        'quantization': 'fp8', 'enable_lora': False, 'max_model_len': 8192, 'gpu_memory_utilization': .85,
        'max_num_seqs': 64, 'speculative_config': None, 'enable_prefix_caching': False})}))
    model = h.loaded_models['qa']
    model.ensure_engine_loaded()
    results['model_load_seconds'] = time.monotonic() - started

    def run(condition, subset, warmup=False):
        chats = [chat_for(case, condition) for case in subset]
        for chat in chats:
            rendered = model.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if len(model.tokenizer.encode(rendered, add_special_tokens=False)) + OUTPUT_LIMITS[condition] > 8192:
                raise ValueError('Evidence exceeds model context; do not truncate checker input')
        reporter.status.clear()
        reporter.set_status(phase='warming both output formats' if warmup else 'comparing checker decisions',
            condition=condition, completed_checks=f'{len(results["records"])}/{2*len(cases)}', active_cases=len(subset))
        reporter.render(force=True)
        call_started = time.monotonic()
        outputs = model.engine_chat_many(chats, chat_kwargs={'chat_template_kwargs': {'enable_thinking': False}, 'sampling_params': {
            'max_tokens': OUTPUT_LIMITS[condition], 'temperature': 0, 'top_p': .8, 'top_k': 20, 'min_p': 0.,
            'presence_penalty': 0., 'seed': 20260926, 'structured_outputs': StructuredOutputsParams(json=SCHEMAS[condition])}})
        seconds = time.monotonic() - call_started
        if len(outputs) != len(subset):
            raise RuntimeError('Checker returned the wrong number of outputs')
        call = {'condition': condition, 'case_ids': [case['id'] for case in subset], 'seconds': seconds,
                'output_tokens': sum(output.output_token_count for output in outputs)}
        records = []
        for case, output in zip(subset, outputs):
            supported, error, parsed = parse_output(output.text, condition)
            records.append({'case_id': case['id'], 'condition': condition, 'expected_supported': case['expected_supported'],
                'supported': supported, 'parse_error': error, 'parsed': parsed, 'raw_output': output.text,
                'prompt_tokens': output.prompt_token_count, 'output_tokens': output.output_token_count,
                'cached_prompt_tokens': output.cached_prompt_token_count, 'finish_reason': output.finish_reason})
        if warmup:
            results['warmups'].append({**call, 'records': records})
        else:
            results['batches'].append(call)
            results['records'].extend(records)
        results['status'] = 'running'
        results['summary'] = summary_of(cases, results['records'])
        write(folder / 'results.json', results)

    shortest = min(cases, key=lambda case: len(case['evidence']))
    for condition in CONDITIONS:
        run(condition, [shortest], warmup=True)
    midpoint = (len(cases) + 1) // 2
    for subset, order in ((cases[:midpoint], CONDITIONS), (cases[midpoint:], tuple(reversed(CONDITIONS)))):
        for condition in order:
            run(condition, subset)
    results['status'] = 'complete'
    results['elapsed_seconds'] = time.monotonic() - started
    results['summary'] = summary_of(cases, results['records'])
    for condition in CONDITIONS:
        metrics = results['summary'][condition]
        metrics['call_seconds'] = sum(batch['seconds'] for batch in results['batches'] if batch['condition'] == condition)
        reporter.add_data_point('summary', dict(zip(columns, [condition, metrics['completed_cases'], metrics['correct_decisions'],
            metrics['false_accepts'], metrics['false_rejects'], metrics['invalid_outputs'], round(metrics['call_seconds'], 3), metrics['output_tokens']])))
    records_by_key = {(record['case_id'], record['condition']): record for record in results['records']}
    for index, case in enumerate(cases):
        a, b = (records_by_key[(case['id'], condition)] for condition in CONDITIONS)
        reporter.add_data_point('cases', {'case': f'{index+1}: {case["id"][:12]}', 'family': case['family'],
            'expected': '?' if case['expected_supported'] is None else case['expected_supported'],
            'boolean': a['supported'] if a['parse_error'] is None else a['parse_error'],
            'brief': b['supported'] if b['parse_error'] is None else b['parse_error'],
            'brief_check': (b['parsed'] or {}).get('check', b['raw_output'])})
    write(folder / 'results.json', results)
    reporter.status.clear()
    reporter.set_status(phase='comparison complete', completed_checks=f'{2*len(cases)}/{2*len(cases)}',
                        model_load_minutes=f'{results["model_load_seconds"]/60:.1f}', total_minutes=f'{results["elapsed_seconds"]/60:.1f}')
    reporter.finish()
    from activation.harness.hf_utils import FREE_DEVICE
    model.engine_to_device(FREE_DEVICE)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--run-name', default='checker_ab_27b_v1')
    raise SystemExit(main(parser.parse_args()))
