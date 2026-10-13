"""CPU checks of matched inputs, fresh engines, resets, decision parsing and publication."""
import argparse
import collections
import copy
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

from transformers import AutoTokenizer
import generate_bank_27b as production
import run_checker_prefill_27b_v1 as worker


def main(args):
    assert worker.BOOLEAN_PROMPT == production.CHECKER_PROMPT
    configs = [worker.engine_kwargs(budget) for budget in (2048, 8192)]
    assert {k: v for k, v in configs[0].items() if k != 'max_num_batched_tokens'} == {
        k: v for k, v in configs[1].items() if k != 'max_num_batched_tokens'}
    assert worker.sampling_kwargs() == {'max_tokens': 48, 'temperature': 0, 'top_p': .8,
                                        'top_k': 20, 'min_p': 0., 'presence_penalty': 0.}
    cases = list(worker.read_batch(args.cases))
    assert len(cases) == len({case['id'] for case in cases}) == 128
    assert all('expected_supported' not in case for case in cases)
    manifest = json.loads((args.cases.parent / 'cases_manifest.json').read_text())
    assert dict(collections.Counter(case['family'] for case in cases)) == manifest['by_family']
    assert manifest['cases_sha256'] == worker.fingerprint(args.cases)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    lengths, payloads = [], {}
    for case in cases:
        assert all(unit['text'] in case['evidence'] for unit in case['source_units'])
        chat = worker.chat_for(case, 'boolean_only')
        assert chat == [{'role': 'system', 'content': production.CHECKER_PROMPT},
                        {'role': 'user', 'content': case['evidence'] + '\nQuestion: ' + case['question'] + '\nAnswer: ' + case['answer']}]
        payloads[chat[1]['content']] = case
        rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        lengths.append(len(tokenizer.encode(rendered, add_special_tokens=False)))
    assert max(lengths) + 48 <= 8192
    assert worker.parse_output('{"supported":false}', 'boolean_only')[:2] == (False, None)
    assert worker.parse_output('{"supported":true}', 'boolean_only')[:2] == (True, None)
    assert worker.parse_output('{"supported":"false"}', 'boolean_only')[1] == 'missing_boolean'
    assert worker.parse_output('{"supported":', 'boolean_only')[1] == 'invalid_json'
    assert worker.parse_output('{"supported":true,"extra":1}', 'boolean_only')[1] == 'unexpected_fields'
    created, calls, freed, reset_events = [], [], [], []

    def create_model(config):
        engine = config['model_configs']['qa']['engine_kwargs']
        budget = engine['max_num_batched_tokens']
        created.append(budget)
        state = {'reset': False}

        def reset():
            assert not state['reset']
            state['reset'] = True
            reset_events.append(budget)
            return True

        def chat_many(chats, chat_kwargs):
            assert state['reset'] and len(chats) == 64
            state['reset'] = False
            assert chat_kwargs['chat_template_kwargs'] == {'enable_thinking': False}
            sampling = dict(chat_kwargs['sampling_params'])
            assert sampling.pop('structured_outputs') == {'json': worker.SCHEMAS['boolean_only']}
            assert sampling == worker.sampling_kwargs()
            selected = [payloads[chat[1]['content']] for chat in chats]
            calls.append((budget, [case['id'] for case in selected]))
            outputs = []
            for case in selected:
                value = {'supported': not (budget == 8192 and case['id'] == cases[0]['id'])}
                raw = '{unfinished' if budget == 8192 and case['id'] == cases[1]['id'] else json.dumps(value)
                outputs.append(types.SimpleNamespace(text=raw, prompt_token_count=1000,
                    output_token_count=8, cached_prompt_token_count=0, finish_reason='stop'))
            return outputs

        config_object = types.SimpleNamespace(scheduler_config=types.SimpleNamespace(
            max_num_batched_tokens=budget, max_num_seqs=64, enable_chunked_prefill=True),
            cache_config=types.SimpleNamespace(enable_prefix_caching=True, cache_dtype='fp8', num_gpu_blocks=1000, block_size=16))
        replica = types.SimpleNamespace(stats=types.SimpleNamespace(_config=config_object),
            engine=types.SimpleNamespace(reset_prefix_cache=reset), run=lambda value: value)
        vllm_model = types.SimpleNamespace(replicas=[replica], metrics=lambda: [{'replica': 0, 'kv_tokens': 16000,
            'samples': [{'kv_usage': .4}]}])
        model = types.SimpleNamespace(tokenizer=tokenizer, ensure_engine_loaded=lambda: None,
            engine_chat_many=chat_many, engine_to_device=lambda _: freed.append(budget), vllm_model=vllm_model)
        return types.SimpleNamespace(loaded_models={'qa': model})

    harness = types.ModuleType('activation.harness')
    harness.HarnessRuntimeConfig = lambda **kwargs: kwargs
    harness.ModelConfig = lambda *args, **kwargs: kwargs
    harness.HarnessRuntime = create_model
    hf_utils = types.ModuleType('activation.harness.hf_utils')
    hf_utils.FREE_DEVICE = 'cpu'
    sampling = types.ModuleType('vllm.sampling_params')
    sampling.StructuredOutputsParams = lambda **kwargs: kwargs
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(device_count=lambda: 1, mem_get_info=lambda _: (10 * 2**30, 96 * 2**30))
    report_path = Path(__file__).resolve().parents[4] / 'activation/common/reporting.py'
    spec = importlib.util.spec_from_file_location('private_checker_reporting', report_path)
    reporting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporting)
    modules = {'activation.harness': harness, 'activation.harness.hf_utils': hf_utils, 'torch': torch,
               'activation.common.reporting': reporting, 'vllm.sampling_params': sampling}
    with tempfile.TemporaryDirectory(prefix='checker-prefill-private-cpu-') as directory, patch.dict(sys.modules, modules), patch.dict(os.environ, QA_REPORT_ROOT=directory):
        model_path = Path(directory) / worker.MODEL_REVISION
        model_path.mkdir()
        (model_path / 'config.json').write_text('{}')
        run_args = types.SimpleNamespace(cases=args.cases, model_path=model_path, run_name='fake_engine_check')
        for budget in (2048, 8192):
            run_args.prefill_tokens = budget
            assert worker.run_arm(run_args) == 0
        assert created == freed == [2048, 8192]
        assert reset_events == [2048] * 5 + [8192] * 5
        expected = [[case['id'] for case in cases[:64]], [case['id'] for case in cases[64:]]]
        assert [ids for _, ids in calls[:5]] == [expected[0]] + expected * 2
        assert [ids for _, ids in calls[5:]] == [ids for _, ids in calls[:5]]
        assert worker.summarize(run_args) == 0
        folder = Path(directory) / run_args.run_name
        result = json.loads((folder / 'comparison.json').read_text())
        assert result['paired_requests'] == 256 and result['valid_paired_requests'] == 254
        assert result['matching_decisions'] == 252 and len(result['changed_decisions']) == 2
        assert result['arms']['8192']['invalid_outputs'] == 2 and result['arms']['2048']['invalid_outputs'] == 0
        assert all(result['arms'][str(b)]['requests'] == 256 for b in (2048, 8192))
        assert result['not_semantic_accuracy'] is True and (folder / 'report.tressoir.html').is_file()
        arms = [json.loads((folder / f'p{budget}/results.json').read_text()) for budget in (2048, 8192)]
        bad = copy.deepcopy(arms)
        bad[1]['recipe']['sampling_kwargs']['temperature'] = .7
        try:
            worker.compare(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Must reject mismatched non-prefill recipe settings')
    receipt = {'cases': 128, 'case_sha256': worker.fingerprint(args.cases), 'production_prompt_and_sampling_match': True,
        'only_explicit_engine_difference_is_prefill_budget': True, 'same_64_request_batches_both_conditions': True,
        'fresh_engines': 2, 'cache_resets': 10, 'measured_requests_per_engine': 256,
        'proxy_9b_prompt_tokens_min_max_total': [min(lengths), max(lengths), sum(lengths)],
        'runtime_27b_tokenizer_check_required': True, 'invalid_output_and_changed_decision_fixtures': True,
        'mismatched_recipe_rejected': True, 'real_reporter_publication': True, 'gpu_tested': False}
    worker.write(args.cases.parent / 'cpu_checks.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    main(parser.parse_args())
