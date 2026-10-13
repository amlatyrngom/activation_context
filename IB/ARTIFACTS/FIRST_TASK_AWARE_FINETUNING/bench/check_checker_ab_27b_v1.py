"""CPU checks of case isolation, checker parsing, and one-engine comparison publication."""
import argparse
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
import run_checker_ab_27b_v1 as worker


def main(args):
    assert worker.BOOLEAN_PROMPT == production.CHECKER_PROMPT
    assert worker.BRIEF_PROMPT.startswith(worker.BOOLEAN_PROMPT.rsplit('Return JSON ', 1)[0])
    cases = list(worker.read_batch(args.cases))
    assert len(cases) == 23
    assert sum(case['expected_supported'] is None for case in cases) == 3
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    lengths = []
    payloads = {}
    for case in cases:
        expected_payload = case['evidence'] + '\nQuestion: ' + case['question'] + '\nAnswer: ' + case['answer']
        payloads[expected_payload] = case
        assert case['manual_reason'] not in expected_payload
        assert all(unit['text'] in case['evidence'] for unit in case['source_units'])
        for condition in worker.CONDITIONS:
            chat = worker.chat_for(case, condition)
            assert chat[1]['content'] == expected_payload
            rendered = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            length = len(tokenizer.encode(rendered, add_special_tokens=False)) + worker.OUTPUT_LIMITS[condition]
            assert 10 < length < 8192
            lengths.append(length)
    assert worker.parse_output('{"supported":false}', 'boolean_only')[:2] == (False, None)
    assert worker.parse_output('{"check":"Exact source contradicts premise.","supported":false}', 'brief_check')[:2] == (False, None)
    assert worker.parse_output('{"supported":false,"check":"Too late."}', 'brief_check')[1] == 'check_not_before_verdict'
    assert worker.parse_output('{"supported":"false"}', 'boolean_only')[1] == 'missing_boolean'
    assert worker.parse_output('{"check":"unfinished', 'brief_check')[1] == 'invalid_json'
    assert worker.parse_output('{"check":"","supported":true}', 'brief_check')[1] == 'missing_check'
    calls, models = [], []
    changed_id = next(case['id'] for case in cases if case['expected_supported'] is False)

    def chat_many(chats, chat_kwargs):
        condition = next(name for name in worker.CONDITIONS if worker.PROMPTS[name] == chats[0][0]['content'])
        sampling = chat_kwargs['sampling_params']
        assert chat_kwargs['chat_template_kwargs']['enable_thinking'] is False
        assert sampling['temperature'] == 0 and sampling['presence_penalty'] == 0
        assert sampling['max_tokens'] == worker.OUTPUT_LIMITS[condition] <= 128
        outputs = []
        ids = []
        for chat in chats:
            case = payloads[chat[1]['content']]
            ids.append(case['id'])
            supported = True if case['expected_supported'] is None else case['expected_supported']
            if case['id'] == changed_id and condition == 'boolean_only':
                supported = True
            value = {'check': 'Exact evidence fixture.', 'supported': supported} if condition == 'brief_check' else {'supported': supported}
            outputs.append(types.SimpleNamespace(text=json.dumps(value), output_token_count=12,
                prompt_token_count=100, cached_prompt_token_count=0, finish_reason='stop'))
        calls.append((condition, ids))
        return outputs

    model = types.SimpleNamespace(tokenizer=tokenizer, ensure_engine_loaded=lambda: None,
                                  engine_chat_many=chat_many, engine_to_device=lambda _: None)

    def create_model(config):
        models.append(config)
        return types.SimpleNamespace(loaded_models={'qa': model})

    harness = types.ModuleType('activation.harness')
    harness.HarnessRuntimeConfig = lambda **kwargs: kwargs
    harness.ModelConfig = lambda *args, **kwargs: kwargs
    harness.HarnessRuntime = create_model
    hf_utils = types.ModuleType('activation.harness.hf_utils')
    hf_utils.FREE_DEVICE = 'cpu'
    sampling = types.ModuleType('vllm.sampling_params')
    sampling.StructuredOutputsParams = lambda **kwargs: kwargs
    report_path = Path(__file__).resolve().parents[4] / 'activation/common/reporting.py'
    spec = importlib.util.spec_from_file_location('private_checker_reporting', report_path)
    reporting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporting)
    modules = {'activation.harness': harness, 'activation.harness.hf_utils': hf_utils,
               'activation.common.reporting': reporting, 'vllm.sampling_params': sampling}
    with tempfile.TemporaryDirectory(prefix='checker-ab-private-cpu-') as directory, patch.dict(sys.modules, modules), patch.dict(os.environ, QA_REPORT_ROOT=directory):
        model_path = Path(directory) / worker.MODEL_REVISION
        model_path.mkdir()
        (model_path / 'config.json').write_text('{}')
        run_args = types.SimpleNamespace(cases=args.cases, model_path=model_path, run_name='fake_engine_check')
        assert worker.main(run_args) == 0 and len(models) == 1
        result = json.loads((Path(directory) / 'fake_engine_check/results.json').read_text())
        assert result['status'] == 'complete' and len(result['records']) == 46
        assert len(result['warmups']) == 2 and len(result['batches']) == 4
        assert [condition for condition, _ in calls] == ['boolean_only', 'brief_check', 'boolean_only', 'brief_check', 'brief_check', 'boolean_only']
        assert result['summary']['boolean_only']['false_accepts'] == 1
        assert result['summary']['brief_check']['false_accepts'] == 0
        assert len(result['summary']['changed_decisions']) == 1
        assert result['summary']['brief_check']['scorable_cases'] == 20
        assert (Path(directory) / 'fake_engine_check/report.tressoir.html').is_file()
    result = {'cases': len(cases), 'production_boolean_prompt_matches': True,
              'same_evidence_question_answer_both_conditions': True, 'manual_labels_excluded_from_chats': True,
              'max_prompt_plus_output_tokens_proxy_9b_tokenizer': max(lengths),
              'runtime_27b_tokenizer_budget_check_required': True,
              'output_parsing_fixtures': 6, 'fake_engine_same_model_publication_and_summary': True,
              'measured_case_condition_pairs': 46, 'gpu_tested': False}
    worker.write(args.cases.parent / 'cpu_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    main(parser.parse_args())
