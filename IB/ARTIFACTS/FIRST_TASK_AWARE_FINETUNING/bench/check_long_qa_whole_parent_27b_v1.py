"""CPU-only checks for temporary labels and portable exact-span evidence."""
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
import generate_long_qa_whole_parent_27b_v1 as worker
import generate_long_qa_whole_parent_v1 as original_worker


def check_publication_and_resume(tokenizer, packets):
    """Exercise the real writer/reporter with a small fake engine, without loading a GPU model."""
    calls = []

    def chat_many(chats, chat_kwargs):
        assert chat_kwargs['chat_template_kwargs'] == {'enable_thinking': False}
        parameters = chat_kwargs['sampling_params']
        assert parameters['top_p'] == .8 and parameters['top_k'] == 20 and parameters['min_p'] == 0.
        assert parameters['presence_penalty'] == 0.
        if chats[0][0]['content'] == worker.GENERATOR_SYSTEM:
            assert parameters['temperature'] == .6 and parameters['max_tokens'] == 1024 and parameters['seed'] == worker.SEED
        else:
            assert parameters['temperature'] == 0 and parameters['max_tokens'] == 96
        calls.append(chats[0]['role'] if isinstance(chats[0], dict) else chats[0][0]['content'])
        if chats[0][0]['content'] == worker.GENERATOR_SYSTEM:
            good = {'question': 'How does the named rule affect the given scenario?',
                    'answer': 'Its exception determines the stated result.', 'evidence_indices': [0, 1]}
            bad = {**good, 'question': 'Which conditions apply to this scenario?', 'answer': 'Source 0 gives the conditions.'}
            rejected = {**good, 'question': 'Does the named function preserve the stated invariant?'}
            values = [{'questions': [good, bad, rejected]}, {'questions': []}]
        else:
            values = [{'supported': True, 'needs_multiple_sections': True, 'self_contained': True},
                      {'supported': False, 'needs_multiple_sections': True, 'self_contained': True}]
        assert len(values) == len(chats)
        return [types.SimpleNamespace(text=json.dumps(value), output_token_count=40, finish_reason='stop') for value in values]

    model = types.SimpleNamespace(tokenizer=tokenizer, ensure_engine_loaded=lambda: None,
                                  engine_chat_many=chat_many, engine_to_device=lambda _: None)
    harness = types.ModuleType('activation.harness')
    harness.HarnessRuntimeConfig = lambda **kwargs: kwargs
    harness.ModelConfig = lambda name, model_id, **kwargs: {'name': name, 'model_id': model_id, **kwargs}

    def create_harness(config):
        qa = config['model_configs']['qa']
        assert Path(qa['model_id']).name == worker.MODEL_REVISION
        assert qa['engine_kwargs'] == {'quantization': 'fp8', 'enable_lora': False,
            'max_model_len': 40960, 'gpu_memory_utilization': .85, 'max_num_seqs': 2,
            'speculative_config': None, 'enable_chunked_prefill': True, 'max_num_batched_tokens': 8192}
        return types.SimpleNamespace(loaded_models={'qa': model})

    harness.HarnessRuntime = create_harness
    hf_utils = types.ModuleType('activation.harness.hf_utils')
    hf_utils.FREE_DEVICE = 'cpu'
    sampling = types.ModuleType('vllm.sampling_params')
    sampling.StructuredOutputsParams = lambda **kwargs: kwargs
    report_path = Path(__file__).resolve().parents[4] / 'activation/common/reporting.py'
    spec = importlib.util.spec_from_file_location('private_pilot_reporting', report_path)
    reporting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporting)
    modules = {'activation.harness': harness, 'activation.harness.hf_utils': hf_utils,
               'activation.common.reporting': reporting, 'vllm.sampling_params': sampling}
    with tempfile.TemporaryDirectory(prefix='long-qa-private-cpu-') as directory, patch.dict(sys.modules, modules), patch.dict(os.environ, QA_REPORT_ROOT=directory):
        source = Path(directory) / 'sources.jsonl'
        source.write_text(''.join(json.dumps(packet) + '\n' for packet in packets[:2]))
        model_path = Path(directory) / worker.MODEL_REVISION
        model_path.mkdir()
        (model_path / 'config.json').write_text('{}')
        args = types.SimpleNamespace(sources=source, run_name='fake_engine_check', batch_size=2, seconds=600, model_path=model_path)
        assert worker.main(args) == 0
        folder = Path(directory) / args.run_name
        manifest = json.loads((folder / 'manifest.json').read_text())
        assert manifest['total_questions'] == 1 and len(manifest['processed_source_ids']) == 2
        assert manifest['counts']['generated_candidates'] == 3
        assert manifest['counts']['temporary_reference'] == 1
        assert manifest['counts']['check_rejected'] == 1
        assert manifest['counts']['parents_with_no_questions'] == 1
        row = next(worker.read_batch(folder / manifest['batches'][0]['file']))
        assert row['question'].startswith('How does') and len(row['source']['units']) == 2
        assert row['provenance']['generator_model'] == 'Qwen/Qwen3.5-27B dynamic FP8'
        assert manifest['recipe']['model_revision'] == worker.MODEL_REVISION
        assert (folder / 'report.tressoir.html').exists() and (folder / 'report_data.json').exists()
        assert worker.main(args) == 0 and len(calls) == 2  # Completed resume does not regenerate or recheck.
        committed = folder / manifest['batches'][0]['file']
        committed.write_text(committed.read_text() + '\n')
        try:
            worker.main(args)
        except ValueError as error:
            assert 'Changed published pilot batch' in str(error)
        else:
            raise AssertionError('A changed committed batch must fail before resuming')


def main(args):
    assert worker.GENERATOR_SYSTEM == original_worker.GENERATOR_SYSTEM
    assert worker.CHECKER_SYSTEM == original_worker.CHECKER_SYSTEM
    assert worker.GENERATOR_SCHEMA == original_worker.GENERATOR_SCHEMA and worker.CHECKER_SCHEMA == original_worker.CHECKER_SCHEMA
    assert worker.SEED == original_worker.SEED
    assert worker.bank.fingerprint(args.sources) == '7e3941fde695ddb3ba75151b1d9fc8cdfae4f789d99e57358b4467d6228565f6'
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    packets = list(worker.read_batch(args.sources))
    checked_spans = 0
    lengths = []
    for packet in packets:
        assert worker.generator_chat(packet) == original_worker.generator_chat(packet)
        assert packet['source_id'] == __import__('hashlib').sha256(packet['text'].encode()).hexdigest()
        assert packet['sections'][0]['start_char'] == 0
        assert packet['sections'][-1]['end_char'] == len(packet['text'])
        assert all(a['end_char'] == b['start_char'] for a, b in zip(packet['sections'], packet['sections'][1:]))
        refs = [0, len(packet['sections']) - 1]
        qa = {'question': 'How does the named rule change the stated result?', 'answer': 'The exception changes the result.', 'evidence_indices': refs}
        candidate, error = worker.parse_candidate(qa, packet, tokenizer)
        assert candidate and error is None
        row = worker.accepted_row(packet, candidate, tokenizer, 'cpu-fixture-source', {})
        assert row['long_context'] == packet['text']
        assert row['evidence_indices'] == refs
        assert row['split_group_id'] == packet['split_group_id']
        for unit in row['source']['units']:
            assert unit['text'] == row['long_context'][unit['start_char']:unit['end_char']]
            assert unit['id'] == row['long_context_sections'][unit['section_index']]['id']
            checked_spans += 1
        assert len(row['source']['units']) == 2
        assert row['source']['text'] == '\n\n'.join(unit['text'] for unit in row['source']['units'])
        checker_text = worker.checker_chat(candidate)[1]['content']
        assert all(unit['text'] in checker_text for unit in candidate['source']['units'])
        lengths.append(worker.prompt_tokens(tokenizer, worker.generator_chat(packet)))
        assert lengths[-1] >= packet['source_tokens']
    packet = packets[0]
    good = {'question': 'How does Section 402A apply to the defect?', 'answer': 'The named manufacturer is strictly liable.', 'evidence_indices': [0, 1]}
    assert worker.parse_candidate(good, packet, tokenizer)[1] is None
    leaks = ['Source 0 says yes.', 'Sources 1 and 2 agree.', 'SOURCE [2] is correct.',
             'EVIDENCE_UNIT_12 says yes.', 'evidence unit 2 says yes.',
             'The other section explains this.', 'According to the provided text, yes.']
    for field in ('question', 'answer'):
        for leak in leaks:
            assert worker.parse_candidate({**good, field: leak}, packet, tokenizer)[1] == 'temporary_reference'
    for refs in ([0, 0], [True, 1], [0, -1], [0, len(packet['sections'])], [0, 1, 2, 3], ['0', 1]):
        assert worker.parse_candidate({**good, 'evidence_indices': refs}, packet, tokenizer)[1] is not None
    assert worker.parse_candidate({'question': [], 'answer': 'yes', 'evidence_indices': [0, 1]}, packet, tokenizer)[1]
    assert max(lengths) + worker.MAX_OUTPUT_TOKENS <= worker.MAX_MODEL_TOKENS
    check_publication_and_resume(tokenizer, packets)
    result = {'parents_checked': len(packets), 'exact_support_spans_checked': checked_spans,
              'label_rejection_fixtures': 2 * len(leaks), 'bad_reference_fixtures': 6,
              'intrinsic_statute_identifier_allowed': True, 'max_prompt_tokens': max(lengths),
              'max_prompt_plus_output_tokens': max(lengths) + worker.MAX_OUTPUT_TOKENS,
              'fake_engine_publication_resume_and_hash_check': True, 'real_html_reporter_checked': True,
              'gpu_generation_tested': False, 'semantic_quality_tested': False, 'same75_parent_bytes_and_order': True,
              'generator_and_checker_prompts_schemas_unchanged': True, 'explicit_sampling_matches_effective9b_defaults': True,
              'pinned27b_provenance_checked': True, 'tokenizer_path': str(args.tokenizer)}
    worker.bank.write(args.sources.parent / 'cpu_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    main(parser.parse_args())
