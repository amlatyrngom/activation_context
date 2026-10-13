"""Batched multi-section QA synthesis with the complete long evidence retained as plain data."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import generate_bank as bank
from select_bank import read_batch


def support_source(packet):
    sections = {section['id']: section for section in packet['sections']}
    units = []
    for ident in packet['suggested_support_section_ids']:
        section = sections[ident]
        units.append({'id': ident, 'document_id': packet['doc_id'],
                      'text': packet['text'][section['start_char']:section['end_char']]})
    if len(units) < 2:
        raise ValueError('Long QA requires at least two support sections')
    text = '\n\n'.join(f'Source {i}:\n{unit["text"]}' for i, unit in enumerate(units))
    return {'units': units, 'text': text, 'mode': 'selected_sections', 'kind': packet['source_kind']}


def main(args):
    if not 1 <= args.batch_size <= 64 or args.seconds <= 180:
        raise ValueError('Batch size must be1–64 and the segment must exceed180seconds')
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.common.reporting import HtmlReporter
    from activation.dataset.dataset import DataModality
    from activation.dataset.dataset_study_prompts import make_source_study_prompt
    from vllm.sampling_params import StructuredOutputsParams

    folder = Path(os.environ['QA_REPORT_ROOT']) / args.run_name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / 'PRESERVE').write_text('Portable long-evidence QA; preserve accepted records independently of recipe changes.\n')
    source_hash = bank.fingerprint(args.sources)
    packets = list(read_batch(args.sources))
    path = folder / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {
        'schema_version': 1, 'source_sha256': source_hash, 'batches': [], 'processed_source_ids': [], 'total_questions': 0}
    if manifest['source_sha256'] != source_hash:
        raise ValueError('Resume requires the same immutable source packet file')
    processed = set(manifest['processed_source_ids'])
    seen = set()
    for entry in manifest['batches']:
        if bank.fingerprint(folder / entry['file']) != entry['sha256']:
            raise ValueError('Changed published QA batch')
        seen.update(row['question'].strip().casefold() for row in read_batch(folder / entry['file']))
    remaining = [packet for packet in packets if packet['source_id'] not in processed]
    reporter = HtmlReporter(str(folder), 'Long-evidence QA synthesis',
        'Questions combine selected support sections; complete long evidence is retained separately. One grounding and multi-section check per question.')
    reporter.set_status(phase='loading QA model', source_packets=len(packets), completed_packets=len(processed), accepted_questions=manifest['total_questions'])
    reporter.render(force=True)
    if not remaining:
        reporter.set_status(phase='generation complete');reporter.finish()
        return 0
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs={'qa': ModelConfig('qa', 'Qwen/Qwen3.5-9B', engine_kwargs={
        'quantization': 'fp8', 'enable_lora': False, 'max_model_len': 8192, 'gpu_memory_utilization': .85, 'max_num_seqs': 64})}))
    model = h.loaded_models['qa']
    model.ensure_engine_loaded()
    started = time.monotonic()
    context = ('Create useful terminal-assistant questions that require combining information from at least two supplied sections. '
        'Prefer code behavior across functions, multi-step procedures, rule/exception interactions, or grounded mathematical reasoning. '
        'Do not merely ask for two unrelated facts. Make assumptions and entity names explicit. Keep answers concise, generally 2–40 words. '
        'Return no question when the sections cannot support a natural combined question. Distinct questions should use different reasoning.')
    recipe = {'generator_script_sha256': bank.fingerprint(Path(__file__)), 'model': 'Qwen/Qwen3.5-9B', 'quantization': 'dynamic FP8',
              'source_sha256': source_hash, 'study_context': context, 'questions_per_source_max': 3, 'temperature': .6,
              'validation': 'one grounding and multi-section check', 'prompt_modes': 'code/numeric/constraints/qa'}
    bank.write(folder / 'recipe.json', recipe)
    checker_schema = {'type': 'object', 'properties': {'supported': {'type': 'boolean'}, 'needs_multiple_sections': {'type': 'boolean'}},
                      'required': ['supported', 'needs_multiple_sections'], 'additionalProperties': False}
    for start in range(0, len(remaining), args.batch_size):
        if bank.STOP or time.monotonic() - started >= args.seconds - 180:
            break
        packets_batch = remaining[start:start + args.batch_size]
        sources, chats = [], []
        schema = None
        for packet in packets_batch:
            source = support_source(packet)
            if len(model.tokenizer.encode(source['text'], add_special_tokens=False)) > 6144:
                raise ValueError('Support bundle exceeds the agreed synthesis budget: ' + packet['source_id'])
            domain = packet['domain']
            kind = 'code' if domain == 'code' else 'numeric' if domain in ('numeric', 'math', 'math_arxiv', 'financial') else 'constraints' if domain in ('legal', 'office') else 'qa'
            system, instructions, schema = make_source_study_prompt(context, DataModality.TEXT, kind, 3, True)
            sources.append(source)
            chats.append([{'role': 'system', 'content': system}, {'role': 'user', 'content': source['text'] + '\n\n' + instructions}])
        reporter.set_status(phase='generating multi-section questions', completed_packets=len(processed), source_packets=len(packets), accepted_questions=manifest['total_questions'])
        reporter.render(force=True)
        outputs = model.engine_chat_many(chats, chat_kwargs={'chat_template_kwargs': {'enable_thinking': False}, 'sampling_params': {
            'max_tokens': 1024, 'temperature': .6, 'seed': 20260924 + len(processed), 'structured_outputs': StructuredOutputsParams(json=schema)}})
        if len(outputs) != len(packets_batch):
            raise RuntimeError('Generator returned the wrong batch size')
        candidates = []
        for packet, source, output in zip(packets_batch, sources, outputs):
            try:
                questions = json.loads(output.text)['questions']
                for qa in questions[:3]:
                    question, answer = qa['question'].strip(), qa['answer'].strip()
                    refs = sorted(set(qa['evidence_indices']))
                    if not question or not answer or len(refs) < 2 or any(type(i) is not int or not 0 <= i < len(source['units']) for i in refs):
                        continue
                    if question.casefold() not in seen:
                        candidates.append((packet, source, question, answer, refs))
            except (ValueError, KeyError, TypeError, AttributeError):
                continue
        accepted = []
        for offset in range(0, len(candidates), 64):
            chunk = candidates[offset:offset + 64]
            reporter.set_status(phase='checking grounding and combined evidence', candidate_questions=len(candidates), checked_questions=offset)
            reporter.render(force=True)
            chats = [[{'role': 'system', 'content': 'Judge the question and proposed answer using only the numbered evidence. Check the question’s factual premises before judging the answer. Reject incorrect premises even when the answer agrees with them. Recompute numerical differences and their direction; trace the exact named code branch and its units rather than borrowing an adjacent branch or guessing missing dependency definitions. supported is true only when both the question’s premises and the answer are correct without unstated facts. needs_multiple_sections is true only when a complete answer requires combining at least two sections; redundant citations or two unrelated trivia questions do not qualify. Return JSON with these two booleans.'},
                      {'role': 'user', 'content': source['text'] + '\nQuestion: ' + question + '\nAnswer: ' + answer}]
                     for _, source, question, answer, _ in chunk]
            checked = model.engine_chat_many(chats, chat_kwargs={'chat_template_kwargs': {'enable_thinking': False}, 'sampling_params': {
                'max_tokens': 64, 'temperature': 0, 'structured_outputs': StructuredOutputsParams(json=checker_schema)}})
            if len(checked) != len(chunk):
                raise RuntimeError('Grounding checker returned the wrong batch size')
            for (packet, source, question, answer, refs), output in zip(chunk, checked):
                try:
                    verdict = json.loads(output.text)
                except (ValueError, TypeError):
                    continue
                if not isinstance(verdict,dict) or verdict.get('supported') is not True or verdict.get('needs_multiple_sections') is not True or question.casefold() in seen:
                    continue
                ident = hashlib.sha256((packet['source_id'] + '\n' + question).encode()).hexdigest()[:24]
                accepted.append({'schema_version': 1, 'id': 'longqa:' + ident, 'domain': packet['domain'], 'source_family': packet['domain'],
                    'question': question, 'answer': answer, 'source': source, 'supporting_unit_ids': [source['units'][i]['id'] for i in refs],
                    'source_tokens': packet['source_tokens'], 'answer_tokens': len(model.tokenizer.encode(answer, add_special_tokens=False)),
                    'long_context': packet['text'], 'complete_parent': packet['complete_parent'], 'split_group_id': packet['split_group_id'],
                    'target_tokens': packet['target_tokens'], 'provenance': {'generator_model': 'Qwen/Qwen3.5-9B dynamicFP8',
                    'source_request_id': packet['source_id'], 'source_file_sha256': source_hash, 'generation_recipe': recipe, 'source': packet['provenance'],
                    'generation_evidence': 'source.units are exactly what generation/checking saw; long_context is the retained full training/evaluation context'}})
                seen.add(question.casefold())
        entry = bank.immutable_rows(folder, f'batches/batch_{len(manifest["batches"]):05d}', accepted)
        entry['source_ids'] = [packet['source_id'] for packet in packets_batch]
        manifest['batches'].append(entry)
        processed.update(entry['source_ids'])
        manifest.update(processed_source_ids=sorted(processed), total_questions=manifest['total_questions'] + len(accepted),
                        updated_utc=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()))
        bank.write(path, manifest)
        reporter.status.clear()
        reporter.set_status(phase='batch published', completed_packets=len(processed), source_packets=len(packets), accepted_questions=manifest['total_questions'])
        reporter.render(force=True)
    complete = len(processed) == len(packets)
    reporter.status.clear()
    reporter.set_status(phase='generation complete' if complete else 'paused; accepted data preserved', completed_packets=len(processed), source_packets=len(packets), accepted_questions=manifest['total_questions'])
    reporter.finish()
    from activation.harness.hf_utils import FREE_DEVICE
    model.engine_to_device(FREE_DEVICE)
    return 0 if complete else 75


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--run-name', default='long_qa_bank')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seconds', type=int, default=6900)
    raise SystemExit(main(parser.parse_args()))
