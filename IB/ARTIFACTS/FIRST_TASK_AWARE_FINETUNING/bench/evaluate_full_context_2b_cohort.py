"""Bare frozen Qwen3.5-2B: exact saved32k development100 full-context evidence."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from evaluate_qualitative_cohort import read, write, digest as file_sha, parsed_answer

MODEL = 'Qwen/Qwen3.5-2B'
INPUT_SHA = '82af75710ca064d052cba8f0349c093ee2636b1be6d84e41eca5650115e9b66e'
CONDITION = 'frozen_qwen35_2b_full_context'


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def validate_rows(source):
    rows = source['rows']
    if (source.get('status') != 'complete' or len(rows) != 100 or len({r['item_id'] for r in rows}) != 100
            or [r['item_id'] for r in rows] != source['selection']['selected_ids']
            or {bank: sum(r['generator_cohort'] == bank for r in rows) for bank in ('9b', '27b')} != {'9b': 50, '27b': 50}):
        raise ValueError('Expected the exact completed32k fixed100 in saved50+50 order')
    for row in rows:
        messages = row['in_context_messages']
        if ([m['role'] for m in messages] != ['system', 'user', 'assistant']
                or len(row['assistant_token_ids']) != 1 or not row['assistant_token_ids'][0]):
            raise ValueError('Unexpected full-context history/targets')
        calls = messages[-1].get('tool_calls', [])
        if (len(calls) != 1 or calls[0]['function']['name'] != 'submit_answer'
                or calls[0]['function']['arguments'] != {'answer': row['gold']}):
            raise ValueError('Saved assistant/gold mismatch')
        if row['full_evidence'] not in messages[1]['content'] or row['question'] not in messages[1]['content']:
            raise ValueError('Saved prompt is missing the exact evidence/question')
        positions = row['answer_positions']
        if not positions or positions != sorted(set(positions)) or not all(0 <= p < len(row['assistant_token_ids'][0]) for p in positions):
            raise ValueError('Invalid saved answer mask')
    return rows


def prepare_rows(rows, tokenizer, tools):
    from activation.ac_model.ac_model_utils import assistant_target_ids, prepare_training_history
    from activation.agent.agent_utils import ModelDialect
    dialect = ModelDialect.for_tokenizer(tokenizer)
    prepared = []
    for row in rows:
        messages = row['in_context_messages']; targets = row['assistant_token_ids'][0]
        if assistant_target_ids(tokenizer, messages[-1], tools, {'enable_thinking': False}) != targets:
            raise ValueError('2B tokenizer does not reproduce saved assistant IDs: '+row['item_id'])
        rendered = tokenizer.decode(targets, skip_special_tokens=False)
        marker = '<parameter=answer>\n'; left = rendered.index(marker)+len(marker)
        right = rendered.index('\n</parameter>', left)
        normalizer = tokenizer.backend_tokenizer.normalizer
        if rendered[left:right] != (normalizer.normalize_str(row['gold']) if normalizer else row['gold']):
            raise ValueError('Rendered gold answer differs')
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        mask = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if a < right and b > left]
        if encoded['input_ids'] != targets or mask != row['answer_positions']:
            raise ValueError('Saved answer-token mask differs under2B tokenizer')
        ids, spans, positions = prepare_training_history(tokenizer, messages, 2, [targets], [],
            tools=tools, chat_template_kwargs={'enable_thinking': False})
        prefix = ids[:positions[0]+1]
        prompt, prompt_spans = dialect.prompt_tokens(tokenizer, messages[:2], tools,
            {'enable_thinking': False}, [], tokenizer.pad_token_id or 0)
        if spans or prompt_spans or prefix != prompt or ids[len(prefix):] != targets:
            raise ValueError('Generation prefix includes gold or differs from the exact prompt')
        prepared.append({'ids': ids, 'positions': positions, 'prefix': prefix, 'targets': targets})
    return prepared


def main(args):
    started = time.monotonic(); deadline = started+args.seconds
    output = args.output/(CONDITION+'.json'); status = args.output/(CONDITION+'_status.json')
    def progress(phase, **extra):
        record = {'condition': CONDITION, 'phase': phase, 'elapsed_seconds': round(time.monotonic()-started, 1), **extra}
        write(status, record); print(json.dumps(record), flush=True)
    progress('checking saved inputs and exact tokenizer/masks')
    if file_sha(args.input) != INPUT_SHA:
        raise ValueError('Input is not the frozen completed32k fixed100 receipt')
    source = read(args.input); rows = validate_rows(source)
    import torch
    import transformers
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from activation.dataset.loaders.trajectory_utils import submit_answer_definition
    from activation.agent.agent_utils import ModelDialect
    import activation.ac_model.ac_model_utils as ac_utils
    import activation.ac_model.ac_model_training as trainer_module
    import activation.harness.loaded_model as loaded_module
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import get_agent_training_autotuning_variables, configure_fla_runtime
    tools = [submit_answer_definition()]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    reference_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-9B', local_files_only=True)
    prepared = prepare_rows(rows, tokenizer, tools)
    original = prepare_rows(rows, reference_tokenizer, tools)
    if prepared != original:
        raise ValueError('2B and original9B rendered prompt/target tokens differ')
    del original, reference_tokenizer
    token_check = {'items': 100, 'assistant_ids_equal': True, 'answer_masks_equal': True,
        'full_prompt_2b_vs_9b_ids_equal': True, 'generation_prefix_excludes_all_gold_tokens': True,
        'input_token_fingerprint': digest(prepared), 'tools_sha256': digest(tools),
        'prompt_tokens': [len(p['prefix']) for p in prepared]}
    if args.cpu_check:
        write(args.output/'tokenizer_check.json', token_check); progress('CPU tokenizer check complete'); return
    if file_sha(trainer_module.__file__) != '38ca9885bfe35e47432352aa716381e53c18aa57748428eedf5f5f6b7fc6b5aa':
        raise ValueError('Expected frozen night trainer scorer')
    device = torch.device('cuda', torch.cuda.current_device())
    torch.set_num_threads(4); torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    snapshot = Path(hf_hub_download(MODEL, 'config.json', local_files_only=True)).parent
    assets = {leaf: file_sha(snapshot/leaf) for leaf in ('config.json', 'generation_config.json',
        'tokenizer_config.json', 'tokenizer.json', 'chat_template.jinja', 'model.safetensors.index.json') if (snapshot/leaf).exists()}
    identity = {'condition': CONDITION, 'model': MODEL, 'model_revision': snapshot.name,
        'model_metadata_sha256': assets, 'adapters': None, 'ac_encoder': None, 'input_sha256': INPUT_SHA,
        'selection_sha256': source['identity']['selection_sha256'], 'input_token_fingerprint': digest(prepared),
        'evaluator_sha256': file_sha(__file__), 'source_sha256': {'trainer': file_sha(trainer_module.__file__),
            'ac_utils': file_sha(ac_utils.__file__), 'loaded_model': file_sha(loaded_module.__file__)},
        'runtime': {'torch': torch.__version__, 'transformers': transformers.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(device)},
        'nll_microbatch': 2, 'nll_batching': 'These exact100 sorted by full sequence length, right padded; not the original400 batch partners.',
        'nll': 'Macro mean gold-answer token NLL; frozen original targets/masks; four-decimal token logp as existing panels.',
        'decoding': 'Greedy, thinking disabled, max160, inherited EOS union chat EOT; prefix contains no gold assistant tokens.',
        'scope': 'Frozen bare2B full-context baseline on original development100; no training or confirmation.'}
    if output.exists():
        result = read(output)
        if result['identity'] != identity:
            raise ValueError('Partial readout identity changed')
        if result['status'] == 'complete':
            progress('already complete'); return
    else:
        result = {'status': 'running', 'identity': identity, 'selection': source['selection'], 'tokenizer_check': token_check,
            'rows': [{**{key: value for key, value in row.items() if key != 'views'}, 'views': {'full context': {}}} for row in rows], 'views': {}}
    write(output, result)
    progress('loading bare frozen2B only')
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs={'frozen_2b': ModelConfig('frozen_2b', MODEL)}))
    target = h.loaded_models['frozen_2b']; target.model_to_device(str(device)); model = target.model
    if prepare_rows(rows, target.tokenizer, tools) != prepared:
        raise ValueError('Loaded-model tokenizer differs from validated tokenizer')
    model.requires_grad_(False); model.eval()
    if getattr(model, 'peft_config', None) or h.module_manager.ac_models or h.module_manager.has_loras('frozen_2b'):
        raise ValueError('Bare baseline unexpectedly contains adapters/AC')
    config = ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0,
        gradient_checkpointing=False, gradient_checkpointing_min_tokens=0, logits_chunk_tokens=64, completion_samples=0)
    trainer = ActivationContextTrainer(h, config)
    variables = get_agent_training_autotuning_variables(target.model_config, config, device=device)
    generation = model.generation_config; inherited = generation.eos_token_id
    eos = [inherited] if isinstance(inherited, int) else list(inherited or [])
    eot = target.model_config.model_description.eot_token
    if eot is None: raise ValueError('Qwen chat EOT is unavailable')
    eot_id = tokenizer.convert_tokens_to_ids(eot)
    if eot_id not in eos: eos.append(eot_id)
    capacity = model.config.get_text_config().max_position_embeddings
    if any(len(p['ids']) > capacity or len(p['prefix'])+160 > capacity for p in prepared):
        raise ValueError('Full evidence or fixed generation budget exceeds model capacity')
    result['generation_provenance'] = {'inherited_generation_config': generation.to_dict(),
        'effective_eos_token_ids': eos, 'model_description_eot_token': eot,
        'max_new_tokens': 160, 'do_sample': False, 'enable_thinking': False, 'autotuning': variables['provenance']}
    study = SimpleNamespace(dialect=ModelDialect.for_tokenizer(tokenizer))
    torch.cuda.reset_peak_memory_stats(device)
    with configure_fla_runtime(variables['fla_config']), torch.no_grad():
        order = sorted(range(100), key=lambda i: len(prepared[i]['ids']))
        for start in range(0, 100, 2):
            group = order[start:start+2]
            if all('answer_nll' in result['rows'][i]['views']['full context'] for i in group): continue
            if time.monotonic() >= deadline: break
            embeds = [model.get_input_embeddings()(torch.tensor(prepared[i]['ids'], device=device)) for i in group]
            hidden = trainer._padded_forward(target, embeds, None)
            for j, i in enumerate(group):
                p = prepared[i]; positions = rows[i]['answer_positions']
                _, logp = trainer._chunked_sft(hidden[j, p['positions']], model.get_output_embeddings(),
                    torch.tensor(p['targets'], device=device))
                values = [round(float(value), 4) for value in logp]
                result['rows'][i]['views']['full context'].update(answer_nll=-statistics.mean(values[k] for k in positions),
                    target_logp=values, gold_token_scores=[{'token_id': p['targets'][k],
                        'text': tokenizer.decode([p['targets'][k]], skip_special_tokens=False), 'nll': -values[k]} for k in positions])
            del hidden, embeds, logp
            write(output, result); progress('scoring exact100 full context', completed=start+len(group), total=100)
        if all('answer_nll' in row['views']['full context'] for row in result['rows']):
            for i, row in enumerate(result['rows']):
                view = row['views']['full context']
                if 'response' in view: continue
                if time.monotonic() >= deadline: break
                prefix = torch.tensor(prepared[i]['prefix'], device=device)
                generated = model.generate(inputs_embeds=model.get_input_embeddings()(prefix)[None],
                    max_new_tokens=160, do_sample=False, use_cache=True, eos_token_id=eos,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)[0].tolist()
                if len(generated) > 160: raise ValueError('Generate unexpectedly returned an input prefix')
                text = tokenizer.decode(generated, skip_special_tokens=True); answer, parsing = parsed_answer(study, text)
                end = text.find('</tool_call>')
                view.update(response=text, answer=answer, parsing=parsing, generated_token_ids=generated,
                    generated_tokens=len(generated), max_new_tokens=160, effective_eos_token_ids=eos,
                    ended_at_eos=bool(generated and generated[-1] in eos), reached_token_budget=len(generated)==160,
                    raw_decode_with_special_tokens=tokenizer.decode(generated, skip_special_tokens=False),
                    post_first_wrapper_text=text[end+len('</tool_call>'):] if end >= 0 else None)
                write(output, result); progress('greedy full-context answers', completed=i+1, total=100)
    complete = all('response' in row['views']['full context'] for row in result['rows'])
    scored = [r for r in result['rows'] if 'answer_nll' in r['views']['full context']]
    result['views']['full context'] = {'reader_identity': 'Bare frozen Qwen3.5-2B; no adapters',
        'subset100_answer_nll': statistics.mean(r['views']['full context']['answer_nll'] for r in scored) if scored else None,
        'subset50_by_generator': {bank: statistics.mean(r['views']['full context']['answer_nll'] for r in scored if r['generator_cohort']==bank)
            for bank in ('9b','27b') if any(r['generator_cohort']==bank for r in scored)},
        'scored_items': len(scored), 'decoded_items': sum('response' in r['views']['full context'] for r in result['rows']), 'complete': complete}
    if trainer.optimizers or trainer.updates_done: raise AssertionError('Frozen evaluation unexpectedly trained')
    result.update(status='complete' if complete else 'partial', optimizer_updates=0,
        elapsed_seconds=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30)
    write(output, result); progress(result['status'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write(args.output/(CONDITION+'_status.json'), {'phase': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
