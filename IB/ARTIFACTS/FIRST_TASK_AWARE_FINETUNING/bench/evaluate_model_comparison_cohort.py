"""Fixed development100: score the original400 before decoding a loss-independent50+50."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

from evaluate_qualitative_cohort import read, write, digest as file_sha, parsed_answer
from evaluate_long_cross import answer_scores, digest
from evaluate_model_confirmation import ARMS, final_pair, shared_fingerprint, split_scores

SELECTION_KEY = 'model-comparison-development100-v1:'
COMMON_QUOTAS = dict(code=15, math_arxiv=6, math_textbook=4, web=10, documents=4, legal=3, office=3)
QUOTAS = {'9b': {**COMMON_QUOTAS, 'financial': 2, 'technical_web': 3},
          '27b': {**COMMON_QUOTAS, 'financial': 3, 'technical_web': 2}}
CONDITIONS = ('frozen_p2', 'frozen_p4', *ARMS)
PIN_KEYS = ('builder', 'harness_4b', 'material_9b', 'material_27b')


def select_ids(material, saved_order):
    """Select by generator cohort/family and immutable ID hash, preserving the saved held order."""
    rows = {r['example']['example_id']: r for r in material['held']}
    cohorts = material['held_cohorts']
    members = [ident for group in cohorts.values() for ident in group]
    if (set(cohorts) != {'9b', '27b'} or any(len(v) != 200 for v in cohorts.values())
            or len(material['held']) != len(rows) or len(rows) != 400 or len(set(members)) != 400
            or set(members) != set(rows) or len(saved_order) != 400 or set(saved_order) != set(rows)):
        raise ValueError('Expected the exact400 development IDs/order and two disjoint200 cohorts')
    chosen = set()
    for bank, quotas in QUOTAS.items():
        families = defaultdict(list)
        for ident in cohorts[bank]:
            families[rows[ident]['source_family']].append(ident)
        for family, count in quotas.items():
            if len(families[family]) < count:
                raise ValueError(f'Insufficient development questions: {bank}/{family}')
            chosen.update(sorted(families[family], key=lambda ident: (hashlib.sha256((SELECTION_KEY+ident).encode()).hexdigest(), ident))[:count])
    if len(chosen) != 100:
        raise ValueError('Selection did not produce100 unique questions')
    return [ident for ident in saved_order if ident in chosen]


def load_inputs(args):
    selection = read(args.selection)
    design = read(args.design)
    pins = {key: design['sha256'][key] for key in PIN_KEYS}
    if selection['pinned_sha256'] != pins or selection['selection_key'] != SELECTION_KEY or selection['quotas'] != QUOTAS:
        raise ValueError('Frozen cohort or material design changed')
    paths = {'builder': args.bench, 'harness_4b': args.harness_4b,
             'material_9b': args.material_dir/'material_9b_16000.json',
             'material_27b': args.material_dir/'material_27b_16000.json'}
    if any(file_sha(path) != pins[name] for name, path in paths.items()):
        raise ValueError('Pinned material or builder bytes changed')
    material = read(paths['material_9b'])
    if selection['selected_ids'] != select_ids(material, selection['original_held_ids']):
        raise ValueError('Frozen100 membership changed')
    by_id = {r['example']['example_id']: r for r in material['held']}
    material['held'] = [by_id[ident] for ident in selection['original_held_ids']]
    return material, selection


def checkpoint(args, selection):
    """Only the requested final must be complete; frozen readers do not depend on training completion."""
    frozen = args.condition.startswith('frozen_')
    if frozen:
        root = args.p2_root if args.condition == 'frozen_p2' else args.p4_root
        encoder = '2B' if args.condition == 'frozen_p2' else '4B'
        pair = {'ac_checkpoint': str(root/'AC_MODELS/rag_poc/epoch_001'),
                'reader_checkpoint': str(root/'LORAS/rag_reader/epoch_001')}
        ratio, update = 1/16, 0
    else:
        bank, ratio, encoder = ARMS[args.condition]
        pair = final_pair(args.condition, args.reports,
                          {name: selection['pinned_sha256']['material_'+name] for name in ('9b', '27b')})
        if read(args.reports/args.condition/'material_ids.json')['held'] != selection['original_held_ids']:
            raise ValueError('Requested final used a different development order')
        update = 2000
    ac, reader = Path(pair['ac_checkpoint']), Path(pair['reader_checkpoint'])
    files = {'ac_modules': ac/'ac_modules.pt'}
    for label, directory, base in (('encoder', ac/'side_lora', 'Qwen/Qwen3.5-'+encoder),
                                   ('reader', reader, 'Qwen/Qwen3.5-9B')):
        config = directory/'adapter_config.json'
        values = read(config)
        if values.get('base_model_name_or_path') != base or values.get('r') != 128:
            raise ValueError('Wrong encoder/reader base or adapter rank: '+str(directory))
        weights = [directory/name for name in ('adapter_model.safetensors', 'adapter_model.bin') if (directory/name).is_file()]
        if len(weights) != 1:
            raise ValueError('Expected one unambiguous adapter weight payload: '+str(directory))
        files[label+'_config'], files[label+'_weights'] = config, weights[0]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files.values()):
        raise ValueError('Missing or empty paired checkpoint')
    # Deliberately done on the allocated GPU node, never the login node.
    hashes = {name: file_sha(path) for name, path in files.items()}
    return pair, encoder, ratio, update, hashes


def optional_panel(args, update):
    arm = {'frozen_p2': 'oracle_27b_16k_r16', 'frozen_p4': 'encoder4b_oracle27b_16k_r16'}.get(args.condition, args.condition)
    path = args.reports/arm/f'panel_{update:05d}.json'
    if not path.exists():
        if update:
            raise FileNotFoundError('Completed final is missing its original panel: '+str(path))
        return None, None
    panel = read(path)
    if panel['progress']['global_step'] != update:
        raise ValueError('Original panel update differs')
    return panel, file_sha(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_with_token_ids(trainer, ac, item, progress, eos_ids):
    """Keep the frozen trainer while explicitly stopping chat turns and retaining actual output IDs."""
    peft = trainer.harness.module_manager.ensure_lora(ac.config.target_model_lora_name)
    original_generate = peft.generate
    sentinel = object()
    previous = peft.__dict__.get('generate', sentinel)
    captured = {}
    def generate(*args, **kwargs):
        if captured:
            raise RuntimeError('Expected one generation call per development sample')
        kwargs['eos_token_id'] = list(eos_ids)
        output = original_generate(*args, **kwargs)
        ids = output[0].detach().cpu().tolist()
        budget = kwargs['max_new_tokens']
        if len(ids) > budget:
            raise ValueError('Returned IDs unexpectedly include an input prefix')
        captured.update(generated_token_ids=ids, generated_tokens=len(ids), max_new_tokens=budget,
            effective_eos_token_ids=list(eos_ids), ended_at_eos=bool(ids and ids[-1] in eos_ids),
            reached_token_budget=len(ids) == budget,
            raw_decode_with_special_tokens=ac.target.tokenizer.decode(ids, skip_special_tokens=False))
        return output
    try:
        peft.generate = generate
        sample = trainer._sample_completions(ac, [item], progress)[0]
    finally:
        if previous is sentinel:
            del peft.generate
        else:
            peft.generate = previous
    if not captured or ac.target.tokenizer.decode(captured['generated_token_ids'], skip_special_tokens=True) != sample['student']:
        raise ValueError('Generated IDs do not match the recorded trainer response')
    return sample, captured


def main(args):
    started = time.monotonic()
    output = args.output/(args.condition+'.json')
    status = args.output/(args.condition+'_status.json')
    def progress(phase, **extra):
        record = {'condition': args.condition, 'phase': phase, 'elapsed_seconds': round(time.monotonic()-started, 1), **extra}
        write(status, record); print(json.dumps(record), flush=True)
    progress('checking frozen development selection and requested checkpoint')
    material, selection = load_inputs(args)
    pair, encoder, ratio, update, hashes = checkpoint(args, selection)
    panel, panel_sha = optional_panel(args, update)
    reader_identity = ('Original P2 epoch1 reader' if encoder == '2B' else 'Original P4 epoch1 reader') if not update else 'Final paired reader: '+args.condition
    identity = {'condition': args.condition, 'update': update, 'encoder': encoder, 'ratio': ratio,
                'reader_identity': reader_identity, 'paired_checkpoint': pair, 'checkpoint_sha256': hashes,
                'selection_sha256': file_sha(args.selection), 'pinned_sha256': selection['pinned_sha256'],
                'evaluator_sha256': file_sha(__file__),
                'helper_sha256': {name: file_sha(Path(__file__).with_name(name)) for name in
                    ('evaluate_qualitative_cohort.py', 'evaluate_long_cross.py', 'evaluate_model_confirmation.py')},
                'microbatch': 2, 'max_new_tokens': 160, 'enable_thinking': False,
                'decoding_policy': 'inherited EOS plus existing model-description chat EOT; actual returned token IDs retained',
                'scope': 'Fixed100 from common400 DEVELOPMENT questions. Confirmation is not read or evaluated.',
                'nll_batching': 'Score all400 in original saved held order with batch2, then select100; original sorted-length batch partners preserved.',
                'baseline_note': 'P4 and P2 are different pretrained AC/reader pairs; this is not an isolated encoder-size comparison.'}
    if output.exists():
        result = read(output)
        if result['identity'] != identity:
            raise ValueError('Existing development evaluation identity changed')
        if result['status'] == 'complete':
            progress('already complete', selected=100); return
    else:
        result = {'status': 'running', 'identity': identity, 'selection': selection, 'views': {}, 'rows': []}
    bench = load_module('development_builder', args.bench)
    harness_factory = bench.harness if encoder == '2B' else load_module('development_harness4b', args.harness_4b).harness
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress, _TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model_training as trainer_module
    runtime_hashes = {'trainer': file_sha(trainer_module.__file__)}
    if runtime_hashes['trainer'] != selection['trainer_sha256']:
        raise ValueError('Pinned training/evaluation implementation changed')
    progress('loading paired checkpoint', reader=reader_identity)
    harness = harness_factory(pair['ac_checkpoint'], pair['reader_checkpoint'])
    ac = harness.module_manager.get_ac_model(bench.AC); ac.prepare()
    peft = harness.module_manager.ensure_lora(bench.READER)
    generation = getattr(peft, 'generation_config', None)
    inherited_eos = generation.eos_token_id
    eos_ids = [inherited_eos] if isinstance(inherited_eos, int) else list(inherited_eos or [])
    eot = ac.target.model_config.model_description.eot_token
    if eot is None:
        raise ValueError('Expected a chat end-of-turn token for these Qwen3.5 readers')
    eot_id = ac.target.tokenizer.convert_tokens_to_ids(eot)
    if eot_id not in eos_ids:
        eos_ids.append(eot_id)
    result['generation_provenance'] = {
        'trainer_sha256': runtime_hashes['trainer'], 'explicit_request': {'max_new_tokens_cap': 160, 'do_sample': False, 'use_cache': True,
            'eos_token_id': eos_ids,
            'pad_token_id': ac.target.tokenizer.pad_token_id if ac.target.tokenizer.pad_token_id is not None else ac.target.tokenizer.eos_token_id},
        'inherited_generation_config': {key: getattr(generation, key, None) for key in
            ('eos_token_id', 'bos_token_id', 'pad_token_id', 'forced_eos_token_id', 'stop_strings')},
        'tokenizer_eos_token_id': ac.target.tokenizer.eos_token_id,
        'model_description_eot_token': eot, 'model_description_eot_token_id': eot_id,
        'note': 'A private generation wrapper adds chat EOT without mutating generation_config. NLL uses the unchanged pinned trainer. Complete returned IDs, special-token decode, EOS termination and budget status are retained per answer.'}
    progress('building original400 development inputs')
    study, _, held = bench.material_items(harness, {'train': material['train'][:1], 'held': material['held'], 'retrieval': material['retrieval']},
                                         evidence_tokens=2048, max_passages=4, ratio=ratio)
    if [item.item_id for item in held] != selection['original_held_ids']:
        raise ValueError('Builder changed the saved development order')
    full = study.transform_for_eval_reference(held, 'full_context')
    noctx = study.transform_for_eval_reference(held, 'no_context')
    common = shared_fingerprint(held, full, noctx)
    ac_hash = digest([(item.item_id, item.ac_messages) for item in held])
    if result.get('shared_input_fingerprint', common) != common or result.get('ac_input_fingerprint', ac_hash) != ac_hash:
        raise ValueError('Rebuilt development inputs changed')
    for condition in CONDITIONS:
        path = args.output/(condition+'.json')
        if path.exists():
            other = read(path)
            if other.get('shared_input_fingerprint', common) != common or other['identity']['selection_sha256'] != identity['selection_sha256']:
                raise ValueError('Development conditions have different common evidence/selection')
    result.update(shared_input_fingerprint=common, ac_input_fingerprint=ac_hash)
    selected = set(selection['selected_ids'])
    indices = [i for i, item in enumerate(held) if item.item_id in selected]
    cohort_of = {ident: bank for bank, ids in material['held_cohorts'].items() for ident in ids}
    if not result['rows']:
        for i in indices:
            item, row = held[i], material['held'][i]
            result['rows'].append({'item_id': item.item_id, 'original_400_index': i, 'generator_cohort': cohort_of[item.item_id],
                'domain': row['domain'], 'source_family': row['source_family'], 'question': item.info['question'], 'gold': item.info['answer'],
                'required_source': row['example']['study_source'], 'full_evidence': item.info['passages_text'],
                'in_context_messages': item.in_context_messages, 'answer_positions': item.info['answer_positions'],
                'assistant_token_ids': item.assistant_token_ids,
                'row_budget': {key: item.info[key] for key in ('encoded_source_tokens','compressed_rows','evidence_source_tokens','passages')},
                'views': {}})
    if [row['item_id'] for row in result['rows']] != selection['selected_ids']:
        raise ValueError('Saved100 output order changed')
    views = {'full context': full, 'no context': noctx} if not update else {'AC': held}
    config = ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0, micro_batch_examples=2,
        max_example_tokens=72000, completion_samples=0, completion_max_new_tokens=160, seed=bench.SEED)
    trainer = ActivationContextTrainer(harness, config); variables = trainer._autotuning_variables(ac)
    trainer._training_deadline = time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    sample_progress = ActivationContextTrainingProgress(epoch=1 if update else 0, epochs=1 if update else 0,
        epoch_number=1 if update else 0, step=update, steps=update,
        global_step=update, epoch_elapsed_s=0., elapsed_s=0.)
    write(output, result)
    try:
        with configure_fla_runtime(variables['fla_config']), torch.no_grad():
            for name, items in views.items():
                if name not in result['views']:
                    progress('scoring original400 before selecting100', view=name, total=400)
                    torch.cuda.synchronize(ac.device); torch.cuda.reset_peak_memory_stats(ac.device); began = time.monotonic()
                    summary = trainer._evaluate(ac, items, ac.device)
                    scored = answer_scores(summary, items, held)
                    torch.cuda.synchronize(ac.device)
                    old = panel['summaries']['held '+name] if panel else None
                    old_scored = answer_scores(old, items, held) if old else None
                    old_by = {r['item_id']: r for r in old_scored['per_item']} if old_scored else {}
                    for record, i in zip(result['rows'], indices):
                        item = held[i]; positions = item.info['answer_positions']; measured = summary['per_item'][i]
                        record['views'][name] = {'answer_nll': scored['per_item'][i]['answer_nll'],
                            'original_panel_answer_nll': old_by.get(item.item_id, {}).get('answer_nll'),
                            'gold_token_scores': [{'token_id': item.assistant_token_ids[0][p],
                                'text': study.tokenizer.decode([item.assistant_token_ids[0][p]], skip_special_tokens=False),
                                'nll': -measured['target_logp'][p]} for p in positions]}
                    delta = [r['answer_nll']-old_by[r['item_id']]['answer_nll'] for r in scored['per_item']] if old_by else []
                    result['views'][name] = {'reader_identity': reader_identity, 'full400': scored,
                        'cohorts200': split_scores(scored, material['held_cohorts']),
                        'subset100_answer_nll': statistics.mean(row['views'][name]['answer_nll'] for row in result['rows']),
                        'subset50_by_generator': {bank: statistics.mean(row['views'][name]['answer_nll'] for row in result['rows'] if row['generator_cohort'] == bank) for bank in QUOTAS},
                        'original_panel_sha256': panel_sha, 'original_panel_max_abs_nll_difference': max(map(abs,delta)) if delta else None,
                        'original_panel_mean_nll_difference': statistics.mean(delta) if delta else None,
                        'seconds': time.monotonic()-began, 'peak_allocated_gib': torch.cuda.max_memory_allocated(ac.device)/2**30,
                        'complete': False}
                    write(output, result)
                for j, i in enumerate(indices):
                    record = result['rows'][j]
                    if 'response' in record['views'][name]:
                        continue
                    progress('free decoding fixed100', view=name, completed=j, total=100)
                    sample, generated = sample_with_token_ids(trainer, ac, items[i], sample_progress, eos_ids)
                    text = sample['student']; answer, parsing = parsed_answer(study, text)
                    end = text.find('</tool_call>')
                    record['views'][name].update(response=text, answer=answer, parsing=parsing, raw_sample=sample, **generated,
                        post_first_wrapper_text=text[end+len('</tool_call>'):] if end >= 0 else None,
                        retokenized_visible_tokens=len(study.tokenizer.encode(text, add_special_tokens=False)))
                    write(output, result)
                result['views'][name]['complete'] = True; write(output, result)
    except _TrainingTimeLimit:
        result['status'] = 'partial'; write(output, result)
        progress('time limit; resumable with the same invocation'); return
    if trainer.optimizers or trainer.updates_done:
        raise AssertionError('Development evaluation unexpectedly trained')
    result.update(status='complete', optimizer_updates=0, elapsed_seconds=round(time.monotonic()-started,1))
    write(output,result); progress('complete', selected=100, views=list(views))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'bench', 'harness-4b', 'material-dir', 'reports', 'selection', 'p2-root', 'p4-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--seconds', type=int, default=2100)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        write(args.output/(args.condition+'_status.json'), {'phase': 'failed', 'condition': args.condition, 'error': f'{type(error).__name__}: {error}'})
        raise
