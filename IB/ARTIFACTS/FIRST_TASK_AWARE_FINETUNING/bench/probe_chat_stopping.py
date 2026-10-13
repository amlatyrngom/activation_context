"""Two actual HF decodes from one frozen QA pair; no scoring or training."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from evaluate_qualitative_cohort import read, write, digest

ITEM = '770164ac47e4be0e8307301a122aa3b8eb6295dc0935f551a819ae5c13eae3df'
TRAINER_SHA = '38ca9885bfe35e47432352aa716381e53c18aa57748428eedf5f5f6b7fc6b5aa'
AC_SHA = 'aaeb3c68225d3092a024d44552b84480dc37cc8319d68c2ab1701fce70980bfd'


def describe_tokens(ids, eos_ids, eot, budget):
    eos = [i for i, token in enumerate(ids) if token in eos_ids]
    ends = [i for i, token in enumerate(ids) if token == eot]
    return {'generated_ids': ids, 'generated_token_count': len(ids),
            'effective_eos_ids': eos_ids, 'eos_positions_zero_based': eos,
            'eot_positions_zero_based': ends,
            'tokens_after_first_eot': len(ids)-ends[0]-1 if ends else None,
            'terminal_eos': bool(ids and ids[-1] in eos_ids),
            'reached_token_budget': len(ids) >= budget}


def main(args):
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError('Preserve existing probe output; choose another output path')
    status = args.output.with_name(args.output.stem+'_status.json')
    def progress(phase, **extra):
        row = {'phase': phase, 'elapsed_seconds': round(time.monotonic()-started, 2), **extra}
        write(status, row); print(json.dumps(row), flush=True)
    progress('validating midpoint pair and frozen400 inputs')
    design = read(args.design)
    if digest(args.bench) != design['sha256']['builder'] or digest(args.material) != design['sha256']['material_27b']:
        raise ValueError('Material or shared input builder changed')
    identity = read(args.run_dir/'run_identity.json')['identity']
    if any(identity[key] != value for key, value in {
        'material_sha256': design['sha256']['material_27b'], 'ratio': 1/16,
        'evidence_tokens': 2048, 'max_passages': 4, 'passage_parts': False}.items()):
        raise ValueError('Original run identity differs from the requested probe')
    order = read(args.run_dir/'material_ids.json')['held']
    samples = read(args.run_dir/'answers_01000.json')
    old = next(row for row in samples if row['item_id'] == ITEM)
    if old['global_step'] != 1000 or len(order) != 400 or len(set(order)) != 400:
        raise ValueError('Expected midpoint sample and common400 development order')
    pair = args.checkpoint_root/'selected/update_01000'
    files = {'ac_modules': pair/'ac/ac_modules.pt'}
    for label, directory, base in [('encoder', pair/'ac/side_lora', 'Qwen/Qwen3.5-2B'),
                                    ('reader', pair/'reader', 'Qwen/Qwen3.5-9B')]:
        cfg = directory/'adapter_config.json'; values = read(cfg)
        if values.get('base_model_name_or_path') != base or values.get('r') != 128:
            raise ValueError('Wrong adapter identity: '+str(directory))
        weights = [directory/name for name in ('adapter_model.safetensors','adapter_model.bin') if (directory/name).is_file()]
        if len(weights) != 1:
            raise ValueError('Expected exactly one adapter payload: '+str(directory))
        files[label+'_config'], files[label+'_weights'] = cfg, weights[0]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files.values()):
        raise ValueError('Missing or empty midpoint pair')
    hashes = {name: digest(path) for name, path in files.items()}
    import torch
    import transformers
    import peft as peft_package
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as trainer_module
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
    from activation.autotuners import configure_fla_runtime
    if digest(trainer_module.__file__) != TRAINER_SHA or digest(ac_module.__file__) != AC_SHA:
        raise ValueError('Probe must use the original train_ws_night_v1 source')
    saved = torch.load(files['ac_modules'], map_location='cpu', weights_only=False)
    saved_version = saved['version']; del saved
    if saved_version != old['ac_version']:
        raise ValueError('Saved AC version differs from the original midpoint sample')
    material = read(args.material)
    by_id = {row['example']['example_id']: row for row in material['held']}
    if len(by_id) != 400 or set(by_id) != set(order):
        raise ValueError('Common400 membership changed')
    material['held'] = [by_id[ident] for ident in order]
    material['train'] = material['train'][:1]
    spec = importlib.util.spec_from_file_location('chat_stop_builder', args.bench)
    bench = importlib.util.module_from_spec(spec); spec.loader.exec_module(bench)
    progress('loading frozen control1000 pair')
    harness = bench.harness(pair/'ac', pair/'reader')
    ac = harness.module_manager.get_ac_model(bench.AC); ac.prepare()
    peft = harness.module_manager.ensure_lora(bench.READER)
    progress('building exact common400 development inputs')
    _, _, held = bench.material_items(harness, material, evidence_tokens=2048, max_passages=4, ratio=1/16)
    if [item.item_id for item in held] != order:
        raise ValueError('Builder changed common400 order')
    item = held[order.index(ITEM)]
    config = ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0, micro_batch_examples=2,
        max_example_tokens=72000, completion_samples=0, completion_max_new_tokens=160, seed=bench.SEED)
    trainer = ActivationContextTrainer(harness, config)
    trainer._training_deadline = time.monotonic()+args.seconds
    variables = trainer._autotuning_variables(ac)
    example = trainer.build_example(ac, item)
    prefix = example.student_ids[:example.student_positions[0]+1]
    tokenizer = ac.target.tokenizer
    eot_string = ac.target.model_config.model_description.eot_token
    if not eot_string or tokenizer.convert_tokens_to_ids(eot_string) is None:
        raise ValueError('No existing model-description end-of-turn token')
    eot = tokenizer.convert_tokens_to_ids(eot_string)
    inherited = peft.generation_config.eos_token_id
    inherited = list(inherited) if isinstance(inherited, (list, tuple)) else ([] if inherited is None else [inherited])
    effective = list(dict.fromkeys([*inherited, eot]))
    result = {'status': 'running', 'item_id': ITEM, 'global_step': 1000,
        'provenance': {'script_sha256': digest(__file__), 'trainer_sha256': TRAINER_SHA, 'ac_model_sha256': AC_SHA,
            'builder_sha256': digest(args.bench), 'material_sha256': digest(args.material),
            'checkpoint_root': str(pair), 'checkpoint_sha256': hashes, 'saved_ac_version': saved_version,
            'runtime_ac_version': ac.version, 'transformers': transformers.__version__, 'peft': peft_package.__version__,
            'original_sample_file_sha256': digest(args.run_dir/'answers_01000.json'),
            'source_root': os.environ.get('QA_SOURCE_ROOT'), 'model_id': ac.target.model_config.model_id,
            'model_revision': getattr(ac.target.model.config, '_commit_hash', None),
            'model_config_sha256': hashlib.sha256(ac.target.model.config.to_json_string().encode()).hexdigest(),
            'tokenizer_name_or_path': tokenizer.name_or_path},
        'inherited_generation_config': json.loads(peft.generation_config.to_json_string()),
        'tokenizer_eos_id': tokenizer.eos_token_id, 'tokenizer_pad_id': tokenizer.pad_token_id,
        'existing_eot_token': eot_string, 'existing_eot_id': eot, 'proposed_eos_ids': effective,
        'input': {'original400_index': order.index(ITEM), 'original400_ids': order,
            'question': item.info['question'], 'gold': item.info['answer'], 'full_evidence': item.info['passages_text'],
            'ac_messages': item.ac_messages, 'prefix_ids': prefix, 'prefix_token_count': len(prefix),
            'row_budget': {key: item.info[key] for key in ('encoded_source_tokens','compressed_rows','evidence_source_tokens','passages')}},
        'original_stored_sample': old, 'decodes': {}, 'interpretation': 'No NLL or correctness scores computed.'}
    write(args.output, result)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    sample_progress = ActivationContextTrainingProgress(epoch=1, epochs=1, epoch_number=1,
        step=1000, steps=2000, global_step=1000, epoch_elapsed_s=0., elapsed_s=0.)
    original_generate = peft.generate
    sentinel = object(); previous_instance_generate = peft.__dict__.get('generate', sentinel)
    try:
        with configure_fla_runtime(variables['fla_config']), torch.no_grad():
            for label, eos_override in [('inherited', None), ('inherited_plus_eot', effective)]:
                capture = {}
                def captured_generate(*positional, **kwargs):
                    if capture:
                        raise RuntimeError('Expected exactly one generation call per sample')
                    if eos_override is not None:
                        kwargs['eos_token_id'] = eos_override
                    embeds = kwargs['inputs_embeds']
                    capture['input_embeds_sha256'] = hashlib.sha256(embeds.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                    capture['input_embeds_shape'] = list(embeds.shape)
                    capture['request'] = {key: kwargs.get(key) for key in ('max_new_tokens','do_sample','use_cache','pad_token_id','eos_token_id')}
                    output = original_generate(*positional, **kwargs)
                    sequences = output.sequences if hasattr(output, 'sequences') else output
                    capture['returned_ids'] = sequences[0].detach().cpu().tolist()
                    return output
                peft.generate = captured_generate
                progress('decoding', condition=label)
                torch.manual_seed(bench.SEED)
                torch.cuda.synchronize(ac.device); torch.cuda.reset_peak_memory_stats(ac.device); began = time.monotonic()
                sample = trainer._sample_completions(ac, [item], sample_progress)[0]
                torch.cuda.synchronize(ac.device)
                ids = capture.pop('returned_ids')
                budget = capture['request']['max_new_tokens']
                if len(ids) > budget:
                    raise ValueError('Returned IDs unexpectedly contain an input prefix; preserve raw capture before interpreting')
                record = describe_tokens(ids, inherited if eos_override is None else effective, eot, budget)
                record.update(capture, elapsed_seconds=time.monotonic()-began,
                    peak_memory_gib=torch.cuda.max_memory_allocated(ac.device)/2**30,
                    raw_decode_with_special_tokens=tokenizer.decode(ids, skip_special_tokens=False),
                    trainer_sample=sample,
                    decode_with_special_tokens_skipped=tokenizer.decode(ids, skip_special_tokens=True))
                if record['decode_with_special_tokens_skipped'] != sample['student']:
                    raise ValueError('Captured returned IDs do not decode to the trainer sample')
                result['decodes'][label] = record; write(args.output, result)
    finally:
        if previous_instance_generate is sentinel:
            del peft.generate
        else:
            peft.generate = previous_instance_generate
    a, b = result['decodes']['inherited'], result['decodes']['inherited_plus_eot']
    ids_a, ids_b = a['generated_ids'], b['generated_ids']
    first_end = a['eot_positions_zero_based'][0] if a['eot_positions_zero_based'] else None
    common = 0
    for x, y in zip(ids_a, ids_b):
        if x != y: break
        common += 1
    result['comparison'] = {'identical_prompt_embeddings': a['input_embeds_sha256'] == b['input_embeds_sha256'],
        'common_generated_prefix_tokens': common,
        'inherited_matches_original_stored_text': a['trainer_sample']['student'] == old['student'],
        'inherited_continued_after_eot': first_end is not None and first_end+1 < len(ids_a),
        'corrected_exactly_ends_original_at_first_eot': first_end is not None and ids_b == ids_a[:first_end+1],
        'inherited_no_eot': first_end is None,
        'generation_config_unchanged': json.loads(peft.generation_config.to_json_string()) == result['inherited_generation_config']}
    result['status'] = 'complete'; result['elapsed_seconds'] = time.monotonic()-started
    write(args.output, result); progress('complete', comparison=result['comparison'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('design','bench','material','run-dir','checkpoint-root','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seconds', type=int, default=600)
    main(p.parse_args())
