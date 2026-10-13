"""Fixed development100 after either inherited-Adam 32k -> fresh10k stage."""
import argparse
from pathlib import Path

import evaluate_model_comparison_cohort as evaluator
from evaluate_long_cross import digest

ARMS = ('continue32k_10k_final_lr', 'continue32k_10k_rewarm')
REQUIRED_HELPERS = ('evaluate_continuation_10k_v1.py', 'evaluate_model_comparison_cohort.py',
    'evaluate_model_confirmation.py', 'evaluate_qualitative_cohort.py', 'evaluate_long_cross.py',
    'run_qa_continuation_10k_v1.py', 'continuation_cohort_csail_v1.sbatch')


def validate_design(path):
    design = evaluator.read(path)
    if (design.get('status') != 'frozen' or set(design['arms']) != set(ARMS)
            or design.get('questions') != 10000 or design.get('parent_questions') != 32000):
        raise ValueError('Expected the frozen two-arm 32k -> 10k continuation design')
    if not set(REQUIRED_HELPERS).issubset(design['helper_sha256']):
        raise ValueError('Continuation design does not pin its evaluation helpers')
    for leaf, expected in design['helper_sha256'].items():
        if Path(leaf).name != leaf or evaluator.file_sha(Path(__file__).with_name(leaf)) != expected:
            raise ValueError('Pinned continuation helper changed: '+leaf)
    return design


def load_inputs(args, design):
    if evaluator.file_sha(args.selection) != design['selection_sha256']:
        raise ValueError('Original fixed100 selection changed')
    if evaluator.file_sha(args.material) != design['material_sha256']:
        raise ValueError('Continuation material changed')
    if evaluator.file_sha(args.bench) != design['helper_sha256']['run_qa_continuation_10k_v1.py']:
        raise ValueError('Continuation input builder changed')
    material, selection = evaluator.read(args.material), evaluator.read(args.selection)
    if (selection['selection_key'] != evaluator.SELECTION_KEY or selection['quotas'] != evaluator.QUOTAS
            or selection['original_held_ids'] != design['original_held_ids']
            or selection['selected_ids'] != evaluator.select_ids(material, selection['original_held_ids'])):
        raise ValueError('Frozen development membership or order changed')
    if len(material['train']) != 10000 or len({r['example']['example_id'] for r in material['train']}) != 10000:
        raise ValueError('Expected the exact fresh10k material')
    by_id = {row['example']['example_id']: row for row in material['held']}
    material['held'] = [by_id[ident] for ident in selection['original_held_ids']]
    # Retain the immutable selection provenance; do not pretend old16k materials were read.
    selection['original_pinned_sha256'] = selection['pinned_sha256']
    selection['pinned_sha256'] = {'builder': evaluator.file_sha(args.bench),
        'continuation_material': design['material_sha256'],
        'continuation_design': evaluator.file_sha(args.design)}
    return material, selection


def final_pair(arm, reports, design, design_path, initial_pair):
    """Reject partial, selected-best, foreign and wrong-schedule stage checkpoints."""
    spec = design['arms'][arm]
    mode = arm.removeprefix('continue32k_10k_')
    if spec['mode'] != mode:
        raise ValueError('Continuation arm/mode differs')
    folder = Path(reports)/arm
    result_path = folder/'result.json'; result = evaluator.read(result_path)
    identity_path = folder/'run_identity.json'; identity = evaluator.read(identity_path)['identity']
    config_path = folder/'training_config.json'; config = evaluator.read(config_path)
    expected = {'arm': 'joint', 'epochs': 1, 'planned_updates': 1250, 'steps_per_epoch': 1250,
        'material_sha256': design['material_sha256'], 'evidence_tokens': 2048, 'max_passages': 4,
        'ratio': 1/16, 'passage_parts': False, 'initial_pair_sha256': design['initial_pair_sha256'],
        'continuation_schedule': mode, 'parent_questions': 32000, 'optimizer_policy': 'inherited'}
    if any(identity.get(k) != v for k, v in expected.items()) or config != spec['training_config']:
        raise ValueError(arm+': training identity/config differs from the frozen continuation')
    if (result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 1250
            or result.get('parent_unique_questions') != 32000
            or result.get('continuation_schedule') != mode or result.get('optimizer_policy') != 'inherited'):
        raise ValueError(arm+': own fixed1250-update continuation final is not complete')
    ids_path = folder/'material_ids.json'; ids = evaluator.read(ids_path)
    if (len(ids['train']) != 10000 or len(set(ids['train'])) != 10000
            or ids['held'] != design['original_held_ids'] or digest(ids) != identity['material_ids_sha256']):
        raise ValueError(arm+': training membership or development order changed')
    receipt_path = folder/'source_receipt.json'; receipt = evaluator.read(receipt_path)
    expected_receipt = {'builder_sha256': design['helper_sha256']['run_qa_continuation_10k_v1.py'],
        'source_sha256': design['source_sha256'], 'design_sha256': evaluator.file_sha(design_path),
        'material_sha256': design['material_sha256'], 'initial_pair_sha256': design['initial_pair_sha256'],
        'mode': mode, 'all_initial_payload_hashes_verified': True}
    if any(receipt.get(k) != v for k, v in expected_receipt.items()):
        raise ValueError(arm+': missing or changed training source/initial-state receipt')
    inherited_path = folder/'inherited_optimizer_check.json'; inherited = evaluator.read(inherited_path)
    provenance = initial_pair['provenance']; optimizer = provenance['optimizer']
    if (not provenance.get('new_stage_not_fresh_adam') or provenance['material_sha256'] != design['material_sha256']
            or not optimizer.get('adam_state_preserved_byte_exact') or not optimizer.get('param_groups_preserved_exact')
            or not optimizer.get('weight_bytes_unchanged') or not optimizer.get('independent_files_not_hardlinks')
            or optimizer.get('stage_updates_done') != 0 or optimizer.get('parent_updates_done') != 4000
            or inherited.get('optimizer_states') != optimizer['optimizer_parameter_states']
            or inherited.get('min_parameter_step') != 4000 or inherited.get('max_parameter_step') != 4000
            or inherited.get('stage_updates_done') != 0):
        raise ValueError(arm+': inherited Adam provenance differs')
    input_path = folder/'parent_input_check.json'; input_check = evaluator.read(input_path)
    for key in ('shared_input_fingerprint', 'ac_input_fingerprint'):
        if input_check.get(key) != initial_pair['parent_'+key]:
            raise ValueError(arm+': development inputs differ from the parent')
    root = (Path(design['run_root'])/arm).resolve()
    ac, reader = Path(result['checkpoint']).resolve(), Path(result['target_lora_checkpoint']).resolve()
    if ac != root/'AC_MODELS/rag_poc/epoch_001' or reader != root/'LORAS/rag_reader/epoch_001':
        raise ValueError(arm+': expected own final epoch1 pair, not selected or foreign weights')
    marker_path = ac.parent/'completed_epoch.json'; marker = evaluator.read(marker_path)
    if (marker.get('epoch_number') != 1 or (root/marker['ac_checkpoint']).resolve() != ac
            or (root/marker['target_lora_checkpoint']).resolve() != reader):
        raise ValueError(arm+': final pair disagrees with epoch publication')
    panel_path = folder/'panel_01250.json'; panel = evaluator.read(panel_path); progress = panel['progress']
    if any(progress.get(key) != value for key, value in
           {'global_step': 1250, 'step': 1250, 'steps': 1250, 'epoch': 1, 'epoch_number': 1, 'epochs': 1}.items()):
        raise ValueError(arm+': final panel has incorrect stage progress')
    if [row['item_id'] for row in panel['summaries']['held AC']['per_item']] != design['original_held_ids']:
        raise ValueError(arm+': final panel does not contain the original400 in saved order')
    files = {'ac_modules': ac/'ac_modules.pt'}
    for label, directory, base in (('encoder', ac/'side_lora', 'Qwen/Qwen3.5-2B'),
                                    ('reader', reader, 'Qwen/Qwen3.5-9B')):
        adapter = directory/'adapter_config.json'; values = evaluator.read(adapter)
        weights = [directory/leaf for leaf in ('adapter_model.safetensors', 'adapter_model.bin') if (directory/leaf).is_file()]
        if values.get('base_model_name_or_path') != base or values.get('r') != 128 or len(weights) != 1:
            raise ValueError(arm+': wrong or ambiguous final adapter payload')
        files[label+'_config'], files[label+'_weights'] = adapter, weights[0]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files.values()):
        raise ValueError(arm+': missing or empty paired weights')
    hashes = {name: evaluator.file_sha(path) for name, path in files.items()}
    return {'ac_checkpoint': str(ac), 'reader_checkpoint': str(reader),
        'parent_questions': 32000, 'stage_unique_questions': 10000, 'stage_epochs': 1,
        'continuation_schedule': mode, 'optimizer_policy': 'inherited Adam, stage-only schedule reset',
        'continuation_design_sha256': evaluator.file_sha(design_path),
        'continuation_evaluator_sha256': evaluator.file_sha(__file__),
        'result_sha256': evaluator.file_sha(result_path), 'marker_sha256': evaluator.file_sha(marker_path),
        'training_identity_sha256': evaluator.file_sha(identity_path), 'training_config_sha256': evaluator.file_sha(config_path),
        'source_receipt_sha256': evaluator.file_sha(receipt_path), 'inherited_optimizer_check_sha256': evaluator.file_sha(inherited_path),
        'parent_input_check_sha256': evaluator.file_sha(input_path), 'final_panel_sha256': evaluator.file_sha(panel_path),
        'initial_pair_sha256': design['initial_pair_sha256'], 'inherited_optimizer': optimizer}, hashes


def main(args):
    design = validate_design(args.design)
    if evaluator.file_sha(args.initial_pair) != design['initial_pair_sha256']:
        raise ValueError('Relocated parent initial-pair receipt changed')
    initial_pair = evaluator.read(args.initial_pair)
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    runtime = {key: evaluator.file_sha(module.__file__) for key, module in
               (('ac_model', ac_module), ('trainer', training_module), ('utils', utils_module))}
    if runtime != design['source_sha256']:
        raise ValueError('Evaluation source differs from frozen continuation training')
    evaluator.CONDITIONS = ARMS
    evaluator.load_inputs = lambda call_args: load_inputs(call_args, design)
    def checkpoint(call_args, selection):
        pair, hashes = final_pair(call_args.condition, call_args.reports, design, args.design, initial_pair)
        return pair, '2B', 1/16, 1250, hashes
    evaluator.checkpoint = checkpoint
    original_shared = evaluator.shared_fingerprint
    def checked_inputs(held, full, noctx):
        shared = original_shared(held, full, noctx)
        if (shared != initial_pair['parent_shared_input_fingerprint']
                or digest([(item.item_id, item.ac_messages) for item in held]) != initial_pair['parent_ac_input_fingerprint']):
            raise ValueError('Rebuilt original400 inputs changed before evaluation')
        return shared
    evaluator.shared_fingerprint = checked_inputs
    original_sample = evaluator.sample_with_token_ids
    saved_progress = evaluator.read(args.reports/args.condition/'panel_01250.json')['progress']
    def sample(trainer, ac, item, progress, eos_ids):
        from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
        return original_sample(trainer, ac, item, ActivationContextTrainingProgress(**saved_progress), eos_ids)
    evaluator.sample_with_token_ids = sample
    evaluator.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'bench', 'material', 'reports', 'selection', 'initial-pair', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', choices=ARMS, required=True)
    parser.add_argument('--seconds', type=int, default=2100)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        evaluator.write(args.output/(args.condition+'_status.json'),
            {'phase': 'failed', 'condition': args.condition, 'error': f'{type(error).__name__}: {error}'})
        raise
