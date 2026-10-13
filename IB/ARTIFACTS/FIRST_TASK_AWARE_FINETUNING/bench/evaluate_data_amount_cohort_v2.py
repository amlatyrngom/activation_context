"""Fixed development100 readout for 32k x 1 and 16k x 2, using their actual final pairs."""
import argparse
from functools import partial
from pathlib import Path

import evaluate_model_comparison_cohort as evaluator
from evaluate_long_cross import digest

ARMS = ('oracle_27b_32k', 'oracle_27b_16k_2ep')
HELPERS = ('evaluate_data_amount_cohort_v2.py', 'evaluate_data_amount_confirmation_v2.py',
           'evaluate_model_comparison_cohort.py', 'evaluate_model_confirmation.py',
           'evaluate_qualitative_cohort.py', 'evaluate_long_cross.py',
           'run_qa_confirmation_v3.py', 'run_qa_confirmation_v4.py', 'run_qa_encoder4b_v1.py')
TRAINING_FILES = ('run_qa_comparison_v2.py', 'run_qa_comparison_v3.py',
                  'model_comparison_32k.sbatch', 'model_comparison_32k_v2.sbatch',
                  'model_comparison_16k_2ep.sbatch')


def validate_design(path):
    design = evaluator.read(path)
    if design.get('status') != 'frozen' or set(design['arms']) != set(ARMS):
        raise ValueError('Expected the frozen two-arm data-amount design')
    if set(design['helper_sha256']) != set(HELPERS) or set(design['training_sha256']) != set(TRAINING_FILES):
        raise ValueError('Frozen design does not pin every evaluation and training entrypoint')
    for name, expected in {**design['helper_sha256'], **design['training_sha256']}.items():
        if Path(name).name != name or evaluator.file_sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('Pinned data-amount helper changed: '+name)
    return design


def final_pair(arm, reports, material_hashes=None, *, design, design_path):
    """Validate the declared full run and its own published final AC/reader pair."""
    spec = design['arms'][arm]
    questions, epochs = (32000, 1) if arm == ARMS[0] else (16000, 2)
    if spec['questions'] != questions or spec['epochs'] != epochs:
        raise ValueError('Declared data-amount arm changed')
    folder = Path(reports)/arm
    result_path = folder/'result.json'; result = evaluator.read(result_path)
    identity_path = folder/'run_identity.json'; identity = evaluator.read(identity_path)['identity']
    config_path = folder/'training_config.json'; config = evaluator.read(config_path)
    expected_identity = {'arm': 'joint', 'epochs': epochs, 'planned_updates': 4000,
        'steps_per_epoch': questions//8, 'material_sha256': spec['material_sha256'],
        'evidence_tokens': 2048, 'max_passages': 4, 'ratio': spec['ratio'],
        'passage_parts': False, 'initial_pair_sha256': None}
    if any(identity.get(k) != v for k, v in expected_identity.items()) or config != spec['training_config']:
        raise ValueError(arm+': training identity/config differs from the frozen design')
    if result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 4000:
        raise ValueError(arm+': full 32,000 question presentations are not complete')
    ids_path = folder/'material_ids.json'; ids = evaluator.read(ids_path)
    if (len(ids['train']) != questions or len(set(ids['train'])) != questions
            or ids['held'] != design['original_held_ids'] or digest(ids) != identity.get('material_ids_sha256')):
        raise ValueError(arm+': training IDs or original held order changed')
    root = (Path(design['run_root'])/arm).resolve()
    expected_ac = root/'AC_MODELS/rag_poc'/f'epoch_{epochs:03d}'
    expected_reader = root/'LORAS/rag_reader'/f'epoch_{epochs:03d}'
    ac = Path(result['checkpoint']).resolve(); reader = Path(result['target_lora_checkpoint']).resolve()
    if ac != expected_ac.resolve() or reader != expected_reader.resolve():
        raise ValueError(arm+': final checkpoint belongs to another run or epoch')
    marker_path = ac.parent/'completed_epoch.json'; marker = evaluator.read(marker_path)
    if (marker.get('epoch_number') != epochs or (root/marker['ac_checkpoint']).resolve() != ac
            or (root/marker['target_lora_checkpoint']).resolve() != reader):
        raise ValueError(arm+': final pair disagrees with its epoch publication')
    panel_path = folder/'panel_04000.json'; progress = evaluator.read(panel_path)['progress']
    if (progress.get('global_step') != 4000 or progress.get('epoch_number') != epochs
            or progress.get('epochs') != epochs or progress.get('step') != questions//8):
        raise ValueError(arm+': final panel is missing or has incorrect epoch progress')
    required = {'ac_modules': ac/'ac_modules.pt'}
    for label, directory, base in (('encoder', ac/'side_lora', 'Qwen/Qwen3.5-2B'),
                                    ('reader', reader, 'Qwen/Qwen3.5-9B')):
        adapter = directory/'adapter_config.json'; values = evaluator.read(adapter)
        if values.get('base_model_name_or_path') != base or values.get('r') != 128:
            raise ValueError(arm+': wrong encoder/reader base or adapter rank')
        weights = [directory/name for name in ('adapter_model.safetensors', 'adapter_model.bin') if (directory/name).is_file()]
        if len(weights) != 1:
            raise ValueError(arm+': expected one unambiguous adapter payload')
        required[label+'_config'], required[label+'_weights'] = adapter, weights[0]
    if any(not p.is_file() or p.stat().st_size == 0 for p in required.values()):
        raise ValueError(arm+': missing or empty final-pair files')
    return {'ac_checkpoint': str(ac), 'reader_checkpoint': str(reader),
        'training_unique_questions': questions, 'training_epochs': epochs,
        'result_sha256': evaluator.file_sha(result_path), 'marker_sha256': evaluator.file_sha(marker_path),
        'training_identity_sha256': evaluator.file_sha(identity_path),
        'training_config_sha256': evaluator.file_sha(config_path), 'material_ids_sha256': evaluator.file_sha(ids_path),
        'final_panel_sha256': evaluator.file_sha(panel_path),
        'data_amount_design_sha256': evaluator.file_sha(design_path),
        'data_amount_entrypoint_sha256': evaluator.file_sha(__file__),
        'files': {name: {'path': str(p), 'bytes': p.stat().st_size, 'sha256': evaluator.file_sha(p)} for name, p in required.items()}}


def main(args):
    design = validate_design(args.evaluation_design)
    if (evaluator.file_sha(args.design) != design['original_design_sha256']
            or evaluator.file_sha(args.selection) != design['selection_sha256']):
        raise ValueError('Original development design or selected100 changed')
    evaluator.ARMS = {arm: ('27b', design['arms'][arm]['ratio'], '2B') for arm in ARMS}
    evaluator.CONDITIONS = ARMS
    evaluator.final_pair = partial(final_pair, design=design, design_path=args.evaluation_design)
    original_checkpoint = evaluator.checkpoint
    def checkpoint(call_args, selection):
        pair, encoder, ratio, _, hashes = original_checkpoint(call_args, selection)
        return pair, encoder, ratio, 4000, hashes
    evaluator.checkpoint = checkpoint
    original_sample = evaluator.sample_with_token_ids
    saved_progress = evaluator.read(args.reports/args.condition/'panel_04000.json')['progress']
    def sample(trainer, ac, item, progress, eos_ids):
        from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
        return original_sample(trainer, ac, item, ActivationContextTrainingProgress(**saved_progress), eos_ids)
    evaluator.sample_with_token_ids = sample
    evaluator.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evaluation-design', 'design', 'bench', 'harness-4b', 'material-dir', 'reports', 'selection', 'p2-root', 'p4-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', choices=ARMS, required=True)
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        evaluator.write(args.output/(args.condition+'_status.json'),
            {'phase': 'failed', 'condition': args.condition, 'error': f'{type(error).__name__}: {error}'})
        raise
