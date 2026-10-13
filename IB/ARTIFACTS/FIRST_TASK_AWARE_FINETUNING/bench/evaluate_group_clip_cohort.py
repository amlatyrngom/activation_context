"""Read the fixed development100 from the completed separate-group-clipping arm."""
import argparse
from pathlib import Path

import evaluate_model_comparison_cohort as evaluator

ARM = 'oracle_27b_16k_r8_group_clip'
CONTROL = 'oracle_27b_16k_r8'
BASE_SHA = '38ca9885bfe35e47432352aa716381e53c18aa57748428eedf5f5f6b7fc6b5aa'
VARIANT_SHA = '8ae0a9fb4c2a514c84e9baee3f3619d5c06961800a7268f36ada85246f2af1de'
original_final_pair = evaluator.final_pair


def final_pair(arm, reports, material_hashes):
    if arm != ARM:
        return original_final_pair(arm, reports, material_hashes)
    original_final_pair(CONTROL, reports, material_hashes)
    read = evaluator.read
    folder, control = Path(reports)/arm, Path(reports)/CONTROL
    for name in ('training_config.json', 'material_ids.json'):
        if read(folder/name) != read(control/name):
            raise ValueError('Clipping experiment changed '+name)
    if read(folder/'run_identity.json')['identity'] != read(control/'run_identity.json')['identity']:
        raise ValueError('Clipping experiment changed data, order, initialization or budget')
    variant = read(folder/'experiment_variant.json')
    if (variant.get('source_sha256') != BASE_SHA
            or variant.get('variant_sha256') != VARIANT_SHA
            or variant.get('run') != ARM
            or variant.get('status') != 'training source selected'):
        raise ValueError('Missing expected training-source variant receipt')
    source = Path(variant['variant_tree'])/'activation/ac_model/ac_model_training.py'
    if evaluator.file_sha(source) != VARIANT_SHA:
        raise ValueError('Separate-clipping training source changed')
    result_path = folder/'result.json'
    result = read(result_path)
    if result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 2000:
        raise ValueError('Clipping experiment is not a completed 16k × 1 final')
    run_root = Path(reports).resolve().parent/ARM
    ac = Path(result['checkpoint']).resolve()
    reader = Path(result['target_lora_checkpoint']).resolve()
    if (ac != run_root/'AC_MODELS/rag_poc/epoch_001'
            or reader != run_root/'LORAS/rag_reader/epoch_001'):
        raise ValueError('Clipping readout must use its own complete final pair')
    marker_path = ac.parent/'completed_epoch.json'
    marker = read(marker_path)
    if (marker.get('epoch_number') != 1
            or (run_root/marker['ac_checkpoint']).resolve() != ac
            or (run_root/marker['target_lora_checkpoint']).resolve() != reader):
        raise ValueError('Clipping final disagrees with paired epoch publication')
    panel = read(folder/'panel_02000.json')['progress']
    if (panel.get('global_step') != 2000 or panel.get('step') != 2000
            or panel.get('epoch') != 1 or panel.get('epochs') != 1):
        raise ValueError('Clipping final panel is incomplete')
    # The original checkpoint() validates both adapter bases/ranks and hashes all payloads.
    return {'ac_checkpoint': str(ac), 'reader_checkpoint': str(reader),
            'result_sha256': evaluator.file_sha(result_path),
            'marker_sha256': evaluator.file_sha(marker_path),
            'auxiliary_entrypoint_sha256': evaluator.file_sha(__file__),
            'training_variant_sha256': evaluator.file_sha(folder/'experiment_variant.json'),
            'training_identity_sha256': evaluator.file_sha(folder/'run_identity.json'),
            'training_config_sha256': evaluator.file_sha(folder/'training_config.json')}


def main(args):
    if evaluator.file_sha(evaluator.__file__) != '167aa74528eeb642f3b0318dc43194ebd2b2ad30d30cb6a07cacab192b003f90':
        raise ValueError('Pinned development evaluator changed')
    evaluator.ARMS = {**evaluator.ARMS, ARM: ('27b', 1/8, '2B')}
    evaluator.CONDITIONS = (*evaluator.CONDITIONS, ARM)
    evaluator.final_pair = final_pair
    evaluator.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'bench', 'harness-4b', 'material-dir', 'reports', 'selection', 'p2-root', 'p4-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--condition', choices=[ARM], default=ARM)
    parser.add_argument('--seconds', type=int, default=1200)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        evaluator.write(args.output/(args.condition+'_status.json'),
                        {'phase': 'failed', 'condition': args.condition,
                         'error': f'{type(error).__name__}: {error}'})
        raise
