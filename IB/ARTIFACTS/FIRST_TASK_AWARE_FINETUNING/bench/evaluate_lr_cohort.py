"""Reuse the fixed development100 evaluator for the two matched half-AC-LR finals.

The original five-arm evaluator and confirmation helpers remain byte-for-byte unchanged.
This private entrypoint only supplies the additional final-checkpoint identities.
"""
import argparse
from pathlib import Path

import evaluate_model_comparison_cohort as evaluator

ARMS = {f'oracle_27b_16k_r{denominator}_half_ac_lr': ('27b', 1/denominator, '2B')
        for denominator in (16, 8)}
original_final_pair = evaluator.final_pair


def final_pair(arm, reports, material_hashes):
    if arm not in ARMS:
        return original_final_pair(arm, reports, material_hashes)
    control = arm.removesuffix('_half_ac_lr')
    # The original final validator verifies the completed control's full recipe.
    original_final_pair(control, reports, material_hashes)
    read = evaluator.read
    folder, prior = Path(reports)/arm, Path(reports)/control
    identity = read(folder/'run_identity.json')['identity']
    if identity != read(prior/'run_identity.json')['identity']:
        raise ValueError('Half-LR experiment changed data, order, initialization or budget')
    expected = {**read(prior/'training_config.json'), 'learning_rate_ac': 2.5e-5}
    if read(folder/'training_config.json') != expected:
        raise ValueError('Half-LR experiment changed more than the AC learning rate')
    result_path = folder/'result.json'
    result = read(result_path)
    if result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 2000:
        raise ValueError('Half-LR 16,000-question final is not complete')
    ac = Path(result['checkpoint']).resolve()
    reader = Path(result['target_lora_checkpoint']).resolve()
    marker_path = ac.parent/'completed_epoch.json'
    marker = read(marker_path)
    if (marker.get('epoch_number') != 1
            or (ac.parents[2]/marker['ac_checkpoint']).resolve() != ac
            or (ac.parents[2]/marker['target_lora_checkpoint']).resolve() != reader):
        raise ValueError('Final paired checkpoint disagrees with its epoch publication')
    # The reused checkpoint() then validates bases/rank/payloads and hashes every model file.
    return {'ac_checkpoint': str(ac), 'reader_checkpoint': str(reader),
            'result_sha256': evaluator.file_sha(result_path),
            'marker_sha256': evaluator.file_sha(marker_path),
            'auxiliary_entrypoint_sha256': evaluator.file_sha(__file__),
            'training_identity_sha256': evaluator.file_sha(folder/'run_identity.json'),
            'training_config_sha256': evaluator.file_sha(folder/'training_config.json')}


def main(args):
    if evaluator.file_sha(evaluator.__file__) != '167aa74528eeb642f3b0318dc43194ebd2b2ad30d30cb6a07cacab192b003f90':
        raise ValueError('Pinned development evaluator changed')
    evaluator.ARMS = {**evaluator.ARMS, **ARMS}
    evaluator.CONDITIONS = (*evaluator.CONDITIONS, *ARMS)
    evaluator.final_pair = final_pair
    evaluator.main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'bench', 'harness-4b', 'material-dir', 'reports', 'selection', 'p2-root', 'p4-root', 'output'):
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
