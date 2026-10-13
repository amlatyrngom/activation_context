"""Freeze the chosen development recipe for the two large runs; CPU-only, create-only."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from evaluate_data_amount_cohort import ARMS, HELPERS, TRAINING_FILES, evaluator
from evaluate_lr_cohort import final_pair


def main(args):
    if args.output.exists():
        raise FileExistsError('Data-amount design is already frozen: '+str(args.output))
    original = evaluator.read(args.design); selection = evaluator.read(args.selection)
    pins = {bank: original['sha256']['material_'+bank] for bank in ('9b', '27b')}
    selected_pair = final_pair(args.control, args.reports, pins)
    config = evaluator.read(args.reports/args.control/'training_config.json')
    identity = evaluator.read(args.reports/args.control/'run_identity.json')['identity']
    if config['learning_rate_ac'] not in (5e-5, 2.5e-5) or identity['ratio'] not in (1/16, 1/8):
        raise ValueError('Select one of the completed matched development recipes')
    if selection['pinned_sha256'] != {k: original['sha256'][k] for k in evaluator.PIN_KEYS}:
        raise ValueError('Original development selection changed')
    material_hashes = {
        'oracle_27b_32k': 'a73de2273d600ab3693540310510e496736c554d1685312132885cc24b5973aa',
        'oracle_27b_16k_2ep': '34c36167c381e3be22a76692f929373fff272eaefcb6ec14515b39fda63683e3'}
    material_paths = {'oracle_27b_32k': args.run_root/'model_comparison_32k_material/material_27b_32000.json',
                      'oracle_27b_16k_2ep': args.run_root/'model_comparison_material/material_27b_16000.json'}
    if any(evaluator.file_sha(material_paths[k]) != v for k, v in material_hashes.items()):
        raise ValueError('Frozen training material changed')
    confirmation = args.run_root/'model_comparison_32k_material/confirmation.json'
    confirmation_hash = evaluator.file_sha(confirmation)
    if confirmation_hash != '0f0a8330c1a442b3f354bfc746780df98b56684406a3aec9ba9fe1c8a13aef25':
        raise ValueError('Fresh reserved confirmation changed')
    bench = Path(__file__).parent
    design = {'status': 'frozen', 'run_root': str(args.run_root.resolve()),
        'selection_control': args.control, 'selection_control_final': selected_pair,
        'selection_basis': 'Completed matched development LR comparisons and source-grounded development answers; no original confirmation tuning.',
        'original_design_sha256': evaluator.file_sha(args.design), 'selection_sha256': evaluator.file_sha(args.selection),
        'original_held_ids': selection['original_held_ids'], 'trainer_sha256': selection['trainer_sha256'],
        'sha256': {'builder': original['sha256']['builder'], 'retrieval_material': original['sha256']['material_9b'], 'confirmation': confirmation_hash},
        'helper_sha256': {name: evaluator.file_sha(bench/name) for name in HELPERS},
        'training_sha256': {name: evaluator.file_sha(bench/name) for name in TRAINING_FILES},
        'initialization': 'Original P2 pair via the pinned launchers QA_P2_ROOT; fresh QA optimizer.',
        'arms': {}}
    for arm, questions, epochs in ((ARMS[0], 32000, 1), (ARMS[1], 16000, 2)):
        expected = {**config, 'num_epochs': epochs, 'schedule_total_updates': 4000,
                    'reporting_interval': .1*epochs, 'max_duration_seconds': 24600}
        design['arms'][arm] = {'questions': questions, 'epochs': epochs, 'ratio': identity['ratio'],
                              'material_sha256': material_hashes[arm], 'training_config': expected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.data_amount_design_', dir=args.output.parent)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(design, handle, indent=2); handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
        os.link(name, args.output)
    finally:
        Path(name).unlink(missing_ok=True)
    print(json.dumps({'status': 'frozen', 'control': args.control, 'sha256': evaluator.file_sha(args.output),
                      'learning_rate_ac': config['learning_rate_ac'], 'ratio': identity['ratio']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('design', 'selection', 'reports', 'run-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--control', choices=[f'oracle_27b_16k_r{r}{suffix}' for r in (16,8) for suffix in ('','_half_ac_lr')], required=True)
    main(parser.parse_args())
