"""Create metadata-only LR candidates from the completed frozen native fresh8k design.

Reuses the exact immutable material and independent Adam5250 initialization.
No material loading, checkpoint copying, remote work, or root approval is performed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

BASELINE_SHA = '31d422fa93d74d07daa63dee9265d6897e2b03498bd4b0c2e5f1babe165b42ab'
SCHEDULER_SHA = 'ca283394391b8b63699357277182d6d35b2c45877e5d8fbeae64b043a833072f'
NEW_HELPERS = ('prepare_fresh42k_native8k_lr_v1.py', 'run_qa_fresh42k_native8k_lr_v1.py',
               'train_fresh42k_native8k_lr_v1.sbatch', 'evaluate_fresh42k_native8k_lr_v1.py',
               'evaluate_fresh42k_native8k_lr_v1.sbatch')
ARMS = {
    'constant': ('CONSTANT2X', 'continue42k_8k_constant_2x', 1.,
                 'constant AC LR 1e-5 / reader LR 4e-6'),
    'cosine': ('COSINE2X', 'continue42k_8k_cosine_2x_floor', .5,
               'cosine AC LR 1e-5 to 5e-6 / reader LR 4e-6 to 2e-6, zero warmup'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_design(baseline, schedule, helpers):
    if (baseline['status'] != 'frozen' or baseline['run_name'] != 'continue42k_8k_final_lr_floor'
            or baseline['parent_condition'] != 'continue32k_10k_final_lr'):
        raise ValueError('Requires the completed constant42k parent native fresh8k baseline')
    expected = {'learning_rate_ac': 5e-6, 'learning_rate_target_lora': 2e-6,
                'lr_schedule': 'constant', 'lr_final_fraction': 1., 'warmup_updates': 0,
                'resume_optimizer': True, 'resume_skip_steps': 0, 'schedule_total_updates': 1000,
                'num_epochs': 1, 'examples_per_update': 8, 'micro_batch_examples': 2,
                'gradient_checkpointing_min_tokens': 1700, 'reporting_interval': .1,
                'max_duration_seconds': 6600, 'shuffle_examples': True}
    if any(baseline['training_config'][key] != value for key, value in expected.items()):
        raise ValueError('Baseline training recipe changed')
    _, name, fraction, description = ARMS[schedule]
    design = deepcopy(baseline)
    design.pop('root_review', None)
    design.update(status='prepared_awaiting_root_review', run_name=name, lr_arm=schedule,
                  lr_description=description, baseline_design_sha256=BASELINE_SHA,
                  baseline_run_name=baseline['run_name'], scheduler_sha256=SCHEDULER_SHA,
                  builder_sha256=helpers['run_qa_fresh42k_native8k_lr_v1.py'])
    design['helper_sha256'].update(helpers)
    design['training_config'].update(learning_rate_ac=1e-5, learning_rate_target_lora=4e-6,
                                     lr_schedule=schedule, lr_final_fraction=fraction)
    design['optimizer_policy'] = 'inherited Adam5250, ' + description
    design['optimizer'] = 'Inherited Adam5250→6250; reset only outer stage clock0→1000; ' + description
    design['learning32'] = 'Exact same fixed32 IDs inside the unchanged baseline fresh8000 material'
    design['comparison'] = ('Only LR changes versus completed constant5e-6/2e-6 baseline. '
                            'Same immutable material bytes, order, held400, learning32, retrieval, '
                            'initial paired weights and actual Adam5250. No new data selection.')
    design['schedule_endpoint'] = ('Existing set_learning_rates indexes applied updates0..999; '
        'cosine approaches its nominal50% floor at update1000. Last applied AC LR '
        '5.000012336995354e-6 and reader LR2.000004934798142e-6; do not modify the scheduler.'
        if schedule == 'cosine' else 'All1000 updates use AC1e-5 / reader4e-6.')
    design['allocation_candidate'] = ('One RTX,64GiB host,2h wall;6600s trainer bound. '
        'Baseline took92m14 including reports; separate20min final readout. '
        'Queue timing/preemption may prevent completion by18UTC; never substitute a partial pair.')
    return design


def main(args):
    if sha(args.baseline_design) != BASELINE_SHA:
        raise ValueError('Baseline frozen design bytes changed')
    baseline = json.loads(args.baseline_design.read_text())
    bench = Path(__file__).parent
    for leaf, expected in baseline['helper_sha256'].items():
        if Path(leaf).name != leaf or sha(bench / leaf) != expected:
            raise ValueError('Original pinned helper changed: ' + leaf)
    helpers = {leaf: sha(bench / leaf) for leaf in NEW_HELPERS}
    outputs = [args.output / ('FRESH42K_NATIVE8K_LR_' + arm[0] + '_DESIGN_V1.json')
               for arm in ARMS.values()]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Do not overwrite prepared or frozen candidates')
    args.output.mkdir(parents=True, exist_ok=True)
    for (schedule, _), path in zip(ARMS.items(), outputs):
        design = build_design(baseline, schedule, helpers)
        with path.open('x') as handle:
            json.dump(design, handle, indent=2, ensure_ascii=False)
            handle.write('\n')
        print(json.dumps({'path': str(path), 'status': design['status'], 'sha256': sha(path)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-design', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
