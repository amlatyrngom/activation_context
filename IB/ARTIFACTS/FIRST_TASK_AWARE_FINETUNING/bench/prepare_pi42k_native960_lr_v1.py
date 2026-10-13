"""Prepare two short paired diagnostic designs; reuse frozen material/subset without regeneration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

BENCH=Path(__file__).resolve().parent
BASE=BENCH/'PI42K_MERGED32K_FRESH960_ORCD_H200_DESIGN_V1.json'
RUNNER='run_qa_pi42k_native960_lr_v1.py'
LAUNCHER='train_pi42k_native960_lr_v1.sbatch'
ROOT='/home/ngom/orcd/pool/activation/taskaware'
ARMS={'floor':('pi42k_native2k_fresh960_floor_v1',5e-6,2e-6),
      '2x':('pi42k_native2k_fresh960_2x_v1',1e-5,4e-6)}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare():
    base=json.loads(BASE.read_text())
    if (base['status']!='frozen' or base['questions']!=960 or base['epochs']!=1
            or base['material_sha256']!='49f711e3eff25b0063f148f850e73ff4fe7917c05dad0898db0167623d7e11a9'
            or base['subset_sha256']!='578a8955664a2b750ea1141433f856a43fa2e2cdb48d38ae4ba1091e2ee650be'
            or base['initial_pair_sha256']!='33fc87f7272acd9ec4140fb3ec5ceb6b98ec47ea6009f4366cedd28bca4d706c'):
        raise ValueError('Expected reviewed frozen960 subset and original ORCD Adam5250 pair')
    results={}
    for arm,(name,encoder_lr,reader_lr) in ARMS.items():
        design=deepcopy(base)
        design.update(status='prepared_pending_root_review',run_name=name,run_root=ROOT+'/'+name,
            evidence_tokens=2048,max_passages=4,passage_parts=False,
            subset_path=ROOT+'/adam_moment_offload_probe_v1_inputs/subset.json',
            builder_sha256=sha(BENCH/RUNNER),baseline_design_sha256=sha(BASE),
            allocation_candidate='ORCD normal L40S,1GPU,64GiB host,2h reservation;6600s trainer including panels',
            diagnostic='Paired short native2k LR diagnostic:960 fresh questions×1. Not equivalent to RTX8k floor, not convergence or a schedule comparison.',
            optimizer=f'Actual inherited Adam5250→5370; stage0→120; constant{encoder_lr:g}/{reader_lr:g}',
            initial_baseline='Fresh same-L40S initial AC and full/no-context scores before optimization; no reused RTX metrics.',
            memory_policy='GPU-resident optimizer; fail immediately on OOM, preserve error; no retry with altered shapes/precision/checkpointing/offload.',
            paired_comparison='Only LR differs between the two L40S arms; same frozen960/order, held100, learning32, retrieval and actualAdam5250.',
            runtime_estimate='Planning estimate60–95min/arm from two instrumented native2k/micro1 L40S updates29.55/26.98s. Full960/worst-case/reporting runtime is unmeasured.',
            geometry_fit='Two actual first-batch updates peaked32.60GiB. Entire960 and full held100 panels have not been memory-probed; no guaranteed fit.',
            readout={'scope':'Fresh matched held100 NLL panels; no inline decoding. Any later decoded100 readout uses own complete120/Adam5370 final pair.',
                     'selection_sha256':base['readout']['selection_sha256']})
        design.pop('root_review',None)
        design['training_config'].update(learning_rate_ac=encoder_lr,learning_rate_target_lora=reader_lr,
                                         max_duration_seconds=6600)
        design['helper_sha256'].update({name:sha(BENCH/name) for name in (RUNNER,LAUNCHER,Path(__file__).name)})
        path=BENCH/f'PI42K_NATIVE960_{arm.upper()}_DESIGN_V1.json'
        if path.exists():raise FileExistsError('Create-only prepared design: '+str(path))
        path.write_text(json.dumps(design,indent=2)+'\n');results[arm]={'path':str(path),'sha256':sha(path)}
    return results

if __name__=='__main__':print(json.dumps(prepare(),indent=2))
