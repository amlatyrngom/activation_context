"""L40S capacity probe: CPU-saved activations for proof only; ordinary smaller-budget timings."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import probe_strong_piecewise_budget_v1 as base

BASE_SHA='377e022e868dce0ae64fed25e639e886276189b67acfc1832003982758faf381'
TIMING_BUDGETS=(32768,16384)
OOM_EXIT=42
write,read,sha=base.write,base.read,base.sha


def offloaded_proof(*args,**kwargs):
    import torch
    ac=args[1]
    if ac.device.type=='cuda':
        torch.cuda.synchronize(ac.device)
        torch.cuda.reset_peak_memory_stats(ac.device)
    # This changes saved-tensor storage only. No hook reads saved CPU tensors;
    # backward restores them through PyTorch on their original device/stream.
    with torch.autograd.graph.save_on_cpu(pin_memory=True):
        result=ORIGINAL_PROOF(*args,**kwargs)
    result['saved_activation_offload']='torch.autograd.graph.save_on_cpu(pin_memory=True); diagnostic only'
    result['peak_host_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    if ac.device.type=='cuda':
        torch.cuda.synchronize(ac.device)
        result['peak_allocated_gib']=torch.cuda.max_memory_allocated(ac.device)/2**30
    return result


ORIGINAL_PROOF=base.prove_case


def worker(args):
    import torch
    proof=args.worker=='proof'
    if not proof and int(args.worker.rsplit('_',1)[1]) not in TIMING_BUDGETS:
        raise ValueError('L40S production timing permits only32768/16384; no65k timing')
    # Preserve all guarded data/model/config behavior in the existing worker.
    # The rebinding is process-local and restored even on a failed measurement.
    original_write=base.write;original_prove=base.prove_case
    def annotated_write(path,value):
        if Path(path)==args.output/(args.worker+'.json'):
            value.update(wrapper_sha256=sha(__file__),base_worker_sha256=BASE_SHA,
                saved_activation_offload=proof,
                offload_scope='numerical proof only' if proof else 'none: ordinary production timing',
                hardware_scope='This L40S only; no RTX throughput comparison implied')
        original_write(path,value)
    base.write=annotated_write
    if proof:base.prove_case=offloaded_proof
    try:
        base.worker(args)
        return 0
    except torch.cuda.OutOfMemoryError as error:
        path=args.output/(args.worker+'.json')
        receipt=read(path) if path.exists() else {}
        # Do not silently classify a missing/changed parent as a usable OOM result.
        if not {'pair','payload_sha256','optimizer_sha256'} <= receipt.keys():raise
        unchanged=(base.paired.payload_hashes(receipt['pair'])==receipt['payload_sha256']
            and sha(Path(receipt['pair']['ac_checkpoint'])/'optimizer.pt')==receipt['optimizer_sha256'])
        if not unchanged:raise AssertionError('Parent payload changed while checking OOM')
        receipt.update(status='oom',error=str(error),parent_payloads_rehashed_unchanged=True,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            usable_for_training=False)
        annotated_write(path,receipt)
        return OOM_EXIT
    finally:
        base.write=original_write;base.prove_case=original_prove


def classify(condition,returncode,receipt):
    if returncode==0 and receipt.get('status')=='complete':return 'complete'
    if (condition!='proof' and returncode==OOM_EXIT and receipt.get('status')=='oom'
            and receipt.get('parent_payloads_rehashed_unchanged') is True
            and receipt.get('saved_activation_offload') is False):return 'oom_unusable'
    return 'failed'


def verify_measurement(record,proof):
    if record.get('saved_activation_offload') is not False:
        raise AssertionError('Timing used saved-activation offload')
    for key in ('payload_sha256','optimizer_sha256'):
        if record[key]!=proof[key]:raise AssertionError('Timing/proof initialization differs: '+key)
    if record.get('completed_runtime_updates')!=4004 or record.get('checkpoint_saves') is not False:
        raise AssertionError('Timing update/save protocol differs')


def controller(args):
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'summary.json'
    if path.exists():raise FileExistsError(path)
    started=time.monotonic()
    summary={'status':'running','wrapper_sha256':sha(__file__),'base_worker_sha256':BASE_SHA,
        'proof_budgets':[65536,32768,16384],'timing_budgets':list(TIMING_BUDGETS),
        'proof_offload':True,'timing_offload':False,'conditions':{},'budgets':{},
        'seconds_budget':args.seconds,'scope':'L40S fit and throughput; no cross-device speedup claim'}
    write(path,summary)
    proof=None
    conditions=['proof',*[f'{case}_{budget}' for budget in TIMING_BUDGETS for case in base.CASES]]
    for condition in conditions:
        budget=None if condition=='proof' else int(condition.rsplit('_',1)[1])
        if budget is not None and summary['budgets'].get(str(budget),{}).get('usable') is False:
            summary['conditions'][condition]={'status':'skipped_budget_already_oom'};write(path,summary);continue
        remaining=args.seconds-(time.monotonic()-started)
        if remaining<120:
            summary['status']='partial_budget';write(path,summary);return 1
        workspace=args.workspace/condition;workspace.mkdir(parents=True,exist_ok=True)
        if (workspace/'AC_MODELS').exists():raise FileExistsError('Use a fresh measurement workspace')
        if not (workspace/'AUTOTUNE').exists():
            (workspace/'AUTOTUNE').symlink_to(args.autotune_root,target_is_directory=True)
        env=dict(os.environ);env['ACTIVATION_SYNC_ROOT']=str(workspace)
        if condition=='proof':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        else:env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',condition]
        for name in ('material','bench','design','parent_receipt','memory_receipt','reports','runs_root','output'):
            command+=['--'+name.replace('_','-'),str(getattr(args,name))]
        with (args.output/(condition+'.log')).open('x') as log:
            try:
                completed=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining-30)
            except subprocess.TimeoutExpired:
                summary['conditions'][condition]={'status':'timeout'}
                summary['status']='partial_budget';write(path,summary);return 1
        result_path=args.output/(condition+'.json');record=read(result_path) if result_path.exists() else {}
        status=classify(condition,completed.returncode,record)
        summary['conditions'][condition]={'returncode':completed.returncode,'status':status,'worker_status':record.get('status')}
        write(path,summary);print(json.dumps({condition:summary['conditions'][condition]}),flush=True)
        if status=='failed':
            summary['status']='proof_failed' if condition=='proof' else 'timing_failed';write(path,summary);return 1
        if condition=='proof':
            if record.get('saved_activation_offload') is not True or not all(record['cases'][case]['passed'] for case in base.CASES):
                raise AssertionError('Expected actual passing offloaded proof for both cases')
            proof=record
        elif status=='oom_unusable':
            summary['budgets'][str(budget)]={'usable':False,'reason':'CUDA OOM without offload','failed_case':condition}
        else:
            if proof is None:raise AssertionError('Timings cannot precede successful proof')
            verify_measurement(record,proof)
        write(path,summary)
    for budget in TIMING_BUDGETS:
        if summary['budgets'].get(str(budget),{}).get('usable') is False:continue
        records=[read(args.output/f'{case}_{budget}.json') for case in base.CASES]
        for record in records:verify_measurement(record,proof)
        summary['budgets'][str(budget)]={'usable':True,
            'case_seconds':{case:record['steady_median_seconds'] for case,record in zip(base.CASES,records)},
            'peak_allocated_gib':max(record['peak_memory_gib'] for record in records)}
    for case in base.CASES:
        records=[read(args.output/f'{case}_{budget}.json') for budget in TIMING_BUDGETS
                 if summary['budgets'][str(budget)]['usable']]
        for key in ('selected_penalty_ids','learning_rates'):
            if records and any(record[key]!=records[0][key] for record in records[1:]):
                raise AssertionError('Candidate training draws/schedule differ: '+key)
    summary.update(status='complete',elapsed_seconds=time.monotonic()-started,
        usable_budgets=[b for b in TIMING_BUDGETS if summary['budgets'][str(b)]['usable']])
    write(path,summary);return 0


def cpu_check():
    original=base.prove_case
    try:
        base.prove_case=offloaded_proof
        base.cpu_check()
    finally:base.prove_case=original


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker');parser.add_argument('--cpu-check',action='store_true')
    for name in ('material','bench','design','parent-receipt','memory-receipt','reports','runs-root','output','workspace','autotune-root'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--seconds',type=int,default=2520);args=parser.parse_args()
    if sha(base.__file__)!=BASE_SHA:raise ValueError('Pinned strong-parent worker changed')
    if (sha(base.previous.__file__)!=base.PREVIOUS_SHA or sha(base.paired.__file__)!=base.PAIRED_SHA
            or sha(Path(__file__).with_name('probe_piecewise_original_v1.py'))!=base.previous.HELPER_SHA):
        raise ValueError('Pinned existing proof helper changed')
    if args.cpu_check:cpu_check()
    elif args.worker:sys.exit(worker(args))
    else:sys.exit(controller(args))
