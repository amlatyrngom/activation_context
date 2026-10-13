"""Finish reporting/publication of the already-trained gentle500 pair; zero updates."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import run_qa_gentle42k_long_v1 as bench

BUILDER_SHA = '2a3060762a16dd66b65c92df23609ef87d94af46300f184c8ec72478ecfdb647'


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def state_digest(value):
    """Exact recursive values, including tensor dtype/shape/raw bytes."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
            h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                h.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            h.update(repr(item).encode())
    visit(value)
    return h.hexdigest()


def weights_and_adam(ac_path, reader_path, *, completed):
    import torch
    from safetensors.torch import load_file
    modules = torch.load(ac_path / 'ac_modules.pt', map_location='cpu', weights_only=False)
    result = {'modules': state_digest(modules['modules'])}
    assert modules['epoch_number'] == (1 if completed else 0)
    result['structure'] = {key: modules[key] for key in ('architecture_version', 'model_ids', 'd_side', 'd_target')}
    del modules; gc.collect()
    for name, folder in [('encoder', ac_path / 'side_lora'), ('reader', reader_path)]:
        tensors = load_file(str(folder / 'adapter_model.safetensors'), device='cpu')
        result[name] = state_digest(tensors)
        del tensors; gc.collect()
    state = torch.load(ac_path / 'optimizer.pt', map_location='cpu', weights_only=False)
    assert state['updates_done'] == 500 and state['epoch_number'] == (1 if completed else 0)
    if not completed: assert state['step'] == 500
    values = state['optimizer']['state']
    assert len(values) == 615 and {float(v['step']) for v in values.values()} == {5750}
    assert [len(g['params']) for g in state['optimizer']['param_groups']] == [374, 256]
    result['optimizer'] = state_digest(state['optimizer'])
    del state, values; gc.collect()
    return result


def final_panel_progress(progress):
    if progress.epoch != 0 or progress.global_step != 500 or progress.steps != 500:
        raise ValueError('Only the saved500-update final reporting baseline may be promoted')
    return replace(progress, epoch=1, epochs=1, epoch_number=1, step=500)


def main(args):
    import torch
    from activation.ac_model import ActivationContextTrainer
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as trainer_module
    import activation.ac_model.ac_model_utils as utils_module
    design_path = args.design.resolve()
    design = json.loads(design_path.read_text())
    root = Path(os.environ['QA_REPORT_ROOT']).parent
    run = root / design['run_name']; report = root / 'reports' / design['run_name']
    assert design['status'] == 'frozen' and design['run_name'] == 'gentle42k_final_lr_familiar4k_8k'
    assert sha(bench.__file__) == BUILDER_SHA == design['builder_sha256']
    assert {key: sha(module.__file__) for key, module in [('ac_model', ac_module), ('trainer', trainer_module), ('utils', utils_module)]} == design['source_sha256']
    for name, expected in design['helper_sha256'].items():
        assert Path(name).name == name and sha(Path(__file__).with_name(name)) == expected
    receipt = json.loads((report / 'source_receipt.json').read_text())
    assert receipt['design_sha256'] == sha(design_path)
    original_result = json.loads((report / 'result.json').read_text())
    assert original_result['stop_reason'] == 'duration limit' and original_result['completed_global_updates'] == 500
    marker_path = run / 'AC_MODELS/rag_poc/partial_epoch.json'
    marker = json.loads(marker_path.read_text())
    assert marker == {'epoch_number': 1, 'step': 500, 'steps': 500,
        'ac_checkpoint': 'AC_MODELS/rag_poc/epoch_001_step_00500',
        'target_lora_checkpoint': 'LORAS/rag_reader/epoch_001_step_00500'}
    assert not (run / 'AC_MODELS/rag_poc/completed_epoch.json').exists()
    assert not (report / 'panel_00500.json').exists()
    backup = run / 'recovery_500_before_finalize_v1'
    backup.mkdir(exist_ok=False)
    (backup / 'PRESERVE').write_text('Original complete500-update pair before zero-update reporting finalization.\n')
    write(backup / 'partial_epoch.json', marker)
    shutil.copy2(report / 'result.json', backup / 'result.json')
    proof_path = report / 'finalization_500_v1.json'
    proof = {'status': 'backing_up', 'wrapper_sha256': sha(__file__), 'builder_sha256': BUILDER_SHA,
        'design_sha256': sha(design_path), 'backup': str(backup), 'original_result_sha256': sha(backup / 'result.json'),
        'purpose': 'All4000 familiar questions already trained; finish missing final reporting and normal epoch publication, zero additional updates.'}
    write(proof_path, proof)
    for name, relative in [('ac', marker['ac_checkpoint']), ('reader', marker['target_lora_checkpoint'])]:
        source = run / relative; destination = backup / name
        shutil.copytree(source, destination)
        for path in source.rglob('*'):
            if path.is_file(): assert sha(path) == sha(destination / path.relative_to(source))
    before = weights_and_adam(backup / 'ac', backup / 'reader', completed=False)
    proof.update(status='finalizing', original_state=before); write(proof_path, proof)
    os.environ['ACTIVATION_SYNC_ROOT'] = str(run)
    original_train = ActivationContextTrainer.train
    original_step = torch.optim.AdamW.step
    callbacks = []
    def reject_step(*unused, **kwargs):
        raise AssertionError('Reporting-only finalization must never call optimizer.step')
    def finish(self, name, *values, **kwargs):
        assert self.config.resume_skip_steps == 500 and self.config.num_epochs == 1
        callback = kwargs['reporting_callback']; reporter = kwargs.get('reporter')
        def final_callback(progress, summary):
            mapped = final_panel_progress(progress)
            callbacks.append(mapped.global_step)
            assert len(callbacks) == 1
            if reporter is not None: reporter.report_eval(mapped, 'reporting', summary)
            callback(mapped, summary)
        kwargs['reporting_callback'] = final_callback
        self.config = replace(self.config, max_duration_seconds=1800)
        stats = original_train(self, name, *values, **kwargs)
        assert stats.steps == 0 and self.updates_done[name] == 500 and stats.stop_reason == 'complete'
        assert callbacks == [500]
        return stats
    ActivationContextTrainer.train = finish
    torch.optim.AdamW.step = reject_step
    try:
        bench.main(SimpleNamespace(arm='joint', items=4000, epochs=1, seconds=design['training_config']['max_duration_seconds'],
            resume=True, design=str(design_path), parent_readout=design['parent_readout_path'], run_name=design['run_name'],
            material=design['material_path'], initial_pair=design['initial_pair_path'], evidence_tokens=8192, max_passages=16,
            ratio=1/16, micro_batch_examples=2, encode_batch_max_tokens=65536, gradient_checkpointing_min_tokens=0,
            passage_parts=True, encoder_lr=5e-6, reader_lr=2e-6))
        after = weights_and_adam(run / 'AC_MODELS/rag_poc/epoch_001', run / 'LORAS/rag_reader/epoch_001', completed=True)
        assert before == after, 'Final weights or exact inherited Adam values changed'
        result = json.loads((report / 'result.json').read_text())
        assert result['stop_reason'] == 'complete' and result['steps'] == 0 and result['completed_global_updates'] == 500
        assert (report / 'panel_00500.json').exists()
        proof.update(status='complete', final_state=after, all_parameter_and_adam_values_exact=True,
            additional_optimizer_updates=0, final_callbacks=callbacks, effective_reporting_budget_seconds=1800,
            original_partial_backup_preserved=True)
    except BaseException as error:
        proof.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        ActivationContextTrainer.train = original_train
        torch.optim.AdamW.step = original_step
        write(proof_path, proof)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', type=Path, required=True)
    main(parser.parse_args())
