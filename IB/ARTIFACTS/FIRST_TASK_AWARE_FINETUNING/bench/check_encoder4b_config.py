"""CPU-only configuration wiring check; deliberately does not claim tensor restoration."""
import ast
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[4]
RUNNER = Path(__file__).with_name('run_qa_encoder4b_v1.py')
RECEIPTS = ROOT / 'IB/TMP/FIRST_TASK_AWARE_FINETUNING/encoder4b_inventory'


def main():
    receipts = json.loads((RECEIPTS / 'remote_metadata.json').read_text())
    # Execute the real config dataclass without importing the GPU-dependent package.
    tree = ast.parse((ROOT / 'activation/ac_model/ac_model.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ActivationContextModelConfig')
    namespace = {'dataclass': dataclasses.dataclass}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<real AC config>', 'exec'), namespace)
    config_type = namespace['ActivationContextModelConfig']

    class ModelConfig:
        def __init__(self, name, model_id):
            self.name, self.model_id = name, model_id

    class HarnessRuntimeConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Manager:
        def register_lora(self, *args, **kwargs):
            self.reader_args, self.reader_kwargs = args, kwargs

        def register_ac_model(self, config):
            self.ac_config = config

    class HarnessRuntime:
        def __init__(self, config):
            self.config, self.module_manager = config, Manager()

    harness_module = types.ModuleType('activation.harness')
    for cls in (ModelConfig, HarnessRuntimeConfig, HarnessRuntime):
        setattr(harness_module, cls.__name__, cls)
    ac_module = types.ModuleType('activation.ac_model')
    ac_module.ActivationContextModelConfig = config_type
    torch_module = types.ModuleType('torch')
    torch_module.set_num_threads = lambda _: None
    torch_module.manual_seed = lambda _: None
    sys.modules.update({'activation.harness': harness_module, 'activation.ac_model': ac_module, 'torch': torch_module})
    spec = importlib.util.spec_from_file_location('encoder4b_runner', RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    checked = []
    with tempfile.TemporaryDirectory() as temporary:
        ac_path, reader_path = Path(temporary) / 'ac', Path(temporary) / 'reader'
        (ac_path / 'side_lora').mkdir(parents=True)
        reader_path.mkdir()
        for receipt in receipts:
            (ac_path / 'side_lora/adapter_config.json').write_text(json.dumps(receipt['side_adapter']))
            (reader_path / 'adapter_config.json').write_text(json.dumps(receipt['reader_adapter']))
            harness = runner.harness(ac_path, reader_path)
            actual = harness.module_manager.ac_config
            metadata = receipt['checkpoint_metadata']
            assert harness.config.model_configs['side_base'].model_id == 'Qwen/Qwen3.5-4B'
            assert harness.config.model_configs['qwen4b'].model_id == 'Qwen/Qwen3.5-9B'
            assert harness.module_manager.reader_kwargs['rank'] == 128
            assert metadata['model_ids'] == [runner.ENCODER_ID, runner.READER_ID]
            assert (metadata['architecture_version'], metadata['d_side'], metadata['d_target']) == (2, 2560, 4096)
            for key in ('default_compression_ratio', 'mixer_layers', 'mixer_heads', 'mixer_window', 'min_view_rows', 'max_view_rows', 'side_lora_rank', 'intervention_frequency'):
                assert getattr(actual, key) == metadata['config'][key], key
            assert actual.input_pooling_stride == 0
            assert metadata['config']['row_head_skip'] is False
            interventions = metadata['interventions']
            assert interventions['layers'] == list(range(3, 32, 4))
            assert [interventions[k] for k in ('source', 'attention_depth', 'target', 'head_style')] == ['passage_attention', 'final', 'residual', 'lora']
            assert not receipt['readout_module_keys']
            checked.append(receipt['arm'])

        # Catch a wrong side base, reader base, or rank before any model construction.
        bad_cases = [(ac_path / 'side_lora/adapter_config.json', 'base_model_name_or_path', 'Qwen/Qwen3.5-2B'),
                     (reader_path / 'adapter_config.json', 'base_model_name_or_path', 'Qwen/Qwen3.5-4B'),
                     (reader_path / 'adapter_config.json', 'r', 64)]
        for path, key, value in bad_cases:
            original = path.read_text()
            bad = json.loads(original)
            bad[key] = value
            path.write_text(json.dumps(bad))
            try:
                runner.harness(ac_path, reader_path)
            except ValueError as error:
                assert 'paired 4B-side/9B-reader rank128' in str(error)
            else:
                raise AssertionError('Invalid adapter identity accepted')
            path.write_text(original)
        try:
            runner.validate_run(Path(temporary), {}, None, {})
        except runner.ResumeError as error:
            assert 'explicit run identity' in str(error)
        else:
            raise AssertionError('Legacy run accepted without encoder identity')

    result = {'status': 'pass', 'candidates': checked, 'adapter_negative_cases': len(bad_cases),
              'legacy_resume_rejected': True, 'scope': 'CPU config wiring and saved metadata only; no tensor restore or GPU execution',
              'runner_sha256': hashlib.sha256(RUNNER.read_bytes()).hexdigest()}
    (RECEIPTS / 'cpu_config_check.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
