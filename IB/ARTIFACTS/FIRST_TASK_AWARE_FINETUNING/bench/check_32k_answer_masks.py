"""CPU-only answer-rendering check over the frozen training/development pool."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.modules.setdefault('fla', None)
root = Path(os.environ['QA_RUN_ROOT'])
bench = Path(__file__).parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, bench/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


old = load('run_qa_comparison_v2')
new = load('run_qa_comparison_v3')
from activation.ac_model import ActivationContextStudyGenerator
from activation.ac_model.ac_model_utils import assistant_target_ids
from activation.dataset.loaders.trajectory_utils import submit_answer_definition

started = time.monotonic()
harness = new.harness()
study = ActivationContextStudyGenerator(harness, new.AC)
material = json.loads((root/'model_comparison_32k_material/material_27b_32000.json').read_bytes())
old_ids = set(json.loads((root/'reports/oracle_27b_16k_r16/material_ids.json').read_text())['train'])
normalizer = study.tokenizer.backend_tokenizer.normalizer
counts = {'train': 0, 'held': 0, 'old_train_exact_masks': 0, 'old_held_exact_masks': 0}
normalized = []
digest = hashlib.sha256()
for split in ('train', 'held'):
    for row in material[split]:
        example = row['example']; answer = example['gold_answers'][0].strip()
        output = study.dialect.rendering.assistant_message('', [{'id': 'answer', 'name': 'submit_answer', 'arguments': {'answer': answer}}])
        ids = assistant_target_ids(study.tokenizer, output, [submit_answer_definition()])
        positions = new.answer_positions(study.tokenizer, ids, answer)
        expected = normalizer.normalize_str(answer) if normalizer is not None else answer
        if expected != answer:
            normalized.append({'split': split, 'item_id': example['example_id']})
        else:
            assert old.answer_positions(study.tokenizer, ids, answer) == positions
        if split == 'held' or example['example_id'] in old_ids:
            assert old.answer_positions(study.tokenizer, ids, answer) == positions
            counts['old_held_exact_masks' if split == 'held' else 'old_train_exact_masks'] += 1
        counts[split] += 1
        digest.update(json.dumps([example['example_id'], ids, positions], separators=(',', ':')).encode())
        if counts[split] % 5000 == 0:
            print(json.dumps({'split': split, 'checked': counts[split]}), flush=True)
try:
    new.answer_positions(study.tokenizer, ids, answer+' deliberately wrong')
except AssertionError:
    pass
else:
    raise AssertionError('A genuine answer mismatch was accepted')
assert counts == {'train': 32000, 'held': 400, 'old_train_exact_masks': 16000, 'old_held_exact_masks': 400}
record = {'status': 'pass', 'counts': counts, 'canonicalized_answers': normalized,
          'normalizer': str(normalizer), 'targets_and_masks_sha256': digest.hexdigest(),
          'builder_sha256': hashlib.sha256((bench/'run_qa_comparison_v3.py').read_bytes()).hexdigest(),
          'seconds': round(time.monotonic()-started, 1), 'model_weights_loaded': False,
          'scope': 'All32000 training and400 development answer targets/masks. No confirmation questions or model weights read. Original16000 and400 masks identical; genuine mismatch still rejected.'}
(root/'answer_mask_repair_check.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
