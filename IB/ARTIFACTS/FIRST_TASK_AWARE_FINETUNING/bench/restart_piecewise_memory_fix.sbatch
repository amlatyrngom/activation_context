#!/bin/bash
#SBATCH --job-name=piecewise_restart_files
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
set -euo pipefail
"$HOME/activation_venv/bin/python" - <<'PY'
import hashlib
import json
import shutil
from pathlib import Path

root = Path.home() / 'orcd/pool/activation/taskaware'
design = json.loads((root / 'bench/PIECEWISE_TRAINING_DESIGN_V2.json').read_text())
assert design['version'] == 2 and design['status'] == 'frozen'
assert design['training']['gradient_checkpointing_min_tokens'] == 0
# Ordinary backward repeats were not numerically repeatable. The separate
# deterministic diagnostic must pass the original thresholds before any restart.
proof_root = root / 'reports/piecewise_backward_deterministic_v1'
proof_path = proof_root / 'proof.json'
proof = json.loads(proof_path.read_text())
settings = json.loads((proof_root / 'deterministic_settings.json').read_text())
assert proof['status'] == 'passed' and proof['equivalence']['passed']
assert settings['status'] == 'complete'
assert settings['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
assert settings['deterministic_algorithms'] and settings['deterministic_algorithms_at_exit']
assert not settings['warn_only'] and not settings['warn_only_at_exit']
validation = design['checkpoint_validation']
assert hashlib.sha256(proof_path.read_bytes()).hexdigest() == validation['deterministic_proof_sha256']
assert proof['source'] == design['source_sha256']
assert proof['P2_pair_sha256'] == design['P2_pair_sha256']
for condition, folder in [('pieces_failing', 'piecewise_memory_only_v1'),
                          ('pieces_quantiles', 'piecewise_memory_only_v1'),
                          ('merged_failing', 'piecewise_memory_remaining_v1'),
                          ('merged_quantiles', 'piecewise_memory_remaining_v1')]:
    path = root / 'reports' / folder / (condition + '.json')
    measured = json.loads(path.read_text())
    assert measured['status'] == 'complete' and measured['threshold'] == 0
    assert measured['source'] == design['source_sha256']
    assert measured['P2_pair_sha256'] == design['P2_pair_sha256']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == validation['memory_case_sha256'][condition]

name = 'piecewise_27b_4k_8k'
run = root / name
report = root / 'reports' / name
archive = root / 'failed_attempts' / (name + '_oom_23485579')
assert run.is_dir() and report.is_dir() and not archive.exists()
assert not list((run / 'AC_MODELS/rag_poc').glob('epoch_*'))
assert not list((run / 'LORAS/rag_reader').glob('epoch_*'))
reference = run / 'AC_MODELS/rag_poc/no_context_reference.pt'
with reference.open('rb') as handle:
    reference_sha = hashlib.file_digest(handle, 'sha256').hexdigest()
assert reference_sha == '5131a2e319f59e7b0400388b5fde3bc946f7cd19defac1dbdc52f6a5ea1ed0f1'
archive.mkdir(parents=True)
run.rename(archive / 'run')
report.rename(archive / 'report')
(archive / 'PRESERVE').write_text('Failed original attempt; keep its report, metadata and frozen reference cache.\n')
destination = run / 'AC_MODELS/rag_poc/no_context_reference.pt'
destination.parent.mkdir(parents=True)
shutil.copy2(archive / 'run/AC_MODELS/rag_poc/no_context_reference.pt', destination)
with destination.open('rb') as handle:
    assert hashlib.file_digest(handle, 'sha256').hexdigest() == reference_sha
(run / 'PRESERVE').write_text('Active fresh-P2 restart after checkpoint-only memory repair.\n')
report.mkdir(parents=True)
receipt = {'archived_attempt': str(archive), 'reference_sha256': reference_sha,
           'reused': 'Unchanged frozen bare-base reference cache only',
           'restart': 'Original P2 pair and fresh QA optimizer; no missing optimizer resume'}
(archive / 'restart_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt), flush=True)
PY
