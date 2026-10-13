#!/bin/bash
#SBATCH --job-name=qa_lr_reference
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
set -euo pipefail
"$HOME/activation_venv/bin/python" - <<'PY'
import hashlib, json, os, shutil, tempfile
from pathlib import Path
root=Path.home()/'orcd/pool/activation/taskaware'
source=root/'oracle_27b_16k_r16/AC_MODELS/rag_poc/no_context_reference.pt'
run=root/'oracle_27b_16k_r16_half_ac_lr'
destination=run/'AC_MODELS/rag_poc/no_context_reference.pt'
assert not (root/'reports'/run.name/'training_config.json').exists()
assert not destination.exists()
destination.parent.mkdir(parents=True,exist_ok=True)
(run/'PRESERVE').write_text('Prepared controlled lower-AC-LR experiment; retain paired state.\n')
fd,name=tempfile.mkstemp(prefix='.reference_',dir=destination.parent);os.close(fd)
temporary=Path(name)
try:
    shutil.copyfile(source,temporary)
    with temporary.open('rb') as handle:
        checksum=hashlib.file_digest(handle,'sha256').hexdigest()
        os.fsync(handle.fileno())
    # Exact bytes already torch-loaded and validated by CPU job23458491.
    assert checksum=='5131a2e319f59e7b0400388b5fde3bc946f7cd19defac1dbdc52f6a5ea1ed0f1'
    os.link(temporary,destination)
    receipt={'source':str(source),'destination':str(destination),'sha256':checksum,
             'bytes':destination.stat().st_size,'validated_source_receipt':'BASE_REFERENCE_REUSE.json'}
    (root/'reports/r16_lr_reference_reuse.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
finally:
    temporary.unlink(missing_ok=True)
PY
