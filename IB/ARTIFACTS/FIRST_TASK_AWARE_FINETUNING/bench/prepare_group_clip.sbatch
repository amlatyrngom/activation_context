#!/bin/bash
#SBATCH --job-name=prepare_group_clip
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
set -euo pipefail
"$HOME/activation_venv/bin/python" - <<'PY'
import hashlib, json, os, shutil, subprocess, tempfile
from pathlib import Path
root=Path.home()/'orcd/pool/activation/taskaware'
source=root/'train_ws_night_v1'; destination=root/'train_ws_group_clip_v1'
relative=Path('activation/ac_model/ac_model_training.py')
expected_before='38ca9885bfe35e47432352aa716381e53c18aa57748428eedf5f5f6b7fc6b5aa'
expected_after='8ae0a9fb4c2a514c84e9baee3f3619d5c06961800a7268f36ada85246f2af1de'
assert not destination.exists()
assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==expected_before
patch=root/'bench/group_clip.patch'
assert hashlib.sha256(patch.read_bytes()).hexdigest()=='95ca2f3cd0cbde521ecd4c5f467394e73d3cd0c747022d30ed76a62d93127fa9'
temporary=Path(tempfile.mkdtemp(prefix='.group_clip_source_',dir=root))
try:
    shutil.copytree(source/'activation',temporary/'activation',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-i',str(patch)],cwd=temporary,check=True)
    assert hashlib.sha256((temporary/relative).read_bytes()).hexdigest()==expected_after
    if destination.exists():raise FileExistsError(destination)
    temporary.rename(destination)
finally:
    if temporary.exists():shutil.rmtree(temporary)
run=root/'oracle_27b_16k_r8_group_clip'
cache=run/'AC_MODELS/rag_poc/no_context_reference.pt'
assert not cache.exists() and not (root/'reports'/run.name/'training_config.json').exists()
cache.parent.mkdir(parents=True,exist_ok=True)
source_cache=root/'oracle_27b_16k_r16/AC_MODELS/rag_poc/no_context_reference.pt'
fd,name=tempfile.mkstemp(prefix='.reference_',dir=cache.parent);os.close(fd)
temporary=Path(name)
try:
    shutil.copyfile(source_cache,temporary)
    with temporary.open('rb') as handle:checksum=hashlib.file_digest(handle,'sha256').hexdigest()
    assert checksum=='5131a2e319f59e7b0400388b5fde3bc946f7cd19defac1dbdc52f6a5ea1ed0f1'
    os.link(temporary,cache)
finally:temporary.unlink(missing_ok=True)
(run/'PRESERVE').write_text('Prepared optional optimizer comparison; preserve its reference and paired saves.\n')
receipt={'status':'prepared; no GPU training submitted','source_tree':str(source),'variant_tree':str(destination),
    'source_sha256':expected_before,'variant_sha256':expected_after,
    'change':'Clip encoder/AC modules and reader separately to norm1; all other source files and training settings unchanged.',
    'reference_sha256':checksum,'reference_bytes':cache.stat().st_size}
(destination/'SOURCE_VARIANT.json').write_text(json.dumps(receipt,indent=2)+'\n')
(root/'reports/group_clip_preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
