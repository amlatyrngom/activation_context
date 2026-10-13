#!/bin/bash
#SBATCH --job-name=prepare_piecewise_runtime
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
source=root/'train_ws_night_v1'; destination=root/'train_ws_piecewise_v1'
relative=Path('activation/ac_model/ac_model.py')
before='aaeb3c68225d3092a024d44552b84480dc37cc8319d68c2ab1701fce70980bfd'
after='bb0b50724a1b7d9f8b88effa6a555e25b4aa7d2ed1e7053629e1c231c55e3541'
assert not destination.exists()
assert hashlib.sha256((source/relative).read_bytes()).hexdigest()==before
patch=root/'bench/piecewise_batching_v1.patch'
assert hashlib.sha256(patch.read_bytes()).hexdigest()=='abfee2f20b6152671c74fbde6f2af1f275e06abeef051d513c9c1c75efa4b51e'
temporary=Path(tempfile.mkdtemp(prefix='.piecewise_source_',dir=root))
try:
    shutil.copytree(source/'activation',temporary/'activation',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    subprocess.run(['patch','--batch','--forward','--fuzz=0','-p1','-i',str(patch)],cwd=temporary,check=True)
    assert hashlib.sha256((temporary/relative).read_bytes()).hexdigest()==after
    if destination.exists():raise FileExistsError(destination)
    temporary.rename(destination)
finally:
    if temporary.exists():shutil.rmtree(temporary)
cache_source=root/'oracle_27b_16k_r16/AC_MODELS/rag_poc/no_context_reference.pt'
reference_sha='5131a2e319f59e7b0400388b5fde3bc946f7cd19defac1dbdc52f6a5ea1ed0f1'
copies={}
for name in ('piecewise_27b_4k_8k','merged_27b_4k_8k'):
    run=root/name; cache=run/'AC_MODELS/rag_poc/no_context_reference.pt'
    assert not cache.exists() and not (root/'reports'/name/'training_config.json').exists()
    cache.parent.mkdir(parents=True,exist_ok=True)
    fd,filename=tempfile.mkstemp(prefix='.reference_',dir=cache.parent);os.close(fd)
    temporary=Path(filename)
    try:
        shutil.copyfile(cache_source,temporary)
        with temporary.open('rb') as handle:checksum=hashlib.file_digest(handle,'sha256').hexdigest()
        assert checksum==reference_sha
        os.link(temporary,cache)
    finally:temporary.unlink(missing_ok=True)
    (run/'PRESERVE').write_text('Authorized matched piecewise QA experiment; preserve references and paired saves.\n')
    copies[name]=cache.stat().st_size
receipt={'source_tree':str(source),'variant_tree':str(destination),
    'changed_file':str(relative),'source_sha256':before,'variant_sha256':after,
    'change':'Side decoder uses causal right-padding without a mask; checkpoint threshold counts total padded batch tokens. Mixer valid mask unchanged. Pending real GPU equivalence and timing proof.',
    'reference_sha256':reference_sha,'reference_copies_bytes':copies}
(destination/'SOURCE_VARIANT.json').write_text(json.dumps(receipt,indent=2)+'\n')
(root/'reports/piecewise_runtime_preparation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
PY
