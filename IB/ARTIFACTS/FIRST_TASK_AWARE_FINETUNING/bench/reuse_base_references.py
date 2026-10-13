"""Copy frozen base-only penalty targets while destination training is held."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from datetime import datetime, timezone

import torch


root = Path.home() / 'orcd/pool/activation/taskaware'
pairs = [
    ('oracle_27b_16k_r16', 'encoder4b_oracle27b_16k_r16'),
    ('oracle_9b_16k_r8', 'oracle_9b_16k_r16'),
]
receipts = []
torch.set_num_threads(2)
for source_run, destination_run in pairs:
    relative = Path('AC_MODELS/rag_poc/no_context_reference.pt')
    source = root / source_run / relative
    destination = root / destination_run / relative
    assert not (root / 'reports' / destination_run / 'training_config.json').exists(), 'Destination already started'
    assert not destination.exists(), 'Never replace an existing cache'
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.base_reference_', dir=destination.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        shutil.copyfile(source, temporary)
        data = torch.load(temporary, map_location='cpu', weights_only=False)
        identity = data['identity']
        assert identity['model'] == 'Qwen/Qwen3.5-9B'
        assert identity['reference'] == 'base_no_context'
        assert identity['revision'] == 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
        assert data['top_k'] == 32
        assert len(data['targets']) == 16000
        assert data['targets'].keys() == data['target_logp'].keys()
        for key, tensors in data['targets'].items():
            assert key[0] == 'rag_poc#base' and key[2] == 32
            assert len(tensors) == 3 and all(isinstance(t, torch.Tensor) for t in tensors)
            assert tensors[0].shape == tensors[1].shape
            assert tensors[0].shape[0] == tensors[2].numel() == data['target_logp'][key].numel()
            assert all(torch.isfinite(t).all() for t in (*tensors, data['target_logp'][key]))
        source_hash = hashlib.file_digest(source.open('rb'), 'sha256').hexdigest()
        copy_hash = hashlib.file_digest(temporary.open('rb'), 'sha256').hexdigest()
        assert source_hash == copy_hash
        with temporary.open('rb') as f:
            os.fsync(f.fileno())
        # Publish atomically without overwriting; the source remains an independent file.
        os.link(temporary, destination)
        receipt = dict(source=str(source), destination=str(destination), sha256=copy_hash,
                       bytes=destination.stat().st_size, identity=identity,
                       entries=len(data['targets']), top_k=data['top_k'],
                       utc=datetime.now(timezone.utc).isoformat())
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    finally:
        temporary.unlink(missing_ok=True)
report = root / 'reports/base_reference_reuse.json'
staged = report.with_suffix('.next.json')
staged.write_text(json.dumps(receipts, indent=2) + '\n')
staged.replace(report)
