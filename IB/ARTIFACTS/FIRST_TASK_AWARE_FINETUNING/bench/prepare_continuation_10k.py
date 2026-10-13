"""Freeze fresh continuation data and a copied final pair with preserved Adam state."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import unicodedata

from prepare_model_comparison import interleave, length_balanced_order, request_id, summarize
from select_bank import quotas, read_batch, training_row

SEED = 20260926
ITEMS = 10000


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def normalized(text):
    return unicodedata.normalize('NFC', text).strip().casefold()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def select_fresh(base, bank, reservations):
    old_ids = {row['example']['example_id'] for row in base['train']}
    old_questions = {normalized(row['example']['query']) for row in base['train']}
    protected = {normalized(row['example']['query']) for row in base['held']}
    protected_requests = {request_id(row) for row in base['held']}
    reservation_receipts = []
    for path in reservations:
        data = json.loads(path.read_bytes())
        if set(data) != {'questions'}:
            raise ValueError('Only question-only reservation lists are accepted: ' + str(path))
        protected.update(normalized(q) for q in data['questions'])
        reservation_receipts.append({'path': str(path), 'sha256': sha(path), 'questions': len(data['questions'])})
    bank_bytes = (bank / 'manifest.json').read_bytes()
    manifest = json.loads(bank_bytes)
    shares = manifest['source_shares']
    if shares != base['recipe']['source_shares']:
        raise ValueError('Source mix differs from parent')
    seen_ids, seen_questions, candidates = set(), set(), collections.defaultdict(list)
    excluded = collections.Counter()
    # Close reserved requests over all batches before selecting: a reserved sibling
    # can occur after an otherwise eligible question. Never inspect its gold answer.
    for entry in manifest['batches']:
        path = bank / entry['file']
        if sha(path) != entry['sha256']:
            raise ValueError('Changed published batch: ' + str(path))
        for raw in read_batch(path):
            if 'example' in raw:
                ident, query = raw['example']['example_id'], raw['example']['query']
                req = request_id(raw)
            else:
                ident, query = raw['id'], raw['question']
                req = raw.get('provenance', {}).get('source_request_id', ident)
            key = normalized(query)
            if key in protected:
                protected_requests.add(req)
                excluded['reserved_question'] += 1
                continue
            if ident in old_ids or key in old_questions:
                excluded['parent_training_question'] += 1
                continue
            if ident in seen_ids or key in seen_questions:
                excluded['duplicate_question'] += 1
                continue
            row = training_row(raw)
            fields = [query, *row['example']['gold_answers']]
            if any(ord(char) < 32 and char not in '\n\t' for field in fields for char in field):
                excluded['corrupted_qa'] += 1
                continue
            seen_ids.add(ident)
            seen_questions.add(key)
            candidates[row['source_family']].append(row)
    eligible = {family: [row for row in candidates[family] if request_id(row) not in protected_requests]
                for family in shares}
    wanted = quotas(ITEMS, shares)
    counts = {family: len(rows) for family, rows in eligible.items()}
    receipt = {'bank_manifest_sha256': hashlib.sha256(bank_bytes).hexdigest(),
               'published_questions': manifest['total_questions'], 'input_batches': manifest['batches'],
               'source_shares': shares, 'eligible_by_family': counts, 'required_by_family': wanted,
               'reserved_questions': len(protected), 'reserved_requests': len(protected_requests),
               'reservation_files': reservation_receipts, 'excluded': dict(excluded),
               'selection_helpers': 'prepare_model_comparison interleave/length_balanced_order/request_id/summarize; select_bank quotas/training_row',
               'normalization': 'NFC + strip + casefold for exclusion only; questions/gold/source unchanged',
               'split_policy': 'Exclude all parent training IDs/questions; reserve whole synthesis requests of held/reserved questions. Other siblings of old training questions remain eligible, matching the existing training split policy.'}
    missing = {family: {'needed': n, 'available': counts[family]} for family, n in wanted.items() if counts[family] < n}
    if missing:
        return None, {**receipt, 'status': 'insufficient', 'missing': missing}
    selected = {family: length_balanced_order(eligible[family], f'27b:continuation10k:{family}')[:n]
                for family, n in wanted.items()}
    train = interleave(selected)
    ids = [row['example']['example_id'] for row in train]
    assert len(ids) == len(set(ids)) == len({normalized(row['example']['query']) for row in train}) == ITEMS
    assert not set(ids) & old_ids
    assert not {normalized(row['example']['query']) for row in train} & (old_questions | protected)
    assert not {request_id(row) for row in train} & protected_requests
    assert dict(collections.Counter(row['source_family'] for row in train)) == wanted
    learning = interleave({family: random.Random(f'{SEED}:{family}:learning').sample(selected[family], n)
                           for family, n in quotas(32, shares).items()})
    return {'train': train, 'learning_ids': [row['example']['example_id'] for row in learning]}, {
        **receipt, 'status': 'complete', 'training': summarize(train), 'seed': SEED,
        'ordered_training_ids_sha256': canonical(ids), 'learning_questions': len(learning)}


def assert_exact(left, right, path='optimizer'):
    """Compare tensors by bytes, including signed zero/NaNs, and nested scalar metadata."""
    import torch
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor) and left.dtype == right.dtype and left.shape == right.shape, path
        assert left.device.type == right.device.type == 'cpu', path
        assert torch.equal(left.contiguous().reshape(-1).view(torch.uint8), right.contiguous().reshape(-1).view(torch.uint8)), path
    elif isinstance(left, dict):
        assert isinstance(right, dict) and list(left) == list(right), path
        for key in left:
            assert_exact(left[key], right[key], f'{path}/{key}')
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            assert_exact(a, b, f'{path}/{i}')
    else:
        assert type(left) is type(right) and left == right, path


def derive_pair(source_ac, source_reader, destination, parent_pin):
    import torch
    torch.set_num_threads(2)
    source_files = {}
    for name, source in [('ac', source_ac), ('reader', source_reader)]:
        for path in sorted(source.rglob('*')):
            if path.is_symlink():
                raise ValueError('Linked input in source pair: ' + str(path))
            if path.is_file():
                relative = f'{name}/{path.relative_to(source)}'
                source_files[relative] = {'sha256': sha(path), 'bytes': path.stat().st_size}
                if relative in parent_pin['files']:
                    assert source_files[relative] == parent_pin['files'][relative], relative
        shutil.copytree(source, destination / name)
    for relative, info in source_files.items():
        copy = destination / relative
        original = (source_ac if relative.startswith('ac/') else source_reader) / relative.split('/', 1)[1]
        assert copy.stat().st_ino != original.stat().st_ino or copy.stat().st_dev != original.stat().st_dev, relative
        assert sha(copy) == info['sha256'], relative
    optimizer_path = destination / 'ac/optimizer.pt'
    parent_state = torch.load(optimizer_path, map_location='cpu', weights_only=False)
    assert parent_state['updates_done'] == 4000 and parent_state['epoch_number'] == 1
    assert parent_state['optimizer']['state'] and len(parent_state['optimizer']['param_groups']) == 2
    stage_state = dict(parent_state)
    stage_state['updates_done'] = 0
    stage_state['epoch_number'] = 0
    temporary = optimizer_path.with_suffix('.next.pt')
    torch.save(stage_state, temporary)
    check = torch.load(temporary, map_location='cpu', weights_only=False)
    assert set(check) == set(parent_state)
    for key in parent_state:
        if key not in ('updates_done', 'epoch_number'):
            assert_exact(parent_state[key], check[key], key)
    assert check['updates_done'] == check['epoch_number'] == 0
    steps = collections.Counter(float(value['step'].item()) if isinstance(value['step'], torch.Tensor) else value['step']
                                for value in parent_state['optimizer']['state'].values())
    temporary.replace(optimizer_path)
    # Parent originals are rehashed after all derivation; no source mutation allowed.
    for relative, info in source_files.items():
        original = (source_ac if relative.startswith('ac/') else source_reader) / relative.split('/', 1)[1]
        assert sha(original) == info['sha256'], 'Parent changed: ' + relative
    derived_files = {relative: {'sha256': sha(destination / relative), 'bytes': (destination / relative).stat().st_size}
                     for relative in source_files}
    for relative in source_files:
        if relative != 'ac/optimizer.pt':
            assert derived_files[relative] == source_files[relative], relative
    return {'parent_files': source_files, 'derived_files': derived_files,
            'parent_training_questions': 32000, 'parent_updates_done': 4000, 'parent_epoch_number': 1,
            'stage_updates_done': 0, 'stage_epoch_number': 0,
            'adam_state_preserved_byte_exact': True, 'param_groups_preserved_exact': True,
            'adam_parameter_step_histogram': dict(steps),
            'optimizer_parameter_states': len(parent_state['optimizer']['state']),
            'parameter_groups': [len(g['params']) for g in parent_state['optimizer']['param_groups']],
            'weight_bytes_unchanged': True, 'independent_files_not_hardlinks': True,
            'ac_modules_saved_epoch_number': 1,
            'runner_requirement': 'Set loaded ac.epoch_number=0 for this stage; resume_optimizer=True and exact group-layout validation. Preserve Adam state and per-parameter steps. Stage schedule overrides group LRs at runtime.'}


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run preparation/hash/copy/torch on CPU compute only')
    root = args.root
    material_out, initial_out = root / 'continuation_10k_material', root / 'continuation_32k_initial'
    if material_out.exists() or initial_out.exists():
        raise FileExistsError('Continuation outputs already exist; never overwrite')
    pin = json.loads(args.parent_pin.read_bytes())
    report = root / 'reports/oracle_27b_32k'
    result, identity = json.loads((report / 'result.json').read_bytes()), json.loads((report / 'run_identity.json').read_bytes())['identity']
    assert sha(report / 'result.json') == pin['files']['result.json']['sha256']
    assert result['stop_reason'] == 'complete' and result['items'] == result['examples'] == 32000
    assert result['completed_global_updates'] == result['planned_updates'] == identity['planned_updates'] == 4000
    assert result['planned_epochs'] == identity['epochs'] == 1
    source_ac, source_reader = Path(result['checkpoint']), Path(result['target_lora_checkpoint'])
    assert str(source_ac) == pin['parent']['checkpoint'] and str(source_reader) == pin['parent']['target_lora_checkpoint']
    assert source_ac.name == source_reader.name == 'epoch_001'
    marker = source_ac.parent / 'completed_epoch.json'
    assert sha(marker) == pin['files']['completed_epoch.json']['sha256']
    assert json.loads(marker.read_bytes()) == {'epoch_number': 1, 'ac_checkpoint': 'AC_MODELS/rag_poc/epoch_001', 'target_lora_checkpoint': 'LORAS/rag_reader/epoch_001'}
    original = root / 'model_comparison_32k_material/material_27b_32000.json'
    assert sha(original) == pin['material_sha256'] == identity['material_sha256']
    base = json.loads(original.read_bytes())
    assert len(base['train']) == 32000 and len(base['held']) == 400
    reservations = [root / 'model_comparison_32k_material/reserved_questions.json',
                    root / 'model_comparison_material/reserved_questions.json', root / 'qa27b_reserved_questions.json']
    selection, recipe = select_fresh(base, root / 'reports/synthesis_bank_27b', reservations)
    if selection is None:
        write(root / 'bench' / f'continuation_10k_insufficient_{os.environ["SLURM_JOB_ID"]}.json', recipe)
        raise RuntimeError('Insufficient fresh mixed questions: ' + json.dumps(recipe['missing']))
    final_readout_path = root / 'reports/data_amount_cohort100/oracle_27b_32k.json'
    assert sha(final_readout_path) == pin['final_readout']['sha256']
    final_readout = json.loads(final_readout_path.read_bytes())
    assert final_readout['status'] == 'complete' and final_readout['identity']['update'] == 4000
    assert final_readout['identity']['condition'] == 'oracle_27b_32k'
    for key in ['shared_input_fingerprint', 'ac_input_fingerprint']:
        assert final_readout[key] == pin['final_readout'][key]
    payload_keys = {'ac_modules': 'ac/ac_modules.pt', 'encoder_config': 'ac/side_lora/adapter_config.json', 'encoder_weights': 'ac/side_lora/adapter_model.safetensors', 'reader_config': 'reader/adapter_config.json', 'reader_weights': 'reader/adapter_model.safetensors'}
    for key, relative in payload_keys.items():
        assert final_readout['identity']['checkpoint_sha256'][key] == pin['files'][relative]['sha256']
    recipe.update(parent_material=str(original), parent_material_sha256=sha(original),
                  parent_train_questions=32000, continuation_questions=ITEMS,
                  preserved_fields=['held', 'held_cohorts', 'retrieval'],
                  planned_epochs=1, selection='Fresh mixed questions only; no model response/loss selection. Both arms share exact material and order.')
    material = {**base, **selection, 'recipe': recipe, 'planned_epochs': 1, 'whole_run_panels': True}
    for key in recipe['preserved_fields']:
        assert material[key] == base[key]
    for key in ['held', 'held_cohorts', 'retrieval']:
        recipe[key + '_sha256'] = canonical(base[key])
    with tempfile.TemporaryDirectory(prefix='.continuation10k.', dir=root) as tmp:
        staging = Path(tmp)
        mat, init = staging / 'material', staging / 'initial'
        mat.mkdir(); init.mkdir()
        material_path = mat / 'material_27b_10000.json'
        with material_path.open('x') as handle:
            json.dump(material, handle, ensure_ascii=False)
            handle.flush(); os.fsync(handle.fileno())
        write(mat / 'summary.json', recipe)
        (mat / 'PRESERVE').write_text('Frozen shared continuation questions, held/retrieval inputs and order.\n')
        optimizer = derive_pair(source_ac, source_reader, init, pin)
        for name, model in [('ac/side_lora', 'Qwen/Qwen3.5-2B'), ('reader', 'Qwen/Qwen3.5-9B')]:
            config = json.loads((init / name / 'adapter_config.json').read_bytes())
            assert config['base_model_name_or_path'] == model and config['r'] == 128
        shutil.copy2(report / 'panel_04000.json', init / 'parent_panel.json')
        assert sha(init / 'parent_panel.json') == pin['files']['panel_04000.json']['sha256']
        write(init / 'parent_readout_receipt.json', {'source': str(final_readout_path), 'sha256': pin['final_readout']['sha256'], 'status': 'complete', 'identity': final_readout['identity'], 'shared_input_fingerprint': final_readout['shared_input_fingerprint'], 'ac_input_fingerprint': final_readout['ac_input_fingerprint']})
        provenance = {'parent': pin['parent'], 'parent_pin_sha256': sha(args.parent_pin),
                      'optimizer': optimizer, 'material': str(material_out / material_path.name),
                      'material_sha256': sha(material_path), 'original_initialization': 'P2 epoch1',
                      'new_stage_not_fresh_adam': True, 'job_id': os.environ['SLURM_JOB_ID']}
        write(init / 'pair.json', {'ac_checkpoint': str(initial_out / 'ac'), 'reader_checkpoint': str(initial_out / 'reader'), 'parent_panel_path': str(initial_out / 'parent_panel.json'), 'parent_shared_input_fingerprint': final_readout['shared_input_fingerprint'], 'parent_ac_input_fingerprint': final_readout['ac_input_fingerprint'], 'provenance': provenance})
        (init / 'PRESERVE').write_text('Shared final32k-derived pair; Adam preserved, trainer stage counters reset only.\n')
        helpers = {name: sha(Path(__file__).parent / name) for name in ['prepare_continuation_10k.py', 'prepare_model_comparison.py', 'prepare_data_budget.py', 'select_bank.py']}
        receipt = {'status': 'complete', 'job_id': os.environ['SLURM_JOB_ID'],
                   'material': provenance['material'], 'material_sha256': provenance['material_sha256'],
                   'pair': str(initial_out / 'pair.json'), 'pair_sha256': sha(init / 'pair.json'),
                   'parent_panel_sha256': sha(init / 'parent_panel.json'),
                   'parent_shared_input_fingerprint': final_readout['shared_input_fingerprint'],
                   'parent_ac_input_fingerprint': final_readout['ac_input_fingerprint'],
                   'source_mix': recipe['training']['by_family'], 'eligible_by_family': recipe['eligible_by_family'],
                   'ordered_training_ids_sha256': recipe['ordered_training_ids_sha256'],
                   'held_sha256': recipe['held_sha256'], 'held_cohorts_sha256': recipe['held_cohorts_sha256'],
                   'retrieval_sha256': recipe['retrieval_sha256'], 'learning_ids': selection['learning_ids'],
                   'optimizer': optimizer, 'helpers': helpers}
        write(mat / 'preparation_receipt.json', receipt)
        # These two create-only directories are complete before publication.
        # Consumers must require the receipt AND both paths/hashes before launch.
        mat.rename(material_out)
        init.rename(initial_out)
    print(json.dumps({k: v for k, v in receipt.items() if k not in ('optimizer', 'learning_ids')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-pin', type=Path, required=True)
    main(parser.parse_args())
