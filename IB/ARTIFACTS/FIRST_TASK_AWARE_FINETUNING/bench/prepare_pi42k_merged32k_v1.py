"""Freeze only subset metadata; reuse the native fresh8k material and Adam5250 pair."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from run_qa_pi42k_merged32k_v1 import json_digest, selected_material

MATERIAL_SHA = '49f711e3eff25b0063f148f850e73ff4fe7917c05dad0898db0167623d7e11a9'
BASELINE_SHA = '31d422fa93d74d07daa63dee9265d6897e2b03498bd4b0c2e5f1babe165b42ab'
SELECTION_SHA = '469320c02db9347d4d93b99c94003f9eb40d738d03ee272ff83a0129fda92051'
COUNTS = {'code':288, 'math_arxiv':115, 'math_textbook':77, 'financial':48, 'web':192,
          'technical_web':48, 'documents':77, 'legal':58, 'office':57}
NEW_HELPERS = ('run_qa_pi42k_merged32k_v1.py', 'prepare_pi42k_merged32k_v1.py',
               'train_pi42k_merged32k_v1.sbatch', 'train_pi42k_merged32k_csail_v1.sbatch')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def interleave(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['domain'], []).append(row)
    return [row for _, _, row in sorted(((i/len(values), domain, row)
        for domain, values in groups.items() for i, row in enumerate(values)), key=lambda x:(x[0],x[1]))]


def select_subset(material, original_selection):
    """Exact family quotas, mandatory original learning32, remaining frozen-order prefix."""
    if len(material['train']) != 8000 or len(material['held']) != 400:
        raise ValueError('Expected unchanged fresh8000/held400 material')
    held = interleave(material['held'])
    if [row['example']['example_id'] for row in held] != original_selection['original_held_ids']:
        raise ValueError('Original400 order differs')
    learning = material['learning_ids']
    if len(learning) != 32 or len(set(learning)) != 32:
        raise ValueError('Expected32 unique frozen learning questions')
    selected = set()
    for family, count in COUNTS.items():
        rows = [row for row in material['train'] if row['source_family'] == family]
        mandatory = [row for row in rows if row['example']['example_id'] in learning]
        remaining = [row for row in rows if row['example']['example_id'] not in learning]
        if len(mandatory) > count or len(rows) < count:
            raise ValueError('Cannot satisfy quota with learning32: ' + family)
        selected.update(row['example']['example_id'] for row in mandatory + remaining[:count-len(mandatory)])
    train = interleave([row for row in material['train'] if row['example']['example_id'] in selected])
    subset = {'train':[row['example']['example_id'] for row in train],
              'held':original_selection['selected_ids'], 'learning_ids':learning,
              'source_material_sha256':MATERIAL_SHA, 'development_selection_sha256':SELECTION_SHA,
              'training_by_family':dict(Counter(row['source_family'] for row in train)),
              'selection':'Keep exact learning32; per-family remaining frozen8k prefix; interleave train only.'}
    if subset['training_by_family'] != COUNTS or len(selected) != 960:
        raise ValueError('Quota selection incomplete or duplicated')
    view = selected_material(material, subset)
    if set(subset['train']) & set(subset['held']):
        raise ValueError('Development question entered training')
    if view['retrieval'] is not material['retrieval']:
        raise AssertionError('Retrieval pool changed')
    return subset


def main(args):
    for path, expected in ((args.material, MATERIAL_SHA), (args.baseline_design, BASELINE_SHA),
                           (args.selection, SELECTION_SHA)):
        if sha(path) != expected:
            raise ValueError('Frozen input changed: ' + str(path))
    subset_path=args.output/'subset.json';design_path=args.output/'PI42K_MERGED32K_FRESH960_DESIGN_V1.json'
    if subset_path.exists() or design_path.exists():
        raise FileExistsError('Never overwrite prepared/frozen selection')
    material=json.loads(args.material.read_text());original=json.loads(args.selection.read_text())
    subset=select_subset(material, original)
    base=json.loads(args.baseline_design.read_text());design=deepcopy(base);design.pop('root_review',None)
    if args.relocated_initial_pair:
        if not args.original_initial_pair or not args.remote_initial_pair:
            raise ValueError('Relocation requires original receipt and explicit remote pair path')
        if sha(args.original_initial_pair)!=base['initial_pair_sha256']:
            raise ValueError('Original initialization receipt changed')
        original_pair=json.loads(args.original_initial_pair.read_text())
        relocated=json.loads(args.relocated_initial_pair.read_text())
        if set(relocated)-set(original_pair) not in (set(),{'deployment'}):
            raise ValueError('Unexpected fields added during initialization relocation')
        unchanged=lambda value:{k:v for k,v in value.items() if k not in ('ac_checkpoint','reader_checkpoint','deployment')}
        if unchanged(original_pair)!=unchanged(relocated):
            raise ValueError('Relocation changed scientific initialization metadata')
        if (relocated['ac_checkpoint']!=str(args.remote_initial_pair.parent/'ac')
                or relocated['reader_checkpoint']!=str(args.remote_initial_pair.parent/'reader')):
            raise ValueError('Relocated pair paths differ from staged host layout')
        design.update(initial_pair_path=str(args.remote_initial_pair),
            initial_pair_sha256=sha(args.relocated_initial_pair),
            original_initial_pair_sha256=base['initial_pair_sha256'],
            relocation='Only checkpoint paths changed; payload and populated Adam5250 hashes unchanged')
    elif args.remote_initial_pair:
        raise ValueError('A remote pair override requires its validated relocated receipt')
    if args.remote_material:design['material_path']=str(args.remote_material)
    for key in ('parent_readout_path','parent_readout_sha256','selection_receipt_sha256'):
        design.pop(key,None)
    bench=Path(__file__).parent
    for leaf,expected in base['helper_sha256'].items():
        if sha(bench/leaf)!=expected:raise ValueError('Original helper changed: '+leaf)
    design['helper_sha256'].update({leaf:sha(bench/leaf) for leaf in NEW_HELPERS})
    design.update(status='prepared_awaiting_root_review', run_name='pi42k_merged32k_fresh960_v1',
        builder_sha256=sha(bench/NEW_HELPERS[0]), questions=960, epochs=1,
        subset_path=str(args.remote_output/'subset.json'), baseline_design_sha256=BASELINE_SHA,
        run_root=str(args.remote_output.parent/'pi42k_merged32k_fresh960_v1'),
        material_ids_sha256=json_digest({split:subset[split] for split in ('train','held')}),
        questions_previously_seen=False, unique_questions_after_stage=42960,
        total_presentations_after_stage=42960, training_by_family=COUNTS,
        learning32='Unchanged original32 learning IDs from frozen fresh8000; all included in960',
        training_exposure='960 fresh questions once;42000→42960 unique questions',
        optimizer='Actual inherited Adam5250→5370; stage clock0→120; constant5e-6/2e-6',
        reporting='Held original100 (50/50) and learning32 every96 new questions; initial frozen reader controls and final current controls only; no inline decoding',
        initial_baseline='Measured locally from exact paired42k initialization before any update; no external readout dependency',
        final_selection='Own completed120-update epoch001 pair and Adam5370 only; no best/partial substitution',
        evidence_tokens=32768, max_passages=64, passage_parts=False,
        allocation_candidate='RTX,64GiB host,3h wall;9000s trainer bound; separate final100 readout',
        readout={'selection_sha256':SELECTION_SHA, 'scope':'Same original100 at merged32k; final panel replay and actual AC answers; no confirmation'})
    design['training_config'].update(micro_batch_examples=1,gradient_checkpointing_min_tokens=0,
        schedule_total_updates=120,checkpoint_every_updates=24,max_duration_seconds=9000)
    args.output.mkdir(parents=True,exist_ok=True)
    with subset_path.open('x') as handle:json.dump(subset,handle,indent=2);handle.write('\n')
    design['subset_sha256']=sha(subset_path)
    with design_path.open('x') as handle:json.dump(design,handle,indent=2,ensure_ascii=False);handle.write('\n')
    print(json.dumps({'status':design['status'],'design':str(design_path),'design_sha256':sha(design_path),
                     'subset':str(subset_path),'subset_sha256':sha(subset_path),'questions':960},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('material','baseline-design','selection','output','remote-output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('remote-material','original-initial-pair','relocated-initial-pair','remote-initial-pair'):
        parser.add_argument('--'+name,type=Path)
    main(parser.parse_args())
