"""CPU-compute-only candidate: half of familiar8k material plus independent Adam5250 pair."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil

import evaluate_continuation_10k_v1 as continuation
import evaluate_strong_long_cohort_v1 as long
from prepare_continuation_10k import assert_exact

SOURCE_MATERIAL_SHA='9a9bb6c1a8e96931983eb606a8b793ae5867ba75b07635fbf4fc213319c9fdad'
COUNTS={'code':1200,'math_arxiv':480,'math_textbook':320,'financial':200,'web':800,
        'technical_web':200,'documents':320,'legal':240,'office':240}
PARENTS=('continue32k_10k_final_lr','continue32k_10k_rewarm')
sha,digest,read,write=long.file_sha,long.digest,long.read,long.write


def select_familiar(material):
    """Exact half-family quotas; keep learning32, use existing interleave order for remainder."""
    ordered=long.interleave(material['train']);groups=defaultdict(list)
    for row in ordered:groups[row['source_family']].append(row)
    if set(groups)!=set(COUNTS) or any(len(groups[k])!=2*n for k,n in COUNTS.items()):
        raise ValueError('Expected the frozen8000-question nine-family mix')
    learning=set(material['learning_ids'])
    if len(learning)!=32:raise ValueError('Expected original32 learning IDs')
    chosen=set()
    for family,count in COUNTS.items():
        mandatory=[r for r in groups[family] if r['example']['example_id'] in learning]
        rest=[r for r in groups[family] if r['example']['example_id'] not in learning]
        if len(mandatory)>count:raise ValueError('Learning32 exceeds family quota')
        chosen.update(r['example']['example_id'] for r in mandatory+rest[:count-len(mandatory)])
    rows=[r for r in material['train'] if r['example']['example_id'] in chosen]
    if len(rows)!=4000 or len(chosen)!=4000 or not learning<=chosen:
        raise ValueError('Incomplete/duplicate familiar4000 selection or lost learning32')
    if chosen & {r['example']['example_id'] for r in material['held']}:
        raise ValueError('Development IDs entered the training subset')
    return rows


def reset_stage_clock(parent, *, expected_states=615):
    """Keep the exact nested optimizer object; only the two outer stage fields change."""
    if parent.get('updates_done')!=1250 or parent.get('epoch_number')!=1:
        raise ValueError('Expected completed1250-update continuation optimizer')
    saved=parent['optimizer'];states=saved['state']
    if (len(saved['param_groups'])!=2 or len(states)!=expected_states
            or any(not {'step','exp_avg','exp_avg_sq'}.issubset(v) for v in states.values())
            or {float(v['step']) for v in states.values()}!={5250}):
        raise ValueError('Expected actual populated Adam5250 states, not original4000 or fresh Adam')
    result=dict(parent);result['updates_done']=0;result['epoch_number']=0
    return result


def derive_pair(pair,destination):
    import torch
    torch.set_num_threads(2)
    source_files={};source_roots={'ac':Path(pair['ac_checkpoint']),'reader':Path(pair['reader_checkpoint'])}
    for label,source in source_roots.items():
        for path in sorted(source.rglob('*')):
            if path.is_symlink():raise ValueError('Linked input is not an independent frozen pair')
            if path.is_file():source_files[label+'/'+str(path.relative_to(source))]={'sha256':sha(path),'bytes':path.stat().st_size}
        shutil.copytree(source,destination/label,copy_function=shutil.copy2)
    for relative,info in source_files.items():
        label,rest=relative.split('/',1);original=source_roots[label]/rest;copied=destination/relative
        if (copied.stat().st_dev,copied.stat().st_ino)==(original.stat().st_dev,original.stat().st_ino) or sha(copied)!=info['sha256']:
            raise ValueError('Derived payload is linked or changed')
    optimizer_path=destination/'ac/optimizer.pt'
    parent=torch.load(optimizer_path,map_location='cpu',weights_only=False)
    derived=reset_stage_clock(parent)
    if [len(g['params']) for g in parent['optimizer']['param_groups']]!=[374,256]:
        raise ValueError('Actual parent parameter groups changed')
    temporary=optimizer_path.with_suffix('.next.pt');torch.save(derived,temporary)
    check=torch.load(temporary,map_location='cpu',weights_only=False)
    if set(parent)!=set(check) or check['updates_done']!=0 or check['epoch_number']!=0:
        raise ValueError('Only the two stage clock fields may change')
    for key in parent:
        if key not in ('updates_done','epoch_number'):assert_exact(parent[key],check[key],key)
    temporary.replace(optimizer_path)
    for relative,info in source_files.items():
        label,rest=relative.split('/',1)
        if sha(source_roots[label]/rest)!=info['sha256']:raise ValueError('Original parent changed')
        if relative!='ac/optimizer.pt' and sha(destination/relative)!=info['sha256']:
            raise ValueError('Copied non-optimizer payload changed')
    return {'parent_files':source_files,'derived_files':{relative:{'sha256':sha(destination/relative),'bytes':(destination/relative).stat().st_size} for relative in source_files},
        'adam_state_preserved_byte_exact':True,'param_groups_preserved_exact':True,'weight_bytes_unchanged':True,
        'independent_files_not_hardlinks':True,'optimizer_parameter_states':615,'adam_parameter_steps':5250,
        'parent_stage_updates_done':1250,'parent_lifetime_updates_done':5250,'stage_updates_done':0,'stage_epoch_number':0}


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Actual material/optimizer transforms require an allocated CPU compute job')
    if args.output.exists():raise FileExistsError('Never overwrite a prepared candidate')
    if sha(args.material)!=SOURCE_MATERIAL_SHA:raise ValueError('Wrong familiar8k source material')
    cdesign=continuation.validate_design(args.continuation_design)
    if sha(args.continuation_initial_pair)!=cdesign['initial_pair_sha256']:raise ValueError('Original continuation initialization changed')
    pair,hashes=continuation.final_pair(args.parent_arm,args.reports,cdesign,args.continuation_design,read(args.continuation_initial_pair))
    baseline=read(args.parent_readout);ident=baseline['identity']
    if (baseline.get('status')!='complete' or ident.get('condition')!=args.parent_arm or ident.get('context')!='8k'
            or ident.get('paired_checkpoint')!=pair or ident.get('checkpoint_sha256')!=hashes
            or ident.get('passage_parts') is not True or ident.get('encode_batch_max_tokens')!=65536
            or len(baseline['scored_ids'])!=400 or len(baseline['rows'])!=100
            or not all(baseline['views'][k]['complete'] for k in ('AC','full context','no context'))):
        raise ValueError('Requires the completed exact42k8k400/100 reference before preparation')
    material=read(args.material)
    preserved={k:digest(material[k]) for k in ('held','held_cohorts','retrieval','learning_ids')}
    material['train']=select_familiar(material);material['planned_epochs']=1;material['whole_run_panels']=True
    parent_train_ids=set(read(args.reports/args.parent_arm/'material_ids.json')['train'])
    if not {row['example']['example_id'] for row in material['train']}<=parent_train_ids:
        raise ValueError('A selected question is not familiar to this parent continuation')
    material['recipe']={**material['recipe'],'adaptation':{
        'source_material_sha256':SOURCE_MATERIAL_SHA,'training_exposure':'4000 familiar questions ×1;42k unique,46k total presentations after stage',
        'selection':'Half of each source-family quota; preserve learning32; remaining rows follow existing interleave; retain source order before runner interleave.',
        'initialization':'Completed42k '+args.parent_arm+' paired weights and actual Adam5250; only stage clock resets; constant5e-6/2e-6.'}}
    if any(digest(material[k])!=value for k,value in preserved.items()):raise AssertionError('Frozen held/retrieval/learning data changed')
    args.output.mkdir(parents=True)
    material_path=args.output/'material.json'
    with material_path.open('x') as handle:json.dump(material,handle,ensure_ascii=False)
    optimizer=derive_pair(pair,args.output/'initial')
    initial={'ac_checkpoint':str(args.output/'initial/ac'),'reader_checkpoint':str(args.output/'initial/reader'),
        'payload_sha256':hashes,'parent_pair':pair,'optimizer_provenance':optimizer,
        'parent_readout_sha256':sha(args.parent_readout),'material_sha256':sha(material_path)}
    initial_path=args.output/'initial/pair.json';write(initial_path,initial)
    # Complete base-only cache is an optional acceleration; exact requested keys/header remain checked by the trainer.
    cache=Path(pair['ac_checkpoint']).parent/'no_context_reference.pt'
    if cache.is_file():
        shutil.copy2(cache,args.output/'initial/no_context_reference.pt')
        if sha(cache)!=sha(args.output/'initial/no_context_reference.pt'):raise ValueError('Base-reference cache copy differs')
    original=read(Path(__file__).with_name('STRONG_LONG_ADAPTATION_DESIGN_V1.json'))
    config=dict(original['training_config']);config.update(learning_rate_ac=5e-6,learning_rate_target_lora=2e-6,
        warmup_updates=0,lr_schedule='constant',lr_final_fraction=1.,resume_optimizer=True,resume_skip_steps=0,
        schedule_total_updates=500,num_epochs=1,max_duration_seconds=17100)
    ids={split:[row['example']['example_id'] for row in long.interleave(material[split])] for split in ('train','held')}
    if ids['held']!=baseline['scored_ids']:raise ValueError('Original400 saved order changed')
    helpers=set(original['helper_sha256'])|{'run_qa_gentle42k_long_v1.py','prepare_gentle42k_long_v1.py','train_gentle42k_long_v1.sbatch',
        'evaluate_continuation_10k_v1.py','evaluate_continuation_long_8k_v1.py','prepare_continuation_10k.py',
        'evaluate_gentle42k_long_v1.py','evaluate_gentle42k_long_v1.sbatch'}
    design={'version':1,'status':'prepared_awaiting_parent_decision','run_name':'gentle42k_'+args.parent_arm.removeprefix('continue32k_10k_')+'_familiar4k_8k','parent_condition':args.parent_arm,'material_sha256':sha(material_path),
        'material_ids_sha256':digest(ids),'material_path':str(material_path),'initial_pair_sha256':sha(initial_path),
        'initial_pair_path':str(initial_path),'initial_payload_sha256':hashes,'source_sha256':long.SOURCE,
        'parent_readout_path':str(args.parent_readout),'parent_readout_sha256':sha(args.parent_readout),
        'encode_batch_max_tokens':65536,'builder_sha256':sha(Path(__file__).with_name('run_qa_gentle42k_long_v1.py')),
        'helper_sha256':{leaf:sha(Path(__file__).with_name(leaf)) for leaf in sorted(helpers)},
        'training_config':config,'questions':4000,'epochs':1,'training_by_family':COUNTS,'learning32_preserved':True,
        'parent_unique_questions':42000,'unique_questions_after_stage':42000,'total_presentations_after_stage':46000,
        'training_exposure':'FAMILIAR questions; context/layout adaptation, not fresh-data continuation',
        'optimizer':'Preserved Adam moments and per-parameter steps5250; stage clock0→500; constant floorLR5e-6/2e-6',
        'reporting':'AC400+learning32 each400 stage questions; frozen initial42k controls; own moving controls0/50/100; same original100 eventual8k answers',
        'final_selection':'Only own completed500-update pair;32k NLL optional after reviewing8k results.',
        'allocation_candidate':'One RTX,64GiB host,5h wall;17100s training bound; paired manual resume supported.',
        'readout':{'selection_sha256':original['readout']['selection_sha256'],
            'initial':'Completed selected42k parent: same8k400 scores and fixed100 actual AC answers.',
            'final':'Own completed500-update pair:8k400 NLL then the same100 AC answers;32k NLL only if justified after review.',
            'semantic_scope':'Familiar-question context/layout adaptation; no isolated layout or fresh-data claim.',
            'confirmation':'No reserved confirmation read.'},'preserved_material':preserved}
    write(args.output/'GENTLE42K_LONG_DESIGN_V1.json',design)
    (args.output/'PRESERVE').write_text('Prepared familiar4000 context adaptation; independent inherited-Adam parent copy. No launch until frozen decision.\n')
    print(json.dumps({'status':design['status'],'run_name':'gentle42k_'+args.parent_arm.removeprefix('continue32k_10k_')+'_familiar4k_8k','parent_condition':args.parent_arm,'material_sha256':design['material_sha256'],
        'initial_pair_sha256':design['initial_pair_sha256'],'actual_adam_steps':5250,'training_exposure':design['training_exposure']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('material','continuation-design','continuation-initial-pair','reports','parent-readout','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--parent-arm',choices=PARENTS,required=True)
    main(parser.parse_args())
