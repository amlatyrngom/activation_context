"""Paired short native2k LR diagnostic:42k/Adam5250 +960fresh×1; same L40S, own finals."""
import os,json,time,math,random,hashlib,shutil,argparse,gc,collections
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict,replace
ROOT=Path(os.environ.get('QA_SOURCE_ROOT','/workspace')); REPORT=Path(os.environ.get('QA_REPORT_ROOT',ROOT/'IB/TMP/SYNC/TASK_AWARE_QA'))
NAME,AC,SIDE,READER='qwen4b','rag_poc','rag_side','rag_reader'; SEED=20260921

def write(path,value):
    from activation.common.reporting import _write_atomic
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);_write_atomic(path,json.dumps(value,indent=2,default=str,allow_nan=False))

def harness(ac_path=None,reader_path=None,*,encode_batch_max_tokens=65536):
    import torch
    from activation.harness import HarnessRuntime,HarnessRuntimeConfig,ModelConfig
    from activation.ac_model import ActivationContextModelConfig
    torch.set_num_threads(4);torch.manual_seed(SEED)
    h=HarnessRuntime(HarnessRuntimeConfig(model_configs={NAME:ModelConfig(NAME,'Qwen/Qwen3.5-9B'),'side_base':ModelConfig('side_base','Qwen/Qwen3.5-2B')},doc_chunk_size_chars=12000,doc_chunk_overlap_chars=0))
    h.module_manager.register_lora(READER,NAME,rank=128,dropout=0,checkpoint_path=str(reader_path) if reader_path else None)
    h.module_manager.register_ac_model(ActivationContextModelConfig(AC,'side_base',SIDE,NAME,READER,min_view_rows=2,default_compression_ratio=1/16,
        checkpoint_path=str(ac_path) if ac_path else None,intervention_frequency=-1,input_pooling_stride=0,mixer_layers=2,
        encode_batch_max_tokens=encode_batch_max_tokens))
    return h

def answer_positions(tokenizer,ids,answer):
    text=tokenizer.decode(ids,skip_special_tokens=False)
    marker='<parameter=answer>\n';left=text.find(marker)
    if left<0:raise ValueError('QA bench expects the Qwen3.5 submit_answer renderer')
    left+=len(marker);right=text.index('\n</parameter>',left)
    normalizer=tokenizer.backend_tokenizer.normalizer
    expected=normalizer.normalize_str(answer) if normalizer is not None else answer
    assert text[left:right]==expected,(text,answer)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids,'Target rendering did not round-trip'
    positions=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<right and b>left]
    assert positions
    return positions

def allocate_passage_rows(source_tokens,total_rows,min_rows):
    """Proportional integer allocation with minimums; deterministic largest remainders."""
    if total_rows<len(source_tokens)*min_rows:
        raise ValueError(f'Merged budget has {total_rows} rows, but {len(source_tokens)} passages require at least {len(source_tokens)*min_rows} rows')
    weights=[max(1,n) for n in source_tokens];rows=[0]*len(weights);active=list(range(len(weights)));remaining=total_rows
    while active:
        mass=sum(weights[i] for i in active)
        desired={i:remaining*weights[i]/mass for i in active}
        clamped=[i for i in active if desired[i]<min_rows]
        if not clamped:break
        for i in clamped:rows[i]=min_rows;remaining-=min_rows;active.remove(i)
    for i in active:rows[i]=math.floor(desired[i])
    order=sorted(active,key=lambda i:(-(desired[i]-rows[i]),i))
    for i in order[:total_rows-sum(rows)]:rows[i]+=1
    assert sum(rows)==total_rows and all(n>=min_rows for n in rows)
    return rows

def material_items(h,material,*,passage_parts=False,evidence_tokens=2048,max_passages=4,ratio=1/16):
    from activation.dataset.dataset import DatasetDocument,LoadedDataset
    from activation.ac_model import ActivationContextStudyGenerator
    from activation.ac_model.ac_model_training import ActivationContextTrainingItem
    from activation.ac_model.ac_model_utils import assistant_target_ids,tokenize_with_parts
    from activation.ac_model.ac_model_study import RAG_QA_SYSTEM_PROMPT,RAG_QA_INSTRUCTIONS
    from activation.common.ac_parts import ac_part
    from activation.dataset.loaders.trajectory_utils import submit_answer_definition
    study=ActivationContextStudyGenerator(h,AC);dataset=LoadedDataset('qa_material')
    for row in material.get('retrieval',material['train']+material['held']):
        for unit in row['example']['study_source']['units']:
            dataset.documents[unit['doc_id']]=DatasetDocument(unit['doc_id'],dataset.dataset_id,text=unit['text'])
    h.dataset_manager.register_dataset(dataset);index=h.dataset_manager._get_or_create_index(dataset.dataset_id);index.build_bm25_index()
    candidate_token_counts={}
    def build(rows):
        rng=random.Random(SEED);items=[]
        ranked=index.bm25_query_many_frozen([r['example']['query'] for r in rows],top_k=max(12,max_passages*3),
                    excluded_doc_ids=[[u['doc_id'] for u in r['example']['study_source']['units']
                                      if u['doc_id'] in index.loaded_dataset.documents] for r in rows])
        for row,candidates in zip(rows,ranked):
            example=row['example'];question=example['query'];answer=example['gold_answers'][0].strip();source=example['study_source']
            passages=[row['long_context']] if row.get('long_context') else [u['text'] for u in source['units']]
            count=sum(study.count_tokens(p) for p in passages)
            # A bounded total evidence budget; gold is always retained, even if it alone exceeds it.
            for candidate in candidates:
                candidate_text=candidate.chunk_text
                n=candidate_token_counts.get(candidate_text)
                if n is None:
                    n=study.count_tokens(candidate_text);candidate_token_counts[candidate_text]=n
                if count+n<=evidence_tokens:passages.append(candidate.chunk_text);count+=n
                if len(passages)>=max_passages:break
            passage_rng=random.Random(f'{SEED}:{example["example_id"]}') if 'retrieval' in material else rng
            passage_rng.shuffle(passages);text='\n\n'.join(f'Passage {i+1}:\n{p}' for i,p in enumerate(passages))
            prefix=f'Task: {question}\n\nPassages:\n';suffix=f'\n\n{RAG_QA_INSTRUCTIONS}\nQuestion: {question}'
            part=ac_part([{'role':'user','content':f'Task: {question}'},{'role':'user','content':text}],AC,ratio)
            parts=([ac_part([{'role':'user','content':f'Task: {question}'},{'role':'user','content':f'Passage {i+1}:\n{p}'}],AC,ratio)
                    for i,p in enumerate(passages)] if passage_parts else [part])
            # Use exactly the tokenizer and row-budget calculation used by AC encoding.
            encoder=study.ac_model
            source_lengths=[len(tokenize_with_parts(encoder.side.tokenizer,p['messages'],[],encoder.pad_id,tools=p.get('tools'))[0]) for p in parts]
            passage_tokens=[len(encoder.side.tokenizer.encode(p,add_special_tokens=False)) for p in passages]
            actual_evidence_tokens=sum(passage_tokens)
            merged_tokens=(len(tokenize_with_parts(encoder.side.tokenizer,part['messages'],[],encoder.pad_id)[0])
                           if passage_parts else source_lengths[0])
            merged_rows=encoder.num_view_rows(merged_tokens,ratio)
            if passage_parts:
                allocation=allocate_passage_rows(passage_tokens,merged_rows,encoder.config.min_view_rows)
                for p,length,rows in zip(parts,source_lengths,allocation):
                    # compression_target is a ratio, not an absolute row count. Use the middle
                    # of the ceil interval to avoid a floating-point boundary adding a row.
                    p['compression_target']=(rows-.5)/length
            view_rows=[encoder.num_view_rows(length,p['compression_target']) for length,p in zip(source_lengths,parts)]
            assert sum(view_rows)==merged_rows,'Passage layout changed the total compressed-row budget'
            output=study.dialect.rendering.assistant_message('',[{'id':'answer','name':'submit_answer','arguments':{'answer':answer}}]);tools=[submit_answer_definition()]
            ids=assistant_target_ids(study.tokenizer,output,tools);system={'role':'system','content':RAG_QA_SYSTEM_PROMPT}
            full=[system,{'role':'user','content':prefix+text+suffix},output]
            compressed=[system,{'role':'user','content':[{'type':'text','text':prefix},*parts,{'type':'text','text':suffix}]},deepcopy(output)]
            items.append(ActivationContextTrainingItem(item_id=example['example_id'],kind=row['domain'],in_context_messages=full,ac_messages=compressed,
                in_context_start=2,ac_start=2,assistant_token_ids=[ids],tools=tools,dataset_id=row['domain'],doc_ids=row['original_doc_ids'],
                info={'answer':answer,'question':question,'answer_positions':answer_positions(study.tokenizer,ids,answer),'source_request':source['request_id'],
                      'passages_text':text,'passages':len(passages),'matched_full_context':True,'ratio':ratio,'source_tokens':count,'passage_parts':passage_parts,
                      'evidence_source_tokens':actual_evidence_tokens,'encoded_source_tokens':sum(source_lengths),'compressed_rows':sum(view_rows),
                      'part_source_tokens':source_lengths,'part_rows':view_rows,
                      'rows_per_evidence_token':sum(view_rows)/max(1,actual_evidence_tokens),'rows_per_encoded_source_token':sum(view_rows)/sum(source_lengths),
                      'merged_source_tokens':merged_tokens,'merged_compressed_rows':merged_rows,
                      'chat_template_kwargs':{'enable_thinking':False}}))
        return items
    return study,build(material['train']),build(material['held'])

def answer_nll(summary,items):
    if len(summary['per_item'])!=len(items):raise ValueError('Evaluation dropped or added reporting items')
    values=[]
    for row,item in zip(summary['per_item'],items):
        assert row['item_id']==item.item_id
        values.append(-sum(row['target_logp'][i] for i in item.info['answer_positions'])/len(item.info['answer_positions']))
    return sum(values)/len(values)

def cohort_answer_nll(summary,items,cohorts,original_items=None):
    """Split existing per-question measurements; no extra model evaluation."""
    if not cohorts:return {}
    originals=items if original_items is None else original_items
    if len(originals)!=len(items):raise ValueError('Reference view changed reporting item count')
    ids=[item.item_id for item in originals]
    members=[ident for group in cohorts.values() for ident in group]
    if not all(cohorts.values()) or len(members)!=len(set(members)) or set(members)!=set(ids):
        raise ValueError('Development cohorts must partition the reporting items exactly')
    if len(summary['per_item'])!=len(items):raise ValueError('Evaluation dropped or added reporting items')
    indexed={original.item_id:(row,item) for original,row,item in zip(originals,summary['per_item'],items)}
    if any(row['item_id']!=item.item_id for row,item in indexed.values()):
        raise ValueError('Evaluation item order changed')
    return {name:answer_nll({'per_item':[indexed[ident][0] for ident in group]},
                           [indexed[ident][1] for ident in group]) for name,group in cohorts.items()}

class ResumeError(ValueError):
    """A requested retry is incompatible or does not have a recoverable paired save."""

def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def validate_initial_references(record,identity,views):
    """A resumed reader can never silently replace the actual initial-reader controls."""
    if record.get('identity')!=identity or record.get('measured_at_update')!=0:
        raise ResumeError('Frozen initial-reader reference identity differs')
    if set(record.get('summaries',{}))!=set(views):
        raise ResumeError('Frozen initial-reader views are incomplete')
    for name,items in views.items():
        rows=record['summaries'][name]['per_item']
        if len(rows)!=len(items):raise ResumeError('Frozen reference item count differs')
        for row,item in zip(rows,items):
            if (row['item_id']!=item.item_id or len(row['target_logp'])!=len(item.assistant_token_ids[0])
                    or not all(math.isfinite(value) for value in row['target_logp'])):
                raise ResumeError('Frozen reference gold IDs/lengths or values differ')
    return record['summaries']

def validate_run(folder,identity,config,material_ids):
    """Validate before changing reports or loading models; legacy runs carry the same facts."""
    from dataclasses import asdict
    path=folder/'run_identity.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous['identity']!=identity:raise ResumeError('Resume settings/material differ from the original run')
    else:
        data=json.loads((folder/'report_data.json').read_text())
        recipe=json.loads(next(w['text'] for w in data['widgets'] if w['name']=='recipe'))
        for key,value in [('material_sha256',identity['material_sha256']),('epochs',identity['epochs'])]:
            if recipe.get(key)!=value:raise ResumeError(f'Legacy run {key} differs from resume request')
        if recipe.get('initialization')!=('fresh' if identity['arm']=='fresh_joint' else 'P2 epoch1'):
            raise ResumeError('Resume initialization differs from the original run')
        if identity['initial_pair_sha256'] is not None:raise ResumeError('Legacy run did not start from this explicit QA pair')
        for key,default in [('evidence_tokens',2048),('max_passages',4),('ratio',1/16)]:
            if recipe.get(key,default)!=identity[key]:raise ResumeError(f'Resume {key} differs from original run')
        layout=recipe.get('passage_layout','merged passages')
        if (layout=='one part per passage')!=identity['passage_parts']:raise ResumeError('Resume passage layout differs')
        previous={'identity':identity,'created_unix':None,'adopted_legacy_identity':True}
    saved=json.loads((folder/'training_config.json').read_text());wanted=asdict(config)
    # Duration is the attempt's safety limit. Every optimization/data-order setting stays fixed.
    for key,value in wanted.items():
        if key in ('max_duration_seconds','resume_optimizer','resume_skip_steps'):continue
        actual=saved.get(key)
        if key=='schedule_total_updates' and actual is None:actual=identity['planned_updates']
        if json_digest(actual)!=json_digest(value):raise ResumeError(f'Resume training config differs at {key}: {actual!r} vs {value!r}')
    if json.loads((folder/'material_ids.json').read_text())!=material_ids:
        raise ResumeError('Resume training or reporting IDs/order differ from the frozen run')
    return previous

def resume_checkpoint(root,steps_per_epoch,total_epochs):
    """Only published markers own recoverable pairs; never silently discard optimizer moments."""
    import torch
    root=Path(root).resolve();ac_root=root/'AC_MODELS'/AC
    candidates=[]
    for name,partial in [('partial_epoch.json',True),('completed_epoch.json',False)]:
        path=ac_root/name
        if not path.exists():continue
        pair=json.loads(path.read_text());epoch=int(pair['epoch_number']);step=int(pair['step']) if partial else steps_per_epoch
        if epoch<1 or epoch>total_epochs or not 0<=step<=steps_per_epoch or (partial and pair.get('steps')!=steps_per_epoch):
            raise ResumeError('Checkpoint epoch/step count disagrees with the frozen run')
        updates=(epoch-1)*steps_per_epoch+step
        candidates.append((updates,not partial,pair,path))
    if not candidates:raise ResumeError('No published partial/completed paired checkpoint exists')
    updates,complete,pair,marker=max(candidates,key=lambda value:(value[0],value[1]))
    ac_path=(root/pair['ac_checkpoint']).resolve();reader_path=(root/pair['target_lora_checkpoint']).resolve()
    if not ac_path.is_relative_to(root) or not reader_path.is_relative_to(root):raise ResumeError('Checkpoint marker points outside this run')
    for directory in (ac_path/'side_lora',reader_path):
        weights=list(directory.glob('adapter_model.*'))
        if not (directory/'adapter_config.json').is_file() or not weights or any(p.stat().st_size==0 for p in weights):
            raise ResumeError('Incomplete adapter checkpoint: '+str(directory))
    module_file=ac_path/'ac_modules.pt';optimizer_file=ac_path/'optimizer.pt'
    if any(not p.is_file() or p.stat().st_size==0 for p in (module_file,optimizer_file)):
        raise ResumeError('Missing or empty AC/optimizer file in published checkpoint')
    try:state=torch.load(optimizer_file,map_location='cpu',weights_only=False)
    except Exception as error:raise ResumeError('Cannot read optimizer checkpoint: '+str(optimizer_file)) from error
    saved_epoch=int(pair['epoch_number'])-(0 if complete else 1)
    if state.get('updates_done')!=updates or state.get('epoch_number')!=saved_epoch:
        raise ResumeError('Optimizer clock disagrees with the paired checkpoint marker')
    if not complete and state.get('step')!=pair['step']:raise ResumeError('Optimizer partial step disagrees with marker')
    if not isinstance(state.get('optimizer'),dict) or not state['optimizer'].get('param_groups'):
        raise ResumeError('Optimizer parameter groups are missing')
    if updates and not state['optimizer'].get('state'):
        raise ResumeError('Updated checkpoint has no optimizer moments; refusing a silent reset')
    first_epoch=int(pair['epoch_number'])+int(complete)
    skip=0 if complete else int(pair['step'])
    return {'updates':updates,'first_epoch':first_epoch,'skip_steps':skip,'remaining_epochs':max(0,total_epochs-first_epoch+1),
            'saved_epoch':saved_epoch,'ac_path':str(ac_path),'reader_path':str(reader_path),'marker':str(marker),
            'optimizer_state':state['optimizer']}

def validate_optimizer_layout(saved,groups):
    if [len(g['params']) for g in saved['param_groups']]!=[len(g) for g in groups]:
        raise ResumeError('Optimizer layout differs; refusing to restart with fresh moments')
    for prior,current in zip(saved['param_groups'],groups):
        for ident,parameter in zip(prior['params'],current):
            for key,value in saved['state'].get(ident,{}).items():
                if key!='step' and hasattr(value,'shape') and value.shape!=parameter.shape:
                    raise ResumeError('Optimizer moment shape differs from its parameter')

def archive_attempt(folder,resume,steps_per_epoch):
    """Keep the previous branch inspectable; active curves continue only from saved weights."""
    prior_numbers=[int(path.name[7:]) for path in (folder/'attempts').glob('resume_*') if path.name[7:].isdigit()]
    attempt=folder/'attempts'/f'resume_{max(prior_numbers,default=0)+1:03d}'
    attempt.mkdir(parents=True,exist_ok=False)
    for name in ('report_data.json','panels.json','result.json','selected_checkpoint.json'):
        path=folder/name
        if path.exists():shutil.copy2(path,attempt/name)
    panels=json.loads((folder/'panels.json').read_text()) if (folder/'panels.json').exists() else []
    retained=[row for row in panels if row['update']<=resume['updates']]
    for pattern in ('panel_*.json','answers_*.json','full_context_answers_*.json'):
        for path in folder.glob(pattern):
            if int(path.stem.rsplit('_',1)[1])>resume['updates']:shutil.move(str(path),attempt/path.name)
    previous=json.loads((folder/'report_data.json').read_text())
    widgets={w['name']:w for w in previous['widgets']}
    for name,widget in list(widgets.items()):
        if widget['type']=='line':
            limit=resume['updates']/steps_per_epoch if name in ('gold_nll','reporting','token_accuracy') else resume['updates']
            if name=='answer_nll' or name.startswith('cohort_'):limit=resume['updates']*8
            for series in widget['series']:
                kept=[(x,y) for x,y in zip(series['x'],series['y']) if x<=limit]
                series['x']=[x for x,_ in kept];series['y']=[y for _,y in kept]
        elif widget['type']=='table':
            if name=='panels':widget['rows']=[row for row in widget['rows'] if row['questions seen']<=resume['updates']*8]
            else:widget['rows']=[row for row in widget['rows'] if row.get('global update',row.get('update',0))<=resume['updates']]
        elif name.startswith(('completions_update_','answer_scores_')) and int(name.rsplit('_',1)[1])>resume['updates']:
            widgets.pop(name)
        elif name.startswith('epoch_') and widget['type']=='text' and json.loads(widget['text']).get('progress',{}).get('global_step',0)>resume['updates']:
            widgets.pop(name)
    if 'training_summary' in widgets:widgets['training_summary']['title']='Previous attempt summary'
    selected=folder/'selected_checkpoint.json'
    if selected.exists() and json.loads(selected.read_text())['update']>resume['updates']:
        record=json.loads(selected.read_text());record['retained_from_prior_attempt']=str(attempt.relative_to(folder))
        write(selected,record)  # These saved weights remain a valid development candidate.
    return attempt,retained,widgets


def whole_run_progress(progress,resume,total_epochs):
    if progress.epoch==0 and resume:
        return replace(progress,epoch=resume['first_epoch'],epoch_number=resume['first_epoch'],epochs=total_epochs,step=resume['skip_steps'])
    return replace(progress,epoch=progress.epoch_number if progress.epoch else 0,epochs=total_epochs)

def check_inherited_optimizer(ac_path, parameter_groups):
    import torch
    state=torch.load(Path(ac_path)/'optimizer.pt',map_location='cpu',weights_only=False)
    if state.get('updates_done')!=0 or state.get('epoch_number')!=0:
        raise ResumeError('Prepared inherited Adam must start only its stage clock at0')
    validate_optimizer_layout(state['optimizer'],parameter_groups)
    values=state['optimizer']['state']
    if len(values)!=615 or any(not {'step','exp_avg','exp_avg_sq'}.issubset(v) for v in values.values()):
        raise ResumeError('Expected615 complete inherited Adam states')
    steps=[float(v['step']) for v in values.values()]
    if min(steps)!=5250 or max(steps)!=5250:
        raise ResumeError('Expected actual42k parent Adam clocks5250, not fresh or original32k state')
    return {'optimizer_states':615,'min_parameter_step':5250,'max_parameter_step':5250,'stage_updates_done':0,
        'parent_unique_questions':42000,'training_exposure':'960 fresh questions once;42960 unique after completion'}


def selected_material(material, selection):
    """Select only predeclared IDs; retain original gold, learning32 and retrieval objects."""
    selected = dict(material)
    for split, count in (('train', 960), ('held', 100)):
        ids = selection[split]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError('Wrong frozen subset size or duplicate IDs: ' + split)
        by_id = {row['example']['example_id']: row for row in material[split]}
        if not set(ids) <= by_id.keys():
            raise ValueError('Subset contains foreign IDs: ' + split)
        selected[split] = [by_id[ident] for ident in ids]
    if len(material['learning_ids']) != 32 or not set(material['learning_ids']) <= set(selection['train']):
        raise ValueError('Original learning32 must remain in fresh960')
    if selection['learning_ids'] != material['learning_ids']:
        raise ValueError('Learning32 order changed')
    selected['held_cohorts'] = {bank: [ident for ident in selection['held'] if ident in group]
                              for bank, group in material['held_cohorts'].items()}
    if sorted(map(len, selected['held_cohorts'].values())) != [50, 50]:
        raise ValueError('Expected original50/50 development cohorts')
    return selected


def main(args):
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig,ActivationContextTrainingReporter
    folder=REPORT/(args.run_name or args.arm);folder.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    existing=(folder/'training_config.json').exists()
    if args.arm!='joint' or args.items!=960 or args.epochs!=1 or not args.initial_pair or (args.encoder_lr,args.reader_lr) not in ((5e-6,2e-6),(1e-5,4e-6)):
        raise ValueError('Requires960 fresh questions ×1, completed42k inherited pair, constant floor or2x LR')
    if torch.cuda.get_device_name()!='NVIDIA L40S':raise ValueError('This paired diagnostic requires the same L40S device family for both arms')
    if torch.are_deterministic_algorithms_enabled() or os.environ.get('CUBLAS_WORKSPACE_CONFIG'):raise ValueError('Use ordinary production numerical flags')
    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
        raise ValueError('Original production BF16 reduction setting is required')
    if existing and not args.resume:raise ResumeError('This run already started. Use --resume for its paired checkpoint, or a new run name.')
    if args.resume and not existing:raise ResumeError('No original training config exists for this resume')
    if args.initial_pair and args.arm=='fresh_joint':raise ValueError('Fresh initialization cannot also load a paired checkpoint')
    if args.evidence_tokens<1 or args.max_passages<1 or not 0<args.ratio<=1 or args.micro_batch_examples<1 or args.encode_batch_max_tokens<1 or args.gradient_checkpointing_min_tokens<0:
        raise ValueError('Evidence, passage and microbatch limits must be positive; ratio must be in (0,1]')
    initial_bytes=Path(args.initial_pair).read_bytes() if args.initial_pair else None
    initial_pair=json.loads(initial_bytes) if initial_bytes is not None else None
    if initial_pair:
        for key in ('ac_checkpoint','reader_checkpoint'):
            if not Path(initial_pair[key]).is_dir():raise ValueError('Missing initial paired checkpoint: '+initial_pair[key])
    material_path=Path(args.material) if args.material else REPORT/'preparation/material.json'
    material=json.loads(material_path.read_text())
    subset=json.loads(Path(args.selection).read_text())
    material=selected_material(material,subset)
    # Weighted interleave is part of the frozen order; per-epoch shuffle remains seed + original epoch number.
    for split in ('train',):
        queues=collections.defaultdict(list)
        for row in material[split]:queues[row['domain']].append(row)
        ordered=sorted(((i/len(rows),domain,row) for domain,rows in queues.items() for i,row in enumerate(rows)),key=lambda z:(z[0],z[1]))
        material[split]=[row for _,_,row in ordered]
    material['train']=material['train'][:args.items]
    if args.arm=='smoke':material['train']=material['train'][:8];material['held']=material['held'][:4]
    steps_per_epoch=math.ceil(len(material['train'])/8);steps=steps_per_epoch*args.epochs;midpoint=math.ceil(steps*.5)
    # Align whole-run panels even when an epoch ends between two 10% boundaries.
    panel_every=max(1,steps//10)
    diagnostic_every=math.gcd(steps_per_epoch,panel_every)
    whole_run_panels=material.get('whole_run_panels',False) and args.arm!='smoke'
    if whole_run_panels and steps%10:raise ValueError('Fixed whole-run panels require a total update count divisible by ten')
    config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=args.encoder_lr,learning_rate_target_lora=0 if args.arm=='reader_frozen' else args.reader_lr,
        examples_per_update=8,micro_batch_examples=args.micro_batch_examples,num_epochs=args.epochs,shuffle_examples=args.epochs>1 or whole_run_panels,warmup_updates=0,lr_schedule='constant',lr_final_fraction=1.0,resume_optimizer=True,
        reporting_interval=diagnostic_every/steps_per_epoch if whole_run_panels else min(1,.1*args.epochs),completion_samples=0,completion_max_new_tokens=160,teacher_cache_top_k=32,
        checkpoint_every_updates=24,checkpoint_every_epoch=True,max_duration_seconds=args.seconds,
        schedule_total_updates=steps,seed=SEED,gradient_checkpointing_min_tokens=args.gradient_checkpointing_min_tokens)
    material_ids={split:[row['example']['example_id'] for row in material[split]] for split in ('train','held')}
    identity={'arm':args.arm,'epochs':args.epochs,'planned_updates':steps,'steps_per_epoch':steps_per_epoch,
              'material_sha256':hashlib.sha256(material_path.read_bytes()).hexdigest(),'material_ids_sha256':json_digest(material_ids),
              'passage_parts':args.passage_parts,'evidence_tokens':args.evidence_tokens,'max_passages':args.max_passages,'ratio':args.ratio,
              'encode_batch_max_tokens':args.encode_batch_max_tokens,
              'initial_pair_sha256':hashlib.sha256(initial_bytes).hexdigest() if initial_bytes is not None else None}
    design=json.loads(Path(args.design).read_text())
    if (args.run_name != design['run_name'] or args.run_name not in ('pi42k_native2k_fresh960_floor_v1','pi42k_native2k_fresh960_2x_v1')
            or hashlib.sha256(Path(args.selection).read_bytes()).hexdigest()!=design['subset_sha256']):
        raise ValueError('Frozen paired native2k diagnostic run/subset changed')
    required={'arm':'joint','epochs':1,'planned_updates':120,'steps_per_epoch':120,
              'material_sha256':design['material_sha256'],'material_ids_sha256':design['material_ids_sha256'],
              'passage_parts':False,'evidence_tokens':2048,'max_passages':4,'ratio':1/16,
              'encode_batch_max_tokens':design['encode_batch_max_tokens'],'initial_pair_sha256':design['initial_pair_sha256']}
    expected_lrs=(5e-6,2e-6) if args.run_name=='pi42k_native2k_fresh960_floor_v1' else (1e-5,4e-6)
    if (args.encoder_lr,args.reader_lr)!=expected_lrs:raise ValueError('LR values disagree with the paired diagnostic arm name')
    if design['encode_batch_max_tokens'] != 65536 or args.gradient_checkpointing_min_tokens!=0:raise ValueError('Expected65536 encoder budget and0 checkpoint threshold')
    if design.get('status')!='frozen' or identity!=required or json_digest(asdict(config))!=json_digest(design['training_config']):
        raise ValueError('Fresh native continuation differs from its frozen material/initialization/configuration')
    if not initial_pair or len(material['train'])!=960 or len(material['held'])!=100:
        raise ValueError('Expected paired42k initialization and frozen960 fresh/100 development material')
    run_record={'identity':identity,'created_unix':time.time()}
    resume=None;attempt=None;previous_widgets={};panel_records=[]
    if args.resume:
        try:
            run_record=validate_run(folder,identity,config,material_ids)
            resume=resume_checkpoint(Path(os.environ['ACTIVATION_SYNC_ROOT']),steps_per_epoch,args.epochs)
        except (ValueError,KeyError,StopIteration,FileNotFoundError) as error:
            raise ResumeError('Cannot resume this run: '+str(error)) from error
        if resume['remaining_epochs']==0:
            print('QA_RUN_ALREADY_COMPLETE',steps,'published optimizer updates; no training performed',flush=True)
            return
        attempt,panel_records,previous_widgets=archive_attempt(folder,resume,steps_per_epoch)
        write(attempt/'resume.json',{key:value for key,value in resume.items() if key!='optimizer_state'})
    class RunReporter(ActivationContextTrainingReporter):
        run_epoch=resume['first_epoch'] if resume else 0
        def render(self,force=False):
            self.set_status(attempt_elapsed_minutes=f'{(time.monotonic()-started)/60:.1f}',epoch=f'{self.run_epoch} of {args.epochs}',
                            global_updates=f'{self.global_step} of {steps}')
            self._clear_status('arm_elapsed_minutes')
            if hasattr(self,'resume_eta'):self.set_status(update_eta=self.resume_eta)
            if run_record['created_unix'] is not None:
                self.set_status(whole_run_wall_minutes=f'{(time.time()-run_record["created_unix"])/60:.1f}')
            super().render(force=force)
        def whole_progress(self,progress):
            return whole_run_progress(progress,resume,args.epochs)
        def report_phase(self,progress,phase):
            mapped=self.whole_progress(progress);self.run_epoch=mapped.epoch
            if resume:
                completed=progress.global_step-resume['updates']
                remaining=(progress.elapsed_s/completed*(steps-progress.global_step)) if completed>0 else None
                self.resume_eta=f'about {remaining/60:.1f}m' if remaining is not None and progress.global_step<steps else 'updates finished; finalizing' if progress.global_step>=steps else 'estimating after resume'
            super().report_phase(mapped,phase)
        def report_eval(self,progress,name,summary):
            if resume and progress is not None and progress.epoch==0:
                self.report_phase(progress,'checking resumed checkpoint')
                write(attempt/'trainer_resume_baseline.json',summary)
                return
            if whole_run_panels and progress is not None and progress.global_step not in (0,steps) and progress.global_step%panel_every:
                return  # Intermediate epoch diagnostics are not whole-run panels.
            super().report_eval(self.whole_progress(progress) if progress is not None else None,name,summary)
        def report_training(self,stats):
            if resume:self.resume_eta='finished' if stats.stop_reason=='complete' else 'stopped'
            super().report_training(stats)
            self.widgets['training_summary']['title']='Latest attempt summary' if resume else 'Training summary'
            self.render(force=True)
    r=RunReporter(str(folder),'Task-aware QA — '+(args.run_name or args.arm),f'42,000 prior unique questions +960 fresh questions ×1;42,960 unique questions and presentations after completion. Short paired LR diagnostic, one encoder sequence with native2k evidence and up to4 passages. Inherited Adam5250; constant AC/reader LR {args.encoder_lr:g}/{args.reader_lr:g}. Both arms use L40S, microbatch1. This is not a convergence or schedule comparison. 2B encoder →9B reader, ratio {args.ratio:g}. Initial42k reader controls stay frozen; moving-reader controls are measured at0% and100% on identical evidence/distractors.')
    if previous_widgets:r.widgets.update(previous_widgets)
    r.global_step=resume['updates'] if resume else 0;r.epoch_origin=0;r.planned_updates=steps
    args._report_started=True
    r.set_status(phase='loading paired resume checkpoint' if resume else 'loading selected QA checkpoint' if initial_pair else 'loading P2 checkpoint' if args.arm!='fresh_joint' else 'initializing fresh adapters');r.render(force=True)
    source=Path(os.environ['QA_P2_ROOT']);ac_path=source/'AC_MODELS'/AC/'epoch_001';reader_path=source/'LORAS'/READER/'epoch_001'
    if initial_pair:ac_path,reader_path=Path(initial_pair['ac_checkpoint']),Path(initial_pair['reader_checkpoint'])
    if resume:ac_path,reader_path=Path(resume['ac_path']),Path(resume['reader_path'])
    h=harness(None if args.arm=='fresh_joint' and not resume else ac_path,None if args.arm=='fresh_joint' and not resume else reader_path,
              encode_batch_max_tokens=args.encode_batch_max_tokens)
    ac=h.module_manager.get_ac_model(AC);ac.prepare();h.module_manager.ensure_lora(READER)
    if resume:
        if ac.epoch_number!=resume['saved_epoch']:raise ResumeError('AC weights and optimizer disagree about the saved epoch')
        moments=resume['optimizer_state']['state']
        if len(moments)!=615 or {float(value['step']) for value in moments.values()}!={5250+resume['updates']}:
            raise ResumeError('Own resume lost inherited Adam clocks5250 plus completed stage updates')
        validate_optimizer_layout(resume.pop('optimizer_state'),[ac.trainable_parameters(),h.module_manager.lora_parameters(READER)])
        # An interrupted final save may reserve a numbered directory without publishing it.
        # Preserve it under an explicit orphan name, so the trainer resumes the original epoch.
        for checkpoint_root in (Path(os.environ['ACTIVATION_SYNC_ROOT'])/'AC_MODELS'/AC,Path(os.environ['ACTIVATION_SYNC_ROOT'])/'LORAS'/READER):
            for path in checkpoint_root.glob('epoch_*'):
                suffix=path.name[6:]
                if suffix.isdigit() and int(suffix)>=resume['first_epoch']:
                    path.rename(path.with_name(f'incomplete_{path.name}_{attempt.name}'))
    else:
        write(folder/'inherited_optimizer_check.json',check_inherited_optimizer(ac_path,[ac.trainable_parameters(),h.module_manager.lora_parameters(READER)]))
        ac.epoch_number=0
    r.set_status(phase='preparing frozen QA inputs',training_questions=len(material['train']),development_questions=len(material['held']));r.render(force=True)
    study,train,held=material_items(h,material,passage_parts=args.passage_parts,evidence_tokens=args.evidence_tokens,max_passages=args.max_passages,ratio=args.ratio)
    if 'learning_ids' in material and args.arm!='smoke':
        by_id={item.item_id:item for item in train}
        learning=[by_id[ident] for ident in material['learning_ids']]
    else:
        learning=random.Random(SEED+7).sample(train,min(32,len(train)))
    effective_config=replace(config,num_epochs=resume['remaining_epochs'],resume_optimizer=True,resume_skip_steps=resume['skip_steps']) if resume else config
    if os.environ.get('QA_CAMPAIGN_DEADLINE'):
        from datetime import datetime,timezone
        remaining=(datetime.fromisoformat(os.environ['QA_CAMPAIGN_DEADLINE'])-datetime.now(timezone.utc)).total_seconds()-120
        if remaining<=0:
            r.set_status(phase='campaign deadline reached; no training started');r.finish()
            return
        effective_config=replace(effective_config,max_duration_seconds=min(effective_config.max_duration_seconds or remaining,remaining))
    trainer=ActivationContextTrainer(h,effective_config)
    if resume and config.penalty_active:
        consumed=(resume['first_epoch']-1)*len(train)+min(len(train),resume['skip_steps']*config.examples_per_update)
        # Every RAG item carries a penalty candidate; reproduce the same deterministic sampling stream.
        for _ in range(consumed):trainer._penalty_rng.random()
    full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
    from evaluate_model_confirmation import shared_fingerprint
    input_check={'shared_input_fingerprint':shared_fingerprint(held,full,noctx),
        'ac_input_fingerprint':json_digest([(item.item_id,item.ac_messages) for item in held]),
        'held_ids':[item.item_id for item in held], 'initial_pair_sha256':identity['initial_pair_sha256']}
    if input_check['held_ids'] != subset['held']:
        raise ResumeError('Frozen development100 order changed')
    if any(item.info['passage_parts'] or len(item.info['part_source_tokens'])!=1 for item in train+held):
        raise ResumeError('Native2k diagnostic requires exactly one encoder part per example')
    input_path=folder/'input_check.json'
    if input_path.exists() and json.loads(input_path.read_text())!=input_check:
        raise ResumeError('Frozen native2k diagnostic inputs changed on resume')
    if resume:
        baseline=json.loads((folder/'initial_ac_baseline.json').read_text())
        if baseline['identity']!=input_check or baseline['measured_at_update']!=0:
            raise ResumeError('Own resume lacks the actual initial42k AC baseline')
    write(input_path,input_check)
    reference_views={'initial reader full context':full,'initial reader no context':noctx}
    reference_identity={'initial_pair_sha256':identity['initial_pair_sha256'],
        'initial_payload_sha256':design['initial_payload_sha256'],
        'material_sha256':identity['material_sha256'],'source_sha256':design['source_sha256'],
        'shared_inputs_sha256':json_digest({name:[(item.item_id,item.in_context_messages,item.ac_messages,
            item.tools,item.assistant_token_ids,item.info['answer_positions']) for item in items] for name,items in reference_views.items()})}
    reference_path=folder/'initial_reader_references.json';initial_references=None
    if reference_path.exists():
        initial_references=validate_initial_references(json.loads(reference_path.read_text()),reference_identity,reference_views)
    elif resume:
        raise ResumeError('Cannot resume without the actual frozen initial-reader reference cache')
    cohorts=material.get('held_cohorts',{})
    # Rebuild metadata before restoring saved curves below; resumed widgets already exist.
    for widget in ['references','answer_nll','panels','row_budget',*['cohort_'+name for name in cohorts]]:
        r.widgets.pop(widget,None)
    curve_names=['held AC','training AC','initial reader full context','initial reader no context','held full context','held no context']
    panel_description='Held AC100 and learning32 every96 new questions (10%). Initial42k reader full/no-context controls are measured once and remain frozen. Current-reader full/no-context controls are measured only at0% and100%; intervening points are absent. Gold and the separate bare-base KL reference stay fixed.'
    r.initialize_line_plot('answer_nll','Answer-only NLL — fixed questions and initial-reader controls',panel_description,'new unique questions seen this stage','nats / answer token',curve_names)
    r.initialize_table('panels','Fixed QA panels',panel_description+' Additional-evaluation seconds exclude the trainer’s preceding held-AC evaluation.',
        ['questions seen','percent','held items','AC answer NLL','training answer NLL','initial full NLL','initial no-context NLL','current full NLL','current no-context NLL','additional evaluation seconds'])
    if cohorts:
        r.widgets['answer_nll']['title']='Both development cohorts combined — answer NLL'
        for name,ids in cohorts.items():
            r.initialize_line_plot('cohort_'+name,f'{name.upper()}-generated development questions — answer NLL',
                f'{len(ids)} fixed questions. '+panel_description,
                'new unique questions seen this stage','nats / answer token',curve_names)
    row_budget={}
    r.initialize_table('row_budget','Actual compression budget','Passage tokens use the encoder tokenizer. Encoder input also includes labels, repeated questions and chat framing. Both layouts use the merged control’s requested row budget. Separate parts split that fixed budget in proportion to passage lengths, respecting minimum rows. Extra rows must be zero.',
        ['split','questions','passage tokens','encoder input tokens','compressed rows','rows / passage token','rows / encoder token','extra rows vs merged'])
    for split,items in [('training',train),('development',held)]:
        totals={key:sum(item.info[key] for item in items) for key in ('evidence_source_tokens','encoded_source_tokens','compressed_rows','merged_source_tokens','merged_compressed_rows')}
        row_budget[split]={'questions':len(items),**totals,'rows_per_evidence_token':totals['compressed_rows']/max(1,totals['evidence_source_tokens']),
            'rows_per_encoded_source_token':totals['compressed_rows']/max(1,totals['encoded_source_tokens']),
            'additional_rows_vs_merged':totals['compressed_rows']-totals['merged_compressed_rows']}
        r.add_data_point('row_budget',{'split':split,'questions':len(items),'passage tokens':totals['evidence_source_tokens'],
            'encoder input tokens':totals['encoded_source_tokens'],'compressed rows':totals['compressed_rows'],
            'rows / passage token':round(row_budget[split]['rows_per_evidence_token'],5),
            'rows / encoder token':round(row_budget[split]['rows_per_encoded_source_token'],5),
            'extra rows vs merged':row_budget[split]['additional_rows_vs_merged']})
    write(folder/'row_budget.json',{'passage_parts':args.passage_parts,'nominal_ratio':args.ratio,'splits':row_budget})
    r.set_text('recipe','Run recipe',json.dumps({'initialization':'selected QA pair' if initial_pair else 'fresh' if args.arm=='fresh_joint' else 'P2 epoch1','initial_pair':initial_pair,'train_questions':len(train),'held_questions':len(held),'learning_questions':len(learning),'updates':steps,
        'training_exposure':'960 fresh questions ×1;42960 unique after completion','questions_previously_seen':False,'parent_unique_questions':42000,'parent_presentations':42000,'optimizer':'Inherited Adam parameter clock5250; only stage clock resets','epochs':args.epochs,'updates_per_epoch':steps_per_epoch,'requested_training_ceiling_minutes':args.seconds/60,'effective_training_ceiling_minutes':effective_config.max_duration_seconds/60,
        'campaign_deadline_utc':os.environ.get('QA_CAMPAIGN_DEADLINE'),
        'material_sha256':hashlib.sha256(material_path.read_bytes()).hexdigest(),'panel_role':'development; separate confirmation questions remain unused',
        'training_by_domain':dict(collections.Counter(i.kind for i in train)),'passage_layout':'one part per passage' if args.passage_parts else 'merged passages','ratio':args.ratio,'evidence_tokens':args.evidence_tokens,'max_passages':args.max_passages,
        'encoder_lr':config.learning_rate_ac,'reader_lr':config.learning_rate_target_lora,'no_context_penalty':'KL(original base || current), λ1 on25%, scaled4' if config.penalty_active else 'off (reader frozen)'},indent=2))
    # Put intuitive panels first, leaving whole-target trainer diagnostics below them.
    front=['recipe',*['cohort_'+name for name in cohorts],'answer_nll','panels','row_budget','loss'];r.widgets={k:r.widgets[k] for k in front}|{k:v for k,v in r.widgets.items() if k not in front}
    if previous_widgets:
        for key,widget in previous_widgets.items():
            if key not in ('recipe','row_budget'):r.widgets[key]=widget
    def panel(progress,summary):
        nonlocal initial_references
        if whole_run_panels and progress.epoch!=0 and progress.global_step not in (0,steps) and progress.global_step%panel_every:
            return
        if not resume and progress.epoch==0 and progress.global_step==0:
            write(folder/'initial_ac_baseline.json',{'identity':input_check,'measured_at_update':0,
                'label':'Actual constant42k paired weights before the short native2k LR diagnostic',
                'summary':summary,'answer_nll':answer_nll(summary,held)})
        began=time.monotonic();r.set_status(evaluating_update=f'{progress.global_step}/{steps}')
        summaries={'held AC':summary,'training AC':trainer._evaluate(ac,learning,ac.device,r,'fixed training panel')}
        panels={'held AC':held,'held full context':full,'held no context':noctx,'training AC':learning,**reference_views}
        if progress.global_step in (0,steps):
            summaries.update({'held full context':trainer._evaluate(ac,full,ac.device,r,'current reader full context'),
                              'held no context':trainer._evaluate(ac,noctx,ac.device,r,'current reader no context')})
        if initial_references is None:
            if resume or progress.global_step!=0:raise ResumeError('Initial controls were not measured before training')
            record={'identity':reference_identity,'measured_at_update':0,'label':'Preselected42k task-aware initial reader, frozen before adaptation',
                'summaries':{'initial reader full context':summaries['held full context'],
                             'initial reader no context':summaries['held no context']}}
            initial_references=validate_initial_references(record,reference_identity,reference_views)
            write(reference_path,record)
        summaries.update(initial_references)
        scores={name:answer_nll(result,panels[name]) for name,result in summaries.items()}
        split_scores={name:cohort_answer_nll(summaries[name],panels[name],cohorts,held)
                      for name in summaries if name!='training AC'}
        cohort_scores={cohort:{**{name:values[cohort] for name,values in split_scores.items()},
                              'training AC':scores['training AC']} for cohort in cohorts}
        if resume and progress.epoch==0:
            write(attempt/'resume_baseline.json',{'update':progress.global_step,'scores':scores,'summaries':summaries})
            r.set_text('resume_check_'+attempt.name,'Resumed checkpoint verification',json.dumps({'update':progress.global_step,'scores':scores,'note':'Reevaluation of saved weights; existing scheduled panels are retained.'},indent=2))
            r._clear_status('evaluation_items','evaluating_update');r.set_status(phase='resumed checkpoint checked');r.render(force=True)
            return
        questions_seen=min(len(train),progress.global_step*config.examples_per_update)
        r.add_data_point('answer_nll',{'x':questions_seen,**scores})
        for name,values in cohort_scores.items():r.add_data_point('cohort_'+name,{'x':questions_seen,**values})
        entry={'update':progress.global_step,'questions seen':questions_seen,'percent':round(100*progress.global_step/steps,1),'held items':len(held),'AC answer NLL':scores['held AC'],
            'current full NLL':scores.get('held full context'),'current no-context NLL':scores.get('held no context'),
            'initial full NLL':scores['initial reader full context'],'initial no-context NLL':scores['initial reader no context'],
            'training answer NLL':scores['training AC'],'additional evaluation seconds':round(time.monotonic()-began,1)}
        if cohort_scores:entry['cohorts']=cohort_scores
        r.add_data_point('panels',{key:value for key,value in entry.items() if key not in ('cohorts','update')});panel_records.append(entry);write(folder/'panels.json',panel_records)
        write(folder/f'panel_{progress.global_step:05d}.json',{'progress':asdict(r.whole_progress(progress)),'scores':scores,'cohorts':cohort_scores,'summaries':summaries})
        # No inline decoding. Any separate answer readout must use the own completed final pair.
        r._clear_status('evaluation_items','evaluating_update');r.set_status(phase='panel complete',last_completed_panel=f'{progress.global_step}/{steps}');r.render(force=True)
    if resume:
        write(attempt/'effective_training_config.json',asdict(effective_config));write(folder/'panels.json',panel_records)
    else:
        write(folder/'training_config.json',asdict(config));write(folder/'material_ids.json',material_ids)
    write(folder/'run_identity.json',run_record)
    stats=trainer.train(AC,train,reporting_data=held,reporter=r,reporting_callback=panel,reset_optimizer=True)
    result=stats.summarize()
    result.update(summary_scope='latest resume attempt' if resume else 'whole run',planned_updates=steps,
                  completed_global_updates=max(trainer.updates_done.get(AC,0),resume['updates'] if resume else 0),planned_epochs=args.epochs,
                  resume_from_update=resume['updates'] if resume else None,
                  parent_unique_questions=42000,parent_presentations=42000,
                  completed_total_presentations=42000+8*max(trainer.updates_done.get(AC,0),resume['updates'] if resume else 0),
                  unique_questions_after_stage=42000+8*max(trainer.updates_done.get(AC,0),resume['updates'] if resume else 0),planned_unique_questions=42960,training_exposure='960 fresh questions once',optimizer_policy=f'inherited Adam5250, constant LR{args.encoder_lr:g}/{args.reader_lr:g}')
    if stats.stop_reason=='complete':
        moments=trainer.optimizers[AC].state
        if (trainer.updates_done.get(AC)!=120 or len(moments)!=615 or {float(value['step']) for value in moments.values()}!={5370.0}
                or not stats.checkpoint_path or not stats.target_lora_checkpoint_path):
            raise ResumeError('Completed diagnostic must publish own120-update paired final with all615 Adam clocks5370')
        result['final_optimizer_guard']={'populated_states':615,'all_parameter_steps':5370,'stage_updates':120}
    write(folder/'result.json',result);r.finish()
    if args.arm=='smoke':
        assert stats.steps==1 and stats.grad_norm_ac[0]>0 and stats.grad_norm_reader[0]>0,stats.summarize()
        # Real GPU proof: the no-context penalty sends gradients only into the reader adapter.
        from activation.autotuners import configure_fla_runtime
        from activation.agent_training.agent_training_utils import disable_dropout
        trainer.config=replace(config,no_context_penalty_fraction=1.0)
        ac.prepare();h.module_manager.ensure_lora(READER);ac.set_mode('training')
        variables=trainer._autotuning_variables(ac)
        with configure_fla_runtime(variables['fla_config']):
            example=trainer.build_example(ac,train[0]);assert example.penalty is not None
            trainer._store_teacher_targets(ac,[example.penalty],ac.device,lora=None)
            ac._training_gradient_checkpointing=config.gradient_checkpointing
            ac._side_gradient_checkpointing_min_tokens=variables['side_gradient_checkpointing_min_tokens']
            trainer._target_gradient_checkpointing_min_tokens=variables['target_gradient_checkpointing_min_tokens']
            for base in (ac.side.model,ac.target.model):base.train();disable_dropout(base)
            for parameter in ac.trainable_parameters()+h.module_manager.lora_parameters(READER):parameter.grad=None
            main_outputs=trainer._example_loss(ac,example,ac.device)
            penalty=trainer._penalty_term;assert penalty is not None and torch.isfinite(penalty)
            del main_outputs
            penalty.backward()
            assert all(p.grad is None for p in ac.trainable_parameters()),'Penalty leaked into encoder'
            reader_gradients=[p.grad for p in h.module_manager.lora_parameters(READER) if p.grad is not None]
            assert reader_gradients and all(torch.isfinite(g).all() for g in reader_gradients)
            assert any(g.abs().sum()>0 for g in reader_gradients),'Penalty did not reach reader'
        del reader_gradients,base,parameter
        del example,penalty
        expected=stats.reporting['gold_nll'];saved_ac,saved_reader=stats.checkpoint_path,stats.target_lora_checkpoint_path
        del trainer,study,ac,h;gc.collect();torch.cuda.empty_cache()
        reload_h=harness(saved_ac,saved_reader);reload_trainer=ActivationContextTrainer(reload_h,replace(config,no_context_penalty_weight=0))
        measured=reload_trainer.eval(AC,held)['gold_nll'];assert abs(measured-expected)<.003,(expected,measured)
        write(folder/'smoke_pass.json',{'updates':stats.steps,'ac_grad':stats.grad_norm_ac[0],'reader_grad':stats.grad_norm_reader[0],'before_reload_nll':expected,'after_reload_nll':measured,'paired_reload':'pass'})
    print('QA_RUN_FINISHED',args.arm,stats.stop_reason,stats.steps,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['smoke','joint','reader_frozen','fresh_joint'],required=True);parser.add_argument('--items',type=int,default=1536);parser.add_argument('--seconds',type=int,default=6600)
    parser.add_argument('--resume',action='store_true',help='Resume this run from its latest published pair, retaining the original schedule and frozen data.')
    parser.add_argument('--design',required=True);parser.add_argument('--selection',required=True);parser.add_argument('--run-name');parser.add_argument('--material');parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--initial-pair',help='Frozen JSON record with ac_checkpoint and reader_checkpoint paths for a new warm-started run.')
    parser.add_argument('--evidence-tokens',type=int,default=2048)
    parser.add_argument('--max-passages',type=int,default=4)
    parser.add_argument('--ratio',type=float,default=1/16)
    parser.add_argument('--micro-batch-examples',type=int,default=1)
    parser.add_argument('--encode-batch-max-tokens',type=int,default=65536)
    parser.add_argument('--gradient-checkpointing-min-tokens',type=int,default=0)
    parser.add_argument('--passage-parts',action='store_true',help='Encode each numbered passage separately while retaining the merged control’s total row budget.')
    parser.add_argument('--encoder-lr',type=float,default=5e-6);parser.add_argument('--reader-lr',type=float,default=2e-6)
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        if isinstance(error,ResumeError) and not getattr(args,'_report_started',False):raise
        from activation.common.reporting import HtmlReporter
        folder=REPORT/(args.run_name or args.arm);data=folder/'report_data.json';previous=json.loads(data.read_text()) if data.exists() else {}
        r=HtmlReporter(str(folder),'Task-aware QA — '+args.arm,'Run failed; see recorded error.')
        r.status=previous.get('status',{});r.widgets={w['name']:w for w in previous.get('widgets',[])}
        completed=folder/'result.json'
        training_complete=args.arm=='smoke' and completed.exists() and json.loads(completed.read_text()).get('stop_reason')=='complete'
        r.set_status(phase='training complete; auxiliary check failed' if training_complete else 'failed',error=str(error));r.render(force=True);raise
