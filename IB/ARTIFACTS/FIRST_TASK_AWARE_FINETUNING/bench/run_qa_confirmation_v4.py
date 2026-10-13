"""One train call per arm, fixed synthesized QA panels and matched current-reader controls."""
import os,json,time,math,random,hashlib,shutil,argparse,gc,collections
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict,replace
ROOT=Path(os.environ.get('QA_SOURCE_ROOT','/workspace')); REPORT=Path(os.environ.get('QA_REPORT_ROOT',ROOT/'IB/TMP/SYNC/TASK_AWARE_QA'))
NAME,AC,SIDE,READER='qwen4b','rag_poc','rag_side','rag_reader'; SEED=20260921

def write(path,value):
    from activation.common.reporting import _write_atomic
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);_write_atomic(path,json.dumps(value,indent=2,default=str,allow_nan=False))

def harness(ac_path=None,reader_path=None):
    import torch
    from activation.harness import HarnessRuntime,HarnessRuntimeConfig,ModelConfig
    from activation.ac_model import ActivationContextModelConfig
    torch.set_num_threads(4);torch.manual_seed(SEED)
    h=HarnessRuntime(HarnessRuntimeConfig(model_configs={NAME:ModelConfig(NAME,'Qwen/Qwen3.5-9B'),'side_base':ModelConfig('side_base','Qwen/Qwen3.5-2B')},doc_chunk_size_chars=12000,doc_chunk_overlap_chars=0))
    h.module_manager.register_lora(READER,NAME,rank=128,dropout=0,checkpoint_path=str(reader_path) if reader_path else None)
    h.module_manager.register_ac_model(ActivationContextModelConfig(AC,'side_base',SIDE,NAME,READER,min_view_rows=2,default_compression_ratio=1/16,
        checkpoint_path=str(ac_path) if ac_path else None,intervention_frequency=-1,input_pooling_stride=0,mixer_layers=2))
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
        # Own-source evidence can lie outside the frozen distractor corpus; only indexed IDs need exclusion.
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
            for series in widget['series']:
                kept=[(x,y) for x,y in zip(series['x'],series['y']) if x<=limit]
                series['x']=[x for x,_ in kept];series['y']=[y for _,y in kept]
        elif widget['type']=='table':
            widget['rows']=[row for row in widget['rows'] if row.get('global update',row.get('update',0))<=resume['updates']]
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

def main(args):
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig,ActivationContextTrainingReporter
    folder=REPORT/(args.run_name or args.arm);folder.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    existing=(folder/'training_config.json').exists()
    if existing and not args.resume:raise ResumeError('This run already started. Use --resume for its paired checkpoint, or a new run name.')
    if args.resume and not existing:raise ResumeError('No original training config exists for this resume')
    if args.initial_pair and args.arm=='fresh_joint':raise ValueError('Fresh initialization cannot also load a paired checkpoint')
    if args.evidence_tokens<1 or args.max_passages<1 or not 0<args.ratio<=1 or args.micro_batch_examples<1:
        raise ValueError('Evidence, passage and microbatch limits must be positive; ratio must be in (0,1]')
    initial_bytes=Path(args.initial_pair).read_bytes() if args.initial_pair else None
    initial_pair=json.loads(initial_bytes) if initial_bytes is not None else None
    if initial_pair:
        for key in ('ac_checkpoint','reader_checkpoint'):
            if not Path(initial_pair[key]).is_dir():raise ValueError('Missing initial paired checkpoint: '+initial_pair[key])
    material_path=Path(args.material) if args.material else REPORT/'preparation/material.json'
    material=json.loads(material_path.read_text())
    # Weighted interleave is part of the frozen order; per-epoch shuffle remains seed + original epoch number.
    for split in ('train','held'):
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
        examples_per_update=8,micro_batch_examples=args.micro_batch_examples,num_epochs=args.epochs,shuffle_examples=args.epochs>1 or whole_run_panels,warmup_updates=50 if args.arm!='smoke' else 0,
        reporting_interval=diagnostic_every/steps_per_epoch if whole_run_panels else min(1,.1*args.epochs),completion_samples=0,completion_max_new_tokens=160,teacher_cache_top_k=32,
        checkpoint_every_updates=min(50,panel_every) if whole_run_panels else min(50,midpoint),checkpoint_every_epoch=True,max_duration_seconds=args.seconds,
        schedule_total_updates=steps,seed=SEED,gradient_checkpointing_min_tokens=1700)
    material_ids={split:[row['example']['example_id'] for row in material[split]] for split in ('train','held')}
    identity={'arm':args.arm,'epochs':args.epochs,'planned_updates':steps,'steps_per_epoch':steps_per_epoch,
              'material_sha256':hashlib.sha256(material_path.read_bytes()).hexdigest(),'material_ids_sha256':json_digest(material_ids),
              'passage_parts':args.passage_parts,'evidence_tokens':args.evidence_tokens,'max_passages':args.max_passages,'ratio':args.ratio,
              'initial_pair_sha256':hashlib.sha256(initial_bytes).hexdigest() if initial_bytes is not None else None}
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
    r=RunReporter(str(folder),'Task-aware QA — '+(args.run_name or args.arm),f'{args.epochs} epochs of synthesized QA, 2B encoder →9B reader, ratio {args.ratio:g}. Fixed development controls use the current reader and identical evidence/distractors.')
    if previous_widgets:r.widgets.update(previous_widgets)
    r.global_step=resume['updates'] if resume else 0;r.epoch_origin=0;r.planned_updates=steps
    args._report_started=True
    r.set_status(phase='loading paired resume checkpoint' if resume else 'loading selected QA checkpoint' if initial_pair else 'loading P2 checkpoint' if args.arm!='fresh_joint' else 'initializing fresh adapters');r.render(force=True)
    source=Path(os.environ['QA_P2_ROOT']);ac_path=source/'AC_MODELS'/AC/'epoch_001';reader_path=source/'LORAS'/READER/'epoch_001'
    if initial_pair:ac_path,reader_path=Path(initial_pair['ac_checkpoint']),Path(initial_pair['reader_checkpoint'])
    if resume:ac_path,reader_path=Path(resume['ac_path']),Path(resume['reader_path'])
    h=harness(None if args.arm=='fresh_joint' and not resume else ac_path,None if args.arm=='fresh_joint' and not resume else reader_path)
    ac=h.module_manager.get_ac_model(AC);ac.prepare();h.module_manager.ensure_lora(READER)
    if resume:
        if ac.epoch_number!=resume['saved_epoch']:raise ResumeError('AC weights and optimizer disagree about the saved epoch')
        validate_optimizer_layout(resume.pop('optimizer_state'),[ac.trainable_parameters(),h.module_manager.lora_parameters(READER)])
        # An interrupted final save may reserve a numbered directory without publishing it.
        # Preserve it under an explicit orphan name, so the trainer resumes the original epoch.
        for checkpoint_root in (Path(os.environ['ACTIVATION_SYNC_ROOT'])/'AC_MODELS'/AC,Path(os.environ['ACTIVATION_SYNC_ROOT'])/'LORAS'/READER):
            for path in checkpoint_root.glob('epoch_*'):
                suffix=path.name[6:]
                if suffix.isdigit() and int(suffix)>=resume['first_epoch']:
                    path.rename(path.with_name(f'incomplete_{path.name}_{attempt.name}'))
    else:ac.epoch_number=0
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
    cohorts=material.get('held_cohorts',{})
    for widget in ['references']:
        r.widgets.pop(widget,None)
    r.initialize_line_plot('answer_nll','Answer-only NLL — current reader','All curves use the current reader LoRA on fixed questions/answers. Full/no-context controls move as the reader trains. SFT gold answers and the original-base KL reference stay fixed. Tool syntax is excluded.','optimizer update','nats / answer token',['held AC','held full context','held no context','training AC'])
    r.initialize_table('panels','Fixed QA panels',f'{len(held)} fixed development questions; current reader in every view. Lower NLL is better.',['update','percent','held items','AC answer NLL','full answer NLL','no-context answer NLL','training answer NLL','panel seconds'])
    if cohorts:
        r.widgets['answer_nll']['title']='Both development cohorts combined — answer NLL'
        for name,ids in cohorts.items():
            r.initialize_line_plot('cohort_'+name,f'{name.upper()}-generated development questions — answer NLL',
                f'{len(ids)} fixed questions shared by both training arms. Compare arms within this cohort. Full/no-context use this arm’s current reader; training AC uses its own fixed learning sample.',
                'optimizer update','nats / answer token',['held AC','held full context','held no context','training AC'])
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
        'epochs':args.epochs,'updates_per_epoch':steps_per_epoch,'requested_training_ceiling_minutes':args.seconds/60,'effective_training_ceiling_minutes':effective_config.max_duration_seconds/60,
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
        if whole_run_panels and progress.epoch!=0 and progress.global_step not in (0,steps) and progress.global_step%panel_every:
            return
        began=time.monotonic();r.set_status(evaluating_update=f'{progress.global_step}/{steps}')
        summaries={'held AC':summary,'held full context':trainer._evaluate(ac,full,ac.device,r,'held full context'),
                   'held no context':trainer._evaluate(ac,noctx,ac.device,r,'held no context'),
                   'training AC':trainer._evaluate(ac,learning,ac.device,r,'fixed training panel')}
        panels={'held AC':held,'held full context':full,'held no context':noctx,'training AC':learning}
        scores={name:answer_nll(result,panels[name]) for name,result in summaries.items()}
        split_scores={name:cohort_answer_nll(summaries[name],panels[name],cohorts,held)
                      for name in ('held AC','held full context','held no context')}
        cohort_scores={cohort:{**{name:values[cohort] for name,values in split_scores.items()},
                              'training AC':scores['training AC']} for cohort in cohorts}
        if resume and progress.epoch==0:
            write(attempt/'resume_baseline.json',{'update':progress.global_step,'scores':scores,'summaries':summaries})
            r.set_text('resume_check_'+attempt.name,'Resumed checkpoint verification',json.dumps({'update':progress.global_step,'scores':scores,'note':'Reevaluation of saved weights; existing scheduled panels are retained.'},indent=2))
            r._clear_status('evaluation_items','evaluating_update');r.set_status(phase='resumed checkpoint checked');r.render(force=True)
            return
        r.add_data_point('answer_nll',{'x':progress.global_step,**scores})
        for name,values in cohort_scores.items():r.add_data_point('cohort_'+name,{'x':progress.global_step,**values})
        entry={'update':progress.global_step,'percent':round(100*progress.global_step/steps,1),'held items':len(held),'AC answer NLL':scores['held AC'],
            'full answer NLL':scores['held full context'],'no-context answer NLL':scores['held no context'],'training answer NLL':scores['training AC'],'panel seconds':round(time.monotonic()-began,1)}
        if cohort_scores:entry['cohorts']=cohort_scores
        r.add_data_point('panels',entry);panel_records.append(entry);write(folder/'panels.json',panel_records)
        write(folder/f'panel_{progress.global_step:05d}.json',{'progress':asdict(r.whole_progress(progress)),'scores':scores,'cohorts':cohort_scores,'summaries':summaries})
        previous_selection=json.loads((folder/'selected_checkpoint.json').read_text()) if (folder/'selected_checkpoint.json').exists() else {}
        if progress.global_step and scores['held AC']<=min(min(p['AC answer NLL'] for p in panel_records),previous_selection.get('development_answer_nll',float('inf'))):
            # Development chooses checkpoints; the untouched confirmation set is scored later.
            selection_root=Path(os.environ['ACTIVATION_SYNC_ROOT'])/'selected'
            destination=selection_root/(attempt.name if resume else '')/f'update_{progress.global_step:05d}'
            destination.mkdir(parents=True,exist_ok=True)
            saved_ac=ac.save(str(destination/'ac'),epoch_number=progress.epoch_number)
            saved_reader=h.module_manager.save_lora(NAME,READER,str(destination/'reader'))
            (destination/'PRESERVE').write_text('Development-best paired weights; retain for confirmation.\n')
            write(folder/'selected_checkpoint.json',{'update':progress.global_step,'development_answer_nll':scores['held AC'],
                'ac_checkpoint':saved_ac,'reader_checkpoint':saved_reader,'selection':'lowest development answer NLL across saved attempt checkpoints; confirmation unused'})
        if progress.global_step in (0,midpoint,steps) and args.arm!='smoke':
            from activation.dataset.scoring import exact_match,token_f1,numeric_exact
            samples=trainer._sample_completions(ac,held[:12],progress)
            for sample,item in zip(samples,held[:12]):
                _,calls=study.dialect.parse(sample['student'])
                prediction=next((str(c['arguments'].get('answer','')) for c in calls if c['name']=='submit_answer'),None)
                if prediction is None:
                    try:
                        name,arguments=study.dialect.preferred_format.parse(sample['student'],None)
                        prediction=str(arguments.get('answer','')) if name=='submit_answer' else ''
                    except ValueError:prediction=''
                gold=item.info['answer'];sample.update(question=item.info['question'],answer=prediction,exact_match=exact_match(prediction,[gold]),token_f1=token_f1(prediction,[gold]))
                if item.kind=='numeric':sample['numeric_exact']=numeric_exact(prediction,[gold])
            r.report_completions(progress,samples)
            r.set_text(f'answer_scores_{progress.global_step:05d}',f'Decoded answer scores · update {progress.global_step}',json.dumps({
                'items':len(samples),'normalized_exact_match':sum(z['exact_match'] for z in samples)/len(samples),'token_f1':sum(z['token_f1'] for z in samples)/len(samples),
                'numeric_exact_on_numeric_items':[z['numeric_exact'] for z in samples if 'numeric_exact' in z]},indent=2))
            write(folder/f'answers_{progress.global_step:05d}.json',samples)
            # The same decoded questions expose how much of the NLL gap affects actual answers.
            full_samples=trainer._sample_completions(ac,full[:12],progress)
            write(folder/f'full_context_answers_{progress.global_step:05d}.json',full_samples)
        if progress.global_step==midpoint and args.arm!='smoke' and progress.step<steps_per_epoch:
            checkpoint=Path(os.environ['ACTIVATION_SYNC_ROOT']);marker=checkpoint/'AC_MODELS'/AC/'partial_epoch.json'
            if marker.exists():
                pair=json.loads(marker.read_text());dest=checkpoint/'midpoint';dest.mkdir(exist_ok=True)
                for name,key in [('ac','ac_checkpoint'),('reader','target_lora_checkpoint')]:
                    shutil.copytree(checkpoint/pair[key],dest/name,ignore=shutil.ignore_patterns('optimizer.pt'),dirs_exist_ok=True)
                write(dest/'pair.json',{'update':midpoint,'ac':str(dest/'ac'),'reader':str(dest/'reader')})
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
                  resume_from_update=resume['updates'] if resume else None)
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
    parser.add_argument('--run-name');parser.add_argument('--material');parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--initial-pair',help='Frozen JSON record with ac_checkpoint and reader_checkpoint paths for a new warm-started run.')
    parser.add_argument('--evidence-tokens',type=int,default=2048)
    parser.add_argument('--max-passages',type=int,default=4)
    parser.add_argument('--ratio',type=float,default=1/16)
    parser.add_argument('--micro-batch-examples',type=int,default=2)
    parser.add_argument('--passage-parts',action='store_true',help='Encode each numbered passage separately while retaining the merged control’s total row budget.')
    parser.add_argument('--encoder-lr',type=float,default=5e-5);parser.add_argument('--reader-lr',type=float,default=2e-5)
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
