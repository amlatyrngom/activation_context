"""Golden reconstruction: one explicit pre-mixing SFT pass, never a task-weight merge.

CPU preparation freezes corpus, sampled ratios, exact target budget and tokenized
examples. Training uses that same source/cache and native paired Adam/RNG resumes.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import signal
import time

from run_qa import AC, NAME, READER, harness, resume_checkpoint, write
from run_trajectory_tasks import compact_example, digest

STOP = False
PREPARATION = None


def stop(signum, frame):
    global STOP
    STOP = True


def file_sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def source_identity():
    import activation
    root=Path(activation.__file__).parent
    return {'activation':digest({str(p.relative_to(root)):file_sha(p) for p in sorted(root.rglob('*.py'))}),
            'worker':file_sha(__file__), 'run_qa':file_sha(Path(__file__).with_name('run_qa.py')),
            'preparation_helper':file_sha(Path(__file__).with_name('run_trajectory_tasks.py'))}


def recipe(steps):
    from activation.ac_model import ActivationContextTrainingConfig
    return ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=5e-5,
        learning_rate_target_lora=2e-5,weight_decay=0,adam_betas=(.9,.95),adam_eps=1e-8,
        max_grad_norm=1,lr_schedule='cosine',lr_final_fraction=.1,warmup_updates=50,
        schedule_total_updates=steps,examples_per_update=8,micro_batch_examples=1,
        num_epochs=1,seed=20260924,shuffle_examples=True,reporting_interval=.1,
        no_context_penalty_weight=1,no_context_penalty_fraction=.25,teacher_cache_top_k=32,
        teacher_cache_block_examples=32,gradient_checkpointing_min_tokens=0,
        logits_chunk_tokens=64,checkpoint_every_updates=100,completion_samples=0,
        completion_max_new_tokens=512)


def frozen_passages(data, held_per_source, seed, smoke=False):
    train=list(data['train']); held=[]; counts=Counter()
    if any(p['source']=='teacher' for p in train+list(data['held'])):
        raise ValueError('Legacy benchmark-conditioned teacher traces are excluded from this warm-up')
    for passage in data['held']:
        if counts[passage['source']] < held_per_source:
            held.append(passage);counts[passage['source']]+=1
    if smoke:
        by_source={}
        for passage in train:by_source.setdefault(passage['source'],[]).append(passage)
        train=[]
        while by_source and len(train)<16:
            for source in list(by_source):
                train.append(by_source[source].pop(0))
                if not by_source[source]:del by_source[source]
                if len(train)==16:break
        held=held[:6]
    if not train or not held:raise ValueError('Need published train and source-held passages')
    if {(p['source'],p['doc_id']) for p in train} & {(p['source'],p['doc_id']) for p in held}:
        raise ValueError('Corpus root/document split changed')
    entries=[]
    for split,passages,split_seed in [('train',train,seed),('held',held,seed+1)]:
        rng=random.Random(split_seed)
        for ordinal,passage in enumerate(passages):
            p=dict(passage)
            if not p['text'].strip():raise ValueError('Empty prepared reconstruction passage')
            p['ratio']=rng.uniform(1/30,1/10)
            p['_id']=f"warmup:{split}:{p['source']}:{ordinal}:{hashlib.sha256(p['text'].encode()).hexdigest()[:20]}"
            entries.append((split,p))
    return entries,len(train)


def prepare_chunk(entries):
    study,builder,ac=PREPARATION
    items=study.generate_reconstruction_samples([p for _,p in entries],variant='reconstruct',seed=0)
    assert len(items)==len(entries)
    examples=[]
    from activation.ac_model.ac_model_utils import validate_part_capacity,tokenize_with_parts
    for item,(split,p) in zip(items,entries):
        item.item_id=p['_id'];item.info.update(source=p['source'],doc_id=p['doc_id'],split=split,
            source_tokens=item.info['passage_tokens'],source_text_sha256=hashlib.sha256(p['text'].strip().encode()).hexdigest())
        example=compact_example(builder.build_example(ac,item))
        if max(len(example.teacher_ids),len(example.student_ids))>72000:raise ValueError('Reader capacity exceeded')
        for request in example.part_requests:validate_part_capacity(ac,request,65536)
        item.info['compressed_rows']=sum(end-start for start,end in example.spans)
        item.info['encoder_tokens']=sum(len(tokenize_with_parts(ac.side.tokenizer,r.messages,[],ac.side.tokenizer.pad_token_id or 0,tools=r.tools)[0]) for r in example.part_requests)
        examples.append(example)
    return items,examples


def token_accounting(items, epochs=1):
    unique={}
    for item in items:unique.setdefault(item.info['source_text_sha256'],item.info['source_tokens'])
    targets=sum(sum(map(len,item.assistant_token_ids)) for item in items)
    encoder_tokens=sum(i.info['encoder_tokens'] for i in items);rows=sum(i.info['compressed_rows'] for i in items)
    return {'encoder_tokens_one_pass':encoder_tokens,'compressed_rows_one_pass':rows,'effective_rows_per_encoder_token':rows/encoder_tokens,
        'training_passages':len(items),'unique_passage_texts':len(unique),
        'unique_source_tokens':sum(unique.values()),'source_tokens_one_pass':sum(i.info['source_tokens'] for i in items),
        'supervised_target_tokens_one_pass':targets,'planned_target_token_exposures':targets*epochs,
        'planned_passage_exposures':len(items)*epochs,'epochs':epochs}



def require_initial_weights(resume):
    if resume is not None and resume['updates'] > 0:
        raise ValueError('Missing initial references cannot be recomputed using resumed trained weights')


def publish_prepared(cache, manifest, identity, prepared):
    tmp=cache.with_suffix('.pkl.tmp')
    with tmp.open('wb') as handle:
        pickle.dump(prepared,handle,protocol=5);handle.flush();os.fsync(handle.fileno())
    write(manifest,identity)  # A visible complete cache must already have its receipt.
    tmp.replace(cache)


def attach_greedy(reporter, path, items=()):
    data=json.loads(path.read_text());by_id={i.item_id:i for i in items}
    rows=[]
    for sample in data['samples']:
        item=by_id.get(sample['item_id'])
        rows.append({'title':sample['item_id'],'state':'finished',
            'stats':{'source':item.info['source'],'ratio':item.info['ratio']} if item else {},
            'prompt':'Reconstruct the passage exactly.','file':path.name,'steps':[
                {'role':'Gold','turn':0,'content':sample['reference'],'calls':[],'results':[]},
                {'role':'AC','turn':0,'content':sample['student'],'calls':[],'results':[]}]})
    reporter.set_trajectories('examples','Greedy reconstruction — '+path.stem.removeprefix('greedy_'),
        f"Fixed source-diverse passages; {data['cap']}-token decode cap explicitly limits exact-copy claims.",rows)
    reporter.widgets['examples']['passages_presented']=data['passages_presented']


def restore_report(reporter, maximum_passages):
    data=reporter.folder/'report_data.json';page=reporter.folder/'report.tressoir.html'
    if data.exists():
        previous=json.loads(data.read_text());widgets={w['name']:w for w in previous['widgets']}
        rows=[p for p in widgets.get('panels',{}).get('rows',[]) if p['passages presented']<=maximum_passages]
        maximum_tokens=max((p['target tokens presented'] for p in rows),default=0)
        for widget in widgets.values():
            if widget['type']=='line':
                for series in widget['series']:
                    kept=[(x,y) for x,y in zip(series['x'],series['y'],strict=True) if x<=maximum_tokens]
                    series['x']=[x for x,_ in kept];series['y']=[y for _,y in kept]
            elif widget['name']=='panels':widget['rows']=rows
            elif widget['type']=='trajectories':widget['items']=[]
        reporter.widgets={**widgets,**{k:v for k,v in reporter.widgets.items() if k not in widgets}}
        if page.exists():
            reporter.last_page_widgets=list(widgets);reporter.last_page_time=time.time()
    valid=[]
    for path in reporter.folder.glob('greedy_*.json'):
        record=json.loads(path.read_text())
        if record['passages_presented']>maximum_passages:
            archive=reporter.folder/'superseded';archive.mkdir(exist_ok=True)
            path.rename(archive/f'{path.stem}_{time.time_ns()}.json')
        else:valid.append((record['passages_presented'],path))
    if valid:attach_greedy(reporter,max(valid,key=lambda pair:pair[0])[1])


def main(args):
    import torch
    from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer
    from activation.ac_model.ac_model_reporter import ActivationContextTrainingReporter
    from activation.common.reporting import HtmlReporter
    started=time.monotonic();out=args.output;out.mkdir(parents=True,exist_ok=True)
    os.environ['ACTIVATION_SYNC_ROOT']=str(out.resolve())
    report=out/'report'
    r=HtmlReporter(str(report),'Golden reconstruction — pre-mixing warm-up',
        '2B encoder → 9B reader · 1/10–1/30 sampled compression · fixed initial-reader controls',page_rewrite_seconds=math.inf)
    r.initialize_line_plot('quality','Held reconstruction NLL','Token-weighted native targets; initial-reader controls stay frozen.',
        'supervised target tokens presented','nats/token',[])
    r.initialize_line_plot('learning','Learning loss and no-context drift','Training NLL is a fixed24-passage sample; drift is base→adapted-reader coarsened KL.',
        'supervised target tokens presented','nats/token',[])
    r.initialize_table('panels','10% quality panels','One corpus pass; ratios sampled once before training.',
        ['passages presented','target tokens presented','epoch','held NLL','training NLL','token accuracy','no-context KL','quality gate'])
    r.set_trajectories('examples','Greedy reconstruction','Fixed source-diverse passages;512-token cap.',[])
    r.set_text('recipe','Pre-mixing warm-up recipe','Checking the frozen corpus and reconstruction initializer.')
    durable=0;markers=out/'AC_MODELS'/AC
    if (markers/'partial_epoch.json').exists():durable=int(json.loads((markers/'partial_epoch.json').read_text())['step'])*8
    if (markers/'completed_epoch.json').exists():durable=json.loads((out/'token_accounting.json').read_text())['training_passages']
    restore_report(r,durable)
    def phase(name,**fields):
        r.status.clear();r.set_status(phase=name,**fields);r.render(force=True)
    phase('checking frozen corpus and reconstruction pair')
    pair=json.loads(args.initial_pair.read_text())
    for path in (pair['ac_checkpoint'],pair['reader_checkpoint']):
        if not Path(path).is_dir():raise FileNotFoundError(path)
    if pair.get('kind')!='reconstruction_initialization':raise ValueError('Explicit reconstruction initializer required')
    for rel,expected in pair['payload_sha256'].items():
        if file_sha(Path(pair['root'])/rel)!=expected:raise ValueError('Initial P2 payload changed: '+rel)
    identity={'phase':'pre-mixing reconstruction warm-up'+(' (GPU gate only)' if args.smoke else ''),'source':source_identity(),'corpus_sha256':file_sha(args.material),
        'initial_pair':digest(pair),'ratios_range':[1/30,1/10],'seed':20260924,'held_per_source':args.held_per_source,
        'minimum_supervised_target_tokens':args.minimum_target_tokens,'smoke':args.smoke}
    cache=out/'prepared_examples.pkl';manifest=out/'prepared_identity.json'
    original=harness(pair['ac_checkpoint'],pair['reader_checkpoint'])
    ac=original.module_manager.get_ac_model(AC);ac.epoch_number=0
    assert ac.side.model_config.model_id=='Qwen/Qwen3.5-2B' and ac.target.model_config.model_id=='Qwen/Qwen3.5-9B'
    assert ac.config.side_lora_rank==128 and original.module_manager.get_lora_config(READER).rank==128
    assert ac.config.input_pooling_stride==0 and ac.config.intervention_frequency==-1 and ac.is_deep
    identity['architecture']={'encoder':'Qwen/Qwen3.5-2B','reader':'Qwen/Qwen3.5-9B','rank':128,
        'input_pooling_stride':0,'input_rows':True,'deep_residuals':ac.intervention_summary()}
    study=ActivationContextStudyGenerator(original,AC)
    identity['tokenizers']={label:{'vocab':digest(t.get_vocab()),'template':digest(t.chat_template)}
        for label,t in [('reader',study.tokenizer),('encoder',ac.side.tokenizer)]}
    identity=json.loads(json.dumps(identity))
    if cache.exists():
        if not manifest.exists() or json.loads(manifest.read_text())!=identity:raise ValueError('Prepared source/recipe changed')
        phase('loading prepared CPU material')
        with cache.open('rb') as handle:prepared=pickle.load(handle)
        if prepared['identity']!=identity:raise ValueError('Cache identity disagrees with receipt')
    else:
        if args.require_prepared:raise ValueError('Complete CPU preparation before allocating a training GPU')
        if not args.prepare_only or os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
            raise ValueError('Initial preparation belongs on CPU: --prepare-only and CUDA_VISIBLE_DEVICES=')
        data=json.loads(args.material.read_text())
        entries,ntrain=frozen_passages(data,args.held_per_source,20260924,args.smoke)
        del data
        global PREPARATION
        torch.set_num_threads(1)
        PREPARATION=(study,ActivationContextTrainer(original,recipe(math.ceil(ntrain/8))),ac)
        chunks=iter(entries[i:i+64] for i in range(0,len(entries),64));items=[];examples=[]
        with ProcessPoolExecutor(max_workers=args.prepare_workers,mp_context=multiprocessing.get_context('fork')) as pool:
            pending=deque(pool.submit(prepare_chunk,c) for c in [next(chunks,None) for _ in range(args.prepare_workers)] if c)
            while pending:
                batch,built=pending.popleft().result();items.extend(batch);examples.extend(built)
                if STOP:return 75
                phase('preparing reconstruction targets on CPU',passages=f'{len(items):,}/{len(entries):,}')
                c=next(chunks,None)
                if c:pending.append(pool.submit(prepare_chunk,c))
        PREPARATION=None
        prepared={'identity':identity,'train':items[:ntrain],'held':items[ntrain:],'examples':examples}
        accounting=token_accounting(prepared['train'])
        if not args.smoke and accounting['supervised_target_tokens_one_pass']<args.minimum_target_tokens:
            write(out/'source_shortfall.json',accounting)
            raise ValueError('Actual native supervised targets below planned warm-up minimum; never silently add epochs')
        publish_prepared(cache,manifest,identity,prepared)
    train,held=prepared['train'],prepared['held'];steps=math.ceil(len(train)/8);config=recipe(steps)
    accounting=token_accounting(train)
    if not args.smoke and accounting['planned_target_token_exposures']<args.minimum_target_tokens:raise ValueError('Frozen target budget is too small')
    write(out/'token_accounting.json',accounting)
    config_identity=json.loads(json.dumps(asdict(config)))
    if (out/'training_config.json').exists() and json.loads((out/'training_config.json').read_text())!=config_identity:
        raise ValueError('Training math or schedule changed')
    write(out/'training_config.json',config_identity)
    write(out/'recipe.json',dict(accounting,phase=identity['phase'],initializer=pair,ratios_range=identity['ratios_range'],
        sources=dict(Counter(i.info['source'] for i in train)),held_items=len(held),training_config=config_identity))
    if args.prepare_only:
        phase('CPU preparation complete; waiting for training',**accounting);return 0
    markers=out/'AC_MODELS'/AC;resume=None
    if any((markers/name).exists() for name in ('partial_epoch.json','completed_epoch.json')):
        resume=resume_checkpoint(out,steps,1)
        if not resume['remaining_epochs']:phase('complete',**accounting);return 0
        resume.pop('optimizer_state',None)
        h=harness(resume['ac_path'],resume['reader_path']);ac=h.module_manager.get_ac_model(AC);ac.epoch_number=resume['saved_epoch']
        study=ActivationContextStudyGenerator(h,AC)
        config=replace(config,resume_optimizer=True,resume_skip_steps=resume['skip_steps'],num_epochs=resume['remaining_epochs'])
        # An unpublished final directory must not move native checkpoint numbering.
        for parent in (out/'AC_MODELS'/AC,out/'LORAS'/READER):
            for path in parent.glob('epoch_*'):
                if path.name[6:].isdigit() and int(path.name[6:])>=resume['first_epoch']:
                    path.rename(path.with_name(f'incomplete_{path.name}_{time.time_ns()}'))
    else:h=original
    config=replace(config,max_duration_seconds=args.seconds)
    class WarmupTrainer(ActivationContextTrainer):
        def build_example(self,model,item,reference=False):
            return lookup[id(item)] if not reference and id(item) in lookup else super().build_example(model,item,reference)
        def _check_capacity(self,model,examples):
            examples=list(examples)
            if all(id(e.item) in lookup and lookup[id(e.item)] is e for e in examples):
                limits=[]
                for base in (model.target.model,model.side.model):
                    c=base.config.get_text_config() if hasattr(base.config,'get_text_config') else base.config
                    limits.append(getattr(c,'max_position_embeddings',0))
                if limits[0]>=72000 and limits[1]>=65536:return
            super()._check_capacity(model,examples)
        def _check_training_deadline(self):
            if STOP:self._training_deadline=0.
            super()._check_training_deadline()
    lookup={id(e.item):e for e in prepared['examples']}
    trainer=WarmupTrainer(h,config)
    if resume:trainer._next_epoch_number=lambda model:resume['first_epoch']
    sft=ActivationContextTrainer(h,replace(config,loss_kind='sft',no_context_penalty_weight=0,resume_optimizer=False))
    drift=ActivationContextTrainer(h,replace(config,loss_kind='kl',no_context_penalty_weight=0,resume_optimizer=False,teacher_cache_block_examples=None))
    noctx=[replace(i,in_context_messages=i.ac_messages,in_context_start=i.ac_start)
           for i in study.transform_for_eval_reference(held,'no_context')]
    learning=random.Random(20260931).sample(train,min(24,len(train)))
    qualitative=list({i.info['source']:i for i in reversed(held)}.values())[:4]
    order=list(range(len(train)));random.Random(20260925).shuffle(order)
    target_prefix=[0];source_prefix=[0]
    for index in order:
        target_prefix.append(target_prefix[-1]+sum(map(len,train[index].assistant_token_ids)))
        source_prefix.append(source_prefix[-1]+train[index].info['source_tokens'])
    panels_path=report/'panels.json';panels=json.loads(panels_path.read_text()) if panels_path.exists() else []
    published=min(len(train),resume['updates']*8) if resume else 0
    panels=[p for p in panels if p['passages presented']<=published]
    write(panels_path,panels)
    r.set_text('recipe','Pre-mixing warm-up recipe',json.dumps(json.loads((out/'recipe.json').read_text()),indent=2))
    fixed_path=report/'fixed_references.json'
    def panel(progress,summary):
        sft._training_deadline=drift._training_deadline=trainer._training_deadline
        n=min(len(train),progress.global_step*8);phase('evaluating fixed reconstruction panel',target_tokens_presented=target_prefix[n])
        if fixed_path.exists():
            fixed=json.loads(fixed_path.read_text())
            if fixed['identity']!=identity:raise ValueError('Frozen baseline identity changed')
        else:
            require_initial_weights(resume)
            fixed={'identity':identity,'views':{}}
            for view in ('full_context','no_context'):
                result=sft.eval(AC,study.transform_for_eval_reference(held,view),release=False)
                write(report/f'frozen_{view}.json',result);fixed['views'][view]=result['gold_nll_token']
            write(fixed_path,fixed)
        if not drift.teacher_cache:
            drift._store_teacher_targets(ac,[drift.build_example(ac,i) for i in noctx],ac.device,lora=None)
        drift_result=drift.eval(AC,noctx,release=False,teacher='stored')
        learned=sft.eval(AC,learning,release=False)
        positions=sum(p['positions'] for p in summary['per_item'])
        accuracy=sum(p['token_accuracy']*p['positions'] for p in summary['per_item'])/positions
        drift_kl=sum(p['kl']*p['positions'] for p in drift_result['per_item'])/sum(p['positions'] for p in drift_result['per_item'])
        record={'passages presented':n,'target tokens presented':target_prefix[n],'source token exposures':source_prefix[n],
            'epoch':n/len(train),'held NLL':summary['gold_nll_token'],'training NLL':learned['gold_nll_token'],
            'token accuracy':accuracy,'no-context KL':drift_kl,'quality gate':summary['gold_nll_token']<.2}
        panels[:]=[p for p in panels if p['passages presented']!=n]+[record];panels.sort(key=lambda p:p['passages presented'])
        write(panels_path,panels);write(report/f'panel_{n:08d}.json',{'held':summary,'learning':learned,'no_context':drift_result,'counts':record})
        xs=[p['target tokens presented'] for p in panels]
        r.widgets['quality']['series']=[{'name':'AC','x':xs,'y':[p['held NLL'] for p in panels]}]+[
            {'name':'frozen '+name,'x':xs,'y':[value]*len(xs)} for name,value in fixed['views'].items()]+[
            {'name':'quality gate <0.2','x':xs,'y':[.2]*len(xs)}]
        r.widgets['learning']['series']=[{'name':name,'x':xs,'y':[p[field] for p in panels]}
            for name,field in [('training NLL','training NLL'),('no-context KL','no-context KL')]]
        r.widgets['panels']['rows']=panels
        if n in (0,len(train)) or (.5*len(train)<=n and not (report/'greedy_halfway.json').exists()):
            stage='initial' if n==0 else 'final' if n==len(train) else 'halfway'
            path=report/f'greedy_{stage}.json'
            if not path.exists():
                samples=sft._sample_completions(ac,qualitative,progress);write(path,{'passages_presented':n,'cap':512,'samples':samples})
            attach_greedy(r,path,qualitative)
        phase('quality panel complete',**record)
    class Progress(ActivationContextTrainingReporter):
        def report_step(self,progress,stats):
            global STOP
            super().report_step(progress,stats)
            if args.stop_after_updates is not None and progress.global_step>=args.stop_after_updates:STOP=True
            n=min(len(train),progress.global_step*8)
            phase('training',training=f'{len(train):,} passages × {n/len(train):.3f} epochs',
                target_tokens_presented=target_prefix[n],source_token_exposures=source_prefix[n],
                unique_source_tokens=accounting['unique_source_tokens'],loss=stats.loss[-1],
                no_context_KL=stats.penalty[-1] if stats.penalty else None)
    native=Progress(str(report/'native'),'Reconstruction training detail','Native paired optimizer/RNG checkpoints.')
    stats=trainer.train(AC,train,held,reporter=native,reporting_callback=panel)
    result=stats.summarize();result.update(accounting,phase=identity['phase'],
        target_tokens_presented=target_prefix[min(len(train),trainer.updates_done.get(AC,0)*8)],
        quality_gate=panels[-1]['quality gate'] if panels else None)
    write(report/'result.json',result)
    if args.smoke and stats.steps:
        assert all(math.isfinite(x) for x in stats.loss)
        assert all(math.isfinite(x) for x in stats.grad_norm_ac+stats.grad_norm_reader)
        assert max(stats.grad_norm_ac)>0 and max(stats.grad_norm_reader)>0
        write(out/f'gpu_gate_{trainer.updates_done.get(AC,0):03d}.json',
            {'finite_losses':True,'nonzero_encoder_and_reader_gradients':True,'attempt_updates':stats.steps,
             'total_updates':trainer.updates_done.get(AC,0),'resumed_from':stats.resumed_from,
             'stop_reason':stats.stop_reason,'peak_memory_bytes':stats.peak_memory_bytes})
    phase('complete' if stats.stop_reason=='complete' else 'paused with paired checkpoint',
          target_tokens_presented=result['target_tokens_presented'],quality_gate=result['quality_gate'])
    return 0 if stats.stop_reason=='complete' else 75


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--material',type=Path);p.add_argument('--initial-pair',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--held-per-source',type=int,default=16)
    p.add_argument('--minimum-target-tokens',type=int,default=100_000_000);p.add_argument('--seconds',type=int,default=20700)
    p.add_argument('--prepare-workers',type=int,default=8);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--require-prepared',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after-updates',type=int,help='Private interruption gate only; preserves the full planned schedule')
    args=p.parse_args()
    if args.material is None or args.initial_pair is None:p.error('The warm-up needs --material and --initial-pair')
    if args.prepare_workers<1:p.error('--prepare-workers must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    try:raise SystemExit(main(args))
    except Exception as error:
        import traceback
        write(args.output/'failure.json',{'error':str(error),'traceback':traceback.format_exc()});raise
