"""Private, reproducible RAG AC experiment; production supplies models, loss, evaluation and reports.

Commands: `prepare` (shared QA material), `run --run S` (smoke run: 256 items, 2 epochs, also the timing calibration),
`budget` (derive the arm size that fits two hours from the smoke run), `run --run A|B|C|D` (the ablation arms), `overview`.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import string
import threading
import time
import traceback

ROOT = Path(os.environ.get('RAG_POC_SOURCE_ROOT') or Path(__file__).resolve().parents[4])
REPORT_ROOT = Path(os.environ.get('RAG_POC_ROOT', ROOT / 'IB/TMP/SYNC/RAG_AC_POC'))
MODEL = 'Qwen/Qwen3.5-4B'
NAME, AC, SIDE, READER = 'qwen4b', 'rag_poc', 'rag_side', 'rag_reader'
SEED = 20260915
TOP_K = 256
# Arm -> (encoder LR, reader LR, loss). S is the smoke run with A's settings.
RUNS = {'S': (1e-4, 2e-5, 'kl'), 'A': (1e-4, 2e-5, 'kl'), 'B': (1e-4, 0.0, 'kl'), 'C': (5e-4, 2e-5, 'kl'), 'D': (1e-4, 2e-5, 'sft')}
SMOKE = {'train_items': 256, 'planned_passes': 2, 'reporting_interval': 0.25, 'completion_samples': 24}
ARM_DEFAULTS = {'planned_passes': 2, 'reporting_interval': 0.25, 'completion_samples': 100}
BUDGET_SECONDS = 7200
spec = importlib.util.spec_from_file_location('poc_reporting', ROOT / 'activation/common/reporting.py')
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    reporting._write_atomic(path, json.dumps(value, indent=2, allow_nan=False, default=str))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def heartbeat(folder):
    stop = threading.Event()
    started = time.time()
    def beat():
        while True:
            write_json(folder / 'heartbeat.json', {'pid': os.getpid(), 'started': started, 'updated': time.time()})
            if stop.wait(10): break
    thread = threading.Thread(target=beat, daemon=True); thread.start()
    try: yield
    finally: stop.set(); thread.join()


def plain_report(folder, title):
    return reporting.HtmlReporter(str(folder), title, 'RAG AC: 4B / 4B, depth 1, 1/16, no distractors.', min_render_interval_seconds=2)


def phase(reporter, name, **values):
    reporter.set_status(phase=name, **values); reporter.render(force=True)


def harness(with_ac=False, initial=None):
    import torch
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.ac_model import ActivationContextModelConfig
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    h = HarnessRuntime(HarnessRuntimeConfig(
        model_configs={NAME: ModelConfig(NAME, MODEL, engine_kwargs={'max_model_len': 4096, 'gpu_memory_utilization': .75})},
        doc_chunk_size_chars=2048, doc_chunk_overlap_chars=0,
        dataset_study_qa_model_name=NAME, dataset_study_qa_batch_size=64,
        dataset_study_chat_kwargs={'chat_template_kwargs': {'enable_thinking': False},
                                  'sampling_params': {'max_tokens': 512, 'temperature': .6}},
    ))
    if with_ac:
        h.module_manager.register_lora(READER, NAME, rank=64, dropout=0,
            checkpoint_path=str(initial/'reader') if initial else None)
        h.module_manager.register_ac_model(ActivationContextModelConfig(AC, NAME, SIDE, NAME, READER,
            min_view_rows=2, default_compression_ratio=1/16,
            checkpoint_path=str(initial/'ac') if initial else None))
    return h


def split_documents(documents):
    """Group exact-normalized duplicates and passages sharing 32 consecutive words."""
    docs = list(documents.values()); parent = list(range(len(docs))); seen = {}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, doc in enumerate(docs):
        words = re.findall(r'\w+', doc.text.casefold())
        keys = [hashlib.sha256(' '.join(words).encode()).digest()]
        keys += [hashlib.sha256(' '.join(words[j:j+32]).encode()).digest() for j in range(len(words)-31)]
        for key in keys:
            previous = seen.setdefault(key, i)
            parent[find(i)] = find(previous)
    groups = {}
    for i, doc in enumerate(docs): groups.setdefault(find(i), []).append(doc)
    values = list(groups.values()); random.Random(SEED).shuffle(values)
    held, train = [], []
    for group in values:
        (held if len(held) < 300 else train).extend(group)
    return train, held


def build_item(h, record, index, split):
    from activation.ac_model import ActivationContextTrainingItem
    from activation.ac_model.ac_model_utils import assistant_target_ids
    from activation.agent.agent_utils import ModelDialect
    from activation.common.ac_parts import ac_part
    from activation.dataset.loaders.trajectory_utils import submit_answer_definition
    tok = h.loaded_models[NAME].tokenizer
    question, answer, text = record['question'], record['answer'], record['passage']
    output = ModelDialect.for_tokenizer(tok).rendering.assistant_message('', [
        {'id':'answer','name':'submit_answer','arguments':{'answer':answer}}])
    tools = [submit_answer_definition()]
    prefix = f'Question: {question}\n\nPassage:\n'
    suffix = '\n\nAnswer from the passage. Call submit_answer with a concise answer.'
    part = ac_part([{'role':'user','content':f'Question: {question}\n\nPassage:\n{text}'}], AC, 1/16)
    system = {'role':'system','content':'You answer questions using the supplied passage.'}
    teacher = [system, {'role':'user','content':prefix+text+suffix}, output]
    student = [deepcopy(system), {'role':'user','content':[{'type':'text','text':prefix}, part, {'type':'text','text':suffix}]}, deepcopy(output)]
    return ActivationContextTrainingItem(f'nq:{split}:{index}', 'rag_qa', teacher, student, 2, 2,
        [assistant_target_ids(tok, output, tools)], tools=tools, dataset_id='nq_poc', doc_ids=[record['doc_id']],
        info={'answer':answer,'question':question,'passage':text,'depth':1,'ratio':1/16,'passages':1,
              'chat_template_kwargs':{'enable_thinking':False}})


def without_context(item):
    """The same item with the compressed passage removed: the no-context control."""
    empty = deepcopy(item.ac_messages)
    empty[1]['content'] = [p for p in empty[1]['content'] if p.get('type') != 'activation_context']
    return replace(item, item_id=item.item_id + ':no_context', ac_messages=empty)


def prepare():
    from activation.dataset.loaders import NqDataset
    from activation.dataset.dataset import LoadedDataset, DataModality
    from activation.harness import FREE_DEVICE
    folder = REPORT_ROOT/'preparation'; r = plain_report(folder,'RAG AC — preparation'); start=time.monotonic()
    with heartbeat(folder):
        phase(r,'initializing tokenizer and harness')
        h=harness()
        phase(r,'loading NQ passages',requested=6000)
        corpus=NqDataset.load(h,max_examples=6000,max_corpus_documents=0)
        phase(r,'grouping overlapping passages',loaded=len(corpus.documents))
        train_docs, held_docs=split_documents(corpus.documents)
        groups={'train':train_docs,'held':held_docs}; material={}
        # Separate indexes prevent the study sampler from crossing the persisted source split.
        h.dataset_manager.loaded_datasets.clear()
        for split,docs in groups.items():
            dataset=LoadedDataset(dataset_id='nq_poc_'+split,documents={d.doc_id:replace(d,dataset_id='nq_poc_'+split,chunks={}) for d in docs})
            h.dataset_manager.register_dataset(dataset)
        phase(r,'building passage indexes',train_sources=len(train_docs),held_sources=len(held_docs))
        h.dataset_manager.build_bm25_indexes()
        write_json(folder/'source_split.json',{s:[d.doc_id for d in docs] for s,docs in groups.items()})
        train_goal=int(os.environ.get('RAG_POC_TRAIN_GOAL','2048'))
        for split,goal in [('held',100),('train',train_goal)]:
            dataset_id='nq_poc_'+split
            def update(values):
                phase(r,values.pop('phase'),split=split,**values)
                if time.monotonic()-start > 1800:
                    raise RuntimeError('Shared preparation exceeded its 30-minute budget; stop before training')
            manager=h.dataset_manager
            count=160 if split=='held' else goal*5//4
            count=min(count,len(manager.dataset_indexes[dataset_id].chunk_ids))
            examples=manager.synthesize_study_examples_qa(dataset_id,count,base_seed=SEED,
                study_context='Ask a specific question answerable from this passage. The answer should be a short phrase or one concise sentence, at most 40 words.',
                caching_id='rag_ac_poc_v1',modality=DataModality.TEXT,progress=update)
            if len(examples)<(goal if split=='held' else goal*3//4): raise RuntimeError(f'{split}: only {len(examples)}/{goal} accepted QA pairs')
            if len(examples)<goal: phase(r,'accepting fewer training items than the goal',split=split,accepted=len(examples),goal=goal)
            material[split]=[{'doc_id':e.positive_doc_ids[0],'chunk_id':e.positive_chunk_ids[0],
                'question':e.query,'answer':e.gold_answers[0],
                'passage':manager.dataset_indexes[dataset_id].chunks[e.positive_chunk_ids[0]].chunk_text} for e in examples[:goal]]
        assert not ({e['doc_id'] for e in material['held']} & {e['doc_id'] for e in material['train']})
        material['preparation_seconds']=time.monotonic()-start
        write_json(folder/'material.json',material)
        h.loaded_models[NAME].engine_to_device(FREE_DEVICE)
        phase(r,'preparation complete',train_items=len(material['train']),held_items=len(material['held']),seconds=f'{material["preparation_seconds"]:.1f}')
        r.finish()


def parameter_hash(ac,h):
    import torch
    params=ac.trainable_parameters()+h.module_manager.lora_parameters(READER)
    value=hashlib.sha256()
    for p in params:
        value.update(str(tuple(p.shape)).encode()); value.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


def answer_positions(tokenizer,item):
    """Select tokens overlapping the serialized answer's value, excluding call framing."""
    ids=item.assistant_token_ids[0]
    text=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    answer=item.info['answer']
    if '<parameter=answer>' in text:
        payload_start=text.index('<parameter=answer>')+len('<parameter=answer>')
        payload_end=text.index('</parameter>',payload_start)
        start=text.find(answer,payload_start,payload_end)
        if start < 0: raise ValueError(f'Answer differs from serialized payload: {item.item_id}')
    else:
        start=text.rfind(answer)
        if start < 0: raise ValueError(f'Cannot locate answer: {item.item_id}')
    end=start+len(answer)
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids'] != ids: raise ValueError(f'Answer mask token roundtrip differs: {item.item_id}')
    indexes=[i for i,(lo,hi) in enumerate(encoded['offset_mapping']) if lo<end and hi>start]
    if not indexes: raise ValueError('Empty answer mask')
    return indexes


def normalize_answer(text):
    """SQuAD-style: lower case, no punctuation, no articles, single spaces."""
    text=''.join(ch for ch in text.casefold() if ch not in string.punctuation)
    text=re.sub(r'\b(a|an|the)\b',' ',text)
    return ' '.join(text.split())


def extract_answer(text):
    """The submitted answer inside a generated completion, or the whole text when no call is found."""
    for pattern in (r'<parameter=answer>(.*?)</parameter>', r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"'):
        found=re.search(pattern,text,re.S)
        if found: return found.group(1).strip()
    return text.strip()


def diagnostics(trainer,ac,items,reporter,label,controls=False):
    """Exact full-vocabulary KL between reader states, whole call and answer tokens only; secondary to the trainer's curves."""
    import torch
    from activation.ac_model.ac_model import TRAINING
    from activation.agent_training.agent_training_utils import disable_dropout
    device=ac.prepare(); ac.set_mode(TRAINING); target=ac.target; base=target.model
    was_training=base.training; base.eval(); disable_dropout(base)
    embedding=base.get_input_embeddings(); head=base.get_output_embeddings()
    rows=[]; started=time.monotonic()
    try:
        with torch.no_grad():
            for index,item in enumerate(items):
                reporter.report_evaluation_progress(label,index,len(items))
                ex=trainer.build_example(ac,item)
                def states(ids,positions,lora):
                    tensor=torch.tensor(ids,device=device)
                    return target.decoder_forward(embedding(tensor)[None],None,lora_name=lora,gradient_checkpointing=False)[0][positions]
                frozen=states(ex.teacher_ids,ex.teacher_positions,None)
                current=states(ex.teacher_ids,ex.teacher_positions,READER)
                encoded=ac.encode_batch(ex.part_requests)
                embs=embedding(torch.tensor(ex.student_ids,device=device)); pieces=[]; cursor=0
                for (lo,hi),row in zip(ex.spans,encoded):
                    pieces.extend([embs[cursor:lo],row.to(embs.dtype)]);cursor=hi
                pieces.append(embs[cursor:]); embs=torch.cat(pieces)
                student=target.decoder_forward(embs[None],None,lora_name=READER,gradient_checkpointing=False)[0][ex.student_positions]
                mask=answer_positions(target.tokenizer,item)
                pairs={'current_ac':(current,student),'fixed_ac':(frozen,student),'fixed_full_text':(frozen,current)}
                if controls:
                    no_ex=trainer.build_example(ac,without_context(item))
                    no_states=states(no_ex.student_ids,no_ex.student_positions,READER)
                    pairs['fixed_no_context']=(frozen,no_states)
                    pairs['current_no_context']=(current,no_states)
                result={'item_id':item.item_id,'positions':len(ex.target_ids),'answer_positions':len(mask),
                        'source_tokens':len(target.tokenizer.encode(item.info['passage'],add_special_tokens=False)),
                        'teacher_tokens':len(ex.teacher_ids),'student_tokens':len(ex.student_ids),
                        'rows':sum(len(x) for x in encoded)}
                # Each unique output head is computed once per chunk, then reused by all comparisons.
                from activation.agent_training.agent_training_utils import head_logits
                import torch.nn.functional as F
                totals={name:[0.,0.] for name in pairs}
                gold={name:[0.,0.] for name in ('teacher','student','current_full_text','no_context')}
                targets=torch.tensor(ex.target_ids,device=device)
                chunk=getattr(trainer,'_logits_chunk_tokens',trainer.config.logits_chunk_tokens or 64)
                for lo in range(0,len(ex.target_ids),chunk):
                    hi=min(len(ex.target_ids),lo+chunk)
                    logps={}
                    for teacher,reader in pairs.values():
                        for hidden in (teacher,reader):
                            if id(hidden) not in logps:
                                logps[id(hidden)]=F.log_softmax(head_logits(hidden[lo:hi],head,trainer.config.fp32_head_matmul),dim=-1)
                    selected=[pos-lo for pos in mask if lo<=pos<hi]
                    for name,(teacher,reader) in pairs.items():
                        t,s=logps[id(teacher)],logps[id(reader)]
                        kl=(t.exp()*(t-s)).sum(dim=-1)
                        totals[name][0]+=float(kl.sum())
                        if selected:totals[name][1]+=float(kl[selected].sum())
                    readers={'teacher':frozen,'student':student,'current_full_text':current}
                    if controls: readers['no_context']=no_states
                    for name,hidden in readers.items():
                        nll=-logps[id(hidden)].gather(-1,targets[lo:hi,None])[:,0]
                        gold[name][0]+=float(nll.sum())
                        if selected:gold[name][1]+=float(nll[selected].sum())
                for name,(whole,answer) in totals.items():
                    result[name]=whole/len(ex.target_ids)
                    result[name+'_answer']=answer/len(mask)
                for name,(whole,answer) in gold.items():
                    if name in ('no_context',) and not controls: continue
                    result['gold_nll_'+name]=whole/len(ex.target_ids)
                    result['gold_nll_'+name+'_answer']=answer/len(mask)
                rows.append(result)
            reporter.report_evaluation_progress(label,len(items),len(items))
    finally:
        base.train(was_training); disable_dropout(base)
    keys=[key for key in rows[0] if key.startswith(('fixed_','current_','gold_nll_'))] if rows else []
    return {'items':len(rows),'seconds':time.monotonic()-started,'means':{key:statistics.mean(r[key] for r in rows) for key in keys},'per_item':rows}


def config(run='A',**overrides):
    from activation.ac_model import ActivationContextTrainingConfig
    encoder_lr,reader_lr,loss=RUNS[run]
    values=dict(loss_kind=loss,learning_rate_ac=encoder_lr,learning_rate_target_lora=reader_lr,
        examples_per_update=8,num_epochs=2,warmup_updates=10,completion_samples=0,reporting_interval=0.25,seed=SEED,
        teacher_cache_top_k=TOP_K if loss=='kl' else 0,gradient_checkpointing_min_tokens=2048)
    values.update(overrides)
    return ActivationContextTrainingConfig(**values)


def settings_for(label):
    """Smoke run: fixed small shape. Arms: the size the smoke run's timing says fits two hours."""
    if label=='S': return dict(SMOKE)
    path=REPORT_ROOT/'preparation'/'calibration_next.json'
    if not path.exists(): raise RuntimeError('Run the smoke run (S) and `budget` before an ablation arm')
    return json.loads(path.read_text())


class RunReporter:
    """Built lazily so the production import happens inside the GPU process."""
    def __new__(cls,folder,title,items,tokenizer,elapsed):
        from activation.ac_model import ActivationContextTrainingReporter
        from activation.ac_model.ac_model_reporter import TEACHER_SERIES
        items_by_id={item.item_id:item for item in items}
        class _Reporter(ActivationContextTrainingReporter):
            def __init__(self):
                super().__init__(str(folder),title,'Qwen3.5-4B reads a 1/16-compressed NQ passage; the teacher is the same reader with the full '
                                 'passage, frozen at training start. Gold answers are NQ answers.',
                                 set_labels={'reporting':'AC student, held-out','validation':'no context, current reader, held-out'})
                self.initialize_line_plot('answer_nll','Gold-answer log-loss, answer tokens only',
                    'Same as the first plot but restricted to the tokens of the answer value inside the submit_answer call.',
                    'epochs in this report','nats per answer token',[])
                self.initialize_line_plot('exact_match','Exact match on held-out items',
                    'Greedy completion whose submitted answer equals the gold answer after SQuAD normalization.',
                    'epochs in this report','fraction',['AC student',TEACHER_SERIES])
                self.initialize_table('completions','Held-out completions','Greedy answers on fixed held-out items, refreshed at each epoch end.',
                    ['question','gold answer','AC student',TEACHER_SERIES])
                self.initialize_table('final','Final exact controls','Filled once after the last epoch: exact full-vocabulary KL and gold log-loss '
                    'over the held-out items; whole call and answer tokens only.',['comparison','items','KL','KL answer','gold NLL','gold NLL answer'])
                # Reading order: the fair metrics first, then the trainer's own widgets.
                self.set_text('summary','Run summary','')
                order=['summary','gold_nll','answer_nll','exact_match','reporting','loss','completions','final']
                self.widgets={name:self.widgets[name] for name in order+[n for n in self.widgets if n not in order]}
            def render(self,force=False):
                self.set_status(run_elapsed=f'{(time.monotonic()-elapsed)/60:.1f} min')
                super().render(force=force)
            def report_eval(self,progress,name,summary):
                super().report_eval(progress,name,summary)
                if not summary['items']: return
                label=self.label(name); values={'student':[],'teacher':[]}
                for row in summary['per_item']:
                    item=items_by_id[row['item_id'].removesuffix(':no_context')]
                    mask=answer_positions(tokenizer,item)
                    values['student'].append(-statistics.mean(row['target_logp'][i] for i in mask))
                    if row.get('teacher_target_logp') is not None:
                        values['teacher'].append(-statistics.mean(row['teacher_target_logp'][i] for i in mask))
                self._ensure_series('answer_nll',label)
                point={'x':self.epoch_x,label:statistics.mean(values['student'])}
                if values['teacher'] and name=='reporting':
                    self._ensure_series('answer_nll',TEACHER_SERIES); point[TEACHER_SERIES]=statistics.mean(values['teacher'])
                self.add_data_point('answer_nll',point); self.render()
            def report_completions(self,progress,rows):
                super().report_completions(progress,rows[:6])
                matches={'AC student':[],TEACHER_SERIES:[]}
                for row in rows:
                    gold=normalize_answer(items_by_id[row['item_id']].info['answer'])
                    matches['AC student'].append(normalize_answer(extract_answer(row['student']))==gold)
                    if row['teacher'] is not None: matches[TEACHER_SERIES].append(normalize_answer(extract_answer(row['teacher']))==gold)
                point={'x':self.epoch_x,'AC student':statistics.mean(matches['AC student'])}
                if matches[TEACHER_SERIES]: point[TEACHER_SERIES]=statistics.mean(matches[TEACHER_SERIES])
                self.add_data_point('exact_match',point)
                self.widgets['completions']['rows']=[{'question':items_by_id[r['item_id']].info['question'],'gold answer':items_by_id[r['item_id']].info['answer'],
                    'AC student':extract_answer(r['student']),TEACHER_SERIES:extract_answer(r['teacher']) if r['teacher'] is not None else ''} for r in rows[:6]]
                self.render()
        return _Reporter()


def run(label):
    import torch
    from activation.ac_model import ActivationContextTrainer
    folder=REPORT_ROOT/label; prep=REPORT_ROOT/'preparation'
    material=json.loads((prep/'material.json').read_text()); settings=settings_for(label)
    os.environ['ACTIVATION_SYNC_ROOT']=str(REPORT_ROOT/'states'/label)
    start=time.monotonic()
    boot=plain_report(folder,'RAG AC — '+label)
    with heartbeat(folder):
        phase(boot,'loading initial checkpoint')
        h=harness(True,prep/'initial');ac=h.module_manager.get_ac_model(AC);ac.prepare();h.module_manager.ensure_lora(READER)
        initial_hash=parameter_hash(ac,h)
        expected=json.loads((prep/'calibration.json').read_text()).get('initial_hash') if (prep/'calibration.json').exists() else None
        assert expected is None or initial_hash==expected,'Initial parameters differ from the shared initialization'
        reader_before=[p.detach().cpu().clone() for p in h.module_manager.lora_parameters(READER)]
        if len(material['train'])<settings['train_items']:
            raise RuntimeError(f"{settings['train_items']} training items requested, {len(material['train'])} prepared")
        train=[build_item(h,v,i,'train') for i,v in enumerate(material['train'][:settings['train_items']])]
        held=[build_item(h,v,i,'held') for i,v in enumerate(material['held'])]
        no_context=[without_context(item) for item in held]
        tokenizer=h.loaded_models[NAME].tokenizer
        r=RunReporter(folder,f'RAG AC — {label}: encoder {RUNS[label][0]:g}, reader {RUNS[label][1]:g}, {RUNS[label][2].upper()}',train+held,tokenizer,start)
        r.set_text('summary','Run summary','\n'.join([
            f'Arm {label} · encoder LR {RUNS[label][0]:g} · reader LR {RUNS[label][1]:g} · loss {RUNS[label][2].upper()}'
            + (f' to the training-start reader (top-{TOP_K} stored targets)' if RUNS[label][2]=='kl' else ' on the gold answer') + f' · seed {SEED}',
            f"Data · {len(train)} training items × {settings['planned_passes']} epochs = {math.ceil(len(train)/8)*settings['planned_passes']} updates of 8 · "
            f'{len(held)} held-out items per panel (every {settings["reporting_interval"]:g} epoch) · no-context control on the same held items',
            f'Initial adapters hash {initial_hash[:12]}… · teacher targets stored before the first update',
            '',
            'Reading guide · the first plots score every reader on the same gold tokens (lower is better); the KL plot is the training objective. '
            'The final exact controls table is filled after the last epoch.']))
        trainer=ActivationContextTrainer(h,config(label,num_epochs=settings['planned_passes'],reporting_interval=settings['reporting_interval'],
                                                  completion_samples=settings['completion_samples'],checkpoint_every_epoch=True))
        stats=trainer.train(AC,train,reporting_data=held,validation_data=no_context,reporter=r)
        write_json(folder/'training_stats.json',stats.summarize())
        phase(r,'final exact controls')
        exact_trainer=trainer if RUNS[label][2]=='kl' else ActivationContextTrainer(h,config('A'))
        final=diagnostics(exact_trainer,ac,held,r,'final exact controls',controls=True)
        write_json(folder/'final_controls.json',final)
        means=final['means']
        for comparison,kl,gold in [('AC student vs original base (the stored teacher when the reader started fresh)','fixed_ac','student'),
                                   ('trained reader with the full passage vs original base (reader drift)','fixed_full_text','current_full_text'),
                                   ('no context (trained reader) vs original base','fixed_no_context','no_context'),
                                   ('original base with the full passage (the teacher)',None,'teacher')]:
            r.add_data_point('final',{'comparison':comparison,'items':final['items'],'KL':round(means[kl],4) if kl else '','KL answer':round(means[kl+'_answer'],4) if kl else '',
                'gold NLL':round(means['gold_nll_'+gold],4),'gold NLL answer':round(means['gold_nll_'+gold+'_answer'],4)})
        phase(r,'final loss over all training items')
        full=trainer.eval(AC,train,release=False,reporter=r,reporting_name='final_all_training')
        write_json(folder/'full_training.json',{k:v for k,v in full.items() if k!='per_item'})
        if RUNS[label][1]==0.0:
            assert all(torch.equal(before,p.detach().cpu()) for before,p in zip(reader_before,h.module_manager.lora_parameters(READER))),'Fixed reader changed'
        result={'run':label,'configuration':asdict(config(label)),'settings':settings,'train_items':len(train),'held_items':len(held),
            'updates':trainer.updates_done.get(AC,0),'initial_hash':initial_hash,'final_hash':parameter_hash(ac,h),
            'checkpoint':stats.checkpoint_path,'teacher_targets':stats.teacher_targets,
            'final_controls_means':means,'full_training_loss':full['loss'],'full_training_gold_nll':full['gold_nll'],
            'last_reporting':{k:v for k,v in (stats.reporting or {}).items() if k!='per_item'},
            'duration_seconds':time.monotonic()-start}
        write_json(folder/'result.json',result)
        r.set_text('result','Final result',json.dumps({k:v for k,v in result.items() if k not in ('last_reporting',)},indent=2,default=str))
        phase(r,'complete',run_elapsed=f'{result["duration_seconds"]/60:.1f} min');r.finish()


def budget():
    """From the smoke run's measured rates, the arm size (multiple of 64, at most 4096) whose two epochs fit the two-hour budget."""
    prep=REPORT_ROOT/'preparation'; smoke=REPORT_ROOT/'S'
    stats=json.loads((smoke/'training_stats.json').read_text()); result=json.loads((smoke/'result.json').read_text())
    settings=SMOKE; items=settings['train_items']; held=result['held_items']; epochs=stats['epochs']
    panels_per_epoch=round(1/settings['reporting_interval'])
    train_s=sum(e['training_seconds'] for e in epochs)/(items*len(epochs))
    panel_s=sum(e['reporting_seconds'] for e in epochs)/(panels_per_epoch*held*len(epochs))
    validation_s=sum(e['validation_seconds'] for e in epochs)/(held*len(epochs))
    sample_s=sum(e['sample_seconds'] for e in epochs)/(settings['completion_samples']*len(epochs))   # student only
    checkpoint_s=max(e['checkpoint_seconds'] for e in epochs)
    targets_s=stats['teacher_targets']['seconds']/stats['teacher_targets']['examples']   # unique teacher views: the no-context items share the held items' teacher
    available=len(json.loads((prep/'material.json').read_text())['train'])
    baseline_s=stats['baseline_seconds']                     # 2 evals of `held` plus teacher+student samples of the smoke count
    load_s=result['duration_seconds']-stats['duration_s']-json.loads((smoke/'final_controls.json').read_text())['seconds']
    final_controls_s=json.loads((smoke/'final_controls.json').read_text())['seconds']
    arm=dict(ARM_DEFAULTS)
    def total(n):
        prep_s=load_s+(n+held)*targets_s+baseline_s-settings['completion_samples']*2*sample_s+arm['completion_samples']*2*sample_s
        epoch_s=n*train_s+panels_per_epoch*held*panel_s+held*validation_s+arm['completion_samples']*sample_s+checkpoint_s
        final_s=final_controls_s+n*panel_s+60
        return 1.15*(prep_s+arm['planned_passes']*epoch_s+final_s)
    fitting=[candidate for candidate in range(64,min(4096,available)+1,64) if total(candidate)<=BUDGET_SECONDS]
    if not fitting: raise RuntimeError(f'No arm size fits {BUDGET_SECONDS} s: 64 items would take {total(64):.0f} s')
    size=fitting[-1]
    arm.update(train_items=size,prepared_items=available,estimated_seconds=round(total(size)),rates={'train_s_per_item':train_s,'panel_s_per_item':panel_s,'validation_s_per_item':validation_s,
        'sample_s_per_item':sample_s,'targets_s_per_item':targets_s,'checkpoint_s':checkpoint_s,'load_s':load_s,'baseline_s':baseline_s,'final_controls_s':final_controls_s},
        smoke=settings,budget_seconds=BUDGET_SECONDS)
    write_json(prep/'calibration_next.json',arm)
    print(json.dumps(arm,indent=2))


def overview():
    r=plain_report(REPORT_ROOT,'RAG AC — overview')
    r.initialize_table('runs','Live progress','Source timestamps distinguish a live job from a fresh local rendering.',
        ['run','phase','global updates','last panel','source updated','heartbeat age s','last observed locally'])
    links=[];states=[]
    for name in ['preparation','S','A','B','C','D']:
        folder=REPORT_ROOT/name;path=folder/'report_data.json';beat=folder/'heartbeat.json'
        data=json.loads(path.read_text()) if path.exists() else {};status=data.get('status',{})
        hb=json.loads(beat.read_text()) if beat.exists() else {}
        age=round(time.time()-hb['updated']) if hb else ''
        states.append(status.get('phase','pending'))
        r.add_data_point('runs',{'run':name,'phase':status.get('phase','pending'),'global updates':status.get('global updates',''),
            'last panel':status.get('last panel',''),'source updated':data.get('updated_at',''),
            'heartbeat age s':age,'last observed locally':time.strftime('%H:%M:%S UTC',time.gmtime())})
        links.append({'title':name,'state':'finished' if data.get('finished') else 'running','stats':'; '.join(f'{k}: {v}' for k,v in status.items()),
                      'prompt':'','file':name+'/report.tressoir.html','steps':[]})
    r.set_trajectories('links','Open reports','Preparation, smoke run and ablation arms.',links)
    r.set_status(phase=' / '.join(states))
    r.render(force=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run','budget','overview']);parser.add_argument('--run',choices=list(RUNS))
    args=parser.parse_args()
    try:
        if args.command=='run':run(args.run)
        else:globals()[args.command]()
    except BaseException as error:
        folder=REPORT_ROOT/(args.run if args.command=='run' else 'preparation')
        write_json(folder/'error.json',{'error':repr(error),'traceback':traceback.format_exc(),'time':time.time()})
        path=folder/'report_data.json'
        if path.exists():
            data=json.loads(path.read_text());data['status'].update(phase='error',error=repr(error));data['finished']=True
            write_json(path,data)
            reporting._write_atomic(folder/'report.tressoir.html',reporting.render_report_page(data))
        else:
            r=plain_report(folder,'RAG AC — error');phase(r,'error',error=repr(error));r.finish()
        raise
