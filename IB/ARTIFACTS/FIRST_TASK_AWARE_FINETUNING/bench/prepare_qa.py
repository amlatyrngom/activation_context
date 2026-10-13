"""Synthesize only from raw corpus text; no benchmark questions, answers or task traces."""
import os,json,time,random,collections,hashlib,importlib.util
from pathlib import Path
from dataclasses import asdict
ROOT=Path(os.environ.get('QA_SOURCE_ROOT','/workspace'))
OUT=Path(os.environ.get('QA_REPORT_ROOT',ROOT/'IB/TMP/SYNC/TASK_AWARE_QA'))/'preparation'
spec=importlib.util.spec_from_file_location('qa_reporting',ROOT/'activation/common/reporting.py'); reporting=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporting)

def main():
    from activation.harness import HarnessRuntime,HarnessRuntimeConfig,ModelConfig
    from activation.dataset.dataset import LoadedDataset,DatasetDocument
    from vllm.sampling_params import StructuredOutputsParams
    OUT.mkdir(parents=True,exist_ok=True)
    r=reporting.HtmlReporter(str(OUT),'Synthesized QA preparation','Raw code, financial/report text, web, documents and a small wiki slice. No native questions or answers.',min_render_interval_seconds=2)
    start=time.monotonic();r.set_status(phase='loading corpus');r.render(force=True)
    raw=json.loads(Path(os.environ['QA_RAW_MATERIAL']).read_text()); pool=raw['train']; rng=random.Random(20260921)
    financial=[p for p in pool if p['source']=='docs' and p['doc_id'].startswith('loong/') and p['doc_id'].endswith('.txt') and sum(c.isdigit() for c in p['text'])>30]
    technical=[p for p in pool if p['source']=='docs' and p not in financial]
    pools={'code':[p for p in pool if p['source']=='code' and p['doc_id'].endswith(('.py','.java','.cpp','.h','.js','.ts'))],
           'numeric':financial,'web':[p for p in pool if p['source']=='web'],'documents':technical,'wiki':raw['wiki']}
    counts={'code':600,'numeric':500,'web':400,'documents':400,'wiki':100}
    selected={}
    for domain,values in pools.items():
        rng.shuffle(values);selected[domain]=values[:counts[domain]]
    report={'requested_sources':counts,'selected_sources':{k:len(v) for k,v in selected.items()},'omitted':['task-conditioned tool traces','Nemotron math solutions'], 'generator':'Qwen/Qwen3.5-9B; vLLM dynamic FP8', 'seed':20260921}
    reporting._write_atomic(OUT/'recipe.json',json.dumps(report,indent=2));r.set_text('recipe','Source selection',json.dumps(report,indent=2))
    h=HarnessRuntime(HarnessRuntimeConfig(model_configs={'qa':ModelConfig('qa','Qwen/Qwen3.5-9B',engine_kwargs={'quantization':'fp8','enable_lora':False,'max_model_len':8192,'gpu_memory_utilization':.85,'max_num_seqs':64})},
        dataset_study_qa_model_name='qa',dataset_study_label_model_name='qa',dataset_study_qa_batch_size=32,
        dataset_study_chat_kwargs={'chat_template_kwargs':{'enable_thinking':False},'sampling_params':{'max_tokens':384,'temperature':.6}},
        doc_chunk_size_chars=12000,doc_chunk_overlap_chars=0))
    accepted=[];attempted=0
    for domain,values in selected.items():
        # Existing reconstruction windows are raw text. They are not passed off as whole original documents.
        dataset=LoadedDataset(dataset_id='taskaware_'+domain)
        originals={}
        for i,p in enumerate(values):
            ident=hashlib.sha256(p['text'].encode()).hexdigest()[:20]
            dataset.documents[ident]=DatasetDocument(ident,dataset.dataset_id,text=p['text'])
            originals[ident]=p['doc_id']
        h.dataset_manager.register_dataset(dataset);h.dataset_manager._get_or_create_index(dataset.dataset_id).build_bm25_index()
        def progress(status):
            status={ {'accepted':'generated_questions','completed':'source_attempts_completed','total':'source_attempts_planned','cached':'cached_source_attempts'}.get(k,k):v for k,v in status.items()}
            r.status.clear();r.set_status(domain=domain,**status,verified_previous_domains=len(accepted),preparation_elapsed_minutes=f'{(time.monotonic()-start)/60:.1f}');r.render(force=True)
        examples=h.dataset_manager.synthesize_study_examples_qa(dataset.dataset_id,len(dataset.documents),base_seed=20260921,caching_id='taskaware_v1',
            source_mode='chunk',max_source_tokens=2400,prompt_kind={'numeric':'numeric','code':'code'}.get(domain,'qa'),
            study_context='Create a useful grounded task with a concise answer (typically 2–30 words). All evidence needed must be present. Avoid general trivia. For numerical questions use supplied values, not outside knowledge.',progress=progress)
        # One bounded batched grounding check; no rewrite or recursive judging.
        model=h.loaded_models['qa'];valid=[]
        schema={'type':'object','properties':{'supported':{'type':'boolean'}},'required':['supported'],'additionalProperties':False}
        for begin in range(0,len(examples),64):
            batch=examples[begin:begin+64];r.set_status(phase='checking answers',checked=f'{begin}/{len(examples)}');r.render(force=True)
            chats=[[{'role':'system','content':'Check whether the answer correctly answers the question using only the supplied evidence. Reject ambiguous, incorrect, or unsupported answers. Return JSON {"supported":true or false}.'},
                    {'role':'user','content':e.study_source['text']+'\nQuestion: '+e.query+'\nAnswer: '+e.gold_answers[0]}] for e in batch]
            outs=model.engine_chat_many(chats,chat_kwargs={'chat_template_kwargs':{'enable_thinking':False},'sampling_params':{'max_tokens':48,'temperature':0,'structured_outputs':StructuredOutputsParams(json=schema)}})
            for e,o in zip(batch,outs):
                try: good=json.loads(o.text).get('supported') is True
                except ValueError: good=False
                if good:valid.append({'domain':domain,'example':asdict(e),'original_doc_ids':[originals[u['doc_id']] for u in e.study_source['units']]})
        accepted.extend(valid);attempted+=len(dataset.documents)
        reporting._write_atomic(OUT/'accepted.json',json.dumps(accepted,ensure_ascii=False))
        r.set_text('examples','Recent accepted questions',json.dumps(valid[:5],indent=2,ensure_ascii=False));r.set_status(accepted_total=len(accepted),source_attempts=attempted);r.render(force=True)
    # Reserve complete synthesis request groups, so sibling QAs cannot cross the train/report boundary.
    grouped=collections.defaultdict(list)
    for row in accepted:grouped[(row['domain'],row['example']['study_source']['request_id'])].append(row)
    bydomain=collections.defaultdict(list)
    for key,rows in grouped.items():bydomain[key[0]].append(rows)
    held=[];training=[]
    for domain,groups in bydomain.items():
        rng.shuffle(groups);target=max(1,round(100*counts[domain]/sum(counts.values())));held_domain=[]
        for group in groups:
            if len(held_domain)<target:held_domain.extend(group)
            else:training.extend(group)
        held.extend(held_domain)
    # Deterministic interleave across domains; trainer preserves this order.
    queues={d:[row for row in training if row['domain']==d] for d in counts};training=[]
    while any(queues.values()):
        for domain in counts:
            if queues[domain]:training.append(queues[domain].pop())
    material={'train':training,'held':held,'recipe':report,'preparation_seconds':time.monotonic()-start,
              'accepted_by_domain':dict(collections.Counter(row['domain'] for row in accepted))}
    reporting._write_atomic(OUT/'material.json',json.dumps(material,ensure_ascii=False));r.set_status(phase='ready for training',train_questions=len(training),held_questions=len(held),preparation_elapsed_minutes=f'{(time.monotonic()-start)/60:.1f}');r.finish()
    from activation.harness.hf_utils import FREE_DEVICE
    h.loaded_models['qa'].engine_to_device(FREE_DEVICE)
    print('QA_MATERIAL_READY',len(training),len(held),flush=True)
if __name__=='__main__':main()
