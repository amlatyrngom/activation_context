"""Fixed family-balanced100 decodes; preserve the original200 development NLL batching."""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import time

from evaluate_qualitative_cohort import read, write, digest, parsed_answer
from evaluate_data_confirmation import ARMS, final_pair

BENCH_SHA = 'b6d22e42ebd6d89b5b2f510d4bc44df426d462444dc041122bcd3ae58eaa8744'
QUOTAS = {'code':30,'math_arxiv':12,'math_textbook':8,'financial':5,'web':20,
          'technical_web':5,'documents':8,'legal':6,'office':6}
SELECTION_KEY = 'data-budget-cohort100-v1:'
LABELS = {'data_2k_4ep':'2,000 questions × 4 passes', 'data_4k_2ep':'4,000 questions × 2 passes',
          'data_8k_1ep':'8,000 questions × 1 pass', 'frozen_p2':'Frozen P2 reader before task-aware training'}


def selection(material):
    """Only immutable item IDs and family membership determine membership; preserve200 order."""
    groups = defaultdict(list)
    held = material['held']
    assert len(held) == len({r['example']['example_id'] for r in held}) == 200
    for row in held:
        groups[row['source_family']].append(row['example']['example_id'])
    assert {k:len(v) for k,v in groups.items()} == {k:2*v for k,v in QUOTAS.items()}
    chosen = set()
    for family, count in QUOTAS.items():
        chosen.update(sorted(groups[family], key=lambda ident:(hashlib.sha256((SELECTION_KEY+ident).encode()).hexdigest(),ident))[:count])
    return [r['example']['example_id'] for r in held if r['example']['example_id'] in chosen]


def inputs(args):
    material_path = args.material_dir/'material_2000.json'
    assert digest(args.bench) == BENCH_SHA, 'Original frozen input builder changed'
    assert digest(material_path) == ARMS['data_2k_4ep'][2], 'Shared material changed'
    material = read(material_path)
    saved_order = read(args.reports/'data_2k_4ep/material_ids.json')['held']
    by_id = {r['example']['example_id']:r for r in material['held']}
    assert len(saved_order) == len(set(saved_order)) == len(by_id) == 200 and set(saved_order) == set(by_id)
    material['held'] = [by_id[ident] for ident in saved_order]
    chosen = read(args.selection)
    assert chosen['selection_key'] == SELECTION_KEY and chosen['quotas'] == QUOTAS
    assert chosen['selected_ids'] == selection(material), 'Frozen100 membership changed'
    held_ids = [r['example']['example_id'] for r in material['held']]
    assert chosen['original_held_ids'] == held_ids
    assert len(material['retrieval']) == 8200
    # All arms must have evaluated the same exact held order. Training order is irrelevant here.
    for arm in ARMS:
        ids = read(args.reports/arm/'material_ids.json')
        assert ids['held'] == held_ids, arm+': held IDs/order changed'
        identity = read(args.reports/arm/'run_identity.json')['identity']
        assert identity['material_sha256'] == ARMS[arm][2]
    return material, chosen


def main(args):
    started=time.monotonic();output=args.output/(args.condition+'.json');status=args.output/(args.condition+'_status.json')
    def progress(phase,**extra):
        record={'phase':phase,'condition':args.condition,'condition_label':LABELS[args.condition],
                'elapsed_seconds':round(time.monotonic()-started,1),**extra}
        write(status,record);print(json.dumps(record),flush=True)
    progress('checking frozen cohort and completed pairs')
    material,chosen=inputs(args)
    pairs={arm:final_pair(arm,args.reports) for arm in ARMS}
    frozen=args.condition=='frozen_p2';arm='data_2k_4ep' if frozen else args.condition
    update=0 if frozen else 1000
    pair={'ac_checkpoint':str(args.p2_root/'AC_MODELS/rag_poc/epoch_001'),
          'reader_checkpoint':str(args.p2_root/'LORAS/rag_reader/epoch_001')} if frozen else pairs[arm]
    ac_path,reader_path=Path(pair['ac_checkpoint']),Path(pair['reader_checkpoint'])
    weight_files={'ac_modules':ac_path/'ac_modules.pt','encoder_lora':ac_path/'side_lora/adapter_model.safetensors',
                  'reader_lora':reader_path/'adapter_model.safetensors'}
    for name,path in list(weight_files.items()):
        if not path.exists() and path.suffix=='.safetensors':weight_files[name]=path.with_name('adapter_model.bin')
    panel_path=args.reports/arm/f'panel_{update:05d}.json';panel=read(panel_path)
    assert panel['progress']['global_step']==update
    identity={'condition':args.condition,'condition_label':LABELS[args.condition],'update':update,
        'reader_identity':LABELS[args.condition] if frozen else 'Paired trained reader: '+LABELS[args.condition],
        'ac_checkpoint':str(ac_path),'reader_checkpoint':str(reader_path),'final_pairs':pairs,
        'checkpoint_sha256':{k:digest(v) for k,v in weight_files.items()},
        'shared_material_sha256':digest(args.material_dir/'material_2000.json'),
        'selection_sha256':digest(args.selection),'original_panel_sha256':digest(panel_path),
        'bench_sha256':digest(args.bench),'evaluator_sha256':digest(__file__),
        'helpers_sha256':{name:digest(Path(__file__).with_name(name)) for name in ('evaluate_qualitative_cohort.py','evaluate_data_confirmation.py')},
        'nll_microbatch':2,'nll_batching':'Evaluate all original200 in original order, preserving sorted batch partners; then select frozen100 scores.',
        'decoding':'Greedy free decoding without gold-prefix tokens; enable_thinking=False; max_new_tokens=160; only fixed100 questions.',
        'cohort':'Deterministic family-balanced half of shared200 development; not reserved confirmation and not old100 cohort.'}
    if output.exists():
        result=read(output);assert result['identity']==identity,'Existing evaluation identity differs'
        if result['status']=='complete':progress('already complete',items=100);return
    else:result={'status':'running','identity':identity,'selection':chosen,'views':{},'rows':[]}
    spec=importlib.util.spec_from_file_location('original_data_budget_bench',args.bench)
    bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
    progress('loading saved paired weights')
    h=bench.harness(ac_path,reader_path);ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();h.module_manager.ensure_lora(bench.READER)
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
    from activation.autotuners import configure_fla_runtime
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    config=replace(ActivationContextTrainingConfig(**read(args.reports/arm/'training_config.json')),
                   micro_batch_examples=2,no_context_penalty_weight=0,completion_max_new_tokens=160)
    trainer=ActivationContextTrainer(h,config);trainer._training_deadline=time.monotonic()+args.seconds
    variables=trainer._autotuning_variables(ac)
    progress('building original200 evidence and distractors')
    # Explicit retrieval plus question-local shuffles make held evidence independent of train-building work.
    study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':material['held'],'retrieval':material['retrieval']})
    assert [item.item_id for item in held]==chosen['original_held_ids']
    selected=set(chosen['selected_ids']);indices=[i for i,item in enumerate(held) if item.item_id in selected]
    assert len(indices)==100
    view_items={'frozen_full_context':study.transform_for_eval_reference(held,'full_context'),
                'frozen_no_context':study.transform_for_eval_reference(held,'no_context')} if frozen else {args.condition:held}
    if not result['rows']:
        for i in indices:
            item=held[i];source=material['held'][i]
            result['rows'].append({'item_id':item.item_id,'original_200_index':i,'domain':item.kind,
                'source_family':source['source_family'],'question':item.info['question'],'gold':item.info['answer'],
                'required_source':source['example']['study_source'],'full_evidence':item.info['passages_text'],'views':{}})
    assert [r['item_id'] for r in result['rows']]==chosen['selected_ids']
    checkpoint_progress=ActivationContextTrainingProgress(**panel['progress'])
    with configure_fla_runtime(variables['fla_config']),torch.no_grad():
        for name,items in view_items.items():
            if name not in result['views']:
                progress('scoring original200 with batch2',view=name,total=200)
                summary=trainer._evaluate(ac,items,ac.device)
                old_name='held full context' if name=='frozen_full_context' else 'held no context' if name=='frozen_no_context' else 'held AC'
                old_rows={r['item_id']:r for r in panel['summaries'][old_name]['per_item']}
                assert len(summary['per_item'])==len(items)==len(old_rows)==200
                all_scores=[];differences=[];records_by={r['item_id']:r for r in result['rows']}
                for original,item,row in zip(held,items,summary['per_item']):
                    assert item.item_id==row['item_id']
                    positions=original.info['answer_positions'];old=old_rows[row['item_id']]
                    nll=-sum(row['target_logp'][p] for p in positions)/len(positions)
                    previous=-sum(old['target_logp'][p] for p in positions)/len(positions)
                    all_scores.append({'item_id':original.item_id,'answer_nll':nll,'original_panel_answer_nll':previous})
                    differences.append(nll-previous)
                    if original.item_id in records_by:
                        records_by[original.item_id]['views'][name]={'answer_nll':nll,'original_panel_answer_nll':previous,
                            'gold_token_scores':[{'text':study.tokenizer.decode([original.assistant_token_ids[0][p]],skip_special_tokens=False),
                                                  'nll':-row['target_logp'][p]} for p in positions]}
                result['views'][name]={'reader_identity':identity['reader_identity'],
                    'mean_answer_nll':sum(r['views'][name]['answer_nll'] for r in result['rows'])/100,
                    'subset100_answer_nll':sum(r['views'][name]['answer_nll'] for r in result['rows'])/100,
                    'full200_answer_nll':sum(r['answer_nll'] for r in all_scores)/200,
                    'original_panel_full200_answer_nll':sum(r['original_panel_answer_nll'] for r in all_scores)/200,
                    'original_panel_max_abs_nll_difference':max(map(abs,differences)),
                    'original_panel_mean_nll_difference':sum(differences)/200,'full200_per_item':all_scores,'complete':False}
                write(output,result)
            for j,i in enumerate(indices):
                record=result['rows'][j]
                if 'response' in record['views'][name]:continue
                progress('free decoding fixed100',view=name,completed=j,total=100)
                sample=trainer._sample_completions(ac,[items[i]],checkpoint_progress)[0]
                answer,parsing=parsed_answer(study,sample['student'])
                record['views'][name].update(response=sample['student'],answer=answer,parsing=parsing,
                    generated_tokens=len(study.tokenizer.encode(sample['student'],add_special_tokens=False)))
                write(output,result)
            result['views'][name]['complete']=True;write(output,result)
    result['status']='complete';result['elapsed_seconds']=round(time.monotonic()-started,1);write(output,result)
    progress('complete',items=100,views=list(view_items))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('bench','material-dir','reports','selection','p2-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--condition',choices=['frozen_p2',*ARMS],required=True)
    p.add_argument('--seconds',type=int,default=600)
    args=p.parse_args()
    try:main(args)
    except Exception as error:
        write(args.output/(args.condition+'_status.json'),{'phase':'failed','condition':args.condition,'error':f'{type(error).__name__}: {error}'})
        raise
