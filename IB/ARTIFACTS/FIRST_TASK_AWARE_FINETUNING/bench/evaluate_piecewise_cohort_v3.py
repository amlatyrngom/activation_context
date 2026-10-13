"""Fixed development100 for fresh-P2 merged/piecewise 4k×1, with 8k replay and 32k transfer.

One condition/context per process. No optimization, checkpoint selection, or confirmation reads.
"""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

from evaluate_qualitative_cohort import read, write, digest as file_sha, parsed_answer
from evaluate_long_cross import answer_scores, digest
from evaluate_model_confirmation import shared_fingerprint
from evaluate_model_comparison_cohort import select_ids, load_module, sample_with_token_ids, SELECTION_KEY, QUOTAS

ARMS = ('piecewise_27b_4k_8k', 'merged_27b_4k_8k')
CONDITIONS = ('frozen_p2', *ARMS)
CONTEXTS = {'8k': (8192, 16, 2), '32k': (32768, 64, 1)}
MATERIAL_SHA = '9eff31670c2206dd9bb1eeda841461c324e75068ea2b5afaacd7f74ed49c05a5'
SOURCE = {'ac_model': 'aaeb3c68225d3092a024d44552b84480dc37cc8319d68c2ab1701fce70980bfd',
          'trainer': '38ca9885bfe35e47432352aa716381e53c18aa57748428eedf5f5f6b7fc6b5aa',
          'utils': '1b464ee4e8cafc6e7582c1c04e27bc95432cb8616b90f5bd90ad8b6c6fd5096c'}
HELPERS = ('evaluate_qualitative_cohort.py', 'evaluate_long_cross.py',
           'evaluate_model_confirmation.py', 'evaluate_model_comparison_cohort.py')


def interleave(rows):
    groups = defaultdict(list)
    for row in rows: groups[row['domain']].append(row)
    return [row for _, _, row in sorted(((i/len(values), domain, row) for domain, values in groups.items()
            for i, row in enumerate(values)), key=lambda item: (item[0], item[1]))]


def load_material(args, design):
    if design.get('version') != 2 or design.get('status') != 'frozen' or design['material_sha256'] != MATERIAL_SHA or design['source_sha256'] != SOURCE:
        raise ValueError('Unexpected piecewise study/version/source identity')
    if set(design['arms']) != set(ARMS): raise ValueError('Expected exactly the paired merged/piecewise arms')
    for name in ARMS:
        arm = design['arms'][name]
        if (arm['passage_parts'] != name.startswith('piecewise_') or arm['micro_batch_examples'] != 2
                or not isinstance(arm['encode_batch_max_tokens'], int) or arm['encode_batch_max_tokens'] < 1):
            raise ValueError('Invalid fixed layout/batching design')
    for path, expected in ((args.material, design['material_sha256']), (args.bench, design['builder_sha256']),
                           (args.selection, design['selection_sha256'])):
        if file_sha(path) != expected: raise ValueError('Frozen input bytes changed: '+str(path))
    selection = read(args.selection)
    if selection['selection_key'] != SELECTION_KEY or selection['quotas'] != QUOTAS:
        raise ValueError('Original loss-independent development selection changed')
    material = read(args.material)
    if len(material['train']) != 4000 or len(material['held']) != 400 or len(material['retrieval']) != 32398:
        raise ValueError('Expected frozen4000 training/400 development/32398 retrieval material')
    for split in ('train', 'held'): material[split] = interleave(material[split])
    ids = {split: [row['example']['example_id'] for row in material[split]] for split in ('train', 'held')}
    if any(len(set(values)) != len(values) for values in ids.values()): raise ValueError('Duplicate frozen item ID')
    if ids['held'] != selection['original_held_ids'] or select_ids(material, ids['held']) != selection['selected_ids']:
        raise ValueError('Original400 order or fixed100 membership changed')
    return material, selection, ids


def paired_final(condition, reports, runs_root, design, ids):
    """Require this arm's own published final500 pair; never a selected or partial save."""
    folder = reports/condition; arm = design['arms'][condition]
    result = read(folder/'result.json'); identity = read(folder/'run_identity.json')['identity']
    config = read(folder/'training_config.json')
    expected = {'arm': 'joint', 'epochs': 1, 'planned_updates': 500, 'steps_per_epoch': 500,
        'material_sha256': design['material_sha256'], 'material_ids_sha256': digest(ids),
        'passage_parts': arm['passage_parts'], 'evidence_tokens': 8192, 'max_passages': 16, 'ratio': 1/16,
        'encode_batch_max_tokens': arm['encode_batch_max_tokens'], 'initial_pair_sha256': None}
    config_expected = {'loss_kind':'sft', 'micro_batch_examples':2, 'examples_per_update':8, 'num_epochs':1,
        'shuffle_examples':True, 'learning_rate_ac':5e-5, 'learning_rate_target_lora':2e-5,
        'schedule_total_updates':500, 'warmup_updates':50, 'lr_schedule':'cosine', 'lr_final_fraction':.1,
        'no_context_penalty_weight':1., 'no_context_penalty_fraction':.25, 'teacher_cache_top_k':32,
        'max_grad_norm':1., 'weight_decay':0., 'gradient_checkpointing':True,
        'gradient_checkpointing_min_tokens':0, 'seed':20260921}
    if (result.get('stop_reason') != 'complete' or result.get('completed_global_updates') != 500
            or result.get('items') != 4000 or result.get('examples') != 4000
            or result.get('planned_epochs') != 1 or result.get('planned_updates') != 500):
        raise ValueError(condition+': full4000-question/500-update final is not complete')
    if identity != expected or any(config.get(k) != value for k, value in config_expected.items()):
        raise ValueError(condition+': training identity or fixed recipe differs')
    if read(folder/'material_ids.json') != ids: raise ValueError('Saved train/development order differs')
    receipt = read(folder/'source_receipt.json')
    if any(receipt.get(key) != design[key] for key in ('builder_sha256','source_sha256')):
        raise ValueError('Training source/runner receipt differs from frozen design')
    ac, reader = Path(result['checkpoint']).resolve(), Path(result['target_lora_checkpoint']).resolve()
    root = (runs_root/condition).resolve()
    if ac != root/'AC_MODELS/rag_poc/epoch_001' or reader != root/'LORAS/rag_reader/epoch_001':
        raise ValueError('Final must be this arm\'s own epoch001 pair, not a selected/other-run checkpoint')
    marker_path = ac.parent/'completed_epoch.json'; marker = read(marker_path)
    if (marker.get('epoch_number') != 1 or (root/marker['ac_checkpoint']).resolve() != ac
            or (root/marker['target_lora_checkpoint']).resolve() != reader):
        raise ValueError('Published completed marker disagrees with paired final')
    return {'ac_checkpoint':str(ac), 'reader_checkpoint':str(reader)}, {
        name:file_sha(folder/name) for name in ('result.json','run_identity.json','training_config.json','material_ids.json','source_receipt.json')
    } | {'completed_marker_sha256':file_sha(marker_path)}


def payload_hashes(pair):
    ac, reader = Path(pair['ac_checkpoint']), Path(pair['reader_checkpoint'])
    files = {'ac_modules':ac/'ac_modules.pt'}
    for label, directory, base in (('encoder',ac/'side_lora','Qwen/Qwen3.5-2B'), ('reader',reader,'Qwen/Qwen3.5-9B')):
        config = directory/'adapter_config.json'; values = read(config)
        if values.get('base_model_name_or_path') != base or values.get('r') != 128:
            raise ValueError('Wrong paired adapter base/rank: '+str(directory))
        weights = [directory/name for name in ('adapter_model.safetensors','adapter_model.bin') if (directory/name).is_file()]
        if len(weights) != 1: raise ValueError('Expected one unambiguous adapter weight payload')
        files[label+'_config'], files[label+'_weights'] = config, weights[0]
    if any(not path.is_file() or path.stat().st_size == 0 for path in files.values()): raise ValueError('Missing/empty paired payload')
    return {name:file_sha(path) for name,path in files.items()}


def resolved_rows(trainer, ac, items):
    from activation.ac_model.ac_model_utils import tokenize_with_parts
    result = []
    for item in items:
        ex = trainer.build_example(ac,item); lengths=[]; rows=[]; uncapped=[]
        for request, span in zip(ex.part_requests,ex.spans):
            n = len(tokenize_with_parts(ac.side.tokenizer,request.messages,[],ac.pad_id,tools=request.tools)[0])
            count = ac.num_view_rows(n,request.compression_ratio)
            if count != span[1]-span[0]: raise ValueError('Resolved rows differ from actual reader placeholder span')
            lengths.append(n);rows.append(count)
            uncapped.append(max(ac.config.min_view_rows,math.ceil(n*request.compression_ratio)))
        if (len(ex.part_requests) != len(ex.spans) or lengths != item.info['part_source_tokens'] or rows != item.info['part_rows']
                or sum(rows) != item.info['compressed_rows'] or sum(rows) != item.info['merged_compressed_rows']):
            raise ValueError('Actual requests/spans disagree with fixed merged row budget: '+item.item_id)
        merged_uncapped=max(ac.config.min_view_rows,math.ceil(item.info['merged_source_tokens']/16))
        result.append({'item_id':item.item_id,'part_encoder_tokens':lengths,'part_rows':rows,'part_uncapped_rows':uncapped,
            'encoder_tokens':sum(lengths),'compressed_rows':sum(rows),'reader_tokens':len(ex.student_ids),
            'evidence_tokens':item.info['evidence_source_tokens'],'passages':item.info['passages'],
            'merged_encoder_tokens':item.info['merged_source_tokens'],'repeated_framing_extra_tokens':sum(lengths)-item.info['merged_source_tokens'],
            'rows_per_encoder_token':sum(rows)/sum(lengths),'rows_per_evidence_token':sum(rows)/max(1,item.info['evidence_source_tokens']),
            'merged_uncapped_rows':merged_uncapped,'max_view_rows':ac.config.max_view_rows,
            'row_cap_applied':merged_uncapped>ac.config.max_view_rows or any(x>ac.config.max_view_rows for x in uncapped)})
    return result


def cohort_summary(scored, cohorts):
    by_id = {row['item_id']:row for row in scored['per_item']}
    return {bank:{'questions':len(ids),'answer_nll':statistics.mean(by_id[i]['answer_nll'] for i in ids)} for bank,ids in cohorts.items()}


def main(args):
    started=time.monotonic(); cell=args.condition+'__'+args.context
    output=args.output/cell/'result.json'; status=args.output/cell/'status.json'
    def progress(phase,**extra):
        value={'cell':cell,'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-started,1),**extra}
        write(status,value);print(json.dumps(value),flush=True)
    progress('validating frozen development inputs and pair')
    design=read(args.design);material,selection,ids=load_material(args,design)
    frozen=args.condition=='frozen_p2'
    arm=design['arms']['merged_27b_4k_8k' if frozen else args.condition]
    if frozen:
        pair={'ac_checkpoint':str(args.p2_root/'AC_MODELS/rag_poc/epoch_001'),
              'reader_checkpoint':str(args.p2_root/'LORAS/rag_reader/epoch_001')};publication={};update=0
    else:
        pair,publication=paired_final(args.condition,args.reports,args.runs_root,design,ids);update=500
    hashes=payload_hashes(pair)
    if frozen and design.get('P2_pair_sha256') is not None and hashes!=design['P2_pair_sha256']:
        raise ValueError('Original P2 payload identity changed')
    evidence_budget,max_passages,micro=CONTEXTS[args.context]
    identity={'condition':args.condition,'context':args.context,'update':update,'paired_checkpoint':pair,
        'checkpoint_sha256':hashes,'publication_sha256':publication,'design_sha256':file_sha(args.design),
        'selection_sha256':file_sha(args.selection),'evaluator_sha256':file_sha(__file__),
        'helper_sha256':{name:file_sha(Path(__file__).with_name(name)) for name in HELPERS},
        'source_sha256':design['source_sha256'],'builder_sha256':design['builder_sha256'],
        'passage_parts':arm['passage_parts'],'encode_batch_max_tokens':arm['encode_batch_max_tokens'],
        'ratio':1/16,'evidence_tokens':evidence_budget,'max_passages':max_passages,'micro_batch_examples':micro,
        'selection_policy':'Original immutable development100 (50 per generator); no confirmation access.',
        'reader':'Original frozen P2 epoch1 reader' if frozen else 'This completed arm\'s paired final reader',
        'nll_scope':'Original400 in original order, then retain selected100' if args.context=='8k' else 'Selected100 in original order, microbatch1',
        'decoding':'8k only: greedy160 tokens, thinking disabled, inherited EOS union chat EOT, actual token IDs retained',
        'transfer_scope':'32k adds retrieval context/distractors; this does not test distributed-required-evidence synthesis.'}
    if output.exists():
        result=read(output)
        if result['identity']!=identity:raise ValueError('Existing evaluation identity changed')
        if result['status']=='complete':progress('already complete');return
    else:result={'status':'running','identity':identity,'views':{},'rows':[]}
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress,_TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    runtime={'ac_model':file_sha(ac_module.__file__),'trainer':file_sha(training_module.__file__),'utils':file_sha(utils_module.__file__)}
    if runtime!=design['source_sha256']:raise ValueError('Runtime source differs from validated piecewise source variant')
    bench=load_module('piecewise_evaluation_builder',args.bench)
    progress('loading frozen paired weights')
    h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=arm['encode_batch_max_tokens'])
    ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();peft=h.module_manager.ensure_lora(bench.READER)
    model_provenance={'side_model':ac.side.model_config.model_id,'reader_model':ac.target.model_config.model_id,
        'side_revision':getattr(ac.side.model.config,'_commit_hash',None),
        'reader_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'side_config_sha256':digest(json.loads(ac.side.model.config.to_json_string())),
        'reader_config_sha256':digest(json.loads(ac.target.model.config.to_json_string()))}
    if result.get('model_provenance',model_provenance)!=model_provenance:
        raise ValueError('Base model identity/config changed during partial evaluation')
    result['model_provenance']=model_provenance
    selected=set(selection['selected_ids'])
    rows=material['held'] if args.context=='8k' else [row for row in material['held'] if row['example']['example_id'] in selected]
    progress('building matched evidence',questions=len(rows),evidence_tokens=evidence_budget)
    study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':rows,'retrieval':material['retrieval']},
        passage_parts=arm['passage_parts'],evidence_tokens=evidence_budget,max_passages=max_passages,ratio=1/16)
    expected=ids['held'] if args.context=='8k' else selection['selected_ids']
    if [item.item_id for item in held]!=expected:raise ValueError('Builder changed frozen question order')
    for item,row in zip(held,rows):
        if any(unit['text'] not in item.info['passages_text'] for unit in row['example']['study_source']['units']):
            raise ValueError('Required source not retained whole: '+item.item_id)
    full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
    shared=shared_fingerprint(held,full,noctx);ac_hash=digest([(item.item_id,item.ac_messages) for item in held])
    trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',no_context_penalty_weight=0,
        micro_batch_examples=micro,max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED))
    row_counts=resolved_rows(trainer,ac,held);budget_hash=digest([(r['item_id'],r['compressed_rows']) for r in row_counts])
    fingerprints={'shared_input_fingerprint':shared,'ac_input_fingerprint':ac_hash,'total_rows_fingerprint':budget_hash}
    for key,value in fingerprints.items():
        if result.get(key,value)!=value:raise ValueError('Rebuilt partial inputs changed: '+key)
    for condition in CONDITIONS:
        path=args.output/(condition+'__'+args.context)/'result.json'
        if path.exists():
            other=read(path)
            if other['identity']['design_sha256']!=identity['design_sha256']:raise ValueError('Other condition belongs to another design')
            if any(other.get(key,value)!=value for key,value in [('shared_input_fingerprint',shared),('total_rows_fingerprint',budget_hash)]):
                raise ValueError('Layouts differ in full/no-context/target inputs or total row capacity')
    result.update(**fingerprints,input_rows=row_counts,max_view_rows=ac.config.max_view_rows,
        selected_ids=selection['selected_ids'],scored_ids=expected,required_sources_retained_whole=True)
    indices=[i for i,item in enumerate(held) if item.item_id in selected]
    cohort_of={ident:bank for bank,group in material['held_cohorts'].items() for ident in group}
    cohorts={bank:[ident for ident in group if ident in set(expected)] for bank,group in material['held_cohorts'].items()}
    if not result['rows']:
        result['rows']=[{'item_id':held[i].item_id,'original_400_index':ids['held'].index(held[i].item_id),
            'generator_cohort':cohort_of[held[i].item_id],'source_family':rows[i]['source_family'],'domain':rows[i]['domain'],
            'question':held[i].info['question'],'gold':held[i].info['answer'],'required_source':rows[i]['example']['study_source'],
            'full_evidence':held[i].info['passages_text'],'in_context_messages':held[i].in_context_messages,
            'answer_positions':held[i].info['answer_positions'],'assistant_token_ids':held[i].assistant_token_ids,
            'row_budget':row_counts[i],'views':{}} for i in indices]
    if [row['item_id'] for row in result['rows']]!=selection['selected_ids']:raise ValueError('Saved fixed100 row order changed')
    panel=None;panel_sha=None
    if args.context=='8k' and not frozen:
        path=args.reports/args.condition/'panel_00500.json';panel=read(path);panel_sha=file_sha(path)
        if panel['progress']['global_step']!=500:raise ValueError('Expected this final500 development panel')
    generation=peft.generation_config;raw_eos=generation.eos_token_id
    eos=[raw_eos] if isinstance(raw_eos,int) else list(raw_eos or [])
    eot=ac.target.model_config.model_description.eot_token;eot_id=ac.target.tokenizer.convert_tokens_to_ids(eot) if eot else None
    if eot_id is None:raise ValueError('Expected existing Qwen chat-end token')
    if eot_id not in eos:eos.append(eot_id)
    result['generation_provenance']={'inherited_generation_config':json.loads(generation.to_json_string()),
        'explicit_eos_ids':eos,'eot':eot,'eot_id':eot_id,'max_new_tokens':160,'thinking':False,'do_sample':False,
        'model_revision':getattr(ac.target.model.config,'_commit_hash',None),'tokenizer':ac.target.tokenizer.name_or_path}
    variables=trainer._autotuning_variables(ac);trainer._training_deadline=time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    sample_progress=ActivationContextTrainingProgress(epoch=1 if update else 0,epochs=1 if update else 0,
        epoch_number=1 if update else 0,step=update,steps=update,global_step=update,epoch_elapsed_s=0.,elapsed_s=0.)
    views={'full context':full,'no context':noctx} if frozen else {'AC':held,'full context':full,'no context':noctx}
    write(output,result)
    try:
        with configure_fla_runtime(variables['fla_config']),torch.no_grad():
            for name,items in views.items():
                if name not in result['views']:
                    progress('scoring '+name,questions=len(items))
                    torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device);begin=time.monotonic()
                    summary=trainer._evaluate(ac,items,ac.device);scored=answer_scores(summary,items,held)
                    torch.cuda.synchronize(ac.device)
                    replay=None
                    if panel is not None:
                        old_summary=panel['summaries']['held '+name]
                        prior=answer_scores(old_summary,items,held)
                        delta=[a['answer_nll']-b['answer_nll'] for a,b in zip(scored['per_item'],prior['per_item'])]
                        if any(len(a['target_logp'])!=len(b['target_logp']) for a,b in zip(summary['per_item'],old_summary['per_item'])):
                            raise ValueError('Original panel gold-token count changed')
                        token_delta=max(abs(x-y) for a,b in zip(summary['per_item'],old_summary['per_item'])
                                        for x,y in zip(a['target_logp'],b['target_logp']))
                        replay={'original_panel_sha256':panel_sha,'maximum_absolute_difference':max(map(abs,delta)),
                            'maximum_absolute_target_logp_difference':token_delta,
                            'mean_difference':statistics.mean(delta),'absolute_tolerance':1e-3,
                            'passed':max(map(abs,delta))<=1e-3 and token_delta<=1e-3}
                    for record,i in zip(result['rows'],indices):
                        item=held[i];token_ids=item.assistant_token_ids[0]
                        record['views'][name]={'answer_nll':scored['per_item'][i]['answer_nll'],
                            'gold_token_scores':[{'token_id':token_ids[p],'text':study.tokenizer.decode([token_ids[p]],skip_special_tokens=False),
                                'nll':-summary['per_item'][i]['target_logp'][p]} for p in item.info['answer_positions']]}
                    result['views'][name]={**scored,'cohorts':cohort_summary(scored,cohorts),
                        'selected100_answer_nll':statistics.mean(record['views'][name]['answer_nll'] for record in result['rows']),
                        'selected50_by_generator':{bank:statistics.mean(record['views'][name]['answer_nll'] for record in result['rows'] if record['generator_cohort']==bank) for bank in cohorts},
                        'panel_replay':replay,'seconds':time.monotonic()-begin,
                        'peak_allocated_gib':torch.cuda.max_memory_allocated(ac.device)/2**30,'complete':False}
                    write(output,result)
                replay=result['views'][name]['panel_replay']
                if replay is not None and not replay['passed']:
                    result['status']='replay_mismatch';write(output,result);raise ValueError('Original400 panel replay differs beyond1e-3; preserve and diagnose')
                # Final AC answers plus original-P2 full/no-context baselines are
                # sufficient for the qualitative comparison; final-reader controls
                # still have all teacher-forced NLLs, without another200 decodes.
                if args.context=='8k' and (frozen or name=='AC'):
                    for j,i in enumerate(indices):
                        record=result['rows'][j]
                        if 'response' in record['views'][name]:continue
                        progress('decoding fixed100',view=name,completed=j,total=100)
                        sample,generated=sample_with_token_ids(trainer,ac,items[i],sample_progress,eos)
                        text=sample['student'];answer,parsing=parsed_answer(study,text);end=text.find('</tool_call>')
                        record['views'][name].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,
                            post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                        write(output,result)
                result['views'][name]['complete']=True;write(output,result)
    except _TrainingTimeLimit:
        result['status']='partial';write(output,result);progress('time limit; resume same condition/context');return
    if trainer.optimizers or trainer.updates_done:raise AssertionError('Evaluation unexpectedly optimized weights')
    result.update(status='complete',optimizer_updates=0,attempt_elapsed_seconds=round(time.monotonic()-started,1))
    write(output,result);progress('complete',questions=len(held),selected=100)


def cpu_check():
    """Five failure gates on a tiny filesystem fixture; no model/data loads."""
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);reports=root/'reports';arm=ARMS[0];folder=reports/arm;folder.mkdir(parents=True)
        ac=root/arm/'AC_MODELS/rag_poc/epoch_001';reader=root/arm/'LORAS/rag_reader/epoch_001';ac.mkdir(parents=True);reader.mkdir(parents=True)
        ids={'train':['t'+str(i) for i in range(4000)],'held':['h'+str(i) for i in range(400)]}
        design={'material_sha256':MATERIAL_SHA,'builder_sha256':'builder','source_sha256':SOURCE,
            'arms':{arm:{'passage_parts':True,'encode_batch_max_tokens':16384}}}
        identity={'arm':'joint','epochs':1,'planned_updates':500,'steps_per_epoch':500,'material_sha256':MATERIAL_SHA,
            'material_ids_sha256':digest(ids),'passage_parts':True,'evidence_tokens':8192,'max_passages':16,'ratio':1/16,
            'encode_batch_max_tokens':16384,'initial_pair_sha256':None}
        config={'loss_kind':'sft','micro_batch_examples':2,'examples_per_update':8,'num_epochs':1,'shuffle_examples':True,
            'learning_rate_ac':5e-5,'learning_rate_target_lora':2e-5,'schedule_total_updates':500,'warmup_updates':50,
            'lr_schedule':'cosine','lr_final_fraction':.1,'no_context_penalty_weight':1.,'no_context_penalty_fraction':.25,
            'teacher_cache_top_k':32,'max_grad_norm':1.,'weight_decay':0.,'gradient_checkpointing':True,'gradient_checkpointing_min_tokens':0,'seed':20260921}
        result={'stop_reason':'complete','completed_global_updates':500,'items':4000,'examples':4000,'planned_epochs':1,
            'planned_updates':500,'checkpoint':str(ac),'target_lora_checkpoint':str(reader)}
        marker={'epoch_number':1,'ac_checkpoint':'AC_MODELS/rag_poc/epoch_001','target_lora_checkpoint':'LORAS/rag_reader/epoch_001'}
        fixtures={folder/'result.json':result,folder/'run_identity.json':{'identity':identity},folder/'training_config.json':config,
            folder/'material_ids.json':ids,folder/'source_receipt.json':{'builder_sha256':'builder','source_sha256':SOURCE},ac.parent/'completed_epoch.json':marker}
        for path,value in fixtures.items():write(path,value)
        paired_final(arm,reports,root,design,ids)
        cases=[(folder/'result.json','completed_global_updates',499),
               (folder/'result.json','stop_reason','duration limit'),
               (folder/'run_identity.json','identity',{**identity,'material_sha256':'wrong'}),
               (folder/'training_config.json','micro_batch_examples',1),
               (folder/'training_config.json','gradient_checkpointing_min_tokens',1700),
               (folder/'source_receipt.json','builder_sha256','wrong'),
               (ac.parent/'completed_epoch.json','target_lora_checkpoint','LORAS/wrong/epoch_001')]
        for path,key,value in cases:
            bad=copy.deepcopy(fixtures[path]);bad[key]=value;write(path,bad)
            try:paired_final(arm,reports,root,design,ids)
            except ValueError:pass
            else:raise AssertionError('Expected failure: '+key)
            write(path,fixtures[path])
        print(json.dumps({'valid_final_metadata':'pass','partial_update_duration_wrong_material_microbatch_source_and_pair_rejected':len(cases),'model_weights_loaded':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-check',action='store_true')
    for name in ('design','material','bench','selection','reports','runs-root','p2-root','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--condition',choices=CONDITIONS)
    parser.add_argument('--context',choices=CONTEXTS)
    parser.add_argument('--seconds',type=int,default=2100)
    args=parser.parse_args()
    if args.cpu_check:cpu_check()
    else:
        if any(getattr(args,name) is None for name in ('design','material','bench','selection','reports','runs_root','p2_root','output','condition','context')):
            parser.error('All input paths, --condition and --context are required')
        main(args)
