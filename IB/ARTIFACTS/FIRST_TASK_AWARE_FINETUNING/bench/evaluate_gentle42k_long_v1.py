"""Own final development400 replay +100 actual AC answers after42k→familiar4k context adaptation.

One condition/context per process. No optimization, checkpoint selection, or confirmation reads.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from evaluate_qualitative_cohort import read, write, digest as file_sha, parsed_answer
from evaluate_long_cross import answer_scores, digest
from evaluate_model_confirmation import shared_fingerprint
from evaluate_strong_long_cohort_v1 import interleave, payload_hashes, resolved_rows, cohort_summary
from evaluate_model_comparison_cohort import select_ids, load_module, sample_with_token_ids, SELECTION_KEY, QUOTAS

ARMS = ('gentle42k_final_lr_familiar4k_8k','gentle42k_rewarm_familiar4k_8k')
CONDITIONS = ARMS
CONTEXTS = {'8k': (8192, 16, 2)}
SOURCE = {'ac_model': 'aaeb3c68225d3092a024d44552b84480dc37cc8319d68c2ab1701fce70980bfd',
          'trainer': '38ca9885bfe35e47432352aa716381e53c18aa57748428eedf5f5f6b7fc6b5aa',
          'utils': '1b464ee4e8cafc6e7582c1c04e27bc95432cb8616b90f5bd90ad8b6c6fd5096c'}
HELPERS = ('evaluate_qualitative_cohort.py', 'evaluate_long_cross.py',
           'evaluate_model_confirmation.py', 'evaluate_model_comparison_cohort.py')


def load_material(args, design):
    if (design.get('version')!=1 or design.get('status')!='frozen' or design['source_sha256']!=SOURCE
            or design['run_name']!=args.condition or args.condition not in ARMS
            or design['encode_batch_max_tokens']!=65536 or design['questions']!=4000 or design['epochs']!=1):
        raise ValueError('Wrong frozen familiar4000 adaptation design')
    for name,expected in design['helper_sha256'].items():
        if Path(name).name!=name or file_sha(Path(__file__).with_name(name))!=expected:
            raise ValueError('Frozen helper changed: '+name)
    for path,expected in ((args.material,design['material_sha256']),(args.bench,design['builder_sha256']),
            (args.selection,design['readout']['selection_sha256']),(args.initial_pair,design['initial_pair_sha256']),
            (args.parent_readout,design['parent_readout_sha256'])):
        if file_sha(path)!=expected:raise ValueError('Frozen input bytes changed: '+str(path))
    selection=read(args.selection);material=read(args.material)
    if selection['selection_key']!=SELECTION_KEY or selection['quotas']!=QUOTAS:
        raise ValueError('Original development100 selection policy changed')
    if len(material['train'])!=4000 or len(material['held'])!=400 or len(material['retrieval'])!=32398:
        raise ValueError('Expected4000 familiar training/400 development/32398 retrieval rows')
    for split in ('train','held'):material[split]=interleave(material[split])
    ids={split:[row['example']['example_id'] for row in material[split]] for split in ('train','held')}
    if any(len(set(values))!=len(values) for values in ids.values()) or digest(ids)!=design['material_ids_sha256']:
        raise ValueError('Frozen material membership/order differs')
    if ids['held']!=selection['original_held_ids'] or select_ids(material,ids['held'])!=selection['selected_ids']:
        raise ValueError('Original400 order/fixed100 membership changed')
    parent=read(args.parent_readout)
    if (parent.get('status')!='complete' or parent['identity']['condition']!=design['parent_condition']
            or parent['identity']['context']!='8k' or parent['identity']['update']!=1250
            or parent['scored_ids']!=ids['held'] or parent['selected_ids']!=selection['selected_ids']
            or not all(parent['views'][name]['complete'] for name in ('AC','full context','no context'))):
        raise ValueError('Own parent42k8k reference is incomplete or changed')
    return material,selection,ids


def validate_final_optimizer(state):
    saved=state['optimizer'];values=saved['state']
    if (state.get('updates_done')!=500 or state.get('epoch_number')!=1
            or [len(group['params']) for group in saved['param_groups']]!=[374,256]
            or len(values)!=615 or any(not {'step','exp_avg','exp_avg_sq'}<=v.keys() for v in values.values())
            or {float(value['step']) for value in values.values()}!={5750}):
        raise ValueError('Expected completed500-stage/5750-lifetime inherited Adam, not reset/partial moments')
    return {'stage_updates':500,'epoch':1,'active_parameter_states':615,'adam_parameter_steps':5750}


def paired_final(condition,reports,runs_root,design,ids):
    """Only this4000-familiar stage's published500-update/Adam5750 own final pair."""
    import torch
    folder=reports/condition;result=read(folder/'result.json');identity=read(folder/'run_identity.json')['identity']
    expected={'arm':'joint','epochs':1,'planned_updates':500,'steps_per_epoch':500,
        'material_sha256':design['material_sha256'],'material_ids_sha256':digest(ids),'passage_parts':True,
        'evidence_tokens':8192,'max_passages':16,'ratio':1/16,'encode_batch_max_tokens':65536,
        'initial_pair_sha256':design['initial_pair_sha256']}
    if (condition!=design['run_name'] or identity!=expected or read(folder/'material_ids.json')!=ids
            or digest(read(folder/'training_config.json'))!=digest(design['training_config'])):
        raise ValueError('Final training identity/configuration/order changed')
    required={'stop_reason':'complete','completed_global_updates':500,'planned_updates':500,'planned_epochs':1,
        'items':4000,'examples':4000,'parent_unique_questions':42000,'unique_questions_after_stage':42000,
        'completed_total_presentations':46000,'optimizer_policy':'inherited Adam5250, constant floor LR'}
    if any(result.get(key)!=value for key,value in required.items()):
        raise ValueError('Familiar4000 stage is incomplete or has different exposure/optimizer policy')
    receipt=read(folder/'source_receipt.json')
    for key in ('builder_sha256','source_sha256','initial_payload_sha256','initial_pair_sha256','material_sha256','parent_condition','parent_readout_sha256'):
        if receipt.get(key)!=design[key]:raise ValueError('Final source/initialization receipt differs: '+key)
    inherited=read(folder/'inherited_optimizer_check.json');replay=read(folder/'initial_ac_replay.json')
    if (inherited.get('optimizer_states')!=615 or inherited.get('min_parameter_step')!=5250
            or inherited.get('max_parameter_step')!=5250 or inherited.get('stage_updates_done')!=0
            or not replay.get('passed') or replay.get('parent_readout_sha256')!=design['parent_readout_sha256']):
        raise ValueError('Missing actual inheritedAdam5250 or passed initial42k numerical replay')
    root=(runs_root/condition).resolve();ac=Path(result['checkpoint']).resolve();reader=Path(result['target_lora_checkpoint']).resolve()
    if ac!=root/'AC_MODELS/rag_poc/epoch_001' or reader!=root/'LORAS/rag_reader/epoch_001':
        raise ValueError('Expected own final epoch001 pair, not best-selected/partial/foreign weights')
    marker_path=ac.parent/'completed_epoch.json';marker=read(marker_path)
    if (marker.get('epoch_number')!=1 or (root/marker['ac_checkpoint']).resolve()!=ac
            or (root/marker['target_lora_checkpoint']).resolve()!=reader):
        raise ValueError('Completed paired publication marker differs')
    panel=read(folder/'panel_00500.json')
    if any(panel['progress'].get(key)!=value for key,value in {'global_step':500,'step':500,'steps':500,'epoch':1,'epoch_number':1,'epochs':1}.items()):
        raise ValueError('Final500 panel progress differs')
    if [row['item_id'] for row in panel['summaries']['held AC']['per_item']]!=ids['held']:
        raise ValueError('Final panel changed original400 order')
    optimizer=ac/'optimizer.pt';state=torch.load(optimizer,map_location='cpu',weights_only=False)
    optimizer_receipt=validate_final_optimizer(state);del state
    publications={name:file_sha(folder/name) for name in ('result.json','run_identity.json','training_config.json',
        'material_ids.json','source_receipt.json','inherited_optimizer_check.json','initial_ac_replay.json','parent_input_check.json','panel_00500.json')}
    publications.update(completed_marker_sha256=file_sha(marker_path),optimizer_sha256=file_sha(optimizer),optimizer=optimizer_receipt)
    return {'ac_checkpoint':str(ac),'reader_checkpoint':str(reader)},publications






def main(args):
    started=time.monotonic(); cell=args.condition+'__'+args.context
    output=args.output/cell/'result.json'; status=args.output/cell/'status.json'
    def progress(phase,**extra):
        value={'cell':cell,'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-started,1),**extra}
        write(status,value);print(json.dumps(value),flush=True)
    progress('validating frozen development inputs and pair')
    design=read(args.design);material,selection,ids=load_material(args,design)
    frozen=False;parent=read(args.parent_readout)
    arm={'passage_parts':True,'encode_batch_max_tokens':design['encode_batch_max_tokens']}
    pair,publication=paired_final(args.condition,args.reports,args.runs_root,design,ids);update=500
    if read(args.reports/args.condition/'source_receipt.json')['design_sha256']!=file_sha(args.design):
        raise ValueError('Training used a different frozen design')
    hashes=payload_hashes(pair)
    evidence_budget,max_passages,micro=CONTEXTS[args.context]
    identity={'condition':args.condition,'context':args.context,'update':update,'paired_checkpoint':pair,
        'checkpoint_sha256':hashes,'publication_sha256':publication,'design_sha256':file_sha(args.design),
        'selection_sha256':file_sha(args.selection),'evaluator_sha256':file_sha(__file__),
        'helper_sha256':{name:file_sha(Path(__file__).with_name(name)) for name in HELPERS},
        'source_sha256':design['source_sha256'],'builder_sha256':design['builder_sha256'],
        'passage_parts':arm['passage_parts'],'encode_batch_max_tokens':arm['encode_batch_max_tokens'],
        'ratio':1/16,'evidence_tokens':evidence_budget,'max_passages':max_passages,'micro_batch_examples':micro,
        'selection_policy':'Original immutable development100 (50 per generator); no confirmation access.',
        'reader':'This completed4000-familiar stage paired final reader',
        'nll_scope':'Original400 in original order, then retain selected100' if args.context=='8k' else 'Selected100 in original order, microbatch1',
        'decoding':'8k only: greedy160 tokens, thinking disabled, inherited EOS union chat EOT, actual token IDs retained',
        'parent_readout_sha256':file_sha(args.parent_readout),
        'training_exposure':'4000 familiar questions once;42k unique/46k presentations; inherited Adam5750',
        'transfer_scope':'8k familiar-question context adaptation; no new unique-data or isolated-layout claim.'}
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
    if model_provenance!=parent['model_provenance']:raise ValueError('Base models/configs differ from exact parent8k readout')
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
    from evaluate_continuation_long_8k_v1 import bind_inputs
    bind_inputs(fingerprints,row_counts,result['rows'],parent)
    panel=None;panel_sha=None
    if args.context=='8k':
        path=args.reports/args.condition/'panel_00500.json'
        if path.exists():
            panel=read(path);panel_sha=file_sha(path)
            if panel['progress']['global_step']!=update:raise ValueError('Wrong initial/final adaptation panel')
        elif not frozen:raise ValueError('Missing completed final development panel')
    generation=peft.generation_config;raw_eos=generation.eos_token_id
    eos=[raw_eos] if isinstance(raw_eos,int) else list(raw_eos or [])
    eot=ac.target.model_config.model_description.eot_token;eot_id=ac.target.tokenizer.convert_tokens_to_ids(eot) if eot else None
    if eot_id is None:raise ValueError('Expected existing Qwen chat-end token')
    if eot_id not in eos:eos.append(eot_id)
    generation_record={'inherited_generation_config':json.loads(generation.to_json_string()),
        'explicit_eos_ids':eos,'eot':eot,'eot_id':eot_id,'max_new_tokens':160,'thinking':False,'do_sample':False,
        'model_revision':getattr(ac.target.model.config,'_commit_hash',None),'tokenizer':ac.target.tokenizer.name_or_path}
    if result.get('generation_provenance',generation_record)!=generation_record:raise ValueError('Generation policy changed on resume')
    result['generation_provenance']=generation_record
    variables=trainer._autotuning_variables(ac);trainer._training_deadline=time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:raise ValueError('Original BF16 reduction policy required')
    result['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(ac.device),
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'autotuning':variables['provenance']}
    sample_progress=ActivationContextTrainingProgress(epoch=1 if update else 0,epochs=1 if update else 0,
        epoch_number=1 if update else 0,step=update,steps=update,global_step=update,epoch_elapsed_s=0.,elapsed_s=0.)
    views={'AC':held,'full context':full,'no context':noctx}
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
                # Initial and final AC answers support the qualitative comparison;
                # both readers also retain matched full/no-context NLLs.
                if args.context=='8k' and name=='AC':
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
    if payload_hashes(pair)!=hashes:raise ValueError('Parent final payload changed during readout')
    result.update(status='complete',optimizer_updates=0,attempt_elapsed_seconds=round(time.monotonic()-started,1))
    write(output,result);progress('complete',questions=len(held),selected=100)



if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('design','material','bench','selection','reports','runs-root','initial-pair','parent-readout','output'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--condition',choices=CONDITIONS)
    parser.add_argument('--context',choices=CONTEXTS)
    parser.add_argument('--seconds',type=int,default=2100)
    args=parser.parse_args()
    if any(getattr(args,name) is None for name in ('design','material','bench','selection','reports','runs_root','initial_pair','parent_readout','output','condition','context')):
        parser.error('All input paths, --condition and --context are required')
    main(args)
