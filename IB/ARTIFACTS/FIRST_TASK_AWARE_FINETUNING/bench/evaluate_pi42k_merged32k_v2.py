"""Own960 merged32k final/retention readout; all cells retain actual training logits chunks."""
import argparse
import json
from pathlib import Path
import statistics
import time

import evaluate_strong_long_cohort_v1 as common

read,write,sha,digest=common.read,common.write,common.file_sha,common.digest
RUN='pi42k_merged32k_fresh960_v1'
MATERIAL_SHA='49f711e3eff25b0063f148f850e73ff4fe7917c05dad0898db0167623d7e11a9'
SELECTION_SHA='469320c02db9347d4d93b99c94003f9eb40d738d03ee272ff83a0129fda92051'
CONTEXTS={'32k':(32768,64),'2k':(2048,4)}
PINNED_HELPERS=('evaluate_pi42k_merged32k_v2.py','evaluate_pi42k_merged32k_orcd_h200_v1.sbatch',
    'evaluate_strong_long_cohort_v1.py','evaluate_qualitative_cohort.py','evaluate_long_cross.py',
    'evaluate_model_confirmation.py','evaluate_model_comparison_cohort.py')


def validate_design(design):
    expected={'version':1,'status':'frozen','run_name':RUN,'questions':960,'epochs':1,
        'material_sha256':MATERIAL_SHA,'encode_batch_max_tokens':65536,'passage_parts':False,
        'evidence_tokens':32768,'max_passages':64,'unique_questions_after_stage':42960,
        'parent_condition':'continue32k_10k_final_lr','source_sha256':common.SOURCE}
    if any(design.get(k)!=v for k,v in expected.items()):raise ValueError('Wrong frozen960 fresh/merged32k design')
    config=design['training_config']
    required={'num_epochs':1,'examples_per_update':8,'micro_batch_examples':1,'schedule_total_updates':120,
        'resume_optimizer':True,'warmup_updates':0,'lr_schedule':'constant','lr_final_fraction':1.0,
        'learning_rate_ac':5e-6,'learning_rate_target_lora':2e-6,'gradient_checkpointing_min_tokens':0}
    if any(config.get(k)!=v for k,v in required.items()):raise ValueError('Wrong120-stage constant-floor inherited-Adam configuration')


def validate_final_optimizer(state):
    saved=state['optimizer'];values=saved['state']
    if (state.get('updates_done')!=120 or state.get('epoch_number')!=1
            or [len(group['params']) for group in saved['param_groups']]!=[374,256]
            or len(values)!=615 or any(not {'step','exp_avg','exp_avg_sq'}<=v.keys() for v in values.values())
            or {float(value['step']) for value in values.values()}!={5370}):
        raise ValueError('Expected own complete120-stage/615 actual Adam5370 final optimizer')
    return {'stage_updates':120,'epoch':1,'active_parameter_states':615,'adam_parameter_steps':5370}


def validate_initial(initial,design):
    p=initial['optimizer_provenance']
    required={'adam_state_preserved_byte_exact':True,'param_groups_preserved_exact':True,'weight_bytes_unchanged':True,
        'independent_files_not_hardlinks':True,'optimizer_parameter_states':615,'adam_parameter_steps':5250,
        'parent_stage_updates_done':1250,'parent_lifetime_updates_done':5250,'stage_updates_done':0}
    if any(p.get(k)!=v for k,v in required.items()) or initial['payload_sha256']!=design['initial_payload_sha256']:
        raise ValueError('Initial paired42k/actual Adam5250 provenance differs')
    parent=initial['parent_pair']
    if (parent.get('continuation_schedule')!='final_lr' or parent.get('parent_questions')!=32000
            or parent.get('stage_unique_questions')!=10000 or parent.get('stage_epochs')!=1):
        raise ValueError('Expected completed constant-floor42k initialization')


def final_metadata(reports,runs_root,design,ids,design_sha):
    folder=reports/RUN;result=read(folder/'result.json');identity=read(folder/'run_identity.json')['identity']
    required_identity={'arm':'joint','epochs':1,'planned_updates':120,'steps_per_epoch':120,
        'material_sha256':design['material_sha256'],'material_ids_sha256':digest(ids),'passage_parts':False,
        'evidence_tokens':32768,'max_passages':64,'ratio':1/16,'encode_batch_max_tokens':65536,
        'initial_pair_sha256':design['initial_pair_sha256']}
    required_result={'stop_reason':'complete','completed_global_updates':120,'planned_updates':120,'planned_epochs':1,
        'items':960,'examples':960,'parent_unique_questions':42000,'unique_questions_after_stage':42960,
        'completed_total_presentations':42960,'optimizer_policy':'inherited Adam5250, constant floor LR'}
    if (identity!=required_identity or read(folder/'material_ids.json')!=ids
            or read(folder/'training_config.json')!=design['training_config']
            or any(result.get(k)!=v for k,v in required_result.items())):
        raise ValueError('Own960×1 stage incomplete or final identity/config/order changed')
    receipt=read(folder/'source_receipt.json')
    for key in ('builder_sha256','source_sha256','initial_payload_sha256','initial_pair_sha256','material_sha256','parent_condition','subset_sha256'):
        if receipt.get(key)!=design[key]:raise ValueError('Training source/initialization receipt changed: '+key)
    if receipt.get('design_sha256')!=design_sha:raise ValueError('Training used a different frozen design')
    inherited=read(folder/'inherited_optimizer_check.json')
    if any(inherited.get(k)!=v for k,v in {'optimizer_states':615,'min_parameter_step':5250,'max_parameter_step':5250,'stage_updates_done':0}.items()):
        raise ValueError('Missing actual inherited Adam5250 at stage start')
    root=(runs_root/RUN).resolve()
    if 'run_root' in design and Path(design['run_root']).resolve()!=root:raise ValueError('Relocated run root differs')
    ac,reader=Path(result['checkpoint']).resolve(),Path(result['target_lora_checkpoint']).resolve()
    if ac!=root/'AC_MODELS/rag_poc/epoch_001' or reader!=root/'LORAS/rag_reader/epoch_001':
        raise ValueError('Expected own completed epoch001 pair, not selected/partial/foreign weights')
    marker_path=ac.parent/'completed_epoch.json';marker=read(marker_path)
    if (marker.get('epoch_number')!=1 or (root/marker['ac_checkpoint']).resolve()!=ac
            or (root/marker['target_lora_checkpoint']).resolve()!=reader):
        raise ValueError('Published paired final marker differs')
    panel=read(folder/'panel_00120.json')
    if any(panel['progress'].get(k)!=v for k,v in {'global_step':120,'step':120,'steps':120,'epoch':1,'epoch_number':1,'epochs':1}.items()):
        raise ValueError('Own120 final panel is incomplete')
    if [r['item_id'] for r in panel['summaries']['held AC']['per_item']]!=ids['held']:
        raise ValueError('Final panel fixed100 order changed')
    input_check=read(folder/'input_check.json');initial=read(folder/'initial_ac_baseline.json')
    if (input_check.get('held_ids')!=ids['held'] or input_check.get('initial_pair_sha256')!=design['initial_pair_sha256']
            or initial.get('identity')!=input_check or initial.get('measured_at_update')!=0):
        raise ValueError('Own initial42k input/baseline publication differs')
    files=('result.json','run_identity.json','training_config.json','material_ids.json','source_receipt.json',
        'inherited_optimizer_check.json','initial_ac_baseline.json','input_check.json','panel_00120.json')
    publication={name:sha(folder/name) for name in files}|{'completed_marker_sha256':sha(marker_path)}
    return {'ac_checkpoint':str(ac),'reader_checkpoint':str(reader)},publication,panel,input_check


def replay_scores(summary,scored,prior_summary,items,held,panel_sha):
    prior=common.answer_scores(prior_summary,items,held)
    delta=[a['answer_nll']-b['answer_nll'] for a,b in zip(scored['per_item'],prior['per_item'])]
    if len(delta)!=100 or any(len(a['target_logp'])!=len(b['target_logp']) for a,b in zip(summary['per_item'],prior_summary['per_item'])):
        raise ValueError('Final panel target dimensions changed')
    token_delta=max(abs(x-y) for a,b in zip(summary['per_item'],prior_summary['per_item']) for x,y in zip(a['target_logp'],b['target_logp']))
    return {'original_panel_sha256':panel_sha,'maximum_absolute_difference':max(map(abs,delta)),
        'maximum_absolute_target_logp_difference':token_delta,'mean_difference':statistics.mean(delta),
        'absolute_tolerance':1e-3,'passed':max(map(abs,delta))<=1e-3 and token_delta<=1e-3}


def saved_training_logits_chunk(training_result):
    tuning=training_result.get('autotuning')
    chunk=tuning.get('logits_chunk_tokens') if isinstance(tuning,dict) else None
    if isinstance(chunk,bool) or not isinstance(chunk,int) or chunk<1:
        raise ValueError('Complete training result must record its actual positive integer logits_chunk_tokens')
    return chunk


def main(args):
    began=time.monotonic();label=RUN if args.checkpoint=='final' else RUN+'__parent42k';cell=label+'__'+args.context
    folder=args.output/cell;output=folder/'result.json'
    def progress(phase,**extra):
        record={'cell':cell,'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-began,1),**extra}
        write(folder/'status.json',record);print(json.dumps(record),flush=True)
    progress('validating own completed960 stage and frozen inputs')
    manifest=read(args.manifest)
    if manifest.get('kind')!='pi42k_merged32k_readout_v2' or not set(PINNED_HELPERS)<=manifest['helper_sha256'].keys():
        raise ValueError('Wrong or incomplete independent readout manifest')
    for leaf,expected in manifest['helper_sha256'].items():
        if Path(leaf).name!=leaf or sha(Path(__file__).with_name(leaf))!=expected:raise ValueError('Readout helper changed: '+leaf)
    design=read(args.design);validate_design(design)
    for leaf,expected in design['helper_sha256'].items():
        if Path(leaf).name!=leaf or sha(Path(__file__).with_name(leaf))!=expected:raise ValueError('Frozen training helper changed: '+leaf)
    for path,expected in ((args.material,MATERIAL_SHA),(args.subset,design['subset_sha256']),
            (args.selection,SELECTION_SHA),(args.bench,design['builder_sha256']),(args.initial_pair,design['initial_pair_sha256'])):
        if sha(path)!=expected:raise ValueError('Frozen input bytes changed: '+str(path))
    selection=read(args.selection);subset=read(args.subset);initial=read(args.initial_pair);validate_initial(initial,design)
    if (selection['selection_key']!=common.SELECTION_KEY or selection['quotas']!=common.QUOTAS
            or subset['held']!=selection['selected_ids'] or len(subset['held'])!=100
            or len(subset['train'])!=960 or len(set(subset['train']))!=960):raise ValueError('Original fixed100/subset changed')
    bench=common.load_module('pi42k_readout_builder',args.bench)
    material=read(args.material)
    if len(material['train'])!=8000 or len(material['held'])!=400 or len(material['retrieval'])!=32398:raise ValueError('Wrong original fresh8k material')
    material=bench.selected_material(material,subset);material['train']=common.interleave(material['train'])
    ids={split:[row['example']['example_id'] for row in material[split]] for split in ('train','held')}
    if digest(ids)!=design['material_ids_sha256']:raise ValueError('Frozen960 training/fixed100 order differs')
    final_pair,publication,panel,input_check=final_metadata(args.reports,args.runs_root,design,ids,sha(args.design))
    training_result_path=args.reports/RUN/'result.json'
    if sha(training_result_path)!=publication['result.json']:raise ValueError('Completed training result changed after final validation')
    training_logits_chunk=saved_training_logits_chunk(read(training_result_path))
    import torch
    optimizer=Path(final_pair['ac_checkpoint'])/'optimizer.pt'
    state=torch.load(optimizer,map_location='cpu',weights_only=False,mmap=True)
    publication.update(optimizer=validate_final_optimizer(state),optimizer_sha256=sha(optimizer));del state
    pair=final_pair if args.checkpoint=='final' else {key:initial[key] for key in ('ac_checkpoint','reader_checkpoint')}
    hashes=common.payload_hashes(pair)
    if args.checkpoint=='parent42k' and hashes!=design['initial_payload_sha256']:raise ValueError('Parent42k weights changed')
    if args.checkpoint=='parent42k' and args.context!='2k':raise ValueError('Only the inexpensive same-host parent2k control is authorized')
    tokens,passages=CONTEXTS[args.context];update=120 if args.checkpoint=='final' else 0
    identity={'condition':label,'checkpoint_role':args.checkpoint,'context':args.context,'update':update,
        'paired_checkpoint':pair,'checkpoint_sha256':hashes,'publication':publication,'design_sha256':sha(args.design),
        'readout_manifest_sha256':sha(args.manifest),'evaluator_sha256':sha(__file__),'builder_sha256':design['builder_sha256'],
        'source_sha256':design['source_sha256'],'material_sha256':MATERIAL_SHA,'subset_sha256':sha(args.subset),'selection_sha256':SELECTION_SHA,
        'passage_parts':False,'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':tokens,'max_passages':passages,'micro_batch_examples':1,
        'training_logits_chunk_tokens':training_logits_chunk,'training_logits_chunk_result_sha256':publication['result.json'],
        'actual_adam_step':5370 if update else 5250,'unique_training_questions':42960 if update else 42000,
        'readout_policy':'Own final only; fresh AC/full/no-context NLL on exact original development100, same-host parent2k control. No RTX score reuse or confirmation.',
        'decoding':'Final32k only:100 greedy AC answers, thinking off,160-token cap, inherited EOS union chat EOT; actual IDs saved.'}
    if output.exists():
        result=read(output)
        if result['identity']!=identity:raise ValueError('Partial readout identity changed')
        if result['status']=='complete':progress('already complete');return
    else:result={'status':'running','identity':identity,'rows':[],'views':{}}
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress,_TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    source={key:sha(module.__file__) for key,module in [('ac_model',ac_module),('trainer',training_module),('utils',utils_module)]}
    if source!=design['source_sha256']:raise ValueError('Runtime source differs from immutable training snapshot')
    progress('loading paired weights')
    h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=65536)
    ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();peft=h.module_manager.ensure_lora(bench.READER)
    model={'side_model':ac.side.model_config.model_id,'reader_model':ac.target.model_config.model_id,
        'side_revision':getattr(ac.side.model.config,'_commit_hash',None),'reader_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'side_config_sha256':digest(json.loads(ac.side.model.config.to_json_string())),
        'reader_config_sha256':digest(json.loads(ac.target.model.config.to_json_string()))}
    if result.get('model_provenance',model)!=model:raise ValueError('Model provenance changed on resume')
    result['model_provenance']=model;progress('building exact selected100 evidence',tokens=tokens)
    study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':material['held'],'retrieval':material['retrieval']},
        passage_parts=False,evidence_tokens=tokens,max_passages=passages,ratio=1/16)
    if [item.item_id for item in held]!=selection['selected_ids']:raise ValueError('Fixed100 held order changed')
    for item,row in zip(held,material['held']):
        if any(unit['text'] not in item.info['passages_text'] for unit in row['example']['study_source']['units']):
            raise ValueError('Required source truncated: '+item.item_id)
    full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
    trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',no_context_penalty_weight=0,
        micro_batch_examples=1,max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED,
        logits_chunk_tokens=training_logits_chunk))
    row_counts=common.resolved_rows(trainer,ac,held)
    if any(len(row['part_rows'])!=1 for row in row_counts):raise ValueError('Expected one monolithic encoder request per item')
    fingerprints={'shared_input_fingerprint':common.shared_fingerprint(held,full,noctx),
        'ac_input_fingerprint':digest([(item.item_id,item.ac_messages) for item in held]),
        'total_rows_fingerprint':digest([(row['item_id'],row['compressed_rows']) for row in row_counts])}
    if args.context=='32k' and any(fingerprints[key]!=input_check[key] for key in ('shared_input_fingerprint','ac_input_fingerprint')):
        raise ValueError('Exact final training panel inputs changed')
    for key,value in fingerprints.items():
        if result.get(key,value)!=value:raise ValueError('Partial readout inputs changed: '+key)
    cohort_of={ident:bank for bank,group in material['held_cohorts'].items() for ident in group}
    rows=[{'item_id':item.item_id,'original_400_index':selection['original_held_ids'].index(item.item_id),
        'generator_cohort':cohort_of[item.item_id],'source_family':raw['source_family'],'domain':raw['domain'],
        'question':item.info['question'],'gold':item.info['answer'],'required_source':raw['example']['study_source'],
        'full_evidence':item.info['passages_text'],'in_context_messages':item.in_context_messages,'answer_positions':item.info['answer_positions'],
        'assistant_token_ids':item.assistant_token_ids,'row_budget':budget,'views':{}} for item,raw,budget in zip(held,material['held'],row_counts)]
    if result['rows']:
        if [{key:value for key,value in row.items() if key!='views'} for row in result['rows']]!=[{key:value for key,value in row.items() if key!='views'} for row in rows]:
            raise ValueError('Partial row evidence or target identity changed')
    else:result['rows']=rows
    result.update(**fingerprints,input_rows=row_counts,max_view_rows=ac.config.max_view_rows,required_sources_retained_whole=True,
        selected_ids=selection['selected_ids'],scored_ids=selection['selected_ids'])
    raw=peft.generation_config.eos_token_id;eos=[raw] if isinstance(raw,int) else list(raw or [])
    eot=ac.target.model_config.model_description.eot_token;eot_id=ac.target.tokenizer.convert_tokens_to_ids(eot) if eot else None
    if eot_id is None:raise ValueError('Missing Qwen chat-end token')
    if eot_id not in eos:eos.append(eot_id)
    generation={'inherited_generation_config':json.loads(peft.generation_config.to_json_string()),'explicit_eos_ids':eos,
        'eot':eot,'eot_id':eot_id,'max_new_tokens':160,'thinking':False,'do_sample':False}
    if result.get('generation_provenance',generation)!=generation:raise ValueError('Generation settings changed')
    result['generation_provenance']=generation
    variables=trainer._autotuning_variables(ac);trainer._training_deadline=began+args.seconds
    if variables['logits_chunk_tokens']!=training_logits_chunk:raise ValueError('Resolved logits chunk differs from saved training policy')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(ac.device),
        'tf32_matmul':True,'tf32_cudnn':True,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'bf16_reduction_split_k':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,'autotuning':variables['provenance'],
        'resolved_logits_chunk_tokens':variables['logits_chunk_tokens'],
        'logits_chunk_policy':'Actual completed960 training result value reused for final32k,parent2k,final2k; no free-memory-derived replacement.'}
    if not runtime['bf16_reduced_precision_reduction']:raise ValueError('Original precision policy required')
    stable={key:value for key,value in runtime.items() if key!='autotuning'}
    if result.get('runtime_policy',stable)!=stable:raise ValueError('Partial readout hardware/runtime changed')
    result.update(runtime=runtime,runtime_policy=stable)
    sample_progress=ActivationContextTrainingProgress(epoch=1 if update else 0,epochs=1,epoch_number=1 if update else 0,
        step=update,steps=120,global_step=update,epoch_elapsed_s=0.,elapsed_s=0.)
    write(output,result)
    try:
        with configure_fla_runtime(variables['fla_config']),torch.no_grad():
            for name,items in {'AC':held,'full context':full,'no context':noctx}.items():
                if name not in result['views']:
                    progress('scoring fresh '+name,questions=100)
                    torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device);start=time.monotonic()
                    summary=trainer._evaluate(ac,items,ac.device);scored=common.answer_scores(summary,items,held);torch.cuda.synchronize(ac.device)
                    replay=replay_scores(summary,scored,panel['summaries']['held '+name],items,held,publication['panel_00120.json']) if update and args.context=='32k' else None
                    for row,item,value,raw_score in zip(result['rows'],held,scored['per_item'],summary['per_item']):
                        token_ids=item.assistant_token_ids[0]
                        row['views'][name]={'answer_nll':value['answer_nll'],'gold_token_scores':[{'token_id':token_ids[p],
                            'text':study.tokenizer.decode([token_ids[p]],skip_special_tokens=False),'nll':-raw_score['target_logp'][p]} for p in item.info['answer_positions']]}
                    result['views'][name]={**scored,'cohorts':common.cohort_summary(scored,material['held_cohorts']),
                        'selected100_answer_nll':statistics.mean(row['views'][name]['answer_nll'] for row in result['rows']),
                        'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated(ac.device)/2**30,
                        'panel_replay':replay,'scoring_complete':True,'complete':False,'fresh_measurement':True}
                    write(output,result)
                replay=result['views'][name]['panel_replay']
                if replay is not None and not replay['passed']:
                    result['status']='replay_mismatch';write(output,result);raise ValueError('Own final100 replay differs beyond1e-3; preserve and diagnose')
                if update and args.context=='32k' and name=='AC':
                    for index,item in enumerate(held):
                        row=result['rows'][index]
                        if 'response' in row['views'][name]:continue
                        progress('decoding final32k AC answers',completed=index,total=100)
                        sample,generated=common.sample_with_token_ids(trainer,ac,item,sample_progress,eos)
                        text=sample['student'];answer,parsing=common.parsed_answer(study,text);end=text.find('</tool_call>')
                        row['views'][name].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,
                            post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                        write(output,result)
                result['views'][name]['complete']=True;write(output,result)
    except _TrainingTimeLimit:
        result['status']='partial';write(output,result);progress('time limit; resume same cell');return
    if trainer.optimizers or trainer.updates_done:raise AssertionError('Readout unexpectedly optimized weights')
    if common.payload_hashes(pair)!=hashes:raise ValueError('Paired payloads changed during readout')
    result.update(status='complete',optimizer_updates=0,parent_payloads_rehashed_unchanged=True,
        attempt_elapsed_seconds=round(time.monotonic()-began,1));write(output,result);progress('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','design','material','subset','bench','selection','reports','runs-root','initial-pair','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--checkpoint',choices=('final','parent42k'),default='final')
    parser.add_argument('--context',choices=tuple(CONTEXTS),default='32k')
    parser.add_argument('--seconds',type=int,default=2100)
    main(parser.parse_args())
