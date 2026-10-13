"""Independent H200 constant42k merged32k readout; no960-stage dependency or RTX score reuse."""
import argparse
import json
from pathlib import Path
import statistics
import time
import evaluate_strong_long_cohort_v1 as common
import evaluate_pi42k_merged32k_v1 as final_helper

read,write,sha,digest=common.read,common.write,common.file_sha,common.digest
ARM='continue32k_10k_final_lr'
MATERIAL_SHA='49f711e3eff25b0063f148f850e73ff4fe7917c05dad0898db0167623d7e11a9'
PAIR_SHA='d423ee26b4404cfc0f78d4ef9299ebbf9e31116c0bf7eaa6cf63a6130bf537f7'
RECEIPT_SHA='77aa6ea2858b1358d575f3b1130ce2bb187758fc1f548f37b1fe56ebb5092ca7'
SELECTION_SHA='469320c02db9347d4d93b99c94003f9eb40d738d03ee272ff83a0129fda92051'
BUILDER_SHA='ee791342b4b21ca22d3be810e34ce1c6c3337cf3921d60bf6b8cfa8cde983054'


def validate_stage(initial,receipt,native,metadata,selection):
    expected={'status':'complete','job_id':'2181811','pair_sha256':PAIR_SHA,'material_sha256':MATERIAL_SHA,
        'optimizer_states':615,'adam_parameter_steps':5250,'stage_updates_done':0,'stage_epoch_number':0,
        'param_group_counts':[374,256],'optimizer_bytes_unchanged':True,'original_orcd_pair_unchanged':True}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Missing actual verified CSAIL5250 staging receipt')
    final_helper.validate_initial(initial,{'initial_payload_sha256':receipt['payload_sha256']})
    if (native.get('status')!='complete' or native['identity'].get('condition')!=ARM or native['identity'].get('update')!=1250
            or native['identity']['checkpoint_sha256']!=receipt['payload_sha256']
            or native['identity']['paired_checkpoint']!=initial['parent_pair']
            or native['selection']['selected_ids']!=selection['selected_ids']):
        raise ValueError('Staged pair does not bind the completed own42k native readout')
    expected_meta={'name':ARM,'is_final_training_checkpoint':True,'total_unique_questions':42000,
        'stage_unique_questions':10000,'stage_epochs':1,'stage_updates':1250,'parent_questions':32000}
    if any(metadata.get(k)!=v for k,v in expected_meta.items()):raise ValueError('Staged parent is not the completed constant42k final')
    if metadata['completed_epoch']!={'epoch_number':1,'ac_checkpoint':'AC_MODELS/rag_poc/epoch_001','target_lora_checkpoint':'LORAS/rag_reader/epoch_001'}:
        raise ValueError('Parent publication metadata differs')


def main(args):
    began=time.monotonic();cell=ARM+'__merged_32k_h200';folder=args.output/cell;output=folder/'result.json'
    def progress(phase,**extra):
        record={'cell':cell,'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-began,1),**extra}
        write(folder/'status.json',record);print(json.dumps(record),flush=True)
    progress('validating independently staged42k final and actual Adam5250 receipt')
    manifest=read(args.manifest)
    if manifest.get('kind')!='constant42k_merged32k_csail_v1':raise ValueError('Wrong readout manifest')
    for leaf,expected in manifest['helper_sha256'].items():
        if Path(leaf).name!=leaf or sha(Path(__file__).with_name(leaf))!=expected:raise ValueError('Pinned helper changed: '+leaf)
    for path,expected in ((args.stage/'pair.json',PAIR_SHA),(args.stage/'CSAIL_RECEIPT.json',RECEIPT_SHA),
            (args.stage/'material.json',MATERIAL_SHA),(args.selection,SELECTION_SHA),(args.bench,BUILDER_SHA)):
        if sha(path)!=expected:raise ValueError('Staged frozen input changed: '+str(path))
    initial=read(args.stage/'pair.json');receipt=read(args.stage/'CSAIL_RECEIPT.json');selection=read(args.selection)
    for leaf in ('transfer_manifest.json','parent_native_readout_identity.json','parent_checkpoint_metadata.json','parent_completed_epoch.json'):
        expected=receipt['manifest_sha256'] if leaf=='transfer_manifest.json' else receipt['files'][leaf]['sha256']
        if sha(args.stage/leaf)!=expected:raise ValueError('Staged provenance bytes changed: '+leaf)
    native=read(args.stage/'parent_native_readout_identity.json');metadata=read(args.stage/'parent_checkpoint_metadata.json')
    validate_stage(initial,receipt,native,metadata,selection)
    if selection['selection_key']!=common.SELECTION_KEY or selection['quotas']!=common.QUOTAS:raise ValueError('Fixed100 policy changed')
    pair={key:initial[key] for key in ('ac_checkpoint','reader_checkpoint')}
    if Path(pair['ac_checkpoint']).resolve()!=args.stage.resolve()/'ac' or Path(pair['reader_checkpoint']).resolve()!=args.stage.resolve()/'reader':
        raise ValueError('Expected locally staged parent pair, not foreign paths')
    hashes=common.payload_hashes(pair)
    if hashes!=receipt['payload_sha256']:raise ValueError('Verified constant42k payloads changed')
    material=read(args.stage/'material.json')
    if len(material['train'])!=8000 or len(material['held'])!=400 or len(material['retrieval'])!=32398:raise ValueError('Wrong exact fresh8k material')
    held_by_id={row['example']['example_id']:row for row in material['held']}
    if set(held_by_id)!=set(selection['original_held_ids']):raise ValueError('Original400 membership changed')
    material['held']=[held_by_id[ident] for ident in selection['selected_ids']]
    material['held_cohorts']={bank:[ident for ident in selection['selected_ids'] if ident in group] for bank,group in material['held_cohorts'].items()}
    if sorted(map(len,material['held_cohorts'].values()))!=[50,50]:raise ValueError('Expected fixed50/50 generator cohorts')
    bench=common.load_module('constant42k_csail_builder',args.bench);tokens=32768;passages=64;update=1250
    identity={'condition':ARM,'context':'32k','update':1250,'lifetime_questions':42000,'actual_adam_step':5250,
        'paired_checkpoint':pair,'checkpoint_sha256':hashes,'staging_receipt_sha256':RECEIPT_SHA,'pair_sha256':PAIR_SHA,
        'readout_manifest_sha256':sha(args.manifest),'evaluator_sha256':sha(__file__),'builder_sha256':BUILDER_SHA,
        'source_sha256':common.SOURCE,'material_sha256':MATERIAL_SHA,'selection_sha256':SELECTION_SHA,
        'passage_parts':False,'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':tokens,'max_passages':passages,'micro_batch_examples':1,
        'readout_policy':'Independent staged constant42k final. Fresh H200 three-view NLL; no960-stage dependency, RTX score reuse, training or confirmation.',
        'decoding':'AC100 first; optional matched full100 afterward. Greedy160, thinking off, inherited EOS plus chat end, actual token IDs.',
        'decode_full_requested':args.decode_full}
    if output.exists():
        result=read(output)
        if result['identity']!=identity:raise ValueError('Partial readout identity changed')
        if result['status']=='complete' and (not args.decode_full or result.get('optional_full_decode_complete')):
            progress('already complete');return
    else:result={'status':'running','identity':identity,'rows':[],'views':{}}
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress,_TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    source={key:sha(module.__file__) for key,module in [('ac_model',ac_module),('trainer',training_module),('utils',utils_module)]}
    if source!=common.SOURCE:raise ValueError('Runtime source differs from immutable training snapshot')
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
        micro_batch_examples=1,max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED))
    row_counts=common.resolved_rows(trainer,ac,held)
    if any(len(row['part_rows'])!=1 for row in row_counts):raise ValueError('Expected one monolithic encoder request per item')
    fingerprints={'shared_input_fingerprint':common.shared_fingerprint(held,full,noctx),
        'ac_input_fingerprint':digest([(item.item_id,item.ac_messages) for item in held]),
        'total_rows_fingerprint':digest([(row['item_id'],row['compressed_rows']) for row in row_counts])}
    if any(fingerprints[key]!=manifest['input_fingerprints'][key] for key in fingerprints):
        raise ValueError('Canonical original100 merged32k inputs or total rows changed')
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
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(ac.device),
        'tf32_matmul':True,'tf32_cudnn':True,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'bf16_reduction_split_k':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,'autotuning':variables['provenance']}
    if not runtime['bf16_reduced_precision_reduction']:raise ValueError('Original precision policy required')
    stable={key:value for key,value in runtime.items() if key!='autotuning'}
    if result.get('runtime_policy',stable)!=stable:raise ValueError('Partial readout hardware/runtime changed')
    result.update(runtime=runtime,runtime_policy=stable)
    sample_progress=ActivationContextTrainingProgress(epoch=1 if update else 0,epochs=1,epoch_number=1 if update else 0,
        step=update,steps=1250,global_step=update,epoch_elapsed_s=0.,elapsed_s=0.)
    write(output,result)
    try:
        with configure_fla_runtime(variables['fla_config']),torch.no_grad():
            for name,items in {'AC':held,'full context':full,'no context':noctx}.items():
                if name not in result['views']:
                    progress('scoring fresh '+name,questions=100)
                    torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device);start=time.monotonic()
                    summary=trainer._evaluate(ac,items,ac.device);scored=common.answer_scores(summary,items,held);torch.cuda.synchronize(ac.device)
                    replay=None  # Fresh H200 measurement: no RTX-score reuse or cross-device numerical replay.
                    for row,item,value,raw_score in zip(result['rows'],held,scored['per_item'],summary['per_item']):
                        token_ids=item.assistant_token_ids[0]
                        row['views'][name]={'answer_nll':value['answer_nll'],'gold_token_scores':[{'token_id':token_ids[p],
                            'text':study.tokenizer.decode([token_ids[p]],skip_special_tokens=False),'nll':-raw_score['target_logp'][p]} for p in item.info['answer_positions']]}
                    result['views'][name]={**scored,'cohorts':common.cohort_summary(scored,material['held_cohorts']),
                        'selected100_answer_nll':statistics.mean(row['views'][name]['answer_nll'] for row in result['rows']),
                        'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated(ac.device)/2**30,
                        'panel_replay':replay,'scoring_complete':True,'complete':False,'fresh_measurement':True}
                    write(output,result)
                if name=='AC':
                    for index,item in enumerate(held):
                        row=result['rows'][index]
                        if 'response' in row['views'][name]:continue
                        progress('decoding parent42k merged32k AC answers',completed=index,total=100)
                        sample,generated=common.sample_with_token_ids(trainer,ac,item,sample_progress,eos)
                        text=sample['student'];answer,parsing=common.parsed_answer(study,text);end=text.find('</tool_call>')
                        row['views'][name].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,
                            post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                        write(output,result)
                result['views'][name]['complete']=True
                if name=='AC':
                    if common.payload_hashes(pair)!=hashes:raise ValueError('Parent weights changed before AC100 publication')
                    result.update(ac_complete=True,optimizer_updates=0,ac_payloads_rehashed_unchanged=True)
                    progress('AC100 published; fresh full/no-context scores next')
                write(output,result)
            if common.payload_hashes(pair)!=hashes:raise ValueError('Parent weights changed before core publication')
            result.update(core_complete=True,parent_payloads_rehashed_unchanged=True)
            write(output,result);progress('Three-view NLL and AC100 complete; optional full answers next')
            if args.decode_full:
                for index,item in enumerate(full):
                    row=result['rows'][index]
                    if 'response' in row['views']['full context']:continue
                    progress('optional matched full-context answers',completed=index,total=100,ac_complete=True,core_complete=True)
                    sample,generated=common.sample_with_token_ids(trainer,ac,item,sample_progress,eos)
                    text=sample['student'];answer,parsing=common.parsed_answer(study,text);end=text.find('</tool_call>')
                    row['views']['full context'].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,
                        post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                    write(output,result)
                result['optional_full_decode_complete']=True
    except _TrainingTimeLimit:
        result['status']='complete' if result.get('core_complete') else 'partial'
        result['optional_full_decode_complete']=all('response' in row['views'].get('full context',{}) for row in result['rows'])
        write(output,result);progress('time limit; core retained, optional answers resumable',core_complete=result.get('core_complete',False));return
    if trainer.optimizers or trainer.updates_done:raise AssertionError('Readout unexpectedly optimized weights')
    if common.payload_hashes(pair)!=hashes:raise ValueError('Paired payloads changed during readout')
    result.update(status='complete',optimizer_updates=0,parent_payloads_rehashed_unchanged=True,
        attempt_elapsed_seconds=round(time.monotonic()-began,1));write(output,result);progress('complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','stage','bench','selection','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=2250)
    parser.add_argument('--decode-full',action='store_true')
    main(parser.parse_args())
