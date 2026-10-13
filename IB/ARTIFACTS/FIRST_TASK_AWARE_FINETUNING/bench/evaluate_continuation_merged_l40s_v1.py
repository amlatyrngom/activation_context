"""Independent ORCD L40S constant42k merged32k readout; all NLL freshly measured."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

import evaluate_strong_long_cohort_v1 as long
import evaluate_continuation_10k_v1 as continuation

ARM = 'continue32k_10k_final_lr'
LONG_SHA = 'a49dec5dc2aab1791b3e2afd746d4f6127ff70a00aa3385c6bedba348578e585'
CONTINUATION_SHA = '7275f69c0f0ca2ce6fd6a74705f0a05cb596d14054790a5bce86e408a217ef1f'
NATIVE_SHA = '1897fcf5b2c75a7fb72638cd658a75e4f7d5f92fce62276dfeb270c2174a06d6'
REFERENCES = {
    '32k': {'sha256':'a25798cdae7fc50ab84f921344c4491bc89ee9fa6859df6dd1fdf5758e8bc6cb',
        'evaluator':'1714f76c71d709a216b671f490b54062879225ae46cf30849d3e15547b62977c',
        'tokens':32768,'passages':64,'micro':1,'scored':100},
    '8k': {'sha256':'3a689635e419edcb2aea2708f5321e1561300c4ded9cc75690bfae76c66b2cbd',
        'evaluator':'c84649afa4743f3dda7126620116d78e1aaaca4c48521ee24d21597749b4be7b',
        'tokens':8192,'passages':16,'micro':2,'scored':400}}
ADAM_RECEIPT_SHA = '7eb8aa8e77f5d9f38ced3ea41e843caea855078470fbf38ab08d8695043d2b0e'
read,write,sha,digest = long.read,long.write,long.file_sha,long.digest
ROW_FIELDS = ('item_id','original_400_index','generator_cohort','source_family','domain','question','gold',
    'required_source','full_evidence','in_context_messages','answer_positions','assistant_token_ids')


def bind_readouts(native, reference, pair, hashes, selection, design, context):
    spec=REFERENCES[context]
    for record in (native,reference):
        if (record.get('status')!='complete' or record['identity'].get('condition')!=ARM
                or record['identity'].get('update')!=1250 or record.get('optimizer_updates')!=0
                or record['identity'].get('checkpoint_sha256')!=hashes or record['identity'].get('paired_checkpoint')!=pair):
            raise ValueError('Readout does not bind the completed42k own final pair')
        if (not record['views']['AC'].get('complete') or len(record['rows'])!=100
                or [row['item_id'] for row in record['rows']]!=selection['selected_ids']
                or any('response' not in row['views']['AC'] for row in record['rows'])):
            raise ValueError('Expected complete native and matched-context fixed100 AC responses')
    expected={'context':context,'passage_parts':True,'encode_batch_max_tokens':65536,'ratio':1/16,
        'evidence_tokens':spec['tokens'],'max_passages':spec['passages'],'micro_batch_examples':spec['micro'],
        'source_sha256':design['source_sha256'],'builder_sha256':design['builder_sha256'],
        'material_sha256':design['material_sha256'],'evaluator_sha256':spec['evaluator']}
    scored_ids=selection['selected_ids'] if context=='32k' else selection['original_held_ids']
    if (any(reference['identity'].get(k)!=v for k,v in expected.items())
            or reference.get('scored_ids')!=scored_ids or reference.get('selected_ids')!=selection['selected_ids']
            or len(reference['input_rows'])!=spec['scored'] or not reference.get('parent_payloads_rehashed_unchanged')
            or not reference.get('required_sources_retained_whole')
            or not all(reference['views'][view].get('complete') for view in ('AC','full context','no context'))):
        raise ValueError('Expected completed matched piecewise42k context/order/source controls')


def bind_adam_receipt(receipt,pair,hashes):
    values=receipt['optimizer_provenance']
    if receipt['parent_pair']!=pair or receipt['payload_sha256']!=hashes:
        raise ValueError('Adam5250 receipt does not describe this completed42k pair')
    expected={'adam_state_preserved_byte_exact':True,'param_groups_preserved_exact':True,
        'weight_bytes_unchanged':True,'independent_files_not_hardlinks':True,
        'optimizer_parameter_states':615,'adam_parameter_steps':5250,'parent_stage_updates_done':1250,
        'parent_lifetime_updates_done':5250}
    if any(values.get(key)!=value for key,value in expected.items()):
        raise ValueError('Completed42k actual Adam5250 provenance missing or changed')
    # This is an evaluation: validated preparation receipt binds optimizer lineage;
    # no optimizer tensor is loaded, rewritten, or used here.


def bind_inputs(fingerprints, row_counts, rows, reference):
    if fingerprints['shared_input_fingerprint']!=reference['shared_input_fingerprint']:
        raise ValueError('Full/no-context/tools/target inputs differ from reusable controls')
    if fingerprints['total_rows_fingerprint']!=reference['total_rows_fingerprint']:
        raise ValueError('Merged layout changed total compressed capacity')
    if len(row_counts)!=len(reference['input_rows']) or len(rows)!=100:
        raise ValueError('Scoring membership changed')
    for current,prior in zip(row_counts,reference['input_rows']):
        if (any(current[key]!=prior[key] for key in ('item_id','compressed_rows','reader_tokens','evidence_tokens',
                'passages','merged_encoder_tokens','merged_uncapped_rows','max_view_rows'))
                or current['part_encoder_tokens']!=[prior['merged_encoder_tokens']]
                or current['part_rows']!=[prior['compressed_rows']]
                or current['repeated_framing_extra_tokens']!=0):
            raise ValueError('Expected exactly one merged request with unchanged total rows/evidence: '+current['item_id'])
    for current,prior in zip(rows,reference['rows']):
        if any(current[field]!=prior[field] for field in ROW_FIELDS):
            raise ValueError('Selected question/gold/evidence/source changed: '+current['item_id'])


def main(args):
    began=time.monotonic();spec=REFERENCES[args.context];cell=ARM+'__merged_32k_l40s';folder=args.output/cell;output=folder/'result.json'
    def progress(phase,**extra):
        value={'cell':cell,'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-began,1),**extra}
        write(folder/'status.json',value);print(json.dumps(value),flush=True)
    progress('validating completed42k pair and canonical32k evidence')
    if sha(long.__file__)!=LONG_SHA or sha(continuation.__file__)!=CONTINUATION_SHA:
        raise ValueError('Pinned long-input or continuation final-guard helper changed')
    design=read(args.design)
    if design['encode_batch_max_tokens']!=65536:raise ValueError('Only the original65536 packing is authorized')
    material,selection,ids=long.load_material(args,design)
    continuation_design=continuation.validate_design(args.continuation_design)
    if sha(args.continuation_initial_pair)!=continuation_design['initial_pair_sha256']:
        raise ValueError('Continuation inherited-Adam initial pair changed')
    pair,hashes=continuation.final_pair(ARM,args.reports,continuation_design,args.continuation_design,read(args.continuation_initial_pair))
    final_result=read(args.reports/ARM/'result.json')
    if any(final_result.get(k)!=v for k,v in {'items':10000,'examples':10000,'planned_epochs':1,'planned_updates':1250}.items()):
        raise ValueError('Continuation final does not contain the full10000-question single stage')
    if sha(args.native_readout)!=NATIVE_SHA or sha(args.piecewise_readout)!=spec['sha256']:
        raise ValueError('Completed comparison readout bytes changed')
    native=read(args.native_readout);initial=read(args.piecewise_readout)
    bind_readouts(native,initial,pair,hashes,selection,design,args.context)
    if sha(args.adam_receipt)!=ADAM_RECEIPT_SHA:raise ValueError('Pinned actual Adam5250 receipt changed')
    bind_adam_receipt(read(args.adam_receipt),pair,hashes)
    if initial['identity']['design_sha256']!=sha(args.design) or initial['identity']['selection_sha256']!=sha(args.selection):
        raise ValueError('Initial32k design/selection publication differs')
    identity={'condition':ARM,'context':args.context,'update':1250,'lifetime_questions':42000,'parent_questions':32000,'stage_questions':10000,
        'paired_checkpoint':pair,'checkpoint_sha256':hashes,'design_sha256':sha(args.design),
        'continuation_design_sha256':sha(args.continuation_design),'selection_sha256':sha(args.selection),
        'material_sha256':design['material_sha256'],'builder_sha256':design['builder_sha256'],'source_sha256':design['source_sha256'],
        'evaluator_sha256':sha(__file__),'long_helper_sha256':LONG_SHA,'continuation_helper_sha256':CONTINUATION_SHA,
        'native2k_readout_sha256':NATIVE_SHA,'piecewise_readout_sha256':spec['sha256'],'adam5250_receipt_sha256':ADAM_RECEIPT_SHA,
        'passage_parts':False,'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':spec['tokens'],'max_passages':spec['passages'],'micro_batch_examples':spec['micro'],
        'reader':'Completed42k constant-LR continuation paired reader; all views freshly measured on L40S',
        'scope':'One merged-input development cell; same evidence and total rows as prior independent passages. Extra retrieved context tests distractor robustness, not distributed required evidence.',
        'actual_adam_step_at_parent':5250,'decode_views':['AC','optional full context'],
        'nll_replay':'All three NLL views freshly measured on this device. Earlier RTX piecewise readout binds only inputs/rows/paired weights; no score reuse or cross-device replay.',
        'decoding':'Greedy160, thinking off, inherited EOS union chat EOT; actual generated IDs retained'}
    if output.exists():
        result=read(output)
        if result['identity']!=identity:raise ValueError('Partial/complete readout identity changed')
        if result['status']=='complete' and result.get('optional_full_decode_complete'):progress('already complete');return
    else:result={'status':'running','identity':identity,'views':{},'rows':[]}
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress,_TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as trainer_module
    import activation.ac_model.ac_model_utils as utils_module
    source={k:sha(module.__file__) for k,module in [('ac_model',ac_module),('trainer',trainer_module),('utils',utils_module)]}
    if source!=design['source_sha256'] or source!=continuation_design['source_sha256']:
        raise ValueError('Runtime source differs from both frozen studies')
    bench=long.load_module('continuation_merged_builder',args.bench)
    progress('loading completed42k paired weights')
    h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=65536)
    ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();peft=h.module_manager.ensure_lora(bench.READER)
    provenance={'side_model':ac.side.model_config.model_id,'reader_model':ac.target.model_config.model_id,
        'side_revision':getattr(ac.side.model.config,'_commit_hash',None),'reader_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'side_config_sha256':digest(json.loads(ac.side.model.config.to_json_string())),
        'reader_config_sha256':digest(json.loads(ac.target.model.config.to_json_string()))}
    if provenance!=initial['model_provenance']:
        raise ValueError('Base model revisions/configs differ from the matched initial32k32k readout')
    if result.get('model_provenance',provenance)!=provenance:raise ValueError('Base model identity changed on readout resume')
    result['model_provenance']=provenance
    selected=set(selection['selected_ids']);rows=material['held'] if args.context=='8k' else [row for row in material['held'] if row['example']['example_id'] in selected]
    scored_ids=selection['original_held_ids'] if args.context=='8k' else selection['selected_ids']
    cohorts={bank:[ident for ident in group if ident in set(scored_ids)] for bank,group in material['held_cohorts'].items()}
    progress('building merged inputs with exact matched evidence',questions=len(rows))
    study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':rows,'retrieval':material['retrieval']},
        passage_parts=False,evidence_tokens=spec['tokens'],max_passages=spec['passages'],ratio=1/16)
    if [item.item_id for item in held]!=scored_ids:raise ValueError('Fixed100 original order changed')
    for item,row in zip(held,rows):
        if any(unit['text'] not in item.info['passages_text'] for unit in row['example']['study_source']['units']):
            raise ValueError('Required source not retained whole: '+item.item_id)
    full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
    trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',no_context_penalty_weight=0,
        micro_batch_examples=spec['micro'],max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED))
    row_counts=long.resolved_rows(trainer,ac,held)
    fingerprints={'shared_input_fingerprint':long.shared_fingerprint(held,full,noctx),
        'ac_input_fingerprint':digest([(item.item_id,item.ac_messages) for item in held]),
        'total_rows_fingerprint':digest([(row['item_id'],row['compressed_rows']) for row in row_counts])}
    indexes=[i for i,item in enumerate(held) if item.item_id in selected]
    cohort_of={ident:bank for bank,cohort in material['held_cohorts'].items() for ident in cohort}
    selected_rows=[{'item_id':held[i].item_id,'original_400_index':ids['held'].index(held[i].item_id),'generator_cohort':cohort_of[held[i].item_id],
        'source_family':rows[i]['source_family'],'domain':rows[i]['domain'],'question':held[i].info['question'],
        'gold':held[i].info['answer'],'required_source':rows[i]['example']['study_source'],
        'full_evidence':held[i].info['passages_text'],'in_context_messages':held[i].in_context_messages,
        'answer_positions':held[i].info['answer_positions'],'assistant_token_ids':held[i].assistant_token_ids,
        'row_budget':row_counts[i],'views':{}} for i in indexes]
    bind_inputs(fingerprints,row_counts,selected_rows,initial)
    if result['rows']:
        bind_inputs(fingerprints,row_counts,result['rows'],initial)
        if any(result.get(key)!=value for key,value in fingerprints.items()):raise ValueError('Partial fingerprints changed')
    else:result['rows']=selected_rows
    result.update(**fingerprints,input_rows=row_counts,selected_ids=selection['selected_ids'],scored_ids=scored_ids,
        required_sources_retained_whole=True,max_view_rows=ac.config.max_view_rows,input_binding='Exact completed piecewise42k full/no-context inputs, targets, evidence and total rows; one merged AC request per item')
    generation=peft.generation_config;raw=generation.eos_token_id;eos=[raw] if isinstance(raw,int) else list(raw or [])
    eot=ac.target.model_config.model_description.eot_token;eot_id=ac.target.tokenizer.convert_tokens_to_ids(eot) if eot else None
    if eot_id is None:raise ValueError('Expected Qwen chat-end token')
    if eot_id not in eos:eos.append(eot_id)
    gen={'inherited_generation_config':json.loads(generation.to_json_string()),'explicit_eos_ids':eos,'eot':eot,'eot_id':eot_id,
        'max_new_tokens':160,'thinking':False,'do_sample':False,'model_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'tokenizer':ac.target.tokenizer.name_or_path}
    if result.get('generation_provenance',gen)!=gen:raise ValueError('Generation settings changed during partial evaluation')
    result['generation_provenance']=gen
    variables=trainer._autotuning_variables(ac);trainer._training_deadline=began+args.seconds
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    result['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(ac.device),
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'bf16_reduction_split_k':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        'autotuning':variables['provenance']}
    if 'L40S' not in result['runtime']['device']:raise ValueError('This independent cell is reserved for the L40S runtime')
    if not result['runtime']['bf16_reduced_precision_reduction']:
        raise ValueError('Experimental BF16 reduction policy is not authorized for this readout')
    policy={key:value for key,value in result['runtime'].items() if key!='autotuning'}
    if result.get('runtime_policy',policy)!=policy:raise ValueError('Partial L40S readout hardware/precision changed')
    result['runtime_policy']=policy
    sample_progress=ActivationContextTrainingProgress(**read(args.reports/ARM/'panel_01250.json')['progress'])
    write(output,result)
    try:
        with configure_fla_runtime(variables['fla_config']),torch.no_grad():
            for name,items in {'AC':held}.items():
                if name in result['views']:continue
                progress('scoring '+name,questions=len(held))
                torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device);start=time.monotonic()
                summary=trainer._evaluate(ac,items,ac.device);scored=long.answer_scores(summary,items,held)
                torch.cuda.synchronize(ac.device)
                for record,i in zip(result['rows'],indexes):
                    item=held[i];token_ids=item.assistant_token_ids[0]
                    record['views'][name]={'answer_nll':scored['per_item'][i]['answer_nll'],
                        'gold_token_scores':[{'token_id':token_ids[p],'text':study.tokenizer.decode([token_ids[p]],skip_special_tokens=False),
                            'nll':-summary['per_item'][i]['target_logp'][p]} for p in item.info['answer_positions']]}
                result['views'][name]={**scored,'cohorts':long.cohort_summary(scored,cohorts),
                    'selected100_answer_nll':statistics.mean(record['views'][name]['answer_nll'] for record in result['rows']),
                    'selected50_by_generator':{bank:statistics.mean(record['views'][name]['answer_nll'] for record in result['rows'] if record['generator_cohort']==bank) for bank in material['held_cohorts']},
                    'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated(ac.device)/2**30,
                    'scoring_complete':True,'complete':name!='AC','panel_replay':None,'fresh_measurement':True}
                write(output,result)
            for j,i in enumerate(indexes):
                record=result['rows'][j]
                if 'response' in record['views']['AC']:continue
                progress('decoding fixed100 AC answers',completed=j,total=100)
                sample,generated=long.sample_with_token_ids(trainer,ac,held[i],sample_progress,eos)
                text=sample['student'];answer,parsing=long.parsed_answer(study,text);end=text.find('</tool_call>')
                record['views']['AC'].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,
                    post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                write(output,result)
            result['views']['AC']['complete']=True
            result['views']['AC']['decode_complete']=True
            if long.payload_hashes(pair)!=hashes:raise ValueError('Final pair changed before AC publication')
            result.update(ac_complete=True,ac_payloads_rehashed_unchanged=True,optimizer_updates=0)
            write(output,result);progress('AC100 published; fresh full/no-context NLL next')
            for name,items in {'full context':full,'no context':noctx}.items():
                if name in result['views']:continue
                progress('scoring fresh '+name,questions=100)
                torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device);start=time.monotonic()
                summary=trainer._evaluate(ac,items,ac.device);scored=long.answer_scores(summary,items,held)
                torch.cuda.synchronize(ac.device)
                for record,i in zip(result['rows'],indexes):
                    item=held[i];token_ids=item.assistant_token_ids[0]
                    record['views'][name]={'answer_nll':scored['per_item'][i]['answer_nll'],
                        'gold_token_scores':[{'token_id':token_ids[p],'text':study.tokenizer.decode([token_ids[p]],skip_special_tokens=False),
                            'nll':-summary['per_item'][i]['target_logp'][p]} for p in item.info['answer_positions']]}
                result['views'][name]={**scored,'cohorts':long.cohort_summary(scored,cohorts),
                    'selected100_answer_nll':statistics.mean(record['views'][name]['answer_nll'] for record in result['rows']),
                    'selected50_by_generator':{bank:statistics.mean(record['views'][name]['answer_nll'] for record in result['rows'] if record['generator_cohort']==bank) for bank in material['held_cohorts']},
                    'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated(ac.device)/2**30,
                    'scoring_complete':True,'complete':True,'panel_replay':None,'fresh_measurement':True}
                write(output,result)
            if long.payload_hashes(pair)!=hashes:raise ValueError('Final pair changed before core publication')
            result.update(core_complete=True,parent_payloads_rehashed_unchanged=True)
            write(output,result);progress('AC100 and three fresh NLL views complete; optional full answers next')
            if args.context=='32k':
                for j,i in enumerate(indexes):
                    record=result['rows'][j]
                    if 'response' in record['views']['full context']:continue
                    progress('optional matched full-context answers',completed=j,total=100,ac_complete=True)
                    sample,generated=long.sample_with_token_ids(trainer,ac,full[i],sample_progress,eos)
                    text=sample['student'];answer,parsing=long.parsed_answer(study,text);end=text.find('</tool_call>')
                    record['views']['full context'].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,
                        post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                    write(output,result)
                result['views']['full context']['decode_complete']=True
                result['optional_full_decode_complete']=True
    except _TrainingTimeLimit:
        result['status']='complete' if result.get('core_complete') else 'partial'
        result['optional_full_decode_complete']=all('response' in row['views'].get('full context',{}) for row in result['rows'])
        write(output,result);progress('time limit; core retained, optional full answers resumable',ac_complete=result.get('ac_complete',False),core_complete=result.get('core_complete',False));return
    if trainer.optimizers or trainer.updates_done:raise AssertionError('Readout unexpectedly optimized weights')
    if long.payload_hashes(pair)!=hashes:raise ValueError('Final pair changed during evaluation')
    result.update(status='complete',optimizer_updates=0,parent_payloads_rehashed_unchanged=True,
        attempt_elapsed_seconds=round(time.monotonic()-began,1))
    write(output,result);progress('complete',questions=100,selected=100)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('design','material','bench','selection','reports','continuation-design','continuation-initial-pair','native-readout','piecewise-readout','adam-receipt','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--context',choices=('32k',),default='32k')
    parser.add_argument('--seconds',type=int,default=2250)
    main(parser.parse_args())
