"""Completed42k rewarm checkpoint on exactly the initial32k 8k development inputs."""
import argparse
import json
from pathlib import Path
import statistics
import time

import evaluate_strong_long_cohort_v1 as long
import evaluate_continuation_10k_v1 as continuation

ARM = 'continue32k_10k_rewarm'
LONG_SHA = 'a49dec5dc2aab1791b3e2afd746d4f6127ff70a00aa3385c6bedba348578e585'
CONTINUATION_SHA = '7275f69c0f0ca2ce6fd6a74705f0a05cb596d14054790a5bce86e408a217ef1f'
NATIVE_SHA = '1e61fafed224a693db578efa4190ae5d12bb1ab5ae2e2d015e4515d99fb7d116'
INITIAL8K_SHA = '59e5d55aae0fe39e992db57fcceb77a78dea5cea17e65da5c69cf9ab5d296af4'
read,write,sha,digest = long.read,long.write,long.file_sha,long.digest
FINGERPRINTS = ('shared_input_fingerprint','ac_input_fingerprint','total_rows_fingerprint')
ROW_FIELDS = ('item_id','original_400_index','generator_cohort','source_family','domain','question','gold',
    'required_source','full_evidence','in_context_messages','answer_positions','assistant_token_ids','row_budget')


def bind_readouts(native, initial, pair, hashes, selection, design):
    if (native.get('status')!='complete' or native['identity'].get('condition')!=ARM
            or native['identity'].get('update')!=1250 or native.get('optimizer_updates')!=0
            or native['identity'].get('checkpoint_sha256')!=hashes or native['identity'].get('paired_checkpoint')!=pair):
        raise ValueError('Completed native2k readout differs from the published continuation final pair')
    if (not native['views']['AC'].get('complete') or len(native['rows'])!=100
            or [row['item_id'] for row in native['rows']]!=selection['selected_ids']
            or any('response' not in row['views']['AC'] for row in native['rows'])):
        raise ValueError('Native2k completed fixed100 response identity is incomplete')
    expected={'condition':'initial32k','context':'8k','update':0,'passage_parts':True,
        'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':8192,'max_passages':16,'micro_batch_examples':2,
        'source_sha256':design['source_sha256'],'builder_sha256':design['builder_sha256'],
        'checkpoint_sha256':design['initial_payload_sha256'],'evaluator_sha256':LONG_SHA}
    if (initial.get('status')!='complete' or any(initial['identity'].get(k)!=v for k,v in expected.items())
            or initial.get('optimizer_updates')!=0 or initial.get('scored_ids')!=selection['original_held_ids']
            or initial.get('selected_ids')!=selection['selected_ids'] or len(initial['input_rows'])!=400
            or len(initial['rows'])!=100 or not all(initial['views'][view].get('complete') for view in ('AC','full context','no context'))):
        raise ValueError('Expected the completed initial32k 8k reference with the exact400/100 inputs')
    if [row['item_id'] for row in initial['rows']]!=selection['selected_ids']:
        raise ValueError('Initial8k fixed100 saved order changed')


def bind_inputs(fingerprints, row_counts, rows, initial):
    if any(fingerprints[key]!=initial[key] for key in FINGERPRINTS):
        raise ValueError('Rebuilt400 shared/AC/row fingerprints differ from initial32k at8k')
    if row_counts!=initial['input_rows']:
        raise ValueError('Actual400 encoder tokens/rows/reader spans differ from initial32k at8k')
    if len(rows)!=100:
        raise ValueError('Expected fixed100 selected rows')
    for current,prior in zip(rows,initial['rows']):
        if any(current[field]!=prior[field] for field in ROW_FIELDS):
            raise ValueError('Selected gold/evidence/source identity changed: '+current['item_id'])


def main(args):
    began=time.monotonic();cell=ARM+'__8k';folder=args.output/cell;output=folder/'result.json'
    def progress(phase,**extra):
        value={'cell':cell,'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-began,1),**extra}
        write(folder/'status.json',value);print(json.dumps(value),flush=True)
    progress('validating completed42k pair and frozen8k reference')
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
    if sha(args.native_readout)!=NATIVE_SHA or sha(args.initial8k_readout)!=INITIAL8K_SHA:
        raise ValueError('Completed comparison readout bytes changed')
    native=read(args.native_readout);initial=read(args.initial8k_readout)
    bind_readouts(native,initial,pair,hashes,selection,design)
    if initial['identity']['design_sha256']!=sha(args.design) or initial['identity']['selection_sha256']!=sha(args.selection):
        raise ValueError('Initial8k design/selection publication differs')
    identity={'condition':ARM,'context':'8k','update':1250,'lifetime_questions':42000,'parent_questions':32000,'stage_questions':10000,
        'paired_checkpoint':pair,'checkpoint_sha256':hashes,'design_sha256':sha(args.design),
        'continuation_design_sha256':sha(args.continuation_design),'selection_sha256':sha(args.selection),
        'material_sha256':design['material_sha256'],'builder_sha256':design['builder_sha256'],'source_sha256':design['source_sha256'],
        'evaluator_sha256':sha(__file__),'long_helper_sha256':LONG_SHA,'continuation_helper_sha256':CONTINUATION_SHA,
        'native2k_readout_sha256':NATIVE_SHA,'initial32k_8k_readout_sha256':INITIAL8K_SHA,
        'passage_parts':True,'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':8192,'max_passages':16,'micro_batch_examples':2,
        'reader':'Completed42k rewarm continuation paired reader, including its own full/no-context controls',
        'scope':'One8k development cell, original400 scored then fixed100 AC decoded; no confirmation,32k cell or training.',
        'nll_replay':'No42k 8k training panel exists; exact input binding is to the completed initial32k 8k readout, not a same-weight NLL replay.',
        'decoding':'Greedy160, thinking off, inherited EOS union chat EOT; actual generated IDs retained'}
    if output.exists():
        result=read(output)
        if result['identity']!=identity:raise ValueError('Partial/complete readout identity changed')
        if result['status']=='complete':progress('already complete');return
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
    bench=long.load_module('continuation_long8k_builder',args.bench)
    progress('loading completed42k paired weights')
    h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=65536)
    ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();peft=h.module_manager.ensure_lora(bench.READER)
    provenance={'side_model':ac.side.model_config.model_id,'reader_model':ac.target.model_config.model_id,
        'side_revision':getattr(ac.side.model.config,'_commit_hash',None),'reader_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'side_config_sha256':digest(json.loads(ac.side.model.config.to_json_string())),
        'reader_config_sha256':digest(json.loads(ac.target.model.config.to_json_string()))}
    if provenance!=initial['model_provenance']:
        raise ValueError('Base model revisions/configs differ from the matched initial32k8k readout')
    if result.get('model_provenance',provenance)!=provenance:raise ValueError('Base model identity changed on readout resume')
    result['model_provenance']=provenance
    rows=material['held'];selected=set(selection['selected_ids'])
    progress('building exact initial32k8k inputs',questions=400)
    study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':rows,'retrieval':material['retrieval']},
        passage_parts=True,evidence_tokens=8192,max_passages=16,ratio=1/16)
    if [item.item_id for item in held]!=ids['held']:raise ValueError('Original400 order changed')
    for item,row in zip(held,rows):
        if any(unit['text'] not in item.info['passages_text'] for unit in row['example']['study_source']['units']):
            raise ValueError('Required source not retained whole: '+item.item_id)
    full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
    trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',no_context_penalty_weight=0,
        micro_batch_examples=2,max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED))
    row_counts=long.resolved_rows(trainer,ac,held)
    fingerprints={'shared_input_fingerprint':long.shared_fingerprint(held,full,noctx),
        'ac_input_fingerprint':digest([(item.item_id,item.ac_messages) for item in held]),
        'total_rows_fingerprint':digest([(row['item_id'],row['compressed_rows']) for row in row_counts])}
    indexes=[i for i,item in enumerate(held) if item.item_id in selected]
    cohort_of={ident:bank for bank,cohort in material['held_cohorts'].items() for ident in cohort}
    selected_rows=[{'item_id':held[i].item_id,'original_400_index':i,'generator_cohort':cohort_of[held[i].item_id],
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
    result.update(**fingerprints,input_rows=row_counts,selected_ids=selection['selected_ids'],scored_ids=ids['held'],
        required_sources_retained_whole=True,max_view_rows=ac.config.max_view_rows,input_binding='Exact completed initial32k8k400 inputs/rows and100 gold/evidence fields')
    generation=peft.generation_config;raw=generation.eos_token_id;eos=[raw] if isinstance(raw,int) else list(raw or [])
    eot=ac.target.model_config.model_description.eot_token;eot_id=ac.target.tokenizer.convert_tokens_to_ids(eot) if eot else None
    if eot_id is None:raise ValueError('Expected Qwen chat-end token')
    if eot_id not in eos:eos.append(eot_id)
    gen={'inherited_generation_config':json.loads(generation.to_json_string()),'explicit_eos_ids':eos,'eot':eot,'eot_id':eot_id,
        'max_new_tokens':160,'thinking':False,'do_sample':False,'model_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'tokenizer':ac.target.tokenizer.name_or_path}
    if result.get('generation_provenance',gen)!=gen:raise ValueError('Generation settings changed during partial evaluation')
    result['generation_provenance']=gen
    variables=trainer._autotuning_variables(ac);trainer._training_deadline=time.monotonic()+args.seconds
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    result['runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(ac.device),
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'bf16_reduction_split_k':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k,
        'autotuning':variables['provenance']}
    if not result['runtime']['bf16_reduced_precision_reduction']:
        raise ValueError('Experimental BF16 reduction policy is not authorized for this readout')
    sample_progress=ActivationContextTrainingProgress(**read(args.reports/ARM/'panel_01250.json')['progress'])
    write(output,result)
    try:
        with configure_fla_runtime(variables['fla_config']),torch.no_grad():
            for name,items in {'AC':held,'full context':full,'no context':noctx}.items():
                if name in result['views']:continue
                progress('scoring '+name,questions=400)
                torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device);start=time.monotonic()
                summary=trainer._evaluate(ac,items,ac.device);scored=long.answer_scores(summary,items,held)
                torch.cuda.synchronize(ac.device)
                for record,i in zip(result['rows'],indexes):
                    item=held[i];token_ids=item.assistant_token_ids[0]
                    record['views'][name]={'answer_nll':scored['per_item'][i]['answer_nll'],
                        'gold_token_scores':[{'token_id':token_ids[p],'text':study.tokenizer.decode([token_ids[p]],skip_special_tokens=False),
                            'nll':-summary['per_item'][i]['target_logp'][p]} for p in item.info['answer_positions']]}
                result['views'][name]={**scored,'cohorts':long.cohort_summary(scored,material['held_cohorts']),
                    'selected100_answer_nll':statistics.mean(record['views'][name]['answer_nll'] for record in result['rows']),
                    'selected50_by_generator':{bank:statistics.mean(record['views'][name]['answer_nll'] for record in result['rows'] if record['generator_cohort']==bank) for bank in material['held_cohorts']},
                    'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated(ac.device)/2**30,
                    'scoring_complete':True,'complete':name!='AC','panel_replay':None}
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
    except _TrainingTimeLimit:
        result['status']='partial';write(output,result);progress('time limit; resume same42k8k cell');return
    if trainer.optimizers or trainer.updates_done:raise AssertionError('Readout unexpectedly optimized weights')
    if long.payload_hashes(pair)!=hashes:raise ValueError('Final pair changed during evaluation')
    result.update(status='complete',optimizer_updates=0,parent_payloads_rehashed_unchanged=True,
        attempt_elapsed_seconds=round(time.monotonic()-began,1))
    write(output,result);progress('complete',questions=400,selected=100)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('design','material','bench','selection','reports','continuation-design','continuation-initial-pair','native-readout','initial8k-readout','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=2100)
    main(parser.parse_args())
