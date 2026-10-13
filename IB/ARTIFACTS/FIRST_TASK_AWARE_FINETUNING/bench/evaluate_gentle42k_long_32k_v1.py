"""Completed gentle42k familiar4000 final at32k on exactly its constant42k parent inputs; conditional on own8k100 completion."""
import argparse
import json
from pathlib import Path
import statistics
import time

import evaluate_strong_long_cohort_v1 as long
import evaluate_gentle42k_long_v1 as gentle
import evaluate_continuation_long_32k_v1 as matched

ARM = 'gentle42k_final_lr_familiar4k_8k'
LONG_SHA = 'a49dec5dc2aab1791b3e2afd746d4f6127ff70a00aa3385c6bedba348578e585'
GENTLE_SHA = '521dd03df081447c7d6fa6f3e8afff9699b2e717d7ea611bd911ecd756a926de'
MATCHED_SHA = '1714f76c71d709a216b671f490b54062879225ae46cf30849d3e15547b62977c'
GENTLE_DESIGN_SHA = 'f7fef157964562fabc7e5fe5fd2a23f1ed24f10b6823612f612ef51f5ab4bb3f'
PARENT32K_SHA = 'a25798cdae7fc50ab84f921344c4491bc89ee9fa6859df6dd1fdf5758e8bc6cb'
read,write,sha,digest = long.read,long.write,long.file_sha,long.digest
FINGERPRINTS = ('shared_input_fingerprint','ac_input_fingerprint','total_rows_fingerprint')
ROW_FIELDS = ('item_id','original_400_index','generator_cohort','source_family','domain','question','gold',
    'required_source','full_evidence','in_context_messages','answer_positions','assistant_token_ids','row_budget')


def bind_readouts(native, initial, pair, hashes, publication, selection, design, gentle_design):
    """Own8k completion gates eligibility; constant42k32k binds inputs, never NLL."""
    expected_native={'condition':ARM,'context':'8k','update':500,'paired_checkpoint':pair,
        'checkpoint_sha256':hashes,'publication_sha256':publication,'design_sha256':GENTLE_DESIGN_SHA,
        'selection_sha256':design['readout']['selection_sha256'],'evaluator_sha256':GENTLE_SHA,
        'source_sha256':design['source_sha256'],'builder_sha256':gentle_design['builder_sha256'],
        'passage_parts':True,'encode_batch_max_tokens':65536,'ratio':1/16,
        'evidence_tokens':8192,'max_passages':16,'micro_batch_examples':2}
    if (native.get('status')!='complete' or any(native['identity'].get(key)!=value for key,value in expected_native.items())
            or native.get('optimizer_updates')!=0 or native.get('scored_ids')!=selection['original_held_ids']
            or native.get('selected_ids')!=selection['selected_ids'] or len(native.get('rows',[]))!=100):
        raise ValueError('Requires own completed gentle8k400/100 readout from the published500/Adam5750 final')
    for name in ('AC','full context','no context'):
        view=native['views'].get(name,{})
        if (not view.get('complete') or not (view.get('panel_replay') or {}).get('passed')
                or [row['item_id'] for row in view.get('per_item',[])]!=selection['original_held_ids']):
            raise ValueError('Own8k400 final replay missing or incomplete: '+name)
    if ([row['item_id'] for row in native['rows']]!=selection['selected_ids']
            or any(not {'response','generated_token_ids','generated_tokens','effective_eos_token_ids'}<=row['views']['AC'].keys() for row in native['rows'])):
        raise ValueError('Own8k100 actual answers missing or reordered')
    expected={'condition':'continue32k_10k_final_lr','context':'32k','update':1250,'passage_parts':True,
        'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':32768,'max_passages':64,'micro_batch_examples':1,
        'source_sha256':design['source_sha256'],'builder_sha256':design['builder_sha256'],
        'checkpoint_sha256':gentle_design['initial_payload_sha256'],'evaluator_sha256':MATCHED_SHA}
    if (initial.get('status')!='complete' or any(initial['identity'].get(k)!=v for k,v in expected.items())
            or initial.get('optimizer_updates')!=0 or initial.get('scored_ids')!=selection['selected_ids']
            or initial.get('selected_ids')!=selection['selected_ids'] or len(initial['input_rows'])!=100
            or len(initial['rows'])!=100 or not all(initial['views'][view].get('complete') for view in ('AC','full context','no context'))
            or [row['item_id'] for row in initial['rows']]!=selection['selected_ids']):
        raise ValueError('Expected completed constant42k32k reference with exact100 inputs')
    # Context/layout changes evidence; the selected questions, sources and gold stay fixed.
    for current,prior in zip(native['rows'],initial['rows']):
        if any(current[field]!=prior[field] for field in ROW_FIELDS if field not in ('full_evidence','in_context_messages','row_budget')):
            raise ValueError('Own8k/parent32k selected question, source or gold identity differs: '+current['item_id'])


bind_inputs=matched.bind_inputs


def main(args):
    began=time.monotonic();cell=ARM+'__32k';folder=args.output/cell;output=folder/'result.json'
    def progress(phase,**extra):
        value={'cell':cell,'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-began,1),**extra}
        write(folder/'status.json',value);print(json.dumps(value),flush=True)
    progress('validating completed gentle pair and frozen parent32k reference')
    if sha(long.__file__)!=LONG_SHA or sha(gentle.__file__)!=GENTLE_SHA or sha(matched.__file__)!=MATCHED_SHA:
        raise ValueError('Pinned long-input or continuation final-guard helper changed')
    design=read(args.design)
    if design['encode_batch_max_tokens']!=65536:raise ValueError('Only the original65536 packing is authorized')
    material,selection,ids=long.load_material(args,design)
    if sha(args.gentle_design)!=GENTLE_DESIGN_SHA:raise ValueError('Frozen gentle adaptation design changed')
    gentle_design=read(args.gentle_design)
    if (gentle_design.get('status')!='frozen' or gentle_design['run_name']!=ARM or gentle_design['questions']!=4000
            or gentle_design['epochs']!=1 or gentle_design['source_sha256']!=design['source_sha256']):
        raise ValueError('Expected frozen4000-familiar adaptation design')
    for leaf,expected in gentle_design['helper_sha256'].items():
        if Path(leaf).name!=leaf or sha(Path(__file__).with_name(leaf))!=expected:raise ValueError('Gentle final-guard dependency changed: '+leaf)
    native=read(args.own8k_readout)
    if native.get('status')!='complete':raise ValueError('Wait for completed own gentle8k100 readout')
    own_ids=read(args.reports/ARM/'material_ids.json')
    if (len(own_ids['train'])!=4000 or len(set(own_ids['train']))!=4000
            or own_ids['held']!=ids['held'] or digest(own_ids)!=gentle_design['material_ids_sha256']):
        raise ValueError('Gentle training membership or shared400 order differs')
    pair,publication=gentle.paired_final(ARM,args.reports,args.runs_root,gentle_design,own_ids)
    if read(args.reports/ARM/'source_receipt.json')['design_sha256']!=GENTLE_DESIGN_SHA:
        raise ValueError('Training used a different gentle frozen design')
    hashes=long.payload_hashes(pair)
    if sha(args.parent32k_readout)!=PARENT32K_SHA:raise ValueError('Completed constant42k32k reference bytes changed')
    initial=read(args.parent32k_readout)
    bind_readouts(native,initial,pair,hashes,publication,selection,design,gentle_design)
    if initial['identity']['design_sha256']!=sha(args.design) or initial['identity']['selection_sha256']!=sha(args.selection):
        raise ValueError('Parent32k input design/selection publication differs')
    identity={'condition':ARM,'context':'32k','update':500,'lifetime_questions':42000,'parent_questions':42000,'stage_questions':4000,'stage_questions_previously_seen':True,'total_presentations':46000,
        'paired_checkpoint':pair,'checkpoint_sha256':hashes,'publication_sha256':publication,'design_sha256':sha(args.design),
        'gentle_design_sha256':GENTLE_DESIGN_SHA,'selection_sha256':sha(args.selection),
        'material_sha256':design['material_sha256'],'builder_sha256':design['builder_sha256'],'source_sha256':design['source_sha256'],
        'evaluator_sha256':sha(__file__),'long_helper_sha256':LONG_SHA,'gentle_helper_sha256':GENTLE_SHA,'matched_helper_sha256':MATCHED_SHA,
        'own8k_readout_sha256':sha(args.own8k_readout),'parent42k_32k_readout_sha256':PARENT32K_SHA,
        'passage_parts':True,'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':32768,'max_passages':64,'micro_batch_examples':1,
        'reader':'Completed gentle4000-familiar adaptation paired reader, including its own full/no-context controls',
        'scope':'One32k development cell: same100 scored and AC decoded after familiar4000 context adaptation.42k unique/46k presentations. Added retrieved context, not distributed required evidence; no confirmation or training.',
        'nll_replay':'Exact32k inputs bind the completed constant42k32k readout. Weights differ; scores are new measurements, not replay targets.',
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
    if source!=design['source_sha256'] or source!=gentle_design['source_sha256']:
        raise ValueError('Runtime source differs from both frozen studies')
    bench=long.load_module('gentle42k_long32k_builder',args.bench)
    progress('loading completed gentle paired weights')
    h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=65536)
    ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();peft=h.module_manager.ensure_lora(bench.READER)
    provenance={'side_model':ac.side.model_config.model_id,'reader_model':ac.target.model_config.model_id,
        'side_revision':getattr(ac.side.model.config,'_commit_hash',None),'reader_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'side_config_sha256':digest(json.loads(ac.side.model.config.to_json_string())),
        'reader_config_sha256':digest(json.loads(ac.target.model.config.to_json_string()))}
    if provenance!=initial['model_provenance']:
        raise ValueError('Base model revisions/configs differ from the matched constant42k32k readout')
    if result.get('model_provenance',provenance)!=provenance:raise ValueError('Base model identity changed on readout resume')
    result['model_provenance']=provenance
    selected=set(selection['selected_ids']);rows=[row for row in material['held'] if row['example']['example_id'] in selected]
    cohorts={bank:[ident for ident in group if ident in selected] for bank,group in material['held_cohorts'].items()}
    progress('building exact constant42k32k inputs',questions=100)
    study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':rows,'retrieval':material['retrieval']},
        passage_parts=True,evidence_tokens=32768,max_passages=64,ratio=1/16)
    if [item.item_id for item in held]!=selection['selected_ids']:raise ValueError('Selected100 original order changed')
    for item,row in zip(held,rows):
        if any(unit['text'] not in item.info['passages_text'] for unit in row['example']['study_source']['units']):
            raise ValueError('Required source not retained whole: '+item.item_id)
    full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
    trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',no_context_penalty_weight=0,
        micro_batch_examples=1,max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED))
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
    result.update(**fingerprints,input_rows=row_counts,selected_ids=selection['selected_ids'],scored_ids=selection['selected_ids'],
        required_sources_retained_whole=True,max_view_rows=ac.config.max_view_rows,input_binding='Exact completed constant42k32k100 inputs/rows and gold/evidence fields')
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
    if not result['runtime']['bf16_reduced_precision_reduction']:
        raise ValueError('Experimental BF16 reduction policy is not authorized for this readout')
    sample_progress=ActivationContextTrainingProgress(**read(args.reports/ARM/'panel_00500.json')['progress'])
    write(output,result)
    try:
        with configure_fla_runtime(variables['fla_config']),torch.no_grad():
            for name,items in {'AC':held,'full context':full,'no context':noctx}.items():
                if name in result['views']:continue
                progress('scoring '+name,questions=100)
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
        result['status']='partial';write(output,result);progress('time limit; resume same gentle42k32k cell');return
    if trainer.optimizers or trainer.updates_done:raise AssertionError('Readout unexpectedly optimized weights')
    if long.payload_hashes(pair)!=hashes:raise ValueError('Final pair changed during evaluation')
    result.update(status='complete',optimizer_updates=0,parent_payloads_rehashed_unchanged=True,
        attempt_elapsed_seconds=round(time.monotonic()-began,1))
    write(output,result);progress('complete',questions=100,selected=100)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('design','material','bench','selection','reports','runs-root','gentle-design','own8k-readout','parent32k-readout','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=1650)
    main(parser.parse_args())
