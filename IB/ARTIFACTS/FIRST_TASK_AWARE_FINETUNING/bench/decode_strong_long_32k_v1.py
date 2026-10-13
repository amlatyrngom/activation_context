"""Decode fixed100 AC answers at32k; reuse only identity-matched completed final NLL."""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import evaluate_strong_long_cohort_v1 as base

BASE_SHA='a49dec5dc2aab1791b3e2afd746d4f6127ff70a00aa3385c6bedba348578e585'
CONDITION='strong32k_piecewise_8k_8k'
write,read,file_sha,digest=base.write,base.read,base.file_sha,base.digest


def validate_nll(nll,expected_identity,selected_ids):
    if nll.get('status')!='complete' or nll.get('optimizer_updates')!=0:
        raise ValueError('Only completed read-only NLL evidence can be reused')
    for key,value in expected_identity.items():
        if nll['identity'].get(key)!=value:raise ValueError('NLL identity differs: '+key)
    if (len(selected_ids)!=100 or len(set(selected_ids))!=100 or nll.get('selected_ids')!=selected_ids
            or nll.get('scored_ids')!=selected_ids or [row['item_id'] for row in nll['rows']]!=selected_ids):
        raise ValueError('NLL fixed100 membership/order changed')
    for name in ('AC','full context','no context'):
        view=nll['views'].get(name,{})
        if view.get('complete') is not True or [row['item_id'] for row in view.get('per_item',[])]!=selected_ids:
            raise ValueError('NLL view incomplete or reordered: '+name)
        for row,score in zip(nll['rows'],view['per_item']):
            value=row['views'][name]['answer_nll']
            if not math.isfinite(value) or value!=score['answer_nll']:
                raise ValueError('Per-question NLL evidence disagrees: '+row['item_id'])
    if nll.get('required_sources_retained_whole') is not True:raise ValueError('Required evidence was not verified')


def bind_inputs(nll,fingerprints,row_counts,held,rows,ids):
    for key,value in fingerprints.items():
        if nll.get(key)!=value:raise ValueError('Rebuilt32k inputs differ from NLL evidence: '+key)
    if nll.get('input_rows')!=row_counts:raise ValueError('Resolved row/token geometry changed')
    for saved,item,row,count in zip(nll['rows'],held,rows,row_counts):
        expected={'item_id':item.item_id,'original_400_index':ids['held'].index(item.item_id),
            'question':item.info['question'],'gold':item.info['answer'],
            'required_source':row['example']['study_source'],'full_evidence':item.info['passages_text'],
            'in_context_messages':item.in_context_messages,'answer_positions':item.info['answer_positions'],
            'assistant_token_ids':item.assistant_token_ids,'row_budget':count}
        if any(saved.get(key)!=value for key,value in expected.items()):
            raise ValueError('Saved NLL row/target/source differs: '+item.item_id)
        if any(unit['text'] not in item.info['passages_text'] for unit in row['example']['study_source']['units']):
            raise ValueError('Required source missing from rebuilt evidence')


def check_decoded(record):
    value=record['views']['AC']
    present='response' in value
    fields=('generated_token_ids','generated_tokens','max_new_tokens','effective_eos_token_ids',
            'ended_at_eos','reached_token_budget','raw_decode_with_special_tokens','raw_sample','answer','parsing')
    if present and any(key not in value for key in fields):raise ValueError('Incomplete saved decoded row')
    if not present and any(key in value for key in fields):raise ValueError('Half-published decoded row')
    if present:
        token_ids=value['generated_token_ids'];cap=value['max_new_tokens']
        if (not isinstance(cap,int) or not 0<cap<=160 or len(token_ids)!=value['generated_tokens'] or len(token_ids)>cap
                or value['ended_at_eos']!=bool(token_ids and token_ids[-1] in value['effective_eos_token_ids'])
                or value['reached_token_budget']!=(len(token_ids)==cap)
                or value['raw_sample'].get('student')!=value['response']):
            raise ValueError('Saved token/termination evidence inconsistent')
    return present


def main(args):
    started=time.monotonic();deadline=started+args.seconds
    output=args.output/'result.json';status=args.output/'status.json'
    if output.resolve()==args.nll_input.resolve():raise ValueError('Decode output must not overwrite NLL evidence')
    def progress(phase,**extra):
        value={'phase':phase,'condition':CONDITION,'context':'32k','elapsed_seconds':round(time.monotonic()-started,1),**extra}
        write(status,value);print(json.dumps(value),flush=True)
    progress('validating completed32k NLL evidence and final pair')
    if file_sha(base.__file__)!=BASE_SHA:raise ValueError('Frozen NLL evaluator changed')
    nll=read(args.nll_input)
    if nll.get('status')!='complete':raise ValueError('Wait for the completed32k NLL readout')
    nll_sha=file_sha(args.nll_input);design=read(args.design)
    material,selection,ids=base.load_material(args,design)
    pair,publication=base.paired_final(CONDITION,args.reports,args.runs_root,design,ids)
    hashes=base.payload_hashes(pair)
    expected={'condition':CONDITION,'context':'32k','update':1000,'paired_checkpoint':pair,
        'checkpoint_sha256':hashes,'publication_sha256':publication,'design_sha256':file_sha(args.design),
        'selection_sha256':file_sha(args.selection),'evaluator_sha256':BASE_SHA,
        'helper_sha256':{name:file_sha(Path(__file__).with_name(name)) for name in base.HELPERS},
        'source_sha256':design['source_sha256'],'builder_sha256':design['builder_sha256'],
        'passage_parts':True,'encode_batch_max_tokens':design['encode_batch_max_tokens'],'ratio':1/16,
        'evidence_tokens':32768,'max_passages':64,'micro_batch_examples':1}
    validate_nll(nll,expected,selection['selected_ids'])
    identity={**expected,'evaluator_sha256':file_sha(__file__),'nll_evidence_path':str(args.nll_input.resolve()),
        'nll_evidence_sha256':nll_sha,'nll_evaluator_sha256':BASE_SHA,
        'nll_identity':nll['identity'],'decoding':'greedy160, thinking disabled, inherited EOS union chat EOT; actual IDs retained',
        'scope':'Final paired AC100 at32k DEVELOPMENT evidence. No NLL rerun, optimizer or confirmation access.'}
    if output.exists():
        result=read(output)
        if result['identity']!=identity:raise ValueError('Partial answer readout identity changed')
        if [row['item_id'] for row in result['rows']]!=selection['selected_ids']:raise ValueError('Partial rows reordered')
        # Only generated fields may have been added; source scores stay immutable.
        if result['views']!=nll['views']:raise ValueError('Reused NLL aggregate evidence changed')
        for row,saved in zip(result['rows'],nll['rows']):
            for key in saved:
                if key!='views' and row.get(key)!=saved[key]:raise ValueError('Partial source row changed')
            for name,values in saved['views'].items():
                if any(row['views'][name].get(key)!=value for key,value in values.items()):raise ValueError('Reused per-token NLL changed')
        completed=sum(check_decoded(row) for row in result['rows'])
        if result['status']=='complete':
            if completed!=100:raise ValueError('Complete marker lacks100 decoded answers')
            progress('already complete',decoded=completed);return
    else:
        result=copy.deepcopy(nll);result.update(status='running',identity=identity,decoded_count=0,
            nll_recomputed=False,nll_evidence_sha256=nll_sha,optimizer_updates=0)
        if any(check_decoded(row) for row in result['rows']):raise ValueError('Expected original32k NLL-only evidence')
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress,_TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model as ac_module
    import activation.ac_model.ac_model_training as training_module
    import activation.ac_model.ac_model_utils as utils_module
    runtime_source={key:file_sha(module.__file__) for key,module in [('ac_model',ac_module),('trainer',training_module),('utils',utils_module)]}
    if runtime_source!=design['source_sha256']:raise ValueError('Runtime source changed')
    bench=base.load_module('strong_long_decode_builder',args.bench)
    progress('loading final paired weights')
    h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=design['encode_batch_max_tokens'])
    ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();peft=h.module_manager.ensure_lora(bench.READER)
    model_provenance={'side_model':ac.side.model_config.model_id,'reader_model':ac.target.model_config.model_id,
        'side_revision':getattr(ac.side.model.config,'_commit_hash',None),'reader_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'side_config_sha256':digest(json.loads(ac.side.model.config.to_json_string())),
        'reader_config_sha256':digest(json.loads(ac.target.model.config.to_json_string()))}
    if model_provenance!=nll['model_provenance']:raise ValueError('Base model/config differs from NLL evidence')
    selected=set(selection['selected_ids']);rows=[row for row in material['held'] if row['example']['example_id'] in selected]
    progress('rebuilding exact32k development inputs',questions=100)
    study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':rows,'retrieval':material['retrieval']},
        passage_parts=True,evidence_tokens=32768,max_passages=64,ratio=1/16)
    if [item.item_id for item in held]!=selection['selected_ids']:raise ValueError('Frozen100 order changed')
    full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
    trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',no_context_penalty_weight=0,
        micro_batch_examples=1,max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED))
    row_counts=base.resolved_rows(trainer,ac,held)
    fingerprints={'shared_input_fingerprint':base.shared_fingerprint(held,full,noctx),
        'ac_input_fingerprint':digest([(item.item_id,item.ac_messages) for item in held]),
        'total_rows_fingerprint':digest([(row['item_id'],row['compressed_rows']) for row in row_counts])}
    bind_inputs(nll,fingerprints,row_counts,held,rows,ids)
    generation=peft.generation_config;raw_eos=generation.eos_token_id
    eos=[raw_eos] if isinstance(raw_eos,int) else list(raw_eos or [])
    eot=ac.target.model_config.model_description.eot_token
    eot_id=ac.target.tokenizer.convert_tokens_to_ids(eot) if eot else None
    if eot_id is None:raise ValueError('Expected existing Qwen chat EOT')
    if eot_id not in eos:eos.append(eot_id)
    generation_provenance={'inherited_generation_config':json.loads(generation.to_json_string()),
        'explicit_eos_ids':eos,'eot':eot,'eot_id':eot_id,'max_new_tokens':160,'thinking':False,'do_sample':False,
        'model_revision':getattr(ac.target.model.config,'_commit_hash',None),'tokenizer':ac.target.tokenizer.name_or_path}
    if generation_provenance!=nll['generation_provenance']:raise ValueError('Inherited generation configuration changed')
    for row in result['rows']:
        if check_decoded(row):
            value=row['views']['AC'];token_ids=value['generated_token_ids']
            if (value['effective_eos_token_ids']!=eos or ac.target.tokenizer.decode(token_ids,skip_special_tokens=True)!=value['response']
                    or ac.target.tokenizer.decode(token_ids,skip_special_tokens=False)!=value['raw_decode_with_special_tokens']):
                raise ValueError('Saved response does not match token IDs or stopping policy')
    result.update(input_binding_verified=True,generation_provenance=generation_provenance,
        decode_runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(ac.device),
            'tf32':True,'note':'Gold NLL reused from pinned evidence; actual responses generated in this process.'})
    variables=trainer._autotuning_variables(ac);trainer._training_deadline=deadline
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    sample_progress=ActivationContextTrainingProgress(epoch=1,epochs=1,epoch_number=1,step=1000,steps=1000,
        global_step=1000,epoch_elapsed_s=0.,elapsed_s=0.)
    write(output,result)
    try:
        with configure_fla_runtime(variables['fla_config']),torch.no_grad():
            for row,item in zip(result['rows'],held):
                if check_decoded(row):continue
                if time.monotonic()>=deadline:raise _TrainingTimeLimit()
                progress('decoding AC answers',completed=sum(check_decoded(saved) for saved in result['rows']),total=100)
                sample,generated=base.sample_with_token_ids(trainer,ac,item,sample_progress,eos)
                text=sample['student'];answer,parsing=base.parsed_answer(study,text);end=text.find('</tool_call>')
                row['views']['AC'].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,
                    post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                check_decoded(row)
                result['decoded_count']=sum(check_decoded(saved) for saved in result['rows']);write(output,result)
    except _TrainingTimeLimit:
        result['status']='partial';write(output,result);progress('time limit; resume same inputs',decoded=result['decoded_count']);return
    if trainer.optimizers or trainer.updates_done:raise AssertionError('Answer readout unexpectedly optimized')
    if file_sha(args.nll_input)!=nll_sha or base.payload_hashes(pair)!=hashes:raise ValueError('Source evidence or parent weights changed')
    if not all(check_decoded(row) for row in result['rows']):raise AssertionError('Missing generated answer')
    result.update(status='complete',decoded_count=100,optimizer_updates=0,nll_recomputed=False,
        checkpoint_payloads_rehashed_unchanged=True,attempt_elapsed_seconds=round(time.monotonic()-started,1))
    write(output,result);progress('complete',decoded=100)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('design','material','bench','selection','reports','runs-root','nll-input','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=1500)
    args=parser.parse_args()
    try:main(args)
    except Exception as error:
        write(args.output/'status.json',{'phase':'failed','error':f'{type(error).__name__}: {error}'})
        raise
