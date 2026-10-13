"""Read-only same-L40S native2k and separate-passage32k comparison; one model load."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import time

import evaluate_continuation_merged_l40s_v1 as merged
import evaluate_continuation_long_32k_v1 as pieces

long,continuation=merged.long,merged.continuation
read,write,sha,digest=merged.read,merged.write,merged.sha,merged.digest
MERGED_SHA='aaf07967a720e1d2efefc6f3f5b5caf3eb245d99c5bad25efeb9d18997966744'
PIECES_SHA='1714f76c71d709a216b671f490b54062879225ae46cf30849d3e15547b62977c'
RUNTIME_KEYS=('torch','cuda','device','tf32_matmul','tf32_cudnn','bf16_reduced_precision_reduction','bf16_reduction_split_k')


def bind_native_inputs(fingerprints,rows,native):
    for key in ('shared_input_fingerprint','ac_input_fingerprint'):
        if fingerprints[key]!=native[key]:raise ValueError('Original native400 input fingerprint changed: '+key)
    if len(rows)!=100:raise ValueError('Expected fixed100 native records')
    for current,prior in zip(rows,native['rows']):
        if any(current[k]!=prior[k] for k in merged.ROW_FIELDS):
            raise ValueError('Native source/gold/evidence changed: '+current['item_id'])
        if current['row_budget']['compressed_rows']!=prior['row_budget']['compressed_rows']:
            raise ValueError('Native actual row count changed')


def bind_l40s_controls(record,cell,pair,hashes,provenance,runtime,variables,max_vocab):
    """Reusing controls is permitted only within the exact L40S execution policy."""
    identity=record['identity']
    required={'condition':merged.ARM,'context':'32k','update':1250,'passage_parts':False,
        'encode_batch_max_tokens':65536,'ratio':1/16,'evidence_tokens':32768,'max_passages':64,
        'micro_batch_examples':1,'evaluator_sha256':MERGED_SHA,**cell['source_identity']}
    if (not record.get('core_complete') or not record.get('parent_payloads_rehashed_unchanged')
            or record.get('optimizer_updates')!=0 or any(identity.get(k)!=v for k,v in required.items())
            or identity['paired_checkpoint']!=pair or identity['checkpoint_sha256']!=hashes
            or record['model_provenance']!=provenance):raise ValueError('L40S control receipt incomplete or wrong final pair')
    if record.get('shared_input_fingerprint')!=cell['shared_input_fingerprint'] or record.get('total_rows_fingerprint')!=cell['total_rows_fingerprint']:
        raise ValueError('L40S control evidence/targets/total rows changed')
    if any(record['runtime'].get(k)!=runtime.get(k) for k in RUNTIME_KEYS) or 'L40S' not in runtime['device']:
        raise ValueError('Control hardware/precision/software differs; no cross-device score reuse')
    old=record['runtime']['autotuning'];new=variables['provenance']
    if any(old.get(k)!=new.get(k) for k in ('fla','checkpoint','logits')):
        raise ValueError('Control autotuning policy differs')
    allowance=min(2**31,old['memory_budget_bytes']//16)
    raw=max(1,min(2048,allowance//(max_vocab*16)));old_chunk=2**(raw.bit_length()-1)
    if variables['logits_chunk_tokens']!=old_chunk:raise ValueError('Control resolved logits chunk differs')
    if len(record['rows'])!=100:raise ValueError('Incomplete controls')
    for a,b in zip(cell['rows'],record['rows']):
        if any(a[k]!=b[k] for k in merged.ROW_FIELDS):raise ValueError('Control fixed100 record changed')
        for name in ('full context','no context'):
            if not record['views'][name].get('complete') or not record['views'][name].get('fresh_measurement'):
                raise ValueError('Controls must be actual complete L40S NLL measurements')
            values=b['views'][name]
            if len(values['gold_token_scores'])!=len(a['answer_positions']):raise ValueError('Control token positions changed')
            tokens=a['assistant_token_ids'][0]
            if any(v['token_id']!=tokens[p] for v,p in zip(values['gold_token_scores'],a['answer_positions'])):
                raise ValueError('Control gold tokens changed')
    return {'same_device_policy':{k:runtime[k] for k in RUNTIME_KEYS},'resolved_logits_chunk_tokens':old_chunk,
        'shared_inputs_and_gold_tokens_exact':True,'earlier_RTX_scores_reused':False}


def main(args):
    began=time.monotonic();folder=args.output;output=folder/'result.json'
    def progress(phase,**extra):
        value={'phase':phase,'attempt_elapsed_seconds':round(time.monotonic()-began,1),**extra}
        write(folder/'status.json',value);print(json.dumps(value),flush=True)
    progress('validating frozen parent and inputs')
    if sha(merged.__file__)!=MERGED_SHA or sha(pieces.__file__)!=PIECES_SHA or sha(long.__file__)!=merged.LONG_SHA or sha(continuation.__file__)!=merged.CONTINUATION_SHA:
        raise ValueError('Pinned validation/helper changed')
    design=read(args.design);material,selection,ids=long.load_material(args,design)
    if design['encode_batch_max_tokens']!=65536:raise ValueError('Only original packing authorized')
    cd=continuation.validate_design(args.continuation_design)
    if sha(args.continuation_initial_pair)!=cd['initial_pair_sha256']:raise ValueError('Continuation initial pair changed')
    pair,hashes=continuation.final_pair(merged.ARM,args.reports,cd,args.continuation_design,read(args.continuation_initial_pair))
    final=read(args.reports/merged.ARM/'result.json')
    if any(final.get(k)!=v for k,v in {'items':10000,'examples':10000,'planned_epochs':1,'planned_updates':1250}.items()):raise ValueError('Not own complete42k final')
    if sha(args.native_readout)!=merged.NATIVE_SHA or sha(args.piecewise_readout)!=merged.REFERENCES['32k']['sha256'] or sha(args.adam_receipt)!=merged.ADAM_RECEIPT_SHA:
        raise ValueError('Pinned native/passage/Adam provenance changed')
    native,reference=read(args.native_readout),read(args.piecewise_readout)
    merged.bind_readouts(native,reference,pair,hashes,selection,design,'32k')
    merged.bind_adam_receipt(read(args.adam_receipt),pair,hashes)
    control=read(args.l40s_merged_readout);control_sha=sha(args.l40s_merged_readout)
    identity={'condition':merged.ARM,'lifetime_unique_questions':42000,'update':1250,'paired_checkpoint':pair,
        'checkpoint_sha256':hashes,'evaluator_sha256':sha(__file__),'merged_helper_sha256':MERGED_SHA,'pieces_helper_sha256':PIECES_SHA,
        'design_sha256':sha(args.design),'material_sha256':design['material_sha256'],'selection_sha256':sha(args.selection),
        'source_sha256':design['source_sha256'],'builder_sha256':design['builder_sha256'],'packing_tokens':65536,
        'native_reference_sha256':merged.NATIVE_SHA,'piecewise_reference_sha256':merged.REFERENCES['32k']['sha256'],
        'l40s_merged_readout_sha256':control_sha,'decoding':'Optional separate-passage32k AC100 only; greedy160, no thinking, EOS union existing chat EOT',
        'scope':'Native2k uses original400/batch2 scoring then fixed100 extraction; passagewise32k scores original fixed100/batch1. All newly scored AC values are L40S. Reused32k controls must pass exact same-L40S/input checks.'}
    result=read(output) if output.exists() else {'status':'running','identity':identity,'cells':{}}
    if result['identity']!=identity:raise ValueError('Partial evaluation identity changed')
    if result.get('status')=='complete' and result.get('optional_decode_complete'):progress('already complete');return
    import torch
    from activation.ac_model import ActivationContextTrainer,ActivationContextTrainingConfig
    from activation.ac_model.ac_model_training import ActivationContextTrainingProgress,_TrainingTimeLimit
    from activation.autotuners import configure_fla_runtime
    import activation.ac_model.ac_model as am
    import activation.ac_model.ac_model_training as tm
    import activation.ac_model.ac_model_utils as um
    if {k:sha(m.__file__) for k,m in [('ac_model',am),('trainer',tm),('utils',um)]}!=design['source_sha256']:raise ValueError('Frozen source changed')
    bench=long.load_module('l40s_layout_builder',args.bench)
    progress('loading the paired models once')
    h=bench.harness(pair['ac_checkpoint'],pair['reader_checkpoint'],encode_batch_max_tokens=65536)
    ac=h.module_manager.get_ac_model(bench.AC);ac.prepare();peft=h.module_manager.ensure_lora(bench.READER)
    provenance={'side_model':ac.side.model_config.model_id,'reader_model':ac.target.model_config.model_id,
        'side_revision':getattr(ac.side.model.config,'_commit_hash',None),'reader_revision':getattr(ac.target.model.config,'_commit_hash',None),
        'side_config_sha256':digest(json.loads(ac.side.model.config.to_json_string())),
        'reader_config_sha256':digest(json.loads(ac.target.model.config.to_json_string()))}
    if provenance!=reference['model_provenance']:raise ValueError('Base model revision/config changed')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    runtime={'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(ac.device),
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'bf16_reduction_split_k':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction_split_k}
    if 'L40S' not in runtime['device'] or not runtime['bf16_reduced_precision_reduction']:raise ValueError('Only unchanged L40S precision authorized')
    if result.get('runtime',runtime)!=runtime:raise ValueError('Partial hardware/precision changed')
    result.update(runtime=runtime,model_provenance=provenance)
    selected=set(selection['selected_ids']);cohort_of={i:bank for bank,group in material['held_cohorts'].items() for i in group}
    contexts={};trainers=[]
    try:
        for cell_name,tokens,parts,micro in [('native2k',2048,False,2),('passagewise32k',32768,True,1)]:
            rows=material['held'] if not parts else [r for r in material['held'] if r['example']['example_id'] in selected]
            progress('building '+cell_name,scored_questions=len(rows))
            study,_,held=bench.material_items(h,{'train':material['train'][:1],'held':rows,'retrieval':material['retrieval']},
                passage_parts=parts,evidence_tokens=tokens,max_passages=64 if parts else 4,ratio=1/16)
            expected=selection['selected_ids'] if parts else selection['original_held_ids']
            if [x.item_id for x in held]!=expected:raise ValueError('Original question order changed')
            for item,row in zip(held,rows):
                if any(u['text'] not in item.info['passages_text'] for u in row['example']['study_source']['units']):raise ValueError('Required source truncated')
            full=study.transform_for_eval_reference(held,'full_context');noctx=study.transform_for_eval_reference(held,'no_context')
            trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',no_context_penalty_weight=0,micro_batch_examples=micro,max_example_tokens=72000,completion_samples=0,completion_max_new_tokens=160,seed=bench.SEED));trainers.append(trainer)
            row_counts=long.resolved_rows(trainer,ac,held);indices=[i for i,x in enumerate(held) if x.item_id in selected]
            fingerprints={'shared_input_fingerprint':long.shared_fingerprint(held,full,noctx),'ac_input_fingerprint':digest([(x.item_id,x.ac_messages) for x in held]),'total_rows_fingerprint':digest([(x['item_id'],x['compressed_rows']) for x in row_counts])}
            built=[{'item_id':held[i].item_id,'original_400_index':ids['held'].index(held[i].item_id),'generator_cohort':cohort_of[held[i].item_id],
                'source_family':rows[i]['source_family'],'domain':rows[i]['domain'],'question':held[i].info['question'],'gold':held[i].info['answer'],
                'required_source':rows[i]['example']['study_source'],'full_evidence':held[i].info['passages_text'],'in_context_messages':held[i].in_context_messages,
                'answer_positions':held[i].info['answer_positions'],'assistant_token_ids':held[i].assistant_token_ids,'row_budget':row_counts[i],'views':{}} for i in indices]
            if parts:pieces.bind_inputs(fingerprints,row_counts,built,reference)
            else:bind_native_inputs(fingerprints,built,native)
            cell=result['cells'].setdefault(cell_name,{'views':{},'rows':built})
            if [{k:v for k,v in r.items() if k!='views'} for r in cell['rows']]!=[{k:v for k,v in r.items() if k!='views'} for r in built]:raise ValueError('Partial source/gold/row records changed')
            cell.update(**fingerprints,input_rows=row_counts,selected_ids=selection['selected_ids'],scored_ids=expected,
                evidence_tokens=tokens,max_passages=64 if parts else 4,passage_parts=parts,micro_batch_examples=micro,required_sources_retained_whole=True,
                source_identity={key:identity[key] for key in ('design_sha256','material_sha256','selection_sha256','source_sha256','builder_sha256')})
            # Release unused allocator cache from the preceding cell before resolving
            # the ordinary device-specific logits scratch size; model weights stay loaded.
            torch.cuda.empty_cache()
            variables=trainer._autotuning_variables(ac);trainer._training_deadline=began+args.seconds
            cell['autotuning']={**variables['provenance'],'resolved_logits_chunk_tokens':variables['logits_chunk_tokens']}
            if parts:
                max_vocab=max(ac.side.model.config.get_text_config().vocab_size,ac.target.model.config.get_text_config().vocab_size)
                proof=bind_l40s_controls(control,cell,pair,hashes,provenance,runtime,variables,max_vocab)
                cell['same_device_control_reuse']={**proof,'source_path':str(args.l40s_merged_readout),'source_sha256':control_sha}
                for name in ('full context','no context'):
                    cell['views'][name]={**deepcopy(control['views'][name]),'fresh_measurement':False,'reused_measurement':True,'source_sha256':control_sha,'source_cell':'merged32k on same L40S runtime; identical full/no-context inputs'}
                    for row,prior in zip(cell['rows'],control['rows']):row['views'][name]=deepcopy(prior['views'][name])
            with configure_fla_runtime(variables['fla_config']),torch.no_grad():
                for name,items in {'AC':held,**({} if parts else {'full context':full,'no context':noctx})}.items():
                    if cell['views'].get(name,{}).get('scoring_complete'):continue
                    progress('scoring '+cell_name+' '+name,scored_questions=len(items),reported_subset=100)
                    torch.cuda.synchronize(ac.device);torch.cuda.reset_peak_memory_stats(ac.device);start=time.monotonic()
                    summary=trainer._evaluate(ac,items,ac.device);scored=long.answer_scores(summary,items,held);torch.cuda.synchronize(ac.device)
                    for row,i in zip(cell['rows'],indices):
                        item=held[i];t=item.assistant_token_ids[0]
                        row['views'][name]={'answer_nll':scored['per_item'][i]['answer_nll'],'gold_token_scores':[{'token_id':t[p],'text':study.tokenizer.decode([t[p]],skip_special_tokens=False),'nll':-summary['per_item'][i]['target_logp'][p]} for p in item.info['answer_positions']]}
                    subset=[scored['per_item'][i] for i in indices]
                    cell['views'][name]={'answer_nll':statistics.mean(x['answer_nll'] for x in subset),'per_item':subset,
                        'selected100_answer_nll':statistics.mean(x['answer_nll'] for x in subset),'scored_question_count':len(items),
                        'selected50_by_generator':{bank:statistics.mean(r['views'][name]['answer_nll'] for r in cell['rows'] if r['generator_cohort']==bank) for bank in material['held_cohorts']},
                        'all_scored_answer_nll':scored['answer_nll'],'all_scored_per_item':scored['per_item'],'denominator':'fixed100 development subset;400 batching retained only for native2k',
                        'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated(ac.device)/2**30,'fresh_measurement':True,'scoring_complete':True,'complete':True}
                    write(output,result)
            contexts[cell_name]=(study,trainer,held,indices,variables)
            cell['nll_complete']=True;write(output,result)
        if long.payload_hashes(pair)!=hashes:raise ValueError('Parent weights changed during NLL scoring')
        result.update(nll_complete=True,parent_payloads_rehashed_unchanged=True,optimizer_updates=0)
        write(output,result);progress('Both NLL cells complete; optional passagewise100 answers next')
        if args.decode:
            study,trainer,held,indices,variables=contexts['passagewise32k'];cell=result['cells']['passagewise32k']
            config=peft.generation_config;raw=config.eos_token_id;eos=[raw] if isinstance(raw,int) else list(raw or [])
            eot=ac.target.model_config.model_description.eot_token;eot_id=ac.target.tokenizer.convert_tokens_to_ids(eot) if eot else None
            if eot_id is None:raise ValueError('Missing model chat EOT')
            if eot_id not in eos:eos.append(eot_id)
            cell['generation_provenance']={'inherited_generation_config':json.loads(config.to_json_string()),'explicit_eos_ids':eos,'eot':eot,'eot_id':eot_id,'max_new_tokens':160,'thinking':False,'do_sample':False}
            progress_value=long.read(args.reports/merged.ARM/'panel_01250.json')['progress'];sample_progress=ActivationContextTrainingProgress(**progress_value)
            with configure_fla_runtime(variables['fla_config']),torch.no_grad():
                for row,i in zip(cell['rows'],indices):
                    if 'response' in row['views']['AC']:continue
                    progress('optional passagewise32k AC answers',completed=sum('response' in r['views']['AC'] for r in cell['rows']),total=100)
                    sample,generated=long.sample_with_token_ids(trainer,ac,held[i],sample_progress,eos)
                    text=sample['student'];answer,parsing=long.parsed_answer(study,text);end=text.find('</tool_call>')
                    row['views']['AC'].update(response=text,answer=answer,parsing=parsing,raw_sample=sample,**generated,post_first_wrapper_text=text[end+len('</tool_call>'):] if end>=0 else None)
                    write(output,result)
            result['optional_decode_complete']=True
        else:result['optional_decode_complete']=False
    except _TrainingTimeLimit:
        result['time_limit_reached']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        if any(t.optimizers or t.updates_done for t in trainers):raise AssertionError('Readout unexpectedly optimized weights')
        if long.payload_hashes(pair)!=hashes:raise ValueError('Parent weights changed during evaluation')
        result.update(status='failed' if result.get('error') else ('complete' if result.get('nll_complete') else 'partial'),optimizer_updates=0,parent_payloads_rehashed_unchanged=True,attempt_elapsed_seconds=round(time.monotonic()-began,1))
        write(output,result);progress(result['status'],nll_complete=result.get('nll_complete',False),optional_decode_complete=result.get('optional_decode_complete',False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('design','material','bench','selection','reports','continuation-design','continuation-initial-pair','native-readout','piecewise-readout','adam-receipt','l40s-merged-readout','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=1400);p.add_argument('--decode',action='store_true')
    main(p.parse_args())
