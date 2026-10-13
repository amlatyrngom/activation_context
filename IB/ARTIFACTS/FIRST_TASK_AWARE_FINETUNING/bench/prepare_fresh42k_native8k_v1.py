"""Allocated-CPU-only fresh8k selection and independent completed42k/Adam5250 copy."""
import argparse,collections,hashlib,json,os,random
from pathlib import Path
from prepare_continuation_10k import normalized
from prepare_model_comparison import interleave,length_balanced_order,request_id,summarize
from select_bank import quotas,training_row
from prepare_gentle42k_long_v1 import derive_pair,PARENTS
import evaluate_continuation_10k_v1 as continuation
import evaluate_strong_long_cohort_v1 as long

ITEMS=8000
SEED=20260927
SOURCE_MATERIAL_SHA='a73de2273d600ab3693540310510e496736c554d1685312132885cc24b5973aa'
sha,digest,read,write=long.file_sha,long.digest,long.read,long.write


def accepted_rows(bank,manifest,receipt):
    """Published rows plus committed arXiv surplus, never republishing/mutating bank state."""
    for entry in manifest['batches']:
        path=bank/entry['file'];hasher=hashlib.sha256();count=0
        with path.open('rb') as handle:
            for line in handle:
                hasher.update(line)
                if not line.strip():continue
                count+=1;yield json.loads(line),entry['file'],False
        if count!=entry['questions'] or hasher.hexdigest()!=entry['sha256']:
            raise ValueError('Changed published batch: '+entry['file'])
        receipt.append({'file':entry['file'],'sha256':entry['sha256'],'questions_read':count,'surplus':False})
    # Producer restore() treats the published family count as the consumed prefix.
    skip=manifest['by_family']['math_arxiv']
    for entry in manifest['microchunks']:
        if entry['source_family']!='math_arxiv':continue
        if skip>=entry['questions']:skip-=entry['questions'];continue
        path=bank/entry['file'];hasher=hashlib.sha256();count=0;emitted=0
        with path.open('rb') as handle:
            for line in handle:
                hasher.update(line)
                if not line.strip():continue
                count+=1
                if skip:skip-=1;continue
                raw=json.loads(line)
                if raw.get('source_family',raw['domain'])!='math_arxiv':raise ValueError('ArXiv surplus family changed')
                emitted+=1;yield raw,entry['file'],True
        if count!=entry['questions'] or hasher.hexdigest()!=entry['sha256']:
            raise ValueError('Changed committed microchunk: '+entry['file'])
        receipt.append({'file':entry['file'],'sha256':entry['sha256'],'questions_read':count,'surplus_questions':emitted,'surplus':True})
    if skip:raise ValueError('Committed arXiv microchunks do not cover the published prefix')


def select_fresh(base,bank,continuation_ids,reservations,*,items=ITEMS):
    old_ids={row['example']['example_id'] for row in base['train']};continued=set(continuation_ids)
    if old_ids&continued or len(continued)!=len(continuation_ids):raise ValueError('Prior training stages overlap or duplicate IDs')
    all_ids=old_ids|continued;old_questions={normalized(row['example']['query']) for row in base['train']}
    protected={normalized(row['example']['query']) for row in base['held']}
    protected_requests={request_id(row) for row in base['held']};reservation_receipts=[]
    for path in reservations:
        data=read(path)
        if set(data)!={'questions'}:raise ValueError('Only question-only reservation lists are accepted')
        protected.update(map(normalized,data['questions']))
        reservation_receipts.append({'path':str(path),'sha256':sha(path),'questions':len(data['questions'])})
    raw_manifest=(bank/'manifest.json').read_bytes();manifest=json.loads(raw_manifest);shares=manifest['source_shares']
    if shares!=base['recipe']['source_shares']:raise ValueError('Source-family mix changed')
    candidates=[];continued_found=set();continued_questions=set();seen_ids=set();seen_questions=set();inputs=[];excluded=collections.Counter()
    for raw,path,surplus in accepted_rows(bank,manifest,inputs):
        if 'example' in raw:
            ex=raw['example'];ident,query,answers=ex['example_id'],ex['query'],ex['gold_answers'];req=request_id(raw)
        else:
            ident,query,answers=raw['id'],raw['question'],[raw['answer']];req=raw.get('provenance',{}).get('source_request_id',ident)
        key=normalized(query)
        if ident in continued:continued_found.add(ident);continued_questions.add(key)
        if key in protected:protected_requests.add(req);excluded['reserved_question']+=1;continue
        if ident in all_ids or key in old_questions:excluded['prior_training']+=1;continue
        if ident in seen_ids or key in seen_questions:excluded['duplicate_question']+=1;continue
        if any(ord(char)<32 and char not in '\n\t' for text in [query,*answers] for char in text):
            excluded['corrupted_qa']+=1;continue
        seen_ids.add(ident);seen_questions.add(key)
        candidates.append((training_row(raw),key,req,path,surplus))
    if continued_found!=continued or len(continued_questions)!=len(continued):raise ValueError('Could not identify all prior10k distinct questions in the immutable bank')
    if len(old_questions)!=len(old_ids) or old_questions&continued_questions:
        raise ValueError('Prior stages do not establish42000 distinct normalized questions')
    eligible=collections.defaultdict(list);provenance={};by_storage=collections.Counter()
    for row,key,req,path,surplus in candidates:
        if key in continued_questions:excluded['prior_training_text']+=1;continue
        if req in protected_requests:excluded['reserved_sibling']+=1;continue
        eligible[row['source_family']].append(row);by_storage['surplus' if surplus else 'published']+=1
        provenance[row['example']['example_id']]={'file':path,'surplus':surplus}
    wanted=quotas(items,shares);counts={family:len(eligible[family]) for family in shares}
    if any(counts[family]<count for family,count in wanted.items()):
        raise ValueError('Insufficient fresh balanced questions: '+json.dumps({'needed':wanted,'available':counts}))
    selected={family:length_balanced_order(eligible[family],f'27b:fresh42k8k:{family}')[:count] for family,count in wanted.items()}
    train=interleave(selected);ids=[row['example']['example_id'] for row in train];questions={normalized(row['example']['query']) for row in train}
    if len(ids)!=len(set(ids)) or len(questions)!=items or set(ids)&all_ids or questions&(old_questions|continued_questions|protected):
        raise ValueError('Fresh selection repeats a prior/protected question')
    if {request_id(row) for row in train}&protected_requests:raise ValueError('Reserved source request entered fresh training')
    learning=interleave({family:random.Random(f'{SEED}:{family}:learning').sample(selected[family],n) for family,n in quotas(min(32,items),shares).items()})
    recipe={'bank_manifest_sha256':hashlib.sha256(raw_manifest).hexdigest(),'published_questions':manifest['total_questions'],
        'accepted_questions':manifest['accepted_questions'],'input_files':inputs,'source_shares':shares,
        'eligible_by_family':counts,'eligible_by_storage':dict(by_storage),'required_by_family':wanted,'excluded':dict(excluded),
        'protected_question_count':len(protected),'protected_request_count':len(protected_requests),'reservation_files':reservation_receipts,
        'selected_by_family':dict(collections.Counter(row['source_family'] for row in train)),
        'selected_surplus_questions':sum(provenance[ident]['surplus'] for ident in ids),
        'selected_source_files':sorted({provenance[ident]['file'] for ident in ids}),
        'selected_ids_sha256':digest(ids),'training':summarize(train),'learning_questions':len(learning),'selection_seed':SEED,
        'freshness':'Exclude all prior42k IDs and NFC/strip/casefold question text, plus existing held/reserved whole synthesis requests. Raw sources and other training-request siblings may recur.',
        'surplus_policy':'Read only immutable committed arXiv suffix; never publish/delete/update producer data or manifest.'}
    return {'train':train,'learning_ids':[row['example']['example_id'] for row in learning]},recipe


def main(args):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Actual material/optimizer transformations require allocated CPU compute')
    if args.output.exists():raise FileExistsError('Never overwrite a prepared candidate')
    if sha(args.material)!=SOURCE_MATERIAL_SHA:raise ValueError('Original32k material changed')
    cdesign=continuation.validate_design(args.continuation_design)
    if sha(args.continuation_initial_pair)!=cdesign['initial_pair_sha256']:raise ValueError('Continuation initial-pair receipt changed')
    pair,hashes=continuation.final_pair(args.parent_arm,args.reports,cdesign,args.continuation_design,read(args.continuation_initial_pair))
    native=read(args.parent_readout)
    if (native.get('status')!='complete' or native['identity']['condition']!=args.parent_arm or native['identity']['update']!=1250
            or native['identity']['paired_checkpoint']!=pair or native['identity']['checkpoint_sha256']!=hashes
            or not native['views']['AC']['complete'] or len(native['rows'])!=100 or len(native['views']['AC']['full400']['per_item'])!=400):
        raise ValueError('Requires the exact completed parent42k native400/100 readout')
    material=read(args.material);prior_ids=read(args.reports/args.parent_arm/'material_ids.json')
    if len(material['train'])!=32000 or len(prior_ids['train'])!=10000:raise ValueError('Expected original32k plus shared10k')
    preserved={key:digest(material[key]) for key in ('held','held_cohorts','retrieval')}
    selection,recipe=select_fresh(material,args.bank,prior_ids['train'],args.reservations)
    material.update(selection,recipe=recipe,planned_epochs=1,whole_run_panels=True)
    if any(digest(material[key])!=value for key,value in preserved.items()):raise AssertionError('Frozen held/retrieval changed')
    ids={split:[row['example']['example_id'] for row in long.interleave(material[split])] for split in ('train','held')}
    if ids['held']!=prior_ids['held'] or ids['held']!=native['selection']['original_held_ids']:raise ValueError('Original400 held order changed')
    args.output.mkdir(parents=True);material_path=args.output/'material.json'
    with material_path.open('x') as handle:json.dump(material,handle,ensure_ascii=False)
    write(args.output/'selection_receipt.json',recipe)
    optimizer=derive_pair(pair,args.output/'initial')
    initial={'ac_checkpoint':str(args.output/'initial/ac'),'reader_checkpoint':str(args.output/'initial/reader'),
        'payload_sha256':hashes,'parent_pair':pair,'optimizer_provenance':optimizer,
        'parent_readout_sha256':sha(args.parent_readout),'material_sha256':sha(material_path)}
    initial_path=args.output/'initial/pair.json';write(initial_path,initial)
    config=read(args.reports/args.parent_arm/'training_config.json');config.update(learning_rate_ac=5e-6,learning_rate_target_lora=2e-6,
        warmup_updates=0,lr_schedule='constant',lr_final_fraction=1.,resume_optimizer=True,resume_skip_steps=0,
        schedule_total_updates=1000,num_epochs=1,max_duration_seconds=6600)
    if config['gradient_checkpointing_min_tokens']!=1700:raise ValueError('Native parent checkpoint threshold changed')
    original=read(Path(__file__).with_name('STRONG_LONG_ADAPTATION_DESIGN_V1.json'))
    helpers=set(original['helper_sha256'])|{'run_qa_gentle42k_long_v1.py','prepare_gentle42k_long_v1.py',
        'evaluate_continuation_10k_v1.py','evaluate_continuation_long_8k_v1.py','prepare_continuation_10k.py',
        'prepare_model_comparison.py','prepare_data_budget.py','select_bank.py',
        'run_qa_fresh42k_native8k_v1.py','prepare_fresh42k_native8k_v1.py','train_fresh42k_native8k_v1.sbatch',
        'evaluate_fresh42k_native8k_v1.py','evaluate_fresh42k_native8k_v1.sbatch'}
    name='continue42k_8k_'+args.parent_arm.removeprefix('continue32k_10k_')+'_floor'
    design={'version':1,'status':'prepared_awaiting_root_review','run_name':name,'parent_condition':args.parent_arm,
        'material_sha256':sha(material_path),'material_ids_sha256':digest(ids),'material_path':str(material_path),
        'initial_pair_sha256':sha(initial_path),'initial_pair_path':str(initial_path),'initial_payload_sha256':hashes,
        'source_sha256':long.SOURCE,'parent_readout_path':str(args.parent_readout),'parent_readout_sha256':sha(args.parent_readout),
        'encode_batch_max_tokens':65536,'builder_sha256':sha(Path(__file__).with_name('run_qa_fresh42k_native8k_v1.py')),
        'helper_sha256':{leaf:sha(Path(__file__).with_name(leaf)) for leaf in sorted(helpers)},'training_config':config,
        'questions':8000,'epochs':1,'training_by_family':recipe['selected_by_family'],'learning32':'New fixed balanced sample from fresh8000',
        'parent_unique_questions':42000,'unique_questions_after_stage':50000,'total_presentations_after_stage':50000,
        'questions_previously_seen':False,'training_exposure':'8000 fresh questions once;42k→50k unique questions',
        'optimizer':'Inherited Adam5250→6250; reset only outer stage clock0→1000; constant5e-6/2e-6',
        'reporting':'HeldAC400/learning32 every800 stage questions; frozen initial controls plus moving controls0/50/100; original100 final actual answers',
        'final_selection':'Own complete1000-update pair only; no best/partial substitution',
        'readout':{'selection_sha256':original['readout']['selection_sha256'],'scope':'Native2k final400 replay plus same100 actual AC answers; no confirmation'},
        'preserved_material':preserved,'selection_receipt_sha256':sha(args.output/'selection_receipt.json'),
        'allocation_candidate':'One RTX,64GiB host,2h wall;6600s trainer bound, paired manual resume; separate20min final readout.'}
    write(args.output/'FRESH42K_NATIVE8K_DESIGN_V1.json',design)
    (args.output/'PRESERVE').write_text('Prepared fresh8000 native continuation; independent inheritedAdam5250 pair. Design awaits root review and freeze.\n')
    print(json.dumps({'status':design['status'],'run_name':name,'material_sha256':design['material_sha256'],
        'initial_pair_sha256':design['initial_pair_sha256'],'selected_surplus_questions':recipe['selected_surplus_questions']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('material','bank','continuation-design','continuation-initial-pair','reports','parent-readout','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reservations',type=Path,nargs='+',required=True)
    parser.add_argument('--parent-arm',choices=PARENTS,required=True)
    main(parser.parse_args())
