"""Long-context reconstruction pass (Sept 29): extends the 75M reconstruction pair to one long AC part per input, the setting
question-blind RAG and trajectory Q/A read (the pair reproduces a gold passage from its own part at 0.06-0.12 NLL/token but
from inside an 8k-token bundle part at 0.74-0.78). CPU only: writes a packet (recipe, items, examples) and its task panel;
`run_channel_mixture.py train` runs it unchanged.

Training examples (--total-examples, every item at --ratio, --epochs passes; more than one is the capacity test: can the
pair fit a small fixed set of long inputs?):
  --passage-share  long_passage: one part of --token-range tokens (log-uniform) of joined passages ("Passage 1:\\n..."),
                   BM25 neighbours of a seed passage or random ones (--bm25-fraction); reproduce Passage k (k uniform)
  --lookup-share   long_lookup: the same inputs; a cue (--cue-tokens, unique in the input) from anywhere inside one passage, continued
  --span-share     long_span: the same inputs; the text between a start cue and an end cue (both unique), so the stop is predictable
  the rest         reconstruction/flat: one warm-up material passage alone, reproduced whole (the warm-up task: short items)
(training lookup continuations and spans are --target-tokens long, 48-256 by default; --mask-stop drops the end-of-turn token from
continuation targets, whose stop position is drawn and so unlearnable; --density-power points training targets at dense windows,
see generate_reconstruction_samples; panel passages are blocked by exact text, as with --seen-gold)
Long inputs come --rag-share from the RAG pool (the evidence passages of the arm packet's training items) and the rest from
the warm-up material's train split. No passage of any task-panel item's evidence, nor any panel target, enters training.

Task panel: the arm packet's panel (Q/A, swapped control, bundle_recon in its layout, lookup, reconstruction; the transfer
baseline) plus long_recon/{passage,lookup[,span]}_{1-4k,4-8k,8-16k}: --held-per-group inputs of the warm-up material's held split
(document-held-out), composed the same way. The lookup probes and held lookups always get `_nostop` twins (the same items
without the stop token: the clean metric, whatever the training targets; the originals stay comparable with earlier runs); with
--span-share, span/mid (+ swapped) probes on the unexposed golds (the mid cue moved earlier when the end cue would not fit) and
long_recon/span_* held groups composed separately, so the passage and lookup groups stay the earlier packets' items. With
--decoy-share S gives that share of multi-passage training passage/lookup/span items a digit-perturbed twin of the target passage
in place of another passage (dropped when the cue is not unique against the twin; training only, every held group unchanged).
--multispan-share S: gold-only multi-read items (one material passage alone, variant "multispan": --reads non-overlapping spans
of --multispan-tokens tokens listed in random order, joined by a --- line; a start read / an end read each with probability
--boundary-share), the passage drawn with weight (floor + dense share)^--multispan-density-power (default: --density-power);
--held-multispan N adds the held group long_recon/multispan_dense (held material at the fixed power, fixed items across arms).
--held-dense N, long_recon/{passage,lookup_nostop,span}_dense: N held inputs over the full bin range with targets drawn at
density power HELD_DENSE_POWER (the same items whatever the training power), the well-powered twins of the dense probes.

  python prepare_long_recon.py --material MATERIAL.json --arm-packet RAG_ARMS/a0 --output RUN --total-examples 16000
"""
import argparse
import json
import math
import os
import pickle
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import run_channel_mixture as mix
from run_channel_mixture import CHUNK,SEED,ForkPool,StageClock,digest,dump,file_sha,nested_parts,reader_tokens,write
from prepare_rag_blind import HEADER,split_passages,text_key
from activation.ac_model.ac_model_study import DENSITY_FLOOR,_dense_mask

BINS=((1024,4096),(4096,8192),(8192,16384))
LOOKUP_TOKENS=(48,256)   # held lookup/span target lengths (every packet's held groups stay comparable); training takes --target-tokens
HELD_DENSE_POWER=2.   # the _dense held groups: targets drawn at this density power whatever --density-power trains with (fixed items across arms)
PANEL_GROUPS=('bundle_recon/unexposed','bundle_recon/original','bundle_recon_passages/original','reconstruction/unexposed',
 'rag_unexposed/uncontextual','rag_unexposed/swapped')


def bin_name(low,high):return f'{low//1024}-{high//1024}k'


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--material',type=Path,required=True,help='the reconstruction warm-up material.json (train/held passages)')
 p.add_argument('--arm-packet',type=Path,required=True,help='a prepared question-blind RAG arm packet (bundle layout): its training evidence is the RAG pool, its panel is kept')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--total-examples',type=int,default=16_000)
 p.add_argument('--passage-share',type=float,default=.4);p.add_argument('--lookup-share',type=float,default=.4);p.add_argument('--span-share',type=float,default=0.)
 p.add_argument('--cue-tokens',type=int,nargs=2,default=(24,24),metavar=('LOW','HIGH'),help='lookup/span cue length range (training and held long_recon; the arm-panel probes keep 24)')
 p.add_argument('--mask-stop',action='store_true',help='drop the end-of-turn token from the training continuation-lookup targets (the panel always carries _nostop twins)')
 p.add_argument('--target-tokens',type=int,nargs=2,default=LOOKUP_TOKENS,metavar=('LOW','HIGH'),help='training lookup/span target length range (the held groups keep %d-%d)'%LOOKUP_TOKENS)
 p.add_argument('--density-power',type=float,default=0.,help='training passage/lookup/span targets drawn with weight (floor + dense share)^power (0: uniform, every earlier packet\'s draws)')
 p.add_argument('--decoy-share',type=float,default=0.,help='share of training passage/lookup/span items whose input holds a digit-perturbed twin of the target passage (0: none)')
 p.add_argument('--multispan-share',type=float,default=0.,help='share of training examples that are gold-only multi-read items (variant multispan; taken from the flat remainder)')
 p.add_argument('--reads',type=int,nargs=2,default=(2,5),metavar=('LOW','HIGH'),help='reads per multispan item')
 p.add_argument('--boundary-share',type=float,default=.25,help='probability of a start read and, independently, of an end read in a multispan item')
 p.add_argument('--multispan-tokens',type=int,nargs=2,default=(16,64),metavar=('LOW','HIGH'),help='multispan read length range')
 p.add_argument('--multispan-cue-tokens',type=int,nargs=2,default=(8,24),metavar=('LOW','HIGH'),help='multispan cue length range')
 p.add_argument('--multispan-density-power',type=float,default=None,help='multispan passages drawn with weight (floor + dense share)^power (default: --density-power)')
 p.add_argument('--held-multispan',type=int,default=0,help='items of the held group long_recon/multispan_dense (held material drawn at power %g)'%HELD_DENSE_POWER)
 p.add_argument('--held-dense',type=int,default=0,help='items per long_recon/{passage,lookup_nostop,span}_dense held group: held material over the whole token range, targets at density power %g (0: no dense groups)'%HELD_DENSE_POWER)
 p.add_argument('--rag-share',type=float,default=1/3,help='share of long inputs composed from the RAG pool (the rest: warm-up material)')
 p.add_argument('--token-range',type=int,nargs=2,default=(1024,16384));p.add_argument('--bm25-fraction',type=float,default=.5)
 p.add_argument('--ratio',type=float,default=1/10);p.add_argument('--held-per-group',type=int,default=16)
 p.add_argument('--panel-groups',nargs='+',default=list(PANEL_GROUPS),help='arm panel groups kept (every arm panel item still blocks training)')
 p.add_argument('--checkpoint-every',type=int,default=200);p.add_argument('--reporting-interval',type=float,default=.125,help='panel spacing, a fraction of each epoch')
 p.add_argument('--epochs',type=int,default=1)
 p.add_argument('--train-seed',type=int,default=0,help='a fresh training draw (an extension run); the panel and held items do not depend on it')
 p.add_argument('--workers',type=int,default=0,help='fork workers (0: every CPU of the allocation; 1: one process)')
 a=p.parse_args()
 shares={'passage':a.passage_share,'lookup':a.lookup_share,'span':a.span_share}
 if not (all(v>=0 for v in shares.values()) and 0<sum(shares.values())<=1):p.error('shares must be nonnegative, positive together, and sum to at most 1')
 if not 0<a.cue_tokens[0]<=a.cue_tokens[1]:p.error('--cue-tokens LOW HIGH needs 0 < LOW <= HIGH')
 if not 0<a.target_tokens[0]<=a.target_tokens[1]:p.error('--target-tokens LOW HIGH needs 0 < LOW <= HIGH')
 if a.density_power<0 or a.held_dense<0:p.error('--density-power and --held-dense are nonnegative')
 if not 0<=a.decoy_share<=1:p.error('--decoy-share is a share in [0, 1]')
 if not 0<=a.multispan_share<=1 or sum(shares.values())+a.multispan_share>1:p.error('--multispan-share is a share in [0, 1] and the shares sum to at most 1')
 if not (0<a.reads[0]<=a.reads[1] and 0<a.multispan_tokens[0]<=a.multispan_tokens[1] and 0<a.multispan_cue_tokens[0]<=a.multispan_cue_tokens[1]):p.error('--reads, --multispan-tokens and --multispan-cue-tokens need 0 < LOW <= HIGH')
 if not 0<=a.boundary_share<=1 or (a.multispan_density_power or 0)<0 or a.held_multispan<0:p.error('--boundary-share in [0, 1]; --multispan-density-power and --held-multispan nonnegative')
 multispan_power=a.density_power if a.multispan_density_power is None else a.multispan_density_power
 if not 0<=a.rag_share<=1:p.error('--rag-share must be in [0, 1]')
 if a.epochs<1:p.error('--epochs must be at least 1')
 out=a.output;out.mkdir(parents=True,exist_ok=True);panel_dir=out/'task_panel'
 if (out/'prepared.json').exists() or (panel_dir/'panel.json').exists():raise FileExistsError('Packet or panel already committed; use a new output')
 panel_dir.mkdir(exist_ok=True)
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.ac_model.ac_model_study import LOOKUP_INSTRUCTION,PASSAGE_INSTRUCTION
 from activation.ac_model.ac_model_utils import validate_part_capacity
 from run_qa import harness,AC
 from run_trajectory_tasks import compact_example
 import prepare_rag_arms
 # The trained prompts are exactly the arm panel's bundle_recon and lookup prompts (their held items measure this task).
 if (PASSAGE_INSTRUCTION,LOOKUP_INSTRUCTION)!=(prepare_rag_arms.BUNDLE_INSTRUCTION,prepare_rag_arms.LOOKUP_INSTRUCTION):raise ValueError('instructions differ from the arm panel\'s')
 workers=a.workers or len(os.sched_getaffinity(0))
 if workers>1:
  import torch
  if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('Parallel preparation needs CUDA_VISIBLE_DEVICES= (a CPU-only job); use --workers 1 otherwise')
  os.environ['TOKENIZERS_PARALLELISM']='false'
 arm_recipe=json.loads((a.arm_packet/'recipe.json').read_text());pair=arm_recipe['initial_pair']
 if arm_recipe['arm']['layout']!='bundle' or arm_recipe.get('ratio')!=a.ratio:raise ValueError('the arm packet must be a bundle-layout packet at --ratio (its panel is the transfer baseline)')
 arm_panel=Path(arm_recipe['held']['panel'])
 with (arm_panel/'panel.pkl').open('rb') as f:arm_panel_data=pickle.load(f)
 if file_sha(arm_panel/'panel.pkl')!=arm_recipe['held']['panel_sha256']:raise ValueError('the arm panel changed')
 # Blocked: every passage a panel item shows or targets (split evidence; whole text for an unsplit bundle).
 blocked=set();unsplit=[]
 for item in arm_panel_data['items']:
  if item.info.get('passages_text'):
   text=item.info['passages_text']   # lookup items carry no passage count: count the headers
   parts=split_passages(item if item.info.get('passages') else SimpleNamespace(info={'passages_text':text,'passages':len(HEADER.findall(text))}))
   if parts is None:unsplit.append(item.info['passages_text'])
   else:blocked.update(text_key(x.strip()) for x in parts)
  if item.info.get('family')=='reconstruction':blocked.add(text_key(item.in_context_messages[-1]['content'].strip()))
 def allowed(text):return text_key(text.strip()) not in blocked and not any(text.strip() in u for u in unsplit)
 with (a.arm_packet/'items.pkl').open('rb') as f:arm_train=pickle.load(f)['train']
 rag_pool={};excluded=Counter()
 for item in arm_train:
  if item.info['family']!='rag_qa' and item.info.get('diagnostic')!='bundle':continue
  parts=split_passages(item)
  if parts is None:excluded['rag_items_unsplit']+=1;continue
  for x in parts:
   x=x.strip()
   if not x or text_key(x) in rag_pool:continue
   if not allowed(x):excluded['rag_passages_blocked']+=1;continue
   rag_pool[text_key(x)]={'text':x,'doc_id':'rag:'+text_key(x)[:16],'source':'rag'}
 rag_pool=sorted(rag_pool.values(),key=lambda x:x['doc_id'])
 material=json.loads(a.material.read_text())
 train_pool=[x for x in material['train'] if allowed(x['text'])];excluded['material_blocked']=len(material['train'])-len(train_pool)
 held_pool=list(material['held'])
 if {x['doc_id'] for x in train_pool}&{x['doc_id'] for x in held_pool}:raise ValueError('material train/held documents overlap')
 print(json.dumps({'rag_pool':len(rag_pool),'material_train':len(train_pool),'material_held':len(held_pool),'excluded':dict(excluded)}),flush=True)
 h=harness(pair['ac_checkpoint'],pair['reader_checkpoint']);study=ActivationContextStudyGenerator(h,AC)
 if workers>1:torch.set_num_threads(1)
 total=a.total_examples;n_long={k:round(total*v) for k,v in shares.items()};n_multispan=round(total*a.multispan_share);n_flat=total-sum(n_long.values())-n_multispan
 if n_flat<0:p.error('the long shares round past the total')
 def density(text):
  """(dense share, tokens) of a passage."""
  mask=_dense_mask(text,study.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']);return sum(mask)/max(1,len(mask)),len(mask)
 multispan_min_tokens=a.reads[0]*(2*a.multispan_cue_tokens[1]+a.multispan_tokens[1])   # the shortest passage that fits the fewest reads at their longest
 def dense_draw(pool,count,power,seed,label):
  """`count` passages of `pool` holding at least multispan_min_tokens tokens, without replacement, each with weight
  (floor + dense share)^power (Efraimidis-Spirakis keys; uniform at power 0)."""
  rng=random.Random(seed);stats=[density(x['text']) for x in pool];keys={i:rng.random()**(1/(DENSITY_FLOOR+share)**power) for i,(share,tokens) in enumerate(stats) if tokens>=multispan_min_tokens}
  if len(keys)<count:raise ValueError(f'{label}: only {len(keys)} passages of {len(pool)} hold {multispan_min_tokens}+ tokens, {count} wanted')
  print(f'{label}: {len(keys)} of {len(pool)} passages hold {multispan_min_tokens}+ tokens',flush=True)
  return [pool[i] for i in sorted(keys,key=lambda i:-keys[i])[:count]]
 multispan_kwargs=dict(target_tokens=tuple(a.multispan_tokens),cue_tokens=tuple(a.multispan_cue_tokens),reads=tuple(a.reads),boundary_share=a.boundary_share)
 spare=lambda n:math.ceil(n*1.05)+8 if n else 0   # a kind with no share draws nothing
 def long_items(pool,counts,seed,label,token_range):
  """Composed inputs of `pool` (at least two passages; topped up before any example is built), consecutive slices as
  `passage`, `lookup` and `span` items (`counts` per variant)."""
  need=sum(spare(n) for n in counts.values());inputs=[]
  for attempt in range(6):
   more=study.compose_reconstruction_inputs(pool,math.ceil((need-len(inputs))*1.1)+8,token_range=token_range,bm25_fraction=a.bm25_fraction,seed=seed+1000*attempt)
   inputs+=[x for x in more if len(x['texts'])>=2]
   if len(inputs)>=need:break
  else:raise ValueError(f'{label}: only {len(inputs)} multi-passage inputs for {need}')
  inputs=inputs[:need];built=[];offset=0
  for variant in ('passage','lookup','span'):
   chosen=inputs[offset:offset+spare(counts[variant])];offset+=spare(counts[variant])
   if not chosen:continue
   for item in study.generate_reconstruction_samples(chosen,ratio=a.ratio,variant=variant,seed=seed,target_tokens=tuple(a.target_tokens),
     cue_tokens=tuple(a.cue_tokens),mask_stop=a.mask_stop and variant=='lookup',density_power=a.density_power,decoy=a.decoy_share):
    item.info.update(family='reconstruction',diagnostic=f'long_{variant}',source=label,pool=label,encoder_input='bare',layout='bundle')
    built.append(item)
  return built
 n_rag={k:round(v*a.rag_share) for k,v in n_long.items()};n_material={k:n_long[k]-n_rag[k] for k in n_long}
 clock=StageClock('composing and rendering',4);stages={}
 T=SEED+10_000*a.train_seed   # training draws only
 pools={'rag':long_items(rag_pool,n_rag,T+1,'rag',tuple(a.token_range)) if sum(n_rag.values()) else []};clock(1)
 pools['material']=long_items(train_pool,n_material,T+2,'material',tuple(a.token_range));clock(2)
 rng=random.Random(f'{SEED}:long_recon:flat'+(f':{a.train_seed}' if a.train_seed else ''))
 flat_rows=[dict(x,ratio=a.ratio) for x in rng.sample(train_pool,spare(n_flat))]
 flat=study.generate_reconstruction_samples(flat_rows,ratio=a.ratio,variant='reconstruct',seed=T+3) if flat_rows else [];clock(3)
 for item in flat:item.info.update(family='reconstruction',source=item.dataset_id,encoder_input='bare')
 multispan=[]
 if n_multispan:   # gold-only multi-read items: dense-weighted passage draw (the whole pool's densities: one tokenization pass)
  rows=[dict(x,ratio=a.ratio) for x in dense_draw(train_pool,math.ceil(n_multispan*1.25)+8,multispan_power,f'{SEED}:long_recon:multispan'+(f':{a.train_seed}' if a.train_seed else ''),'multispan')]   # 25% margin: ~8% of the drawn passages yield no item (no unique cue in 8 draws)
  multispan=study.generate_reconstruction_samples(rows,ratio=a.ratio,variant='multispan',seed=T+4,**multispan_kwargs)
  for item in multispan:item.info.update(family='reconstruction',diagnostic='multispan',source=item.dataset_id,encoder_input='bare')
  print(f'multispan: {len(multispan)} items from {len(rows)} passages at power {multispan_power}, mean reads {sum(len(i.info["reads"]) for i in multispan)/max(1,len(multispan)):.2f}, mean window density {sum(i.info["window_density"] for i in multispan)/max(1,len(multispan)):.3f}',flush=True)
 stages['compose']=clock.seconds()
 wanted={**{(pool,f'long_{k}'):n[k] for pool,n in (('rag',n_rag),('material',n_material)) for k in n if n[k]},('material','flat'):n_flat,**({('material','multispan'):n_multispan} if n_multispan else {})}
 drawn=pools['rag']+pools['material']+flat+multispan;seen=set()
 output=[i for i in drawn if not (i.item_id in seen or seen.add(i.item_id))]   # a repeated draw (same input, same target) once: the spares cover it
 if len(output)<len(drawn):print(f'{len(drawn)-len(output)} repeated items dropped',flush=True)
 config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=5e-5,learning_rate_target_lora=2e-5,examples_per_update=8,num_epochs=a.epochs,completion_samples=0,
  teacher_cache_block_examples=32,logits_chunk_tokens=64,gradient_checkpointing_min_tokens=0,seed=SEED,deterministic=False,
  schedule_total_updates=math.ceil(total/8)*a.epochs,checkpoint_every_updates=a.checkpoint_every,reporting_interval=a.reporting_interval,shuffle_examples=True)
 trainer=ActivationContextTrainer(h,config)
 clock=StageClock('examples',len(output));done=0;kept=[];failures=[]
 mix.BUILD_CONTEXT={'trainer':trainer,'study':study,'scan':True,'output':output}
 with ForkPool(workers) as pool:
  for results in pool.map(mix.example_chunk,[range(i,min(i+CHUNK,len(output))) for i in range(0,len(output),CHUNK)]):
   for value,failure in results:
    if failure:failures.append(failure);continue
    item,example,_=value;kept.append((item,example))
   done+=len(results);clock(done)
 mix.BUILD_CONTEXT=None;stages['examples']=clock.seconds()
 chosen=[]
 for (label,kind),n in wanted.items():
  have=[x for x in kept if x[0].info.get('pool','material')==label and (x[0].info.get('diagnostic') or 'flat')==kind]
  if len(have)<n:raise ValueError(f'{label}/{kind}: only {len(have)} built for {n}: {failures[:3]}')
  chosen+=have[:n]
 train_items=[i for i,_ in chosen];examples=[e for _,e in chosen]
 if len(train_items)!=total:raise ValueError(f'{len(train_items)} training examples, expected {total}')
 if any(i.info['ratio']!=a.ratio for i in train_items):raise ValueError('a training item is not at --ratio')
 # Held long inputs (document-held-out material) per length bin: reproduce Passage k and cued lookup, both as trained.
 builder=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',max_example_tokens=1_000_000))
 panel=[i for i in arm_panel_data['items'] if i.info['panel_group'] in a.panel_groups];kept_ids={id(i) for i in panel}
 panel_examples=[e for e in arm_panel_data['examples'] if id(e.item) in kept_ids];held_long=[]
 if len(panel_examples)!=len(panel) or set(a.panel_groups)-{i.info['panel_group'] for i in panel}:raise ValueError('arm panel groups or examples missing')
 # Lookup probe in the bundle layout (the arm packet predates it): prepare_rag_arms' items and seeds on the unexposed golds
 # (bundle_recon/unexposed keeps the unexposed order): cue at the passage start, mid-passage, each also with the next item's rows
 # (the swapped controls: what the cue alone predicts).
 unexposed=[i for i in arm_panel_data['items'] if i.info['panel_group']=='bundle_recon/unexposed']
 for index,item in enumerate(unexposed):
  parts=split_passages(item);k=item.info['passage_index'];row_id=item.item_id.split(':',2)[2]
  length=len(study.tokenizer(parts[k],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping'])
  if length<2*(prepare_rag_arms.LOOKUP_CUE+prepare_rag_arms.LOOKUP_TARGET):continue
  mid=random.Random(f'{SEED}:lookup:{row_id}').randrange(length//4,length-prepare_rag_arms.LOOKUP_CUE-prepare_rag_arms.LOOKUP_TARGET+1)
  span_mid=min(mid,length-2*prepare_rag_arms.LOOKUP_CUE-prepare_rag_arms.LOOKUP_TARGET)   # the end cue (LOOKUP_CUE tokens) must fit after the target
  evidence=split_passages(unexposed[(index+1)%len(unexposed)])
  probes=[('lookup/start',0,None,{}),('lookup/start_swapped',0,evidence,{}),('lookup/mid',mid,None,{}),('lookup/mid_swapped',mid,evidence,{}),
   ('lookup/start_nostop',0,None,{'mask_stop':True}),('lookup/mid_nostop',mid,None,{'mask_stop':True}),('lookup/mid_nostop_swapped',mid,evidence,{'mask_stop':True})]
  if a.span_share:probes+=[('span/mid',span_mid,None,{'end_cue_tokens':prepare_rag_arms.LOOKUP_CUE}),('span/mid_swapped',span_mid,evidence,{'end_cue_tokens':prepare_rag_arms.LOOKUP_CUE})]
  for group,start,shown,extra in probes:   # the _nostop twins: the same probe without its stop token; span/mid: the same cue and target (moved earlier when needed) with the end cue given
   probe=prepare_rag_arms.lookup_item(study,item,parts,k,start,'bundle',evidence=shown,**extra)
   if probe is None:raise ValueError(f'{group}: the probe does not fit in passage {k} of {row_id}')
   probe.item_id=f'panel:{group}:{row_id}';probe.info.update(panel_group=group,panel=True,reconstruction_overlap=False)
   example=builder.build_example(study.ac_model,probe)
   if reader_tokens(builder,study,probe)>72000:raise ValueError('Panel reader history exceeds 72000 tokens')
   for request in example.part_requests:validate_part_capacity(study.ac_model,request,65536)
   probe.info.update(target_tokens=len(example.target_ids),reader_tokens=len(example.student_ids))
   panel.append(probe);panel_examples.append(compact_example(example))
 for index,(low,high) in enumerate(BINS):
  compose=lambda count,seed:[x for x in study.compose_reconstruction_inputs(held_pool,count,token_range=(low,high),bm25_fraction=a.bm25_fraction,seed=seed) if len(x['texts'])>=2 and x['input_tokens']>=low]
  inputs=compose(6*a.held_per_group,SEED+100+index)[:4*a.held_per_group]   # labelled by actual size; the passage and lookup inputs of every packet since LC
  span_inputs=compose(4*a.held_per_group,SEED+300+index)[:2*a.held_per_group] if a.span_share else []   # drawn apart: the earlier groups stay the same items
  held_variants=[('passage',inputs[:2*a.held_per_group],{}),('lookup',inputs[2*a.held_per_group:],{}),('lookup_nostop',inputs[2*a.held_per_group:],{'mask_stop':True})]+([('span',span_inputs,{})] if a.span_share else [])
  for name,subset,extra in held_variants:   # lookup and lookup_nostop: the same inputs, seed and draws
   variant=name.split('_')[0];group=f'long_recon/{name}_{bin_name(low,high)}'
   items=study.generate_reconstruction_samples(subset,ratio=a.ratio,variant=variant,seed=SEED+200+index,target_tokens=LOOKUP_TOKENS,cue_tokens=tuple(a.cue_tokens),**extra)[:a.held_per_group]
   if len(items)<a.held_per_group:raise ValueError(f'{group}: only {len(items)} held items')
   for item in items:
    item.item_id=f'panel:{group}:{item.item_id.rsplit(":",1)[1]}'
    item.info.update(family='reconstruction',diagnostic=f'long_{variant}',panel_group=group,panel=True,reconstruction_overlap=False,
     source='material_held',encoder_input='bare',layout='bundle')
    example=builder.build_example(study.ac_model,item)
    if reader_tokens(builder,study,item)>72000:raise ValueError('Panel reader history exceeds 72000 tokens')
    for request in example.part_requests:validate_part_capacity(study.ac_model,request,65536)
    item.info.update(target_tokens=len(example.target_ids),reader_tokens=len(example.student_ids))
    panel.append(item);panel_examples.append(compact_example(example));held_long.append(item)
 if a.held_dense:   # dense held groups: held material over the full bin range (fixed, whatever --token-range trains), targets pointed at dense windows at the fixed power
  compose=lambda count,seed:[x for x in study.compose_reconstruction_inputs(held_pool,count,token_range=(BINS[0][0],BINS[-1][1]),bm25_fraction=a.bm25_fraction,seed=seed) if len(x['texts'])>=2 and x['input_tokens']>=BINS[0][0]]
  dense_inputs=compose(8*a.held_dense,SEED+400)
  if len(dense_inputs)<6*a.held_dense:raise ValueError(f'dense held groups: only {len(dense_inputs)} composed inputs for {6*a.held_dense}')
  dense_variants=[('passage',dense_inputs[:2*a.held_dense],{}),('lookup_nostop',dense_inputs[2*a.held_dense:4*a.held_dense],{'mask_stop':True})]+([('span',dense_inputs[4*a.held_dense:6*a.held_dense],{})] if a.span_share else [])
  for name,subset,extra in dense_variants:
   variant=name.split('_')[0];group=f'long_recon/{name}_dense'
   items=study.generate_reconstruction_samples(subset,ratio=a.ratio,variant=variant,seed=SEED+500,target_tokens=LOOKUP_TOKENS,cue_tokens=tuple(a.cue_tokens),density_power=HELD_DENSE_POWER,**extra)[:a.held_dense]
   if len(items)<a.held_dense:raise ValueError(f'{group}: only {len(items)} held items')
   for item in items:
    item.item_id=f'panel:{group}:{item.item_id.rsplit(":",1)[1]}'
    item.info.update(family='reconstruction',diagnostic=f'long_{variant}',panel_group=group,panel=True,reconstruction_overlap=False,
     source='material_held',encoder_input='bare',layout='bundle')
    example=builder.build_example(study.ac_model,item)
    if reader_tokens(builder,study,item)>72000:raise ValueError('Panel reader history exceeds 72000 tokens')
    for request in example.part_requests:validate_part_capacity(study.ac_model,request,65536)
    item.info.update(target_tokens=len(example.target_ids),reader_tokens=len(example.student_ids))
    panel.append(item);panel_examples.append(compact_example(example));held_long.append(item)
   print(f'{group}: {len(items)} items, mean window density {sum(i.info["window_density"] for i in items)/len(items):.3f}',flush=True)
 if a.held_multispan:   # the multispan held group: held passages drawn at the fixed power, the training read shape
  group='long_recon/multispan_dense'
  rows=[dict(x,ratio=a.ratio) for x in dense_draw(held_pool,4*a.held_multispan,HELD_DENSE_POWER,f'{SEED}:long_recon:held_multispan',group)]
  items=study.generate_reconstruction_samples(rows,ratio=a.ratio,variant='multispan',seed=SEED+600,**multispan_kwargs)[:a.held_multispan]
  if len(items)<a.held_multispan:raise ValueError(f'{group}: only {len(items)} held items')
  for item in items:
   item.item_id=f'panel:{group}:{item.item_id.rsplit(":",1)[1]}'
   item.info.update(family='reconstruction',diagnostic='multispan',panel_group=group,panel=True,reconstruction_overlap=False,source='material_held',encoder_input='bare')
   example=builder.build_example(study.ac_model,item)
   for request in example.part_requests:validate_part_capacity(study.ac_model,request,65536)
   item.info.update(target_tokens=len(example.target_ids),reader_tokens=len(example.student_ids))
   panel.append(item);panel_examples.append(compact_example(example));held_long.append(item)
  print(f'{group}: {len(items)} items, mean reads {sum(len(i.info["reads"]) for i in items)/len(items):.2f}, mean window density {sum(i.info["window_density"] for i in items)/len(items):.3f}',flush=True)
 if any(nested_parts(i) for i in panel+train_items):raise ValueError('an item nests an AC part (depth-1 rule)')
 if len({i.item_id for i in panel})!=len(panel):raise ValueError('Duplicate task-panel item ids')
 held_docs={d for i in held_long for d in i.doc_ids}
 if held_docs&{d for i in train_items for d in i.doc_ids}:raise ValueError('a held long-input document trains')
 held=json.loads((arm_panel/'held.json').read_text())
 held['long_recon']={'rule':'warm-up material held split (document-held-out), composed per length bin'+(' and, for the _dense groups, over the full range' if a.held_dense else '')+(' and, for multispan_dense, single passages' if a.held_multispan else ''),'doc_ids':sorted(held_docs)}
 write(panel_dir/'held.json',held)
 dump(panel_dir/'panel.pkl',{'items':panel,'examples':panel_examples})
 groups=Counter(i.info['panel_group'] for i in panel)
 arm_receipt=json.loads((arm_panel/'panel.json').read_text())
 write(panel_dir/'panel.json',{'sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json'),'items':len(panel),'groups':dict(sorted(groups.items())),
  'target_tokens':sum(i.info['target_tokens'] for i in panel),'reader_tokens':sum(i.info['reader_tokens'] for i in panel),
  'initial_pair':pair,'seed':SEED,'ratio':a.ratio,'arm_panel':{'path':str(arm_panel.resolve()),'sha256':arm_receipt['sha256'],'held_sha256':arm_receipt['held_sha256']},
  'rule':'the arm packet\'s panel groups listed in panel_groups (unchanged), lookup/* (and span/mid) on its unexposed golds in the bundle layout with _nostop twins (the same items without the stop token), and long_recon groups of held material'+(' (the _dense groups: targets at the fixed density power)' if a.held_dense else ''),'panel_groups':a.panel_groups,
  'cue_tokens':list(a.cue_tokens),'mask_stop':a.mask_stop,'span_share':a.span_share,'held_dense':a.held_dense,'held_dense_power':HELD_DENSE_POWER if a.held_dense else None,'held_multispan':a.held_multispan})
 actual=Counter(mix.panel_group(i)+'/'+i.info.get('pool','material') for i in train_items)
 by_kind=Counter();tokens=Counter();passages=Counter();inputs=Counter();compositions=Counter()
 for item in train_items:
  kind=mix.panel_group(item);by_kind[kind]+=1;tokens[kind]+=item.info['target_tokens']
  if 'passages' in item.info:passages[kind]+=item.info['passages'];inputs[kind]+=item.info['passage_tokens'];compositions[f"{kind}/{item.info['composition']}"]+=1
 arm={'task':'long-context reconstruction','shares':{'long_passage':a.passage_share,'long_lookup':a.lookup_share,**({'long_span':a.span_share} if a.span_share else {}),**({'multispan':a.multispan_share} if a.multispan_share else {}),'flat':1-sum(shares.values())-a.multispan_share},
  'rag_share':a.rag_share,'token_range':list(a.token_range),'bm25_fraction':a.bm25_fraction,'layout':'bundle','lr_ac':config.learning_rate_ac,**({'train_seed':a.train_seed} if a.train_seed else {}),
  'cue_tokens':list(a.cue_tokens),'mask_stop':a.mask_stop,
  **({'target_tokens':list(a.target_tokens)} if tuple(a.target_tokens)!=LOOKUP_TOKENS else {}),**({'density_power':a.density_power} if a.density_power else {}),**({'decoy_share':a.decoy_share} if a.decoy_share else {}),
  **({'reads':list(a.reads),'boundary_share':a.boundary_share,'multispan_tokens':list(a.multispan_tokens),'multispan_cue_tokens':list(a.multispan_cue_tokens),'multispan_density_power':multispan_power} if a.multispan_share else {})}
 recipe={'seed':SEED,'epochs':a.epochs,'phase':'Long-context reconstruction at 1/%d: %s'%(round(1/a.ratio),json.dumps(arm,sort_keys=True)),'ratio':a.ratio,'arm':arm,
  'counts':{'wanted':{'/'.join(k):v for k,v in wanted.items()}},'actual_train':dict(by_kind),'unique_sources':dict(by_kind),'by_pool':dict(actual),
  'target_tokens':dict(tokens),'mean_passages':{k:round(passages[k]/by_kind[k],2) for k in passages},'mean_input_tokens':{k:round(inputs[k]/by_kind[k]) for k in inputs},
  'compositions':dict(compositions),'excluded':dict(excluded),'failures':failures,'loss_shares':'natural: every example weighs 1','global_batch':8,
  'pools':{'rag':len(rag_pool),'material_train':len(train_pool),'material_held':len(held_pool),'rag_from':str(a.arm_packet.resolve()),
   'material':{'path':str(a.material.resolve()),'sha256':file_sha(a.material)}},
  'held':{'rule':'task_panel/held.json','panel':str(panel_dir.resolve()),'panel_sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json')},
  'headline_note':'the headline held NLL averages every panel group: read the groups','blocked_panel_passages':len(blocked),'panel_unsplit':len(unsplit),
  'initial_pair':pair,'training_config':asdict(config)}
 for item in train_items:item.weight=1.0
 write(out/'recipe.json',recipe)
 bundle_out={'train':train_items,'held':[]}
 dump(out/'items.pkl',bundle_out);dump(out/'prepared_examples.pkl',{**bundle_out,'examples':examples})
 write(out/'prepared.json',{'recipe_sha256':digest(recipe),'items_sha256':file_sha(out/'items.pkl'),'items_bytes':(out/'items.pkl').stat().st_size,
  'prepared_examples_sha256':file_sha(out/'prepared_examples.pkl'),'validated_capacity':{'reader_tokens':72000,'encoder_tokens':65536},
  'training_examples':total,'held_examples':0,'one_pass':True,'preparation':{'workers':workers,'stage_seconds':stages}})
 print(json.dumps({'actual_train':dict(by_kind),'by_pool':dict(actual),'mean_input_tokens':recipe['mean_input_tokens'],'panel':dict(groups),'failures':len(failures)}),flush=True)

if __name__=='__main__':main()
