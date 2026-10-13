"""DRAFT packet builder for a document corpus with multiple-choice questions (QuALITY, LongHealth), modelled on
prepare_rag_arms.py: the packet (items.pkl, prepared_examples.pkl, prepared.json, recipe.json, task_panel/, exposure.json) is the
one `run_channel_mixture.py train` and `review_snapshot.py` read. Not yet run end to end (see "Where it runs").

Training items (--train-form), every one at --ratio, the encoder reading the bare whole document (no chunking), no question ever:
  spans (default)  generate_reconstruction_samples' "multispan" variant on the whole document: --reads (2-5) non-overlapping
                   stretches of --span-tokens (32-128) tokens, each named by its first and last words (--cue-tokens, 16-24),
                   listed in random order and reproduced in that order. ONE long pass of fresh examples: --total-examples N
                   (default: --examples-per-doc x --passes-per-epoch x documents), shared among the training documents in
                   proportion to their token counts, ids span:<doc>#<n>; the stretches of different examples overlap, so every
                   token is a target many times while no request repeats.
                   Held requests (panel): --held-spans N fresh examples of the same form per document, never asked in training
                   (checked), for up to --span-docs training documents (spans/heldout_requests: the text trains, the request does
                   not) and as many never-trained documents (spans/heldout_docs).
                   --reserve (off by default) instead sets text aside: the held items are single stretches drawn FIRST, their
                   token ranges (cue, target, end cue) are never read by a training example (redrawn), panel groups
                   spans/heldout_text (reserved) and spans/trained_text (stretches of the same documents that do train).
                   --multi-part-share S with --train-parts P (LongHealth: .25 and 5): that share of the examples shows P
                   documents, one part each (the target document and P-1 other training documents, seeded, in random order;
                   the panel's "Passages: / Passage k:" layout); the request names the target passage, its cues occur in no
                   other shown document, and the item has no in-context view (never read by SFT; it would double the packet).
  whole            whole-document reconstruction, one part per document (`mix.make_item(study,'reconstruction',...)`, the
                   "reconstruct" variant), --passes-per-epoch copies. --epochs is the trainer's own epoch count (recipe
`epochs`, training_config.num_epochs; schedule_total_updates = ceil(items/8) x epochs; a reshuffle per epoch).
 CAUTION (ac_model_training.py): the trainer evaluates the whole task panel at every reporting boundary of EVERY epoch (at
 least once per epoch, whatever --reporting-interval) and saves a pair per epoch. 50 documents are 7 updates per epoch, so 40
 epochs would mean 40 panel evaluations and 40 saved pairs around 280 updates. --passes-per-epoch K puts every document K times
 in one epoch (ids <doc>#pass<k>; total passes = K x epochs), which spaces panels and checkpoints as in the one-pass packets.
 --dry-run prints the resulting counts.

Held task panel (task_panel/, evaluation only), groups for --benchmark B:
  mc/B                      every question of the evaluation documents as an uncontextual Q/A item: the encoder reads the bare
                            document(s), never the question; the reader sees the rows and mc_eval.question_text (question,
                            lettered options, "answer with the single letter"); the gold submit_answer argument is the letter.
                            Order: round-robin over documents, so the first N items (review_snapshot.py --answers N) cover them all.
  mc_swapped/B              a seeded subsample (--mc-swapped, 128) of the same questions with the NEXT context's rows: how much
                            of the answer comes from the rows.
  mc_norows/B               the same questions (all, or --mc-norows N) with NO AC part: the trained reader alone, the
                            memorization check. The item is the mc item's `no_context` reference view
                            (ActivationContextStudyGenerator.transform_for_eval_reference: the user content keeps its text
                            pieces, the parts are dropped), under its own item id (a ':no_context' suffix would be folded
                            into the mc group by run_channel_mixture.base_id). It keeps the full-document in-context view,
                            because review_snapshot.py --mode frozen asks every panel item for one.
  reconstruction/train_docs, reconstruction/heldout_docs   whole-document reconstruction of up to --recon-held (16) training
                            documents and of as many documents that never train (--heldout-docs, default: the evaluation
                            documents outside training); either group is left out when it has no document.
Scoring: review_snapshot.py's exact_match / token_f1 are NOT valid for letters (normalize_answer drops "a"); rescore its output
with `python mc_eval.py --rescore OUT.json --questions QUESTIONS.json`. Its --answers must be at least the group size (894 for
QuALITY's 50 articles; ec2_review_node.sh defaults to 448). Its frozen mode treats mc_swapped and mc_norows as families of their
own (their names do not end in /swapped) and decodes them under both views too: wasteful, not wrong.

Contexts: --parts-per-item P groups the evaluation documents, in the order given, into contexts of P documents; a question's rows
are its whole context, one part per document (P=1: QuALITY, one article; P=5: the LongHealth candidate, 5 patients per context).
Multi-part TRAINING items exist in the spans form only (--multi-part-share / --train-parts above: stretches of one of P shown
documents, fresh random company per example); the whole form trains one part per document whatever --parts-per-item.
--mc-panel N keeps the first N questions (round-robin over the documents) in the mc / mc_norows groups: the in-training panel of
a long-context benchmark (LongHealth: every question re-encodes five ~12k-token records). Build the full panel as a second packet
of the same arguments (few passes) for the decodes.

Inputs: --documents FILE... (a list, or {"documents": [...]} / {"patients": [...]}, rows {id, text}); --questions FILE (a list or
{"questions": [...]}, rows {id, doc_id | patient_id, question, options, gold [, gold_letters, hard]}); --train-docs / --eval-docs
/ --heldout-docs: `all`, comma-separated ids, or a file (a documents file, a JSON list of ids, or one id per line). Evaluation
documents default to the documents that have questions. Training and evaluation documents may overlap (the "memorize the corpus"
arm) or not (the held-out arm); the overlap is recorded per panel item (exposure.json; `clean_*` panel rows exclude exposed items).

Where it runs. The build loads the harness (run_qa.harness: Qwen3.5-9B reader, Qwen3.5-2B side, the initial pair), so it needs
the model weights: a CPU node with the HF cache, as rag_arms/prepare.sbatch (CUDA_VISIBLE_DEVICES= for --workers > 1), not the
container (tokenizers only). No GPU is needed before `run_channel_mixture.py train`. --dry-run needs neither: it builds every
item with the same code on the two tokenizers alone and prints item counts by group, updates per epoch, target tokens, the
largest reader / encoder token counts against 72,000 / 65,536, and panel evaluations and saved pairs per run. Not covered by
the dry run: ActivationContextTrainer.build_example's own checks, the execution records, the pair's checkpoint shape.

  python prepare_doc_corpus.py --dry-run --benchmark quality --documents quality_eval50.json quality_train_docs.json \
    --questions quality_eval50.json --train-docs quality_train_docs.json --ratio 0.05 --epochs 4 --passes-per-epoch 5
  python prepare_doc_corpus.py ... --initial-pair PAIR.json --output PACKET        (on a node)
"""
import argparse
import json
import math
import os
import random
import statistics
from collections import Counter
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path

import mc_eval
import run_channel_mixture as mix
from run_channel_mixture import CHUNK,SEED,ForkPool,StageClock,digest,dump,file_sha,write

READER,SIDE='Qwen/Qwen3.5-9B','Qwen/Qwen3.5-2B'   # run_qa.harness's models; the dry run loads their tokenizers alone
READER_LIMIT,ENCODER_LIMIT=72000,65536
MIN_ROWS,MAX_ROWS=2,4096   # run_qa.harness's min_view_rows and ActivationContextModelConfig's max_view_rows (dry run only; the build asks the model)
DOC_QA_INSTRUCTIONS='Answer the question from the {noun} above. Call submit_answer with the answer.'

def rows_of(value,*names):
 if isinstance(value,list):return value
 for name in names:
  if name in value:return value[name]
 raise ValueError(f'expected a list or one of {names}')

def load_documents(paths):
 documents={}
 for path in paths:
  for row in rows_of(json.loads(path.read_text()),'documents','patients'):
   text=row['text'].strip()
   if not text:raise ValueError(f'{path}: document {row["id"]} is empty')
   if documents.setdefault(row['id'],{'id':row['id'],'text':text,'title':row.get('title') or row.get('name')})['text']!=text:
    raise ValueError(f'document {row["id"]} appears with two texts')
 return documents

def load_questions(path,documents):
 out=[];seen=set()
 for row in rows_of(json.loads(path.read_text()),'questions'):
  doc=row.get('doc_id',row.get('patient_id'))
  if row['id'] in seen or ':' in row['id']:raise ValueError(f'question id {row["id"]!r} repeats or holds a colon')
  if doc not in documents:raise ValueError(f'question {row["id"]}: document {doc!r} is not among --documents')
  letters=mc_eval.gold_letters(row.get('gold_letters') or row['gold'],row['options'])   # raises on a gold that is not a letter of the question
  if row['gold'] not in letters:raise ValueError(f'question {row["id"]}: gold {row["gold"]!r}')
  seen.add(row['id']);out.append({'id':row['id'],'doc_id':doc,'question':row['question'],'options':list(row['options']),'gold':row['gold'],
   'gold_letters':sorted(letters),'hard':row.get('hard')})
 return out

def doc_ids(spec,documents,default=None):
 if spec is None:return default
 if spec=='all':return list(documents)
 path=Path(spec)
 if path.is_file():
  text=path.read_text()
  try:value=json.loads(text)
  except json.JSONDecodeError:value=[line.strip() for line in text.splitlines() if line.strip()]
  ids=[r['id'] if isinstance(r,dict) else r for r in rows_of(value,'documents','patients','ids')]
 else:ids=[x.strip() for x in spec.split(',') if x.strip()]
 missing=[i for i in ids if i not in documents]
 if missing or len(set(ids))!=len(ids) or not ids:raise ValueError(f'{spec}: {len(missing)} ids not among --documents (first: {missing[:3]}), or repeated / no ids')
 return ids

def mc_item(study,row,texts,docs,ratio,benchmark,*,question_first=True):
 """An uncontextual multiple-choice Q/A item (material_items' item form): the reader sees `texts` as one AC part each (the
 encoder's input is the bare text, exactly the reconstruction item's part) and answers `row` by calling submit_answer with the
 gold letter. One text: "Passage:" + rows (the reconstruction item's label); several: "Passages:" then "Passage k:" + rows
 (prepare_rag_arms.rows_content's passages layout). The in-context view holds the same texts inline."""
 from activation.ac_model import ActivationContextTrainingItem
 from activation.ac_model.ac_model_study import RAG_QA_SYSTEM_PROMPT
 from activation.ac_model.ac_model_utils import assistant_target_ids
 from activation.common.ac_parts import ac_part
 from activation.dataset.loaders.trajectory_utils import submit_answer_definition
 from run_qa import AC,answer_positions
 free='options' not in row   # a synthetic free-text item (prepare --train-form synth): the bare request, the answer text as the target
 question=mc_eval.clean(row['question']) if free else mc_eval.question_text(row['question'],row['options']);gold=row['answer'] if free else row['gold'];several=len(texts)>1
 head=(f'Task: {question}\n\n' if question_first else '')+('Passages:\n' if several else 'Passage:\n')   # the question first, as in material_items' prefix
 suffix='\n\n'+DOC_QA_INSTRUCTIONS.format(noun='passages' if several else 'passage')+f'\nQuestion: {question}'
 content=[];plain=''
 for i,text in enumerate(texts):
  label=(head if i==0 else '\n\n')+(f'Passage {i+1}:\n' if several else '')
  content+=[{'type':'text','text':label},ac_part([{'role':'user','content':text}],AC,ratio)];plain+=label+text
 content.append({'type':'text','text':suffix})
 # `think` (synth form, --think-first): the writer's short reasoning as plain text before the call, as the untrained reader reasons before it calls
 output=study.dialect.rendering.assistant_message(row.get('think') or '',[{'id':'answer','name':'submit_answer','arguments':{'answer':gold}}]);tools=[submit_answer_definition()]
 ids=assistant_target_ids(study.tokenizer,output,tools);system={'role':'system','content':RAG_QA_SYSTEM_PROMPT}
 return ActivationContextTrainingItem(item_id=row['id'],kind='rag_qa',in_context_messages=[system,{'role':'user','content':plain+suffix},output],
  ac_messages=[system,{'role':'user','content':content},deepcopy(output)],in_context_start=2,ac_start=2,assistant_token_ids=[ids],tools=tools,
  dataset_id=benchmark,doc_ids=list(docs),
  info={'answer':gold,'question':question,'answer_positions':answer_positions(study.tokenizer,ids,gold),'family':'rag_qa','encoder_conditioning':'summary',
   'encoder_input':'bare','ratio':ratio,'matched_full_context':True,'passages':len(texts),'chat_template_kwargs':{'enable_thinking':False},
   'multiple_choice':not free,'benchmark':benchmark,'question_id':row['id'],'question_doc':row['doc_id'],'gold_letters':row.get('gold_letters'),
   'options':row.get('options'),'hard':row.get('hard'),'question_first':question_first,'source':benchmark,'portable_id':row['id']})

def without_rows(study,item):
 """The item with no AC part at all: its `no_context` reference view (the text pieces of the user content, joined)."""
 from activation.ac_model.ac_model_utils import direct_parts
 bare=study.transform_for_eval_reference([item],'no_context')[0]
 if list(direct_parts(bare.ac_messages)):raise ValueError(f'{item.item_id}: the no-context view still carries an AC part')
 return replace(bare,item_id=item.item_id,info=dict(bare.info,encoder_input=None,passages=0,rows='none'))

def recon_item(study,doc,benchmark,split):
 return mix.make_item(study,'reconstruction',{'text':doc['text'],'doc_id':doc['id'],'source':benchmark},split)

OFFSETS={}   # document id -> reader-token character offsets (span_item's cue texts)

def span_item(study,doc,benchmark,ratio,tag,*,reads,span_tokens,cue_tokens,boundary_share,avoid=(),elsewhere=(),tries=64):
 """One "multispan" item on the whole document `doc`, redrawn (seed digest((SEED,tag,doc id,try))) until none of its reads (cue,
 target and end cue: reader-token ranges of the document) touches a range of `avoid` and none of its cues occurs in a text of
 `elsewhere` (the other documents shown next to it). info['read_extents'] holds its ranges."""
 for k in range(tries):
  got=study.generate_reconstruction_samples([{'text':doc['text'],'doc_id':doc['id'],'source':benchmark}],ratio=ratio,variant='multispan',
   seed=int(digest((SEED,tag,doc['id'],k))[:8],16),cue_tokens=tuple(cue_tokens),target_tokens=tuple(span_tokens),reads=tuple(reads),boundary_share=boundary_share)
  if not got:continue
  item=got[0];runs=item.info['reads']
  extents=[(r['start'],r['start']+r['cue_tokens']+r['tokens']+r['end_cue_tokens']) for r in runs]
  if any(lo<hi2 and lo2<hi for lo,hi in extents for lo2,hi2 in avoid):continue
  if elsewhere:
   text=doc['text']
   if doc['id'] not in OFFSETS:OFFSETS[doc['id']]=study.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
   offsets=OFFSETS[doc['id']];cut=lambda i:offsets[i][0] if i<len(offsets) else len(text)
   cues=[text[cut(a):cut(b)] for r in runs for a,b in ((r['start'],r['start']+r['cue_tokens']),
    (r['start']+r['cue_tokens']+r['tokens'],r['start']+r['cue_tokens']+r['tokens']+r['end_cue_tokens'])) if b>a]
   if any(not c.strip() or text.count(c)!=1 for c in cues):raise ValueError(f'{doc["id"]}: a recomputed cue is not the generator\'s (tokenizer offsets differ)')
   if any(c in other for c in cues for other in elsewhere):continue
  item.info.update(family='reconstruction',source=benchmark,document=doc['id'],read_extents=extents,span_try=k)
  return item
 raise ValueError(f'{doc["id"]}: no span item in {tries} draws (tag {tag}; {len(avoid)} reserved ranges; the document may be too short)')

def in_company(study,item,documents,company,ratio):
 """The one-part span item `item` shown among `company` (document ids, the item's own among them): one part per document in
 the panel's "Passages: / Passage k:" layout, the request naming the target passage. No in-context view."""
 from activation.common.ac_parts import ac_part
 from run_qa import AC
 target=item.info['document'];k=company.index(target);system,user,output=item.ac_messages
 label,part,tail=user['content']
 if label['text']!='Passage:\n' or not tail['text'].startswith('\n\n') or tail['text'].count('of the passage above')!=1:raise ValueError('unexpected span item layout')
 instruction=tail['text'].replace('of the passage above',f'of Passage {k+1} of the passages above')
 content=[]
 for i,d in enumerate(company):
  content+=[{'type':'text','text':('Passages:\n' if i==0 else '\n\n')+f'Passage {i+1}:\n'},
   part if d==target else ac_part([{'role':'user','content':documents[d]['text']}],AC,ratio)]
 content.append({'type':'text','text':instruction})
 return replace(item,in_context_messages=None,in_context_start=0,ac_messages=[system,{'role':'user','content':content},output],doc_ids=list(company),
  info=dict(item.info,passages=len(company),passage_index=k,company=list(company)))

SPAN_CONTEXT=None   # set before the fork: {'study','documents','train_ids','reserved','args','benchmark','span_opts'}

def span_chunk(task):
 """Spans-form training examples `first`..`first+count-1` of one document, each from its own seeds (so the result does not
 depend on the chunking or the worker count)."""
 i,first,count=task;c=SPAN_CONTEXT;a=c['args'];study=c['study'];documents=c['documents'];train_ids=c['train_ids'];out=[]
 for j in range(first,first+count):
  company=None
  if a.train_parts>1 and int(digest((SEED,'parts',i,j))[:8],16)%10_000<a.multi_part_share*10_000:
   rng=random.Random(digest((SEED,'company',i,j)));company=rng.sample([d for d in train_ids if d!=i],a.train_parts-1)+[i];rng.shuffle(company)
  item=span_item(study,documents[i],c['benchmark'],a.ratio,f'train:{j}',avoid=c['reserved'].get(i,()),elsewhere=[documents[d]['text'] for d in company or () if d!=i],**c['span_opts'])
  if company:item=in_company(study,item,documents,company,a.ratio)
  item.item_id=f'span:{i}#{j+1}';item.info.update(portable_id=item.item_id)
  out.append(item)
 return out

def dry_study():
 """Tokenizers only: an ActivationContextStudyGenerator without its harness (the item generators used here read its tokenizer,
 dialect and AC name, nothing else), and the side tokenizer."""
 from transformers import AutoTokenizer
 from activation.ac_model import ActivationContextStudyGenerator
 from activation.agent.agent_utils import ModelDialect
 from run_qa import AC
 study=ActivationContextStudyGenerator.__new__(ActivationContextStudyGenerator)
 study.ac_name=AC;study.tokenizer=AutoTokenizer.from_pretrained(READER);study.dialect=ModelDialect.for_tokenizer(study.tokenizer)
 study._token_cache={};study.harvest_report={}
 return study,AutoTokenizer.from_pretrained(SIDE)

def measure(study,side,item):
 """Dry-run sizes of one item: rows and encoder positions per part (the encoder rule: clamp(ceil(side tokens x ratio)); tokens +
 rows per side forward), reader tokens of the compressed and the in-context history (prepare_training_history), target tokens."""
 from activation.ac_model.ac_model_utils import direct_parts,prepare_training_history,tokenize_with_parts
 rows=[];encoder=[]
 for part in direct_parts(item.ac_messages):
  n=len(tokenize_with_parts(side,part['messages'],[],side.pad_token_id or 0,tools=part.get('tools'))[0])
  v=min(MAX_ROWS,max(MIN_ROWS,math.ceil(n*part['compression_target'])));rows.append(v);encoder.append(n)
 kwargs=item.info.get('chat_template_kwargs')
 history=lambda messages,start,lengths:len(prepare_training_history(study.tokenizer,messages,start,item.assistant_token_ids,lengths,tools=item.tools,chat_template_kwargs=kwargs)[0])
 compressed=history(item.ac_messages,item.ac_start,rows)
 full=history(item.in_context_messages,item.in_context_start,[]) if item.in_context_messages is not None else 0
 return {'rows':rows,'encoder_tokens':encoder,'encoder_positions':max([n+v for n,v in zip(encoder,rows)],default=0),'reader_compressed':compressed,
  'reader_full':full,'reader_tokens':max(compressed,full),'target_tokens':sum(map(len,item.assistant_token_ids))}

def loss_weights(n,answer,reasoning,answer_share,split=None):
 """Per-target-token loss multipliers (mean 1) of a synthetic item's target of `n` tokens. `answer`: the positions of the
 submit_answer argument; `reasoning`: the positions of the plain text before the call (none: a bare call). Without `split`
 (the rule until Oct 7) argument and reasoning together carry `answer_share`, equal per token, the call's wrapper tokens the
 rest. With `split` = (answer share, reasoning share) (user decision D6, Oct 7: .3 and .5, the wrapper .2) an item that HAS
 reasoning gives each of the three its own fixed share, so a one-token letter weighs as much as a long prose answer; an item
 without reasoning (a quote; a bare call) keeps `answer_share`."""
 answer=set(answer);reasoning=set(reasoning)-answer
 parts=[(answer,split[0]),(reasoning,split[1])] if split and reasoning else [(answer|reasoning,answer_share)]
 rest=n-sum(len(p) for p,_ in parts)
 if any(not p or max(p)>=n or min(p)<0 for p,_ in parts) or rest<1:raise ValueError(f'answer positions {sorted(answer)} / reasoning positions {len(reasoning)} of {n} target tokens')
 weights=[(1-sum(share for _,share in parts))*n/rest]*n
 for p,share in parts:
  for i in p:weights[i]=share*n/len(p)
 return weights

def spread(values):return {'min':min(values),'median':statistics.median(values),'max':max(values)} if values else None

def main():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--benchmark',required=True,help='group suffix: quality, longhealth, ...');p.add_argument('--documents',type=Path,nargs='+',required=True)
 p.add_argument('--questions',type=Path,required=True);p.add_argument('--train-docs',required=True);p.add_argument('--eval-docs');p.add_argument('--heldout-docs')
 p.add_argument('--ratio',type=float,required=True,help='rows per encoder token, e.g. 0.05 for 1/20');p.add_argument('--epochs',type=int,default=1)
 p.add_argument('--passes-per-epoch',type=int,default=1,help='copies of every training document inside one epoch')
 p.add_argument('--train-form',choices=('spans','whole','mc','synth'),default='spans',help='mc: a supervised diagnostic, the --train-questions asked on their own (training) document; synth: the question phase, synthetic study items about the training documents (which may be the evaluation documents)')
 p.add_argument('--train-questions',type=Path,help='mc form: the questions that train (none on an evaluation document)')
 p.add_argument('--train-items',type=Path,nargs='+',help='synth form: combined synthetic study items (synth_slice.py combine, JSONL): split "train" trains, "held" goes to the panel')
 p.add_argument('--styles',nargs='+',help='synth form: only these styles (prose choice quote)');p.add_argument('--free-target',choices=('reader','writer'),default='writer',help='synth form, prose-answer items: the writer\'s answer (it comes with a checked evidence sentence), or the frozen reader\'s full-document answer')
 p.add_argument('--recon-examples',type=int,default=0,help='synth form: multispan reconstruction examples mixed in (a multiple of 8)');p.add_argument('--synth-held',type=int,default=48,help='synth form: held synthetic items per style in the panel')
 p.add_argument('--answer-share',type=float,default=.8,help='synth form: share of a training example\'s loss carried by the tokens of the submit_answer argument (the letter, the prose answer, the quote); the 26 wrapper tokens of the call share the rest (0: plain token mean)')
 p.add_argument('--loss-split',type=float,nargs=2,metavar=('ANSWER','REASONING'),help='synth form: an item with reasoning before the call gives its submit_answer argument and its reasoning these fixed shares of the example\'s loss, the wrapper tokens the rest (e.g. .3 .5: user decision D6); items without reasoning keep --answer-share. Default: --answer-share for argument and reasoning together')
 p.add_argument('--synth-controls',type=int,default=0,help='synth form: control groups in the panel, N items per kind: synth_held_swapped/<kind> (the first N held items of the kind with ANOTHER document\'s rows), synth_trained/<kind> (N seeded TRAINING items with their own rows) and synth_trained_swapped/<kind> (the same N with another document\'s rows: answers that come without the rows were learned by heart)')
 p.add_argument('--schedule-total',type=int,default=0,help='planned updates of the WHOLE run for the learning-rate schedule (0: this packet\'s updates): a staged run trains several packets on one schedule, the decay (--landing) falling at the end of the last one')
 p.add_argument('--updates-offset',type=int,default=0,help='a LATER stage of a staged run: updates already done by the earlier stages. The starting pair must be the previous stage\'s final pair WITH its optimizer.pt (Adam moments and the update count are restored: no new warm-up; recipe updates_offset, training_config.resume_optimizer)')
 p.add_argument('--evidence-first',action='store_true',help='synth form, with --think-first (hill-climb arm, Oct 7): a question item\'s target starts with its checked evidence sentence(s), quoted, before the reasoning and the call; quote items are unchanged')
 p.add_argument('--think-first',action='store_true',help='synth form: the target is the item\'s short reasoning (plain text), then the submit_answer call; without it the call alone')
 p.add_argument('--synth-kind-cap',nargs='+',metavar='KIND=N',help='synth form: at most N training items of a kind (choice, prose, quote, multiquote = a quote row with `quotes` > 1), earliest rounds first, e.g. quote=16800 multiquote=8400; kinds not named are not capped')
 p.add_argument('--synth-cap',type=int,default=0,help='synth form: at most N training items (earliest rounds first, every document alike; 0: all)');p.add_argument('--examples-per-doc',type=int,default=16,help='spans form: fresh examples per document and pass')
 p.add_argument('--reads',type=int,nargs=2,default=(2,5),metavar=('LOW','HIGH'));p.add_argument('--span-tokens',type=int,nargs=2,default=(32,128),metavar=('LOW','HIGH'))
 p.add_argument('--total-examples',type=int,default=0,help='spans form: training examples of the run, a multiple of 8, shared among the documents by token count (0: --examples-per-doc x --passes-per-epoch x documents)')
 p.add_argument('--reserve',action='store_true',help='spans form: set the held stretches\' text aside (never a training target)')
 p.add_argument('--cue-tokens',type=int,nargs=2,default=(16,24),metavar=('LOW','HIGH'));p.add_argument('--boundary-share',type=float,default=.25)
 p.add_argument('--held-spans',type=int,default=2,help='reserved single-stretch items per training document (spans/heldout_text), and items per document of the other span groups')
 p.add_argument('--span-docs',type=int,default=50,help='documents per span panel group (seeded draw)')
 p.add_argument('--multi-part-share',type=float,default=0.,help='spans form: share of the training examples shown among --train-parts documents')
 p.add_argument('--train-parts',type=int,default=1,help='documents shown by a multi-part training example (5: LongHealth)')
 p.add_argument('--mc-panel',type=int,default=0,help='keep the first N questions (round-robin over the documents) in the panel; 0: every question')
 p.add_argument('--landing',type=int,default=500,help='linear decay of both rates to 1 %% over the last N updates (0: constant after the warm-up)')
 p.add_argument('--cycles',type=int,default=0,help='cyclical schedule: the schedule\'s updates (--schedule-total, else this packet\'s) are cut into N cycles of equal length, the last one taking the remainder; every cycle is a 50-update warm-up from 1 %% to the full rates, constant, then --landing as its linear decay back to 1 %% (0: one warm-up and one landing, as before)')
 p.add_argument('--cycle-reset-optimizer',action='store_true',help='with --cycles: the Adam moments start fresh at every cycle start after the first, so a cycle is a restart from the previous cycle\'s final pair (default: the moments carry over)')
 p.add_argument('--report-every',type=int,default=0,help='panel spacing in updates (overrides --reporting-interval)')
 p.add_argument('--lr-ac',type=float,default=2e-5);p.add_argument('--lr-reader',type=float,default=2e-5,help='0: the reader adapter is frozen')
 p.add_argument('--parts-per-item',type=int,default=1,help='documents per evaluation context, one part each (5: the LongHealth candidate)')
 p.add_argument('--no-question-first',action='store_true',help='the question only after the rows (default: also before, as the Q/A items of the starting pairs)')
 p.add_argument('--mc-swapped',type=int,default=128);p.add_argument('--mc-norows',type=int,default=0,help='0: every evaluation question; -1: no mc_norows group')
 p.add_argument('--recon-held',type=int,default=16);p.add_argument('--checkpoint-every',type=int,default=200)
 p.add_argument('--reporting-interval',type=float,default=.125,help='panel spacing, a fraction of each epoch')
 p.add_argument('--initial-pair',type=Path);p.add_argument('--output',type=Path);p.add_argument('--dry-run',action='store_true')
 p.add_argument('--workers',type=int,default=0,help='fork workers (0: every CPU of the allocation; 1: one process)')
 a=p.parse_args()
 if not 0<a.ratio<=1:p.error('--ratio is rows per encoder token, in (0, 1]')
 if a.epochs<1 or a.passes_per_epoch<1 or a.parts_per_item<1:p.error('--epochs, --passes-per-epoch and --parts-per-item are at least 1')
 if a.total_examples<0 or a.total_examples%8 or (a.total_examples and a.train_form not in ('spans','synth')):p.error('--total-examples is a nonnegative multiple of 8 (spans form)')
 if a.examples_per_doc<1 or a.held_spans<0 or a.span_docs<1 or a.mc_panel<0 or a.landing<0 or a.report_every<0:p.error('--examples-per-doc and --span-docs are at least 1; --held-spans, --mc-panel, --landing and --report-every nonnegative')
 if a.total_examples and a.landing and a.landing>a.total_examples//8-50:p.error(f'--landing {a.landing} does not fit after the 50-update warm-up in {a.total_examples//8} updates')
 if a.cycles<0 or (a.cycles and not a.landing) or (a.cycle_reset_optimizer and not a.cycles):p.error('--cycles is nonnegative and needs --landing > 0 (the decay of every cycle); --cycle-reset-optimizer goes with --cycles')
 if not 0<=a.multi_part_share<=1 or a.train_parts<1 or (a.multi_part_share>0)!=(a.train_parts>1) or (a.multi_part_share and a.train_form not in ('spans','synth')):p.error('--multi-part-share in (0, 1] goes with --train-parts > 1 and the spans form')
 if not (0<a.reads[0]<=a.reads[1] and 0<a.span_tokens[0]<=a.span_tokens[1] and 0<a.cue_tokens[0]<=a.cue_tokens[1] and 0<=a.boundary_share<=1):p.error('--reads, --span-tokens, --cue-tokens: 0 < LOW <= HIGH; --boundary-share in [0, 1]')
 if a.lr_ac<0 or a.lr_reader<0 or a.lr_ac==a.lr_reader==0:p.error('learning rates are nonnegative and not both 0')
 if not a.dry_run and (a.initial_pair is None or a.output is None):p.error('the build needs --initial-pair and --output (or --dry-run)')
 if ':' in a.benchmark or '/' in a.benchmark:p.error('--benchmark: no colon or slash')
 if (a.train_form=='mc')!=(a.train_questions is not None):p.error('--train-questions goes with --train-form mc')
 if (a.train_form=='synth')!=(a.train_items is not None) or a.recon_examples<0 or a.recon_examples%8 or (a.recon_examples and a.train_form!='synth'):p.error('--train-items goes with --train-form synth; --recon-examples is a multiple of 8 of that form')
 if not 0<=a.answer_share<1:p.error('--answer-share in [0, 1)')
 if a.loss_split and (min(a.loss_split)<=0 or sum(a.loss_split)>=1 or a.train_form!='synth' or not a.answer_share):p.error('--loss-split: two positive shares below 1 in sum, synth form, with a positive --answer-share (the rule of the items without reasoning)')
 if a.synth_controls<0 or a.schedule_total<0 or a.updates_offset<0 or (a.synth_controls and a.train_form!='synth') or (a.updates_offset and not a.schedule_total):p.error('--synth-controls (synth form), --schedule-total and --updates-offset are nonnegative; --updates-offset needs --schedule-total')
 if a.train_form=='synth' and a.total_examples:p.error('synth form: --recon-examples, not --total-examples')
 if a.train_form=='synth':a.total_examples=a.recon_examples   # the mixed-in reconstruction runs through the spans branch
 documents=load_documents(a.documents);questions=load_questions(a.questions,documents)
 asked=list(dict.fromkeys(q['doc_id'] for q in questions))
 train_ids=doc_ids(a.train_docs,documents);eval_ids=doc_ids(a.eval_docs,documents,asked)
 train_rows=[]
 if a.train_form=='mc':
  train_rows=load_questions(a.train_questions,documents);known={q['id'] for q in questions};texts_asked={(q['question'].strip(),tuple(q['options'])) for q in questions}
  if set(train_ids)&set(eval_ids):raise ValueError('mc form: a training document is an evaluation document')
  if any(q['doc_id'] not in set(train_ids) for q in train_rows) or any(q['id'] in known for q in train_rows):raise ValueError('mc form: a training question is off the training documents, or carries an evaluation question id')
  dropped=[q['id'] for q in train_rows if (q['question'].strip(),tuple(q['options'])) in texts_asked];train_rows=[q for q in train_rows if q['id'] not in set(dropped)]
  if dropped:print(f'NOTE: {len(dropped)} training questions dropped (same question and options as an evaluation question): {dropped[:5]}',flush=True)
 synth_rows=[];synth_held=[];synth_dropped={}
 if a.train_form=='synth':
  bench_words={};free_fields={'reader':'reader_answer','writer':'answer'}
  for q in questions:bench_words.setdefault(q['doc_id'],[]).append(set(mc_eval.clean(q['question']).casefold().split()))
  seen=set()
  for path in a.train_items:
   for line in path.read_text().splitlines():
    if not line.strip():continue
    r=json.loads(line)
    if r['id'] in seen or r['doc_id'] not in set(train_ids) or (a.styles and r['style'] not in a.styles):synth_dropped['duplicate_id_or_other_document_or_style']=synth_dropped.get('duplicate_id_or_other_document_or_style',0)+1;continue
    seen.add(r['id']);words=set(mc_eval.clean(r['question']).casefold().split())
    # Look-alike check: a synthetic question that is (nearly) a benchmark question of its document never trains.
    if any(len(words&w)/max(1,len(words|w))>=.8 for w in bench_words.get(r['doc_id'],())):synth_dropped['benchmark_look_alike']=synth_dropped.get('benchmark_look_alike',0)+1;continue
    evidence_first=lambda r:('The text says: "'+' ... '.join(x.strip() for x in (r.get('evidence'),r.get('evidence_b')) if x and x.strip())+'"\n') if a.evidence_first and r['style']!='quote' and (r.get('evidence') or '').strip() else ''
    row={'id':r['id'].replace(':','.').replace('/','_'),'doc_id':r['doc_id'],'question':r['question'],'style':r['style'],'round':r['round'],'hard':None,'think':(evidence_first(r)+(r.get('reasoning') or '').strip()) if a.think_first else '','kind':'multiquote' if r['style']=='quote' and (r.get('quotes') or 1)>1 else r['style'],'company':[x for x in (r.get('company') or ()) if x in set(train_ids) and x!=r['doc_id']]}
    if r['style']=='choice':row.update(options=list(r['options']),gold=r['gold'],gold_letters=sorted(mc_eval.gold_letters(r['gold'],r['options'])))
    else:
     answer=r['answer'] if r['style']=='quote' else r.get(free_fields[a.free_target])
     if not answer or not str(answer).strip():synth_dropped['no_target']=synth_dropped.get('no_target',0)+1;continue
     row.update(answer=str(answer).strip())
    (synth_held if r.get('split')=='held' else synth_rows).append(row)
  if a.synth_cap and len(synth_rows)>a.synth_cap:   # earliest rounds first, so a later packet with a larger cap holds this one's items plus new ones
   synth_rows=sorted(synth_rows,key=lambda r:(r['round'],digest((SEED,'synth_cap',r['id']))))[:a.synth_cap]
  if a.synth_kind_cap:
   caps={k:int(v) for k,v in (x.split('=',1) for x in a.synth_kind_cap)};kinds=sorted({r['kind'] for r in synth_rows})
   if set(caps)-set(kinds) or min(caps.values())<1:raise ValueError(f'--synth-kind-cap {caps}: the release holds the kinds {kinds}; caps are at least 1')
   synth_rows=[r for kind in kinds for r in sorted([r for r in synth_rows if r['kind']==kind],key=lambda r:(r['round'],digest((SEED,'synth_cap',r['id']))))[:caps.get(kind)]]
  if not synth_rows:raise ValueError('synth form: no training item left')
  print(f'synth items: {len(synth_rows)} train {dict(sorted(Counter(r["kind"] for r in synth_rows).items()))}, {len(synth_held)} held, dropped {synth_dropped}',flush=True)
 trained=set(train_ids);heldout_ids=doc_ids(a.heldout_docs,documents,[i for i in eval_ids if i not in trained])
 if trained&set(heldout_ids):raise ValueError('a --heldout-docs document trains')
 if len(eval_ids)%a.parts_per_item:raise ValueError(f'{len(eval_ids)} evaluation documents do not fill contexts of {a.parts_per_item}')
 contexts=[eval_ids[i:i+a.parts_per_item] for i in range(0,len(eval_ids),a.parts_per_item)];context_of={d:k for k,c in enumerate(contexts) for d in c}
 by_doc={d:[q for q in questions if q['doc_id']==d] for d in eval_ids}
 if any(not v for v in by_doc.values()):raise ValueError(f'evaluation documents without questions: {[d for d,v in by_doc.items() if not v][:3]}')
 # Round-robin over the documents: a prefix of the group covers every document.
 asked_rows=[by_doc[d][r] for r in range(max(map(len,by_doc.values()))) for d in eval_ids if r<len(by_doc[d])]
 if a.mc_panel:asked_rows=asked_rows[:a.mc_panel]
 drawn=lambda name,rows,count,key=lambda r:r['id']:(lambda keep:[r for r in rows if key(r) in keep])(set(sorted(map(key,rows),key=lambda i:digest((SEED,name,i)))[:count]))
 swapped_rows=drawn('mc_swapped',asked_rows,a.mc_swapped) if len(contexts)>1 else []
 norows_rows=[] if a.mc_norows<0 else drawn('mc_norows',asked_rows,a.mc_norows) if a.mc_norows else asked_rows
 recon_train=drawn('recon_train',[documents[i] for i in train_ids],a.recon_held);recon_heldout=drawn('recon_heldout',[documents[i] for i in heldout_ids],a.recon_held)
 B=a.benchmark;question_first=not a.no_question_first
 if a.parts_per_item>1 and a.train_parts!=a.parts_per_item:
  print(f'NOTE: the panel contexts hold {a.parts_per_item} parts; training examples show {a.train_parts} (share {a.multi_part_share})',flush=True)
 if a.train_parts>len(train_ids):raise ValueError('--train-parts exceeds the training documents')
 mix.RAG_VIEW,mix.RATIO,mix.BARE_BLIND='summary',a.ratio,True   # every training and held item: uncontextual, bare, this ratio
 from activation.ac_model import ActivationContextTrainingConfig
 if a.dry_run:
  study,side=dry_study();h=pair=None;workers=a.workers or len(os.sched_getaffinity(0))
  if workers>1:os.environ['TOKENIZERS_PARALLELISM']='false'
 else:
  out=a.output;out.mkdir(parents=True,exist_ok=True);panel_dir=out/'task_panel'
  if (out/'prepared.json').exists() or (panel_dir/'panel.json').exists():raise FileExistsError('Packet or panel already committed; use a new output')
  panel_dir.mkdir(exist_ok=True)
  from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer
  from run_qa import harness,AC
  workers=a.workers or len(os.sched_getaffinity(0))
  if workers>1:
   import torch
   if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():raise ValueError('Parallel preparation needs CUDA_VISIBLE_DEVICES= (a CPU-only job); use --workers 1 otherwise')
   os.environ['TOKENIZERS_PARALLELISM']='false'
  pair=json.loads(a.initial_pair.read_text())
  if a.updates_offset:   # a later stage: the starting pair carries the earlier stages' optimizer (a missing or foreign one would silently restart the schedule)
   import torch as _torch
   saved=Path(pair['ac_checkpoint'])/'optimizer.pt'
   if not saved.is_file():raise FileNotFoundError(f'--updates-offset: {saved} is missing (keep the previous stage\'s final pair WITH optimizer.pt)')
   done=int(_torch.load(saved,map_location='cpu',weights_only=False)['updates_done'])
   if done!=a.updates_offset:raise ValueError(f'--updates-offset {a.updates_offset}: the pair\'s optimizer.pt has done {done} updates')
  h=harness(pair['ac_checkpoint'],pair['reader_checkpoint']);study=ActivationContextStudyGenerator(h,AC)
  if workers>1:torch.set_num_threads(1)   # after harness(), which sets 4: the forked workers inherit this
 stages={}
 # Held span items first: their token ranges are reserved (no training example reads them).
 spans=a.train_form=='spans' or (a.train_form=='synth' and a.recon_examples>0);span_opts={'reads':a.reads,'span_tokens':a.span_tokens,'cue_tokens':a.cue_tokens,'boundary_share':a.boundary_share}
 one_read={**span_opts,'reads':(1,1),'boundary_share':0.}
 def stretches(doc,tag,count,avoid):
  """`count` single-stretch items of `doc`, apart from each other and from `avoid`."""
  made=[];taken=list(avoid)
  for j in range(count):
   item=span_item(study,doc,B,a.ratio,f'{tag}:{j}',avoid=taken,**one_read);taken+=item.info['read_extents'];made.append(item)
  return made
 requests=lambda doc,tag,count:[span_item(study,doc,B,a.ratio,f'{tag}:{j}',**span_opts) for j in range(count)]   # fresh examples of the training form
 reserved={};span_groups={}
 if spans and a.held_spans:
  for doc in drawn('span_docs',[documents[i] for i in train_ids],a.span_docs):
   if not a.reserve:span_groups.setdefault('spans/heldout_requests',[]).extend(requests(doc,'heldout_requests',a.held_spans));continue
   held_items=stretches(doc,'heldout_text',a.held_spans,());reserved[doc['id']]=[e for item in held_items for e in item.info['read_extents']]
   span_groups.setdefault('spans/heldout_text',[]).extend(held_items)
   span_groups.setdefault('spans/trained_text',[]).extend(stretches(doc,'trained_text',a.held_spans,reserved[doc['id']]))
  for doc in drawn('span_heldout_docs',[documents[i] for i in heldout_ids],a.span_docs):
   span_groups.setdefault('spans/heldout_docs',[]).extend(stretches(doc,'heldout_docs',a.held_spans,()) if a.reserve else requests(doc,'heldout_docs',a.held_spans))
 # Training: every training document, --passes-per-epoch times (spans: --examples-per-doc fresh examples each time).
 global SPAN_CONTEXT
 output=[];shares={}
 if spans:
  # One long pass: the run's examples shared among the documents by token count (largest remainders), in chunks of 16 per task.
  wanted=a.total_examples or a.examples_per_doc*a.passes_per_epoch*len(train_ids)
  tokens={i:len(study.tokenizer(documents[i]['text'],add_special_tokens=False)['input_ids']) for i in train_ids};whole=sum(tokens.values())
  exact={i:wanted*tokens[i]/whole for i in train_ids};shares={i:int(exact[i]) for i in train_ids}
  for i in sorted(train_ids,key=lambda i:(shares[i]-exact[i],digest((SEED,'share',i))))[:wanted-sum(shares.values())]:shares[i]+=1
  if sum(shares.values())!=wanted or min(shares.values())<1:raise ValueError(f'{wanted} examples do not give every one of {len(train_ids)} documents at least one')
  tasks=[(i,first,min(16,shares[i]-first)) for i in train_ids for first in range(0,shares[i],16)]
 elif a.train_form=='mc':
  # Every training question, --passes-per-epoch times, cut to whole updates of 8.
  tasks=[(k,row) for k in range(a.passes_per_epoch) for row in train_rows];tasks=tasks[:len(tasks)//8*8]
 elif a.train_form=='synth':tasks=[]   # no reconstruction mixed in: the synthetic items are appended below
 else:tasks=[(k,i) for k in range(a.passes_per_epoch) for i in train_ids]
 clock=StageClock('training items',len(tasks))
 if spans:
  SPAN_CONTEXT={'study':study,'documents':documents,'train_ids':train_ids,'reserved':reserved,'args':a,'benchmark':B,'span_opts':span_opts}
  try:
   with ForkPool(workers) as pool:
    for done,made in enumerate(pool.map(span_chunk,tasks),1):output.extend(made);clock(done)
  finally:SPAN_CONTEXT=None
 elif a.train_form=='mc':
  for done,(k,row) in enumerate(tasks,1):
   item=mc_item(study,row,[documents[row['doc_id']]['text']],[row['doc_id']],a.ratio,B,question_first=not a.no_question_first)
   item.item_id=f'mcq:{row["id"]}'+(f'#pass{k+1}' if k else '');item.info.update(document=row['doc_id'],epoch_pass=k+1,portable_id=item.item_id);output.append(item);clock(done)
 else:
  for k,i in tasks:
   item=recon_item(study,documents[i],B,'train');item.info.update(document=i,epoch_pass=k+1)
   if k:item.item_id+=f'#pass{k+1}'
   output.append(item)
 def synth_item(row,tag,wrong=False):
  """A synthetic study item on its own document, or (LongHealth: --train-parts, share --multi-part-share) among other training documents.
  `wrong` (a panel control): the rows of OTHER training documents only (one part: the next training document; --train-parts P: P seeded others)."""
  d=row['doc_id'];company=[d]
  if wrong:
   others=[x for x in train_ids if x!=d]
   if not others:raise ValueError('a wrong-rows control needs a second training document')
   company=random.Random(digest((SEED,'synth_wrong',tag,row['id']))).sample(others,a.train_parts) if a.train_parts>1 else [train_ids[(train_ids.index(d)+1)%len(train_ids)] if d in train_ids else others[0]]
  elif a.train_parts>1 and int(digest((SEED,'synth_parts',tag,row['id']))[:8],16)%10_000<a.multi_part_share*10_000:
   rng=random.Random(digest((SEED,'synth_company',tag,row['id'])));shown=row['company'][:a.train_parts-1]   # the other records the writer saw (a 'multi' item) are among the documents shown
   company=shown+rng.sample([x for x in train_ids if x!=d and x not in shown],a.train_parts-1-len(shown))+[d];rng.shuffle(company)
  item=mc_item(study,row,[documents[x]['text'] for x in company],company,a.ratio,B,question_first=not a.no_question_first)
  item.info.update(document=d,synth_style=row['style'],synth_round=row['round'],company=company)
  if row['kind']!=row['style']:item.info.update(synth_kind=row['kind'])
  if wrong:item.info.update(wrong_rows=True)
  return item
 if a.train_form=='synth':
  keep=len(synth_rows)-(len(output)+len(synth_rows))%8   # whole updates of 8
  for row in synth_rows[:keep]:
   item=synth_item(row,'train');item.item_id=f'syn:{row["id"]}';item.info.update(portable_id=item.item_id);output.append(item)
   if a.answer_share>0:   # the answer is the argument of a call whose 26 wrapper tokens carry no information (a letter is 1 token of 27): give the content (the argument, and the reasoning before the call) a fixed share of the example's loss (weights of mean 1)
    ids=list(item.assistant_token_ids[0]);n=len(ids)
    before=range(ids.index(study.tokenizer.convert_tokens_to_ids('<tool_call>'))) if row.get('think') else ()   # the reasoning before the call is content too
    try:item.info['target_weights']=loss_weights(n,item.info['answer_positions'],before,a.answer_share,a.loss_split)
    except ValueError as error:raise ValueError(f'{item.item_id}: {error}') from None
    item.info['loss_positions']={'answer':len(set(item.info['answer_positions'])),'reasoning':len(before)}
 if len({i.item_id for i in output})!=len(output):raise ValueError('Duplicate training item ids (two documents with one id and text?)')
 total=len(output);updates=math.ceil(total/8)
 planned=updates*a.epochs;whole=a.schedule_total or planned   # `whole`: the updates the learning-rate schedule spans (a staged run: every stage's)
 if a.updates_offset+planned>whole:raise ValueError(f'--schedule-total {whole} is less than the {a.updates_offset} earlier updates plus this packet\'s {planned}')
 if a.landing and a.landing>whole-50:raise ValueError(f'--landing {a.landing} does not fit after the 50-update warm-up in {whole} updates')
 schedule={'lr_schedule':'cooldown','cooldown_updates':a.landing,'cooldown_shape':'linear','lr_final_fraction':.01} if a.landing else {'lr_schedule':'constant'}
 if a.cycles:   # cycles start at multiples of cycle_updates on the schedule's clock (a later stage of a staged run keeps counting); the last cycle runs to `whole`
  cycle=whole//a.cycles
  if a.landing+50>cycle:raise ValueError(f'--cycles {a.cycles}: a cycle of {cycle} updates ({whole} in {a.cycles}) does not hold the 50-update warm-up and --landing {a.landing}')
  schedule={**schedule,'lr_schedule':'cyclical','cycle_updates':cycle,'cycle_reset_optimizer':a.cycle_reset_optimizer}
 reporting=min(1.,math.floor(1e9*a.report_every/updates)/1e9) if a.report_every else a.reporting_interval   # floored: reporting_boundaries re-reads the float as a fraction and a rounded-up one lands an update late
 config=ActivationContextTrainingConfig(loss_kind='sft',learning_rate_ac=a.lr_ac,learning_rate_target_lora=a.lr_reader,examples_per_update=8,num_epochs=a.epochs,completion_samples=0,
  teacher_cache_block_examples=32,logits_chunk_tokens=64,gradient_checkpointing_min_tokens=0,seed=SEED,deterministic=False,warmup_updates=50,**schedule,
  schedule_total_updates=whole,checkpoint_every_updates=a.checkpoint_every,reporting_interval=reporting,shuffle_examples=True,**({'resume_optimizer':True} if a.updates_offset else {}))
 # The task panel.
 panel=[];overlap={}
 def add(group,item,ident,shown,target):
  item.item_id=f'panel:{group}:{ident}';item.info.update(panel_group=group,panel=True)
  stats={'documents':len(shown),'documents_in_training':sum(d in trained for d in shown),'target_document':target,'target_document_in_training':target in trained}
  item.info.update(exposure=stats,reconstruction_overlap=stats['target_document_in_training'])   # the question's (or the reproduced) document trains
  overlap[item.item_id]={'group':group,**stats};panel.append(item)
 texts=lambda context:[documents[d]['text'] for d in context]
 own={}
 for row in asked_rows:
  context=contexts[context_of[row['doc_id']]]
  own[row['id']]=mc_item(study,row,texts(context),context,a.ratio,B,question_first=question_first);add(f'mc/{B}',own[row['id']],row['id'],context,row['doc_id'])
 for row in swapped_rows:
  k=(context_of[row['doc_id']]+1)%len(contexts);other=contexts[k]
  item=mc_item(study,row,texts(other),other,a.ratio,B,question_first=question_first);item.info.update(swapped_evidence_of=k)
  add(f'mc_swapped/{B}',item,row['id'],other,row['doc_id'])
 for row in norows_rows:add(f'mc_norows/{B}',without_rows(study,own[row['id']]),row['id'],[],row['doc_id'])
 for group,made in span_groups.items():
  for n,item in enumerate(made):
   item.info.update(portable_id=item.item_id);add(group,item,f'{item.info["document"]}#{n}',[item.info['document']],item.info['document'])
 for style in sorted({r['kind'] for r in synth_held}):   # kind = the style; a multi-quote item (release field `quotes` > 1) is its own kind `multiquote`
  chosen=drawn(f'synth_held/{style}',[r for r in synth_held if r['kind']==style],a.synth_held)
  for row in chosen:
   item=synth_item(row,'held');add(f'synth_held/{style}',item,row['id'],item.info['company'],row['doc_id'])
  for row in chosen[:a.synth_controls]:   # the same held items with another document's rows (compare on the same ids)
   item=synth_item(row,'held',wrong=True);add(f'synth_held_swapped/{style}',item,row['id'],item.info['company'],row['doc_id'])
 if a.train_form=='synth' and a.synth_controls:
  trained_rows=synth_rows[:keep]
  for style in sorted({r['kind'] for r in trained_rows}):
   for row in drawn(f'synth_trained/{style}',[r for r in trained_rows if r['kind']==style],a.synth_controls):
    item=synth_item(row,'train');add(f'synth_trained/{style}',item,row['id'],item.info['company'],row['doc_id'])   # the item as it trains
    item=synth_item(row,'train',wrong=True);add(f'synth_trained_swapped/{style}',item,row['id'],item.info['company'],row['doc_id'])
 for group,chosen in (('reconstruction/train_docs',recon_train),('reconstruction/heldout_docs',recon_heldout)):
  for doc in chosen:add(group,recon_item(study,doc,B,'held'),doc['id'],[doc['id']],doc['id'])
 if any(i.info.get('ratio')!=a.ratio for i in panel+output):raise ValueError('panel or training ratio mismatch')
 if len({i.item_id for i in panel})!=len(panel):raise ValueError('Duplicate task-panel item ids')
 if a.train_form not in ('mc','synth') and any('question' in i.info for i in output):raise ValueError('a question trains')
 if a.train_form=='synth':
  bench={mc_eval.clean(q['question']).casefold() for q in questions};held_ids={i.info['question_id'] for i in panel if i.info['panel_group'].startswith('synth_held/')}
  if any(mc_eval.clean(i.info['question'].split('\n\n'+mc_eval.OPTIONS_HEADER)[0]).casefold() in bench for i in output if 'question' in i.info) or any(i.info.get('question_id') in held_ids for i in output):raise ValueError('a benchmark question, or a held synthetic item, trains')
 if a.train_form=='mc' and ({i.info['question'] for i in output}&{i.info['question'] for i in panel if 'question' in i.info} or {i.info['question_doc'] for i in output}&set(eval_ids)):raise ValueError('a panel question, or a question of an evaluation document, trains')
 groups=Counter(i.info['panel_group'] for i in panel);actual=Counter(mix.panel_group(i) for i in output)
 summary={}
 for stats in overlap.values():
  s=summary.setdefault(stats['group'],Counter());s['items']+=1
  for k in ('documents','documents_in_training','target_document_in_training'):s[k]+=int(stats[k])
 summary={g:dict(v) for g,v in sorted(summary.items())}
 inputs={'documents':{str(x):file_sha(x) for x in a.documents},'questions':{str(a.questions):file_sha(a.questions)}}
 snapshots={'documents':digest([[d,documents[d]['text']] for d in sorted(set(train_ids)|set(eval_ids)|set(heldout_ids))]),'questions':digest(questions)}
 arm={'benchmark':B,'lr_ac':a.lr_ac,'lr_reader':a.lr_reader,'reader':'frozen' if a.lr_reader==0 else 'trained','passes_per_epoch':a.passes_per_epoch,
  'train_documents':len(train_ids),'eval_documents':len(eval_ids),'eval_documents_in_training':len(trained&set(eval_ids)),'parts_per_item':a.parts_per_item,
  'question_first':question_first,'mc_panel':a.mc_panel or 'all',
  'train_item_form':('QUESTION PHASE: %d synthetic study items (%s; free-text target: %s; reasoning before the call: '+str(bool(a.think_first))+'; content share of the loss: '+str(a.answer_share)+'; dropped %s), each on its own document (share %g among %d documents), plus %d multispan reconstruction examples')%(len(output)-a.recon_examples,dict(sorted(Counter(r['style'] for r in synth_rows).items())),a.free_target,synth_dropped,a.multi_part_share,a.train_parts,a.recon_examples) if a.train_form=='synth' else
   'SUPERVISED DIAGNOSTIC: %d real multiple-choice questions of the training documents, each on its own document, %d passes'%(len(train_rows),a.passes_per_epoch) if a.train_form=='mc' else
   'whole-document reconstruction, one part per document' if not spans else
   'multispan on the whole document, one pass of fresh examples shared by document tokens (%s per document): reads %s of %s tokens, cues %s, boundary share %g; held: %s; multi-part share %g of %d parts'
   %(spread(list(shares.values())),tuple(a.reads),tuple(a.span_tokens),tuple(a.cue_tokens),a.boundary_share,
     ('%d reserved stretches per document on %d documents'%(a.held_spans,len(reserved))) if a.reserve else '%d fresh requests per document, never asked in training'%a.held_spans,a.multi_part_share,a.train_parts),
  'schedule':{**schedule,'warmup_updates':50,'updates':planned,'report_every':a.report_every or None,**({'cycles':'%d cycles of %d updates (the last one %d), starting at updates %s of %d: each a 50-update warm-up from 1 %% to the full rates, constant, then a linear decay to 1 %% over its last %d updates; the Adam moments %s'%(a.cycles,cycle,whole-(a.cycles-1)*cycle,', '.join(str(k*cycle+1) for k in range(a.cycles)),whole,a.landing,'start fresh at every cycle start after the first' if a.cycle_reset_optimizer else 'carry over from cycle to cycle')} if a.cycles else {}),**({'schedule_total_updates':whole,'updates_offset':a.updates_offset,'stage':('updates %d-%d of %d on one schedule ('+('the cycles are cut on the whole schedule' if a.cycles else 'the decay falls in the last stage')+'); a later stage restores the optimizer of its starting pair')%(a.updates_offset+1,a.updates_offset+planned,whole)} if whole!=planned else {})},
  **({'loss_split':{'answer':a.loss_split[0],'reasoning':a.loss_split[1],'wrapper':round(1-sum(a.loss_split),6),'items_without_reasoning':'--answer-share'}} if a.loss_split else {}),**({'synth_controls':a.synth_controls} if a.synth_controls else {})}
 if spans and any(lo<hi2 and lo2<hi for item in output for lo,hi in item.info.get('read_extents',()) for lo2,hi2 in reserved.get(item.info['document'],())):
  raise ValueError('a training example reads a reserved stretch')
 if spans:   # a held request is one the training never asks (the same document and the same stretches)
  asked={(item.info['document'],tuple(sorted(item.info['read_extents']))) for item in output if 'read_extents' in item.info}
  same=[item.item_id for made in span_groups.values() for item in made if (item.info['document'],tuple(sorted(item.info['read_extents']))) in asked]
  if same:raise ValueError(f'{len(same)} held requests are also training requests (first: {same[:3]}); change --held-spans or the seed tag')
 if a.dry_run:
  from activation.ac_model.ac_model_training import reporting_boundaries
  first_pass=output[:len(train_ids)] if not spans else output if len(output)<=240 else random.Random(SEED).sample(output,240)   # whole form: the later passes are the same items; spans: a seeded sample
  clock=StageClock('dry-run sizes',len(first_pass)+len(panel));sizes={};done=0
  for item in first_pass+panel:
   sizes[item.item_id]=measure(study,side,item);done+=1;clock(done)
  def table(items,scale=1):
   m=[sizes[i.item_id] for i in items]
   over=[i.item_id for i,x in zip(items,m) if x['reader_tokens']>READER_LIMIT or x['encoder_positions']>ENCODER_LIMIT]
   capped=sum(v==MAX_ROWS for x in m for v in x['rows'])
   return {'items':len(items)*scale,'target_tokens':sum(x['target_tokens'] for x in m)*scale,'encoder_tokens':sum(sum(x['encoder_tokens']) for x in m)*scale,
    'rows_per_part':spread([v for x in m for v in x['rows']]),'rows_per_item':spread([sum(x['rows']) for x in m]),
    'reader_tokens_compressed':spread([x['reader_compressed'] for x in m]),'reader_tokens_in_context':spread([x['reader_full'] for x in m]),
    'largest_reader_tokens':max(x['reader_tokens'] for x in m),'largest_encoder_positions':max(x['encoder_positions'] for x in m),
    'over_limits':len(over),'over_limits_first':over[:3],'parts_at_row_cap':capped}
  scale=len(output)/len(first_pass);train_table=table(first_pass,scale)
  multi=[sizes[i.item_id] for i in first_pass if i.info.get('passages',1)>1]
  train_table.update(sampled_items=len(first_pass),multi_part_items_sampled=len(multi),multi_part_reader_tokens=spread([x['reader_compressed'] for x in multi]),
   multi_part_encoder_tokens=spread([sum(x['encoder_tokens']) for x in multi]),target_tokens_per_item=spread([sizes[i.item_id]['target_tokens'] for i in first_pass]))
  panel_tables={g:table([i for i in panel if i.info['panel_group']==g]) for g in sorted(groups)}
  evaluations=len(reporting_boundaries(updates,reporting))
  loss_table={}   # per kind of synthetic training item: mean share of the example's loss on the argument / the reasoning / the wrapper, and their token counts
  for item in output:
   w=item.info.get('target_weights')
   if not w:continue
   n=len(w);ans=set(item.info['answer_positions']);think=set(range(item.info['loss_positions']['reasoning']));t=loss_table.setdefault(item.info.get('synth_kind') or item.info.get('synth_style'),Counter())
   t['items']+=1;t['answer']+=sum(w[i] for i in ans)/n;t['reasoning']+=sum(w[i] for i in think)/n;t['wrapper']+=sum(w[i] for i in range(n) if i not in ans and i not in think)/n
   t['answer_tokens']+=len(ans);t['reasoning_tokens']+=len(think);t['target_tokens']+=n
  loss_table={k:{'items':v['items'],**{x:round(v[x]/v['items'],4) for x in ('answer','reasoning','wrapper','answer_tokens','reasoning_tokens','target_tokens')}} for k,v in sorted(loss_table.items())}
  seconds=sum(mix.step_seconds({'reader_tokens':sizes[i.item_id]['reader_compressed'],'encoder_tokens':sum(sizes[i.item_id]['encoder_tokens'])}) for i in first_pass)*scale
  report={'dry_run':True,'benchmark':B,'ratio':a.ratio,'arm':arm,'train':{'reconstruction/flat':train_table},'panel':panel_tables,'panel_items':len(panel),
   'updates_per_epoch':updates,'epochs':a.epochs,'total_updates':updates*a.epochs,'document_passes':a.passes_per_epoch*a.epochs,
   'reserved_tokens_per_document':spread([sum(hi-lo for lo,hi in v) for v in reserved.values()]),'examples_per_document':spread(list(shares.values())),
   'target_coverage':(lambda t:None if not spans else round(t/sum(tokens.values()),2))(train_table['target_tokens']),
   'span_draws':dict(sorted(Counter(i.info.get('span_try',0) for i in output).items())) if spans else None,
   'target_tokens_per_epoch':train_table['target_tokens'],'target_tokens_total':train_table['target_tokens']*a.epochs,
   'limits':{'reader_tokens':READER_LIMIT,'encoder_positions':ENCODER_LIMIT,'rows_per_part':MAX_ROWS},
   'largest_reader_tokens':max([train_table['largest_reader_tokens']]+[t['largest_reader_tokens'] for t in panel_tables.values()]),
   'largest_encoder_positions':max([train_table['largest_encoder_positions']]+[t['largest_encoder_positions'] for t in panel_tables.values()]),
   'items_over_limits':train_table['over_limits']+sum(t['over_limits'] for t in panel_tables.values()),
   'panel_evaluations':{'per_epoch':evaluations,'total':evaluations*a.epochs,'encoder_tokens_each':sum(t['encoder_tokens'] for t in panel_tables.values())},
   'saved_pairs':{'per_epoch_end':a.epochs,'mid_epoch':((updates-1)//a.checkpoint_every if a.checkpoint_every else 0)*a.epochs},
   'estimated_train_hours_2_gpus':round(seconds*a.epochs/3600,2),
   'estimate_note':'mix.step_seconds, fitted on other packets on 2 x H200; training updates only (no panel evaluation, no checkpoint); not measured for these items',
   'exposure_by_group':summary,'training_config':{k:asdict(config)[k] for k in ('learning_rate_ac','learning_rate_target_lora','num_epochs','schedule_total_updates','checkpoint_every_updates','reporting_interval','examples_per_update','lr_schedule','cooldown_updates','lr_final_fraction','warmup_updates')+(('cycle_updates','cycle_reset_optimizer') if a.cycles else ())},
   'loss_shares':loss_table,'tokenizers':{'reader':READER,'side':SIDE},'not_checked':'trainer.build_example validations, execution records, the pair'}
  if a.output:a.output.mkdir(parents=True,exist_ok=True);write(a.output/'dry_run.json',report)
  print(json.dumps(report,indent=1),flush=True);return
 # --- the build (a node with the model weights) ---
 from activation.ac_model.ac_model_utils import validate_part_capacity
 from run_trajectory_tasks import compact_example
 trainer=ActivationContextTrainer(h,config);failures=[];kept=[]
 clock=StageClock('examples',len(output));done=0
 mix.BUILD_CONTEXT={'trainer':trainer,'study':study,'scan':True,'output':output}
 chunks=[range(i,min(i+CHUNK,len(output))) for i in range(0,len(output),CHUNK)]
 with ForkPool(workers) as pool:
  for results in pool.map(mix.example_chunk,chunks):
   for value,failure in results:
    if failure:failures.append(failure);continue
    item,example,_=value;kept.append((item,example))
   done+=len(results);clock(done)
 mix.BUILD_CONTEXT=None;stages['examples']=clock.seconds()
 if failures:raise ValueError(f'{len(failures)} training documents failed to build (every document must train): {failures[:3]}')
 train_items=[i for i,_ in kept];examples=[e for _,e in kept]
 # Panel examples, as prepare_rag_arms builds them (a builder without the 72,000 guard, so the check below names the item).
 builder=ActivationContextTrainer(h,ActivationContextTrainingConfig(loss_kind='sft',max_example_tokens=1_000_000))
 panel_examples=[];clock=StageClock('panel',len(panel))
 for item in panel:
  example=builder.build_example(study.ac_model,item)
  if mix.reader_tokens(builder,study,item,example)>READER_LIMIT:raise ValueError(f'{item.item_id}: panel reader history exceeds {READER_LIMIT} tokens')
  for request in example.part_requests:validate_part_capacity(study.ac_model,request,ENCODER_LIMIT)
  if mix.nested_parts(item):raise ValueError('Panel item nests an AC part (depth-1 rule)')
  item.info.update(target_tokens=len(example.target_ids),reader_tokens=len(example.student_ids))
  panel_examples.append(compact_example(example));clock(len(panel_examples))
 stages['panel']=clock.seconds()
 held={'rule':'evaluation only: every question of the evaluation documents (mc), a seeded subsample with the next context\'s rows (mc_swapped), '
   'the questions without rows (mc_norows), whole-document reconstruction of training and never-trained documents; '+('the training questions are other documents\' questions' if a.train_form=='mc' else 'no question trains'),
  'eval_docs':eval_ids,'contexts':contexts,'train_docs':train_ids,'heldout_docs':heldout_ids,
  **{g:{'ids':[i.item_id.split(':',2)[2] for i in panel if i.info['panel_group']==g]} for g in sorted(groups)}}
 held[f'mc/{B}']['questions']=[i.info['question'] for i in panel if i.info['panel_group']==f'mc/{B}']
 write(panel_dir/'held.json',held)
 dump(panel_dir/'panel.pkl',{'items':panel,'examples':panel_examples})
 write(panel_dir/'panel.json',{'sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json'),'items':len(panel),'groups':dict(sorted(groups.items())),
  'target_tokens':sum(i.info['target_tokens'] for i in panel),'reader_tokens':sum(i.info['reader_tokens'] for i in panel),
  'snapshots':snapshots,'initial_pair':pair,'seed':SEED,'ratio':a.ratio,'encoder_input':'bare','exposure_by_group':summary,'inputs':inputs,
  'scoring':'letters: rescore review_snapshot.py outputs with mc_eval.py --rescore (its exact_match / token_f1 are not valid here)'})
 write(out/'exposure.json',{'by_group':summary,'items':overlap,'train_documents':len(train_ids),'eval_documents_in_training':len(trained&set(eval_ids))})
 recipe={'seed':SEED,'epochs':a.epochs,'phase':('Supervised question diagnostic' if a.train_form=='mc' else 'Question phase on synthetic study items' if a.train_form=='synth' else 'Document corpus reconstruction')+' (%s) at 1/%g: %s'%(B,round(1/a.ratio,2),json.dumps(arm,sort_keys=True)),'ratio':a.ratio,'arm':arm,
  'encoder_view':'uncontextual (the encoder never sees the question); the bare document, as in reconstruction',
  'counts':{'rag_qa':total-a.recon_examples,'reconstruction':a.recon_examples} if a.train_form=='synth' else {'rag_qa' if a.train_form=='mc' else 'reconstruction':total},'actual_train':dict(actual),'unique_sources':dict(actual),'excluded':{},
  'target_tokens':dict(Counter({g:sum(i.info['target_tokens'] for i in train_items if mix.panel_group(i)==g) for g in actual})),
  'loss_shares':'natural: every example weighs 1','global_batch':8,
  'reconstruction':{'documents':len(train_ids),'passes_per_epoch':a.passes_per_epoch,'examples_per_epoch':total,'document_passes':a.passes_per_epoch*a.epochs,
   'form':a.train_form,'reserved_ranges':reserved,'examples_per_document':shares},
  'held':{'rule':'task_panel/held.json','panel':str(panel_dir.resolve()),'panel_sha256':file_sha(panel_dir/'panel.pkl'),'held_sha256':file_sha(panel_dir/'held.json')},
  'clean_split':'clean_* in panel rows excludes held items whose question (or reproduced) document is a training document',
  'headline_note':'the headline held NLL averages every panel item (mostly mc questions, whole submit_answer call): read the groups, and score letters with mc_eval.py',
  'failures':failures,'inputs':inputs,'initial_pair':pair,'snapshots':snapshots,'training_config':asdict(config)}
 if a.updates_offset:recipe['updates_offset']=a.updates_offset   # run_channel_mixture.train: the update clock of a requeued attempt (run_qa.resume_checkpoint)
 for item in train_items:item.weight=1.0
 write(out/'recipe.json',recipe)
 bundle_out={'train':train_items,'held':[]}
 dump(out/'items.pkl',bundle_out);dump(out/'prepared_examples.pkl',{**bundle_out,'examples':examples})
 write(out/'prepared.json',{'recipe_sha256':digest(recipe),'items_sha256':file_sha(out/'items.pkl'),'items_bytes':(out/'items.pkl').stat().st_size,
  'prepared_examples_sha256':file_sha(out/'prepared_examples.pkl'),'validated_capacity':{'reader_tokens':READER_LIMIT,'encoder_tokens':ENCODER_LIMIT},
  'training_examples':total,'held_examples':0,'one_pass':a.epochs==1 and a.passes_per_epoch==1,'preparation':{'workers':workers,'stage_seconds':stages}})
 print(json.dumps({'actual_train':dict(actual),'panel':dict(groups),'exposure_by_group':summary,'updates_per_epoch':updates,'epochs':a.epochs,'arm':arm}),flush=True)

if __name__=='__main__':main()
