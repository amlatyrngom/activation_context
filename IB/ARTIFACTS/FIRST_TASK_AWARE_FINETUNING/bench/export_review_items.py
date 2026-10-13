"""Export what the off-cluster review of a mixed run needs from its frozen run folder (MIXED_RUN_PLAN.md, 25% review): the fixed
learning sample every panel re-scores (the same seeded draw as `train`, checked against the run's panel rows), and how much of
each task-panel item's source text also occurs in training. CPU only; reads the run and panel, writes only --output.

A source text is any string of 500+ characters inside an activation-context part (a passage, a trajectory message). Held
questions are excluded from training by question, not by source, so a panel item can share its source with training items.

  python export_review_items.py --run RUN --task-panel PANEL --output OUT_DIR
"""
import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

from run_channel_mixture import SEED,panel_group,write

MIN_SOURCE_CHARS=500


def source_texts(messages):
 """Every string of MIN_SOURCE_CHARS+ characters inside activation-context parts, in order."""
 from activation.common.ac_parts import is_ac_part
 found=[]
 def walk(value,inside):
  if isinstance(value,str):
   if inside and len(value)>=MIN_SOURCE_CHARS:found.append(value)
  elif isinstance(value,dict):
   inside=inside or is_ac_part(value)
   for child in value.values():walk(child,inside)
  elif isinstance(value,list):
   for child in value:walk(child,inside)
 walk(messages,False);return found


def digest(text):return hashlib.sha256(text.encode()).hexdigest()


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run',type=Path,required=True);p.add_argument('--task-panel',type=Path,required=True)
 p.add_argument('--learning-items',type=int,default=100,help='the run\'s train --learning-items');p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 with (a.run/'prepared_examples.pkl').open('rb') as f:train=pickle.load(f)['train']
 # The same draw as train(): groups through its id-keyed dict (evidence-QA and RAG items can share an id: the later item's
 # group wins for both), items in packet order, groups sorted, one seeded sample per group.
 train_groups={i.item_id:panel_group(i) for i in train};by_group={}
 for item in train:by_group.setdefault(train_groups[item.item_id],[]).append(item)
 learning=[]
 for name,group in sorted(by_group.items()):
  learning.extend(random.Random(f'{SEED}:learning:{name}').sample(group,min(len(group),max(1,a.learning_items//len(by_group)))))
 misfiled=[i.item_id for i in learning if panel_group(i)!=train_groups[i.item_id]]
 if misfiled:raise ValueError(f'learning items reported under another family\'s group: {misfiled[:3]}')
 counts={}
 for item in learning:counts[train_groups[item.item_id]]=counts.get(train_groups[item.item_id],0)+1
 shared_ids=len(train)-len(train_groups)
 expected={g:v['items'] for g,v in json.loads((a.run/'panels.json').read_text())[0]['learning by group'].items()}
 if counts!=expected:raise ValueError(f'learning sample differs from the run\'s panel rows: {counts} vs {expected}')
 seen={digest(text) for item in train for text in source_texts(item.ac_messages)}
 with (a.task_panel/'panel.pkl').open('rb') as f:panel=pickle.load(f)['items']
 overlap={}
 for item in panel:
  texts=source_texts(item.ac_messages);shared=[t for t in texts if digest(t) in seen]
  overlap[item.item_id]={'group':item.info['panel_group'],'source_texts':len(texts),'shared_texts':len(shared),
   'source_chars':sum(map(len,texts)),'shared_chars':sum(map(len,shared))}
 groups={}
 for row in overlap.values():
  g=groups.setdefault(row['group'],{'items':0,'items_sharing_any':0,'source_chars':0,'shared_chars':0})
  g['items']+=1;g['items_sharing_any']+=row['shared_texts']>0;g['source_chars']+=row['source_chars'];g['shared_chars']+=row['shared_chars']
 with (a.output/'learning_items.pkl').open('wb') as f:pickle.dump({'items':learning,'groups':{i.item_id:panel_group(i) for i in learning}},f)
 write(a.output/'review_items.json',{'run':str(a.run.resolve()),'task_panel':str(a.task_panel.resolve()),'learning_by_group':counts,
  'learning_ids':[i.item_id for i in learning],'train_items_sharing_an_id':shared_ids,'min_source_chars':MIN_SOURCE_CHARS,'training_source_texts':len(seen),
  'panel_overlap_by_group':groups,'panel_overlap':overlap})
 print(json.dumps({'learning_by_group':counts,'panel_overlap_by_group':groups}),flush=True)

if __name__=='__main__':main()
