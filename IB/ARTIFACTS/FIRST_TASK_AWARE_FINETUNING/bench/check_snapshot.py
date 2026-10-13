"""Reload one evaluation snapshot of a mixed run and read it out (MIXED_RUN_PLAN.md, D8): task-panel NLL by group and
decoded answers on the first --answers items of each group, scored by normalized exact match and token F1.

With --run, the snapshot's panel NLL is compared with the run's own panel row at the same update (a reload-fidelity check:
the in-training panel and the snapshot see the same weights). One GPU.

  python check_snapshot.py --snapshot RUN/SNAPSHOTS/epoch_001_050pct --task-panel PANEL --run RUN --output out.json
"""
import argparse
import json
import pickle
import time
import unicodedata
from dataclasses import fields
from pathlib import Path

from run_channel_mixture import group_summary,write


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--task-panel',type=Path,required=True)
 p.add_argument('--run',type=Path,help='the run folder: its training config and panels.json (fidelity check)')
 p.add_argument('--answers',type=int,default=16,help='decoded items per panel group');p.add_argument('--max-new-tokens',type=int,default=256)
 p.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 from activation.ac_model import ActivationContextStudyGenerator,ActivationContextTrainer,ActivationContextTrainingConfig
 from activation.ac_model.ac_model_training import ActivationContextTrainingProgress
 from activation.autotuners import configure_fla_runtime
 from activation.dataset.scoring import exact_match,token_f1
 from run_qa import harness,AC
 receipt=json.loads((a.snapshot/'snapshot.json').read_text())
 config={'loss_kind':'sft'}
 if a.run:
  names={f.name for f in fields(ActivationContextTrainingConfig)}
  config={k:v for k,v in json.loads((a.run/'recipe.json').read_text())['training_config'].items() if k in names}
 config.update(distributed=False,completion_max_new_tokens=a.max_new_tokens)
 started=time.monotonic()
 h=harness(a.snapshot/'ac',a.snapshot/'reader');study=ActivationContextStudyGenerator(h,AC)
 trainer=ActivationContextTrainer(h,ActivationContextTrainingConfig(**config))
 with (a.task_panel/'panel.pkl').open('rb') as f:items=pickle.load(f)['items']
 groups={i.item_id:i.info['panel_group'] for i in items}
 summary=trainer.eval(AC,items,release=False)
 nll=group_summary(summary['per_item'],groups);eval_seconds=time.monotonic()-started
 fidelity=None
 if a.run and (a.run/'panels.json').exists():
  row=next((r for r in json.loads((a.run/'panels.json').read_text()) if r['examples presented']==receipt['examples_presented']),None)
  if row:fidelity={g:{'snapshot':v['nll_token'],'in_training':row['held by group'].get(g,{}).get('nll_token')} for g,v in nll.items()}
 chosen=[]
 for group in sorted(set(groups.values())):chosen.extend([i for i in items if groups[i.item_id]==group][:a.answers])
 ac=study.ac_model;progress=ActivationContextTrainingProgress(1,1,receipt['epoch'],receipt['step'],receipt['steps'],receipt['global_update'],0.,0.)
 decode_started=time.monotonic()
 with configure_fla_runtime(trainer._autotuning_variables(ac)['fla_config']):
  h.module_manager.ensure_lora(ac.config.target_model_lora_name);ac.prepare()
  samples=trainer._sample_completions(ac,chosen,progress)
 answers={}
 for sample,item in zip(samples,chosen,strict=True):
  _,calls=study.dialect.parse(sample['student'])
  prediction=next((str(c['arguments'].get('answer','')) for c in calls if c['name']=='submit_answer'),'')
  gold=unicodedata.normalize('NFC',item.info['answer']);prediction=unicodedata.normalize('NFC',prediction);group=groups[item.item_id]
  answers.setdefault(group,[]).append({'item_id':item.item_id,'question':item.info['question'],'gold':gold,'answer':prediction,
   'submitted':any(c['name']=='submit_answer' for c in calls),'exact_match':exact_match(prediction,[gold]),'token_f1':token_f1(prediction,[gold])})
 scores={g:{'items':len(v),'exact_match':sum(x['exact_match'] for x in v)/len(v),'token_f1':sum(x['token_f1'] for x in v)/len(v),
  'submitted':sum(x['submitted'] for x in v)/len(v)} for g,v in sorted(answers.items())}
 write(a.output,{'snapshot':receipt,'panel_nll':nll,'fidelity':fidelity,'answer_scores':scores,'answers':answers,
  'seconds':{'load_and_panel_eval':round(eval_seconds,1),'decode':round(time.monotonic()-decode_started,1)}})
 print(json.dumps({'panel_nll':{g:round(v['nll_token'],4) for g,v in nll.items()},'fidelity':fidelity,'answer_scores':scores}),flush=True)

if __name__=='__main__':main()
