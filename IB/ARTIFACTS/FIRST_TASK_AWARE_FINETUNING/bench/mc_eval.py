"""Multiple-choice questions (QuALITY: A-D, LongHealth: A-E) for the reader: the question text it is shown, and the scoring of
its answer.

  question_text(question, options)   the question, its lettered options and the instruction to answer with the single letter;
                                     this is the `question` of a panel item (prepare_doc_corpus.py), whose gold answer is the letter.
  choice(answer, options)            the letter the answer chooses, or None: a bare letter ("B", "b", "B.", "(B)", "B,"), a marked one
                                     ("Answer: B", "the answer is (B)", "Option B", "B is correct"), a letter with its own option's
                                     text ("B. Rituximab", "b Rituximab"), a letter followed by other text ("B - because ..."), or
                                     exactly one option's text. No answer (the careful reading, counted in `no_answer`): empty,
                                     several letters, a letter the question does not have, a letter with ANOTHER option's text or
                                     letter, a fragment of an option's text that merely starts like a letter ("E. faecium" when an
                                     option reads "E. faecium infection"), and any non-bare answer that negates or contrasts
                                     ("C, not option A", "B rather than D": which letter is meant is not safe to guess).
  correct(answer, gold, options)     True when the chosen letter is a gold letter. No answer is wrong. `gold` is a letter or
                                     several; an option whose text equals the gold option's text (up to whitespace and case)
                                     counts as gold: LongHealth has three such questions (patient_16-q11, patient_20-q11, and
                                     patient_09-q19 by spacing only). --rescore also reports `accuracy_keyed`: the benchmark's one
                                     keyed letter alone (at most 0.75 points lower on LongHealth).

The scoring never goes through activation/dataset/scoring.py: its normalize_answer drops the articles a/an/the, so the gold
letter "A" normalizes to the empty string, an empty prediction then matches it exactly, and a predicted "A" is compared as empty
too. review_snapshot.py's exact_match / token_f1 columns are therefore NOT valid for these groups; rescore its output here:

  python mc_eval.py --rescore out/review_<label>.json [--questions quality_eval50.json] [--output scores.json]
  python mc_eval.py --selftest

--rescore reads the review's `answers` (the submit_answer argument per item, its question text and gold letter), takes the
options back from the question text (or from --questions, by the question id that ends the item id; that also brings the
gold letters and QuALITY's `hard` flag) and prints accuracy per group. A reader that never called submit_answer has an empty
answer there: it counts as no answer, unless its whole reply (`completion`) is one bare letter, which counts as that letter.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

LETTERS='ABCDE'
OPTIONS_HEADER='Options:'
INSTRUCTION='Answer with the single letter of the correct option ({letters}).'

def clean(text):return ' '.join(unicodedata.normalize('NFC',str(text)).split())   # options and questions on one line each
def key(text):return clean(text).casefold().strip(' \t"\'“”‘’`.,;:!')              # option-text identity

def letters_of(options):
 if not 2<=len(options)<=len(LETTERS):raise ValueError(f'{len(options)} options: 2 to {len(LETTERS)} are lettered')
 return LETTERS[:len(options)]

def question_text(question,options):
 """What the reader is asked: the question, one lettered option per line, and the answer instruction."""
 letters=letters_of(options);listed=', '.join(letters[:-1])+' or '+letters[-1]
 return (clean(question)+'\n\n'+OPTIONS_HEADER+'\n'+'\n'.join(f'{l}. {clean(o)}' for l,o in zip(letters,options))
  +'\n\n'+INSTRUCTION.format(letters=listed))

def parse_question_text(text):
 """(question, options) of a question_text() string (its inverse up to whitespace), or None when it is not one."""
 head,found,rest=text.rpartition('\n\n'+OPTIONS_HEADER+'\n')
 if not found:return None
 body,found,_=rest.rpartition('\n\n'+INSTRUCTION.split('{')[0])
 if not found:return None
 lines=body.split('\n')
 if not 2<=len(lines)<=len(LETTERS) or any(not line.startswith(f'{l}. ') for l,line in zip(LETTERS,lines)):return None
 return head,[line[3:] for line in lines]

BARE=re.compile(r'^[\s\(\[\*"\'`]*([A-Za-z])[\s\)\]\.,\*:"\'`]*$')
OWN=re.compile(r'^[\(\[\*]*([A-Ea-e])[\)\]\*]*\s*[\.:\)\]\-–—,]?\s*(\S.*)$',re.S)   # a letter, then (checked by the caller) its own option's text
VERDICT=re.compile(r'^[\(\[\*]*([A-E])[\)\]\*]*\s+is\s+(?:correct|right|the\s+(?:correct\s+|right\s+)?(?:answer|option|choice))\b[\s\.!]*$')
NEGATION=re.compile(r"(?i)\b(?:not|no|never|wrong|incorrect|rather\s+than|instead\s+of|except|neither|nor)\b|n't\b")
LEADING=re.compile(r'^[\(\[\*]*([A-E])[\)\]\*]*\s*(?:[\.:\)\]]|[-–—])\s+(\S.*)$',re.S)
# A marker word, then a capital letter ("the answer is a drug" names no option); a small letter only when it ends the answer.
MARKED=re.compile(r'(?i:\banswer|\boption|\bchoice|\bletter)\b(?i:\s+is)?\s*:?\s*[\(\[\*"\'`]*(?:([A-E])(?![A-Za-z0-9])|([a-e])[\)\]\*"\'`\.]*$)')
def marked(text):return {(a or b).upper() for a,b in MARKED.findall(text)}

def choice(answer,options):
 """The letter `answer` chooses among `options`, or None (no answer)."""
 letters=letters_of(options);text=clean(answer if answer is not None else '')
 if not text:return None
 keys=[key(o) for o in options]
 by_text=[l for l,k in zip(letters,keys) if k and k==key(text)]
 if by_text:return by_text[0]   # exactly an option's text (identical options: the first letter; correct() treats them alike)
 bare=BARE.match(text)
 if bare:
  letter=bare.group(1).upper()
  return letter if letter in letters else None
 own=OWN.match(text)
 if own and own.group(1).upper() in letters and key(own.group(2)) and key(own.group(2))==keys[letters.index(own.group(1).upper())]:return own.group(1).upper()
 # A fragment of an option's text is not a letter ("E. faecium" of "E. faecium infection"); a negated or contrasted letter is not safe to read.
 if any(k and key(text) in k for k in keys) or NEGATION.search(text):return None
 verdict=VERDICT.match(text)
 if verdict:return verdict.group(1) if verdict.group(1) in letters else None
 leading=LEADING.match(text)
 if leading:
  letter,rest=leading.group(1),key(leading.group(2))
  if letter not in letters:return None
  other=BARE.match(leading.group(2))
  if other and other.group(1).upper()!=letter:return None   # "A - B", "A. B": two letters
  # "B. <the text of option C>" names two options: no answer.
  if any(k and k==rest and k!=keys[letters.index(letter)] for k in keys):return None
  return None if marked(leading.group(2))-{letter} else letter
 named=marked(text)
 if len(named)==1:
  letter=named.pop()
  return letter if letter in letters else None
 return None

def gold_letters(gold,options):
 """Every letter that counts as correct: the gold letter(s) and any option with the same text as one of them."""
 letters=letters_of(options);given=[gold] if isinstance(gold,str) else list(gold)
 if not given or any(len(g)!=1 or g not in letters for g in given):raise ValueError(f'gold {gold!r} is not a letter of {letters}')
 keys=[key(o) for o in options];texts={keys[letters.index(g)] for g in given}
 return {l for l,k in zip(letters,keys) if l in given or k in texts}

def correct(answer,gold,options):
 chosen=choice(answer,options)
 return chosen is not None and chosen in gold_letters(gold,options)

def rescore(review,questions=None):
 """Per group of a review_snapshot.py output: items, accuracy, the share without a usable answer and the chosen-letter counts."""
 by_id={q['id']:q for q in questions or []};out={};rows={}
 for group,answers in sorted(review['answers'].items()):
  scored=[]
  for row in answers:
   source=next((q for i,q in by_id.items() if row['item_id']==i or row['item_id'].split(':',2)[-1].split(':')[0]==i),None)   # panel:<group>:<question id>[:view]
   parsed=parse_question_text(row['question'])
   if source is None and parsed is None:
    if scored:raise ValueError(f'{group}: {row["item_id"]} is not a multiple-choice item in a multiple-choice group')
    break   # not a multiple-choice group
   options=source['options'] if source else parsed[1];gold=(source.get('gold_letters') or source['gold']) if source else row['gold']
   keyed=source['gold'] if source else row['gold']   # the benchmark's one keyed letter
   # A reader that writes the bare letter as its whole reply, without the submit_answer call, has answered (Oct 7: after long
   # reconstruction-only training a third of the LongHealth replies were just "A"): the letter counts, and `bare_letter` reports the share.
   answer=row['answer'];bare_reply=not clean(answer or '') and bool(BARE.match(clean(row.get('completion') or '')))
   if bare_reply:answer=row['completion']
   chosen=choice(answer,options)
   scored.append({'bare_letter':bare_reply,'item_id':row['item_id'],'gold':sorted(gold_letters(gold,options)),'keyed':keyed,'choice':chosen,'correct':chosen is not None and chosen in gold_letters(gold,options),
    'correct_keyed':chosen is not None and chosen==keyed,'submitted':row.get('submitted'),'hard':source.get('hard') if source else None})
  else:
   if not scored:continue
   n=len(scored);entry={'items':n,'accuracy':sum(r['correct'] for r in scored)/n,'accuracy_keyed':sum(r['correct_keyed'] for r in scored)/n,'no_answer':sum(r['choice'] is None for r in scored)/n,
    'submitted':sum(bool(r['submitted']) for r in scored)/n,'bare_letter':sum(r['bare_letter'] for r in scored)/n,'choices':{l:sum(r['choice']==l for r in scored) for l in LETTERS if any(r['choice']==l for r in scored)}}
   for name,flag in (('hard',True),('easy',False)):
    part=[r for r in scored if r['hard'] is flag]
    if part:entry[name]={'items':len(part),'accuracy':sum(r['correct'] for r in part)/len(part)}
   out[group]=entry;rows[group]=scored
 return out,rows

def selftest():
 four=['7 years','10 hours','12 years','1 hour'];five=['Polatuzumab vedotin','Rituximab','Cyclophosphamide','Vincristine','Doxorubicin']
 cases=[   # (answer, options, expected choice)
  ('A',five,'A'),('B',five,'B'),('C',five,'C'),('D',five,'D'),('E',five,'E'),('A',four,'A'),('D',four,'D'),
  ('b',five,'B'),('B.',five,'B'),('(B)',five,'B'),(' B ',five,'B'),('**B**',five,'B'),('"C"',five,'C'),('[D]',five,'D'),('B)',five,'B'),('E:',five,'E'),
  ('Answer: B',five,'B'),('answer: (C)',five,'C'),('The answer is D.',five,'D'),('The correct answer is: **E**',five,'E'),('Option A',five,'A'),('option b',five,'B'),('Answer: b',five,'B'),('the answer is a',five,'A'),
  ('The answer is a drug',five,None),('The answer is an antibody, option B',five,'B'),('I think the answer is B because A is wrong',five,None),
  ('B. Rituximab',five,'B'),('B) Rituximab',five,'B'),('(D) Vincristine',five,'D'),('C: 12 years',four,'C'),('B - because the text says so',five,'B'),
  ('Rituximab',five,'B'),('rituximab.',five,'B'),(' "Vincristine" ',five,'D'),('10 hours',four,'B'),('1 hour',four,'D'),
  ('',five,None),('   ',five,None),(None,five,None),('E',four,None),('(E)',four,None),('Answer: E',four,None),('F',five,None),('Z',four,None),
  ('A, B',five,None),('A and C',five,None),('AB',five,None),('A or B',five,None),('B. Vincristine',five,None),('Answer: A or answer: B',five,None),
  ('A. I think, but option C is also possible',five,None),('A patient with DLBCL',five,None),('I do not know',five,None),('None of the above',five,None),
  ('Rituximab and Vincristine',five,None),('The hour',four,None),('1',four,None),('12',four,None),('B C',five,None),
  ('Answer: B\nRituximab',five,'B'),('The answer is B, Rituximab.',five,'B'),
  ('C because option A is wrong',five,None),('C, not option A',five,None),('It is not option B',five,None),('B rather than option D',five,None),('Option B is wrong, the answer is C',five,None),
  ('A - B',five,None),('A. B',five,None),('b. Rituximab',five,'B'),('B Rituximab',five,'B'),('B,',five,'B'),('B is correct',five,'B'),('B\nRituximab',five,'B'),('B is the correct answer.',five,'B'),
  ('E. faecium',['E. coli infection','Staphylococcus','E. faecium infection','Candida','None'],None),('E. faecium infection',['E. coli infection','Staphylococcus','E. faecium infection','Candida','None'],'C'),
  ('C. E. faecium infection',['E. coli infection','Staphylococcus','E. faecium infection','Candida','None'],'C'),('No',['Yes','No'],'B'),('B. No',['Yes','No'],'B'),
 ]
 failures=[(a,o,e,choice(a,o)) for a,o,e in cases if choice(a,o)!=e]
 # Scoring: the gold letter "A" and an empty answer must NOT match (the normalize_answer failure), every gold letter is reachable.
 checks=[(correct('',g,five) is False and correct(None,g,five) is False and correct(g,g,five) is True and correct(f'({g})',g,five) is True
  and all(correct(o,g,five) is False for o in LETTERS if o!=g) and correct(five[LETTERS.index(g)],g,five) is True) for g in LETTERS]
 checks+=[correct('A','A',four),not correct('',' A'.strip(),four),not correct('a patient','A',four),not correct('A, B','A',four),not correct('E','A',four)]
 # LongHealth's doubled options: the correct text under two letters, both count; identical wrong options stay wrong.
 doubled=['13.5 g/dL','14.3 g/dL','7.6 g/dL','7.8 g/dL','14.3 g/dL']
 checks+=[correct('B',['B','E'],doubled),correct('E',['B','E'],doubled),correct('E','B',doubled),correct('14.3 g/dL','B',doubled),not correct('A',['B','E'],doubled),
  gold_letters('B',doubled)=={'B','E'},choice('14.3 g/dL',doubled)=='B']
 twin=['Active upper GI bleed','Gastric ulcer','Gastric ulcer','Intestinal obstruction','Esophageal varices'];checks+=[not correct('Gastric ulcer','A',twin),not correct('C','A',twin),correct('C','B',twin)]
 # The question text round-trips, lists every letter once, and names the letters in its instruction.
 for options in (four,five):
  text=question_text('How  much time\nhas passed?',options);parsed=parse_question_text(text)
  checks+=[parsed==('How much time has passed?',options),text.count('\n'+LETTERS[len(options)-1]+'. ')==1,(LETTERS[len(options)-2]+' or '+LETTERS[len(options)-1]) in text,
   ('E. ' in text)==(len(options)==5)]
 checks+=[parse_question_text('What is the capital of France?') is None]
 try:gold_letters('E',four);checks.append(False)
 except ValueError:checks.append(True)
 # A review_snapshot.py-shaped output: the letter groups are rescored, a free-text group is left out.
 q=question_text('Which drug is NOT part of the regimen?',five)
 review={'answers':{'mc/x':[{'item_id':'panel:mc/x:p-q00','question':q,'gold':'D','answer':'D','submitted':True},
  {'item_id':'panel:mc/x:p-q01','question':q,'gold':'A','answer':'','submitted':False},{'item_id':'panel:mc/x:p-q02','question':q,'gold':'A','answer':'Vincristine','submitted':True}],
  'rag_qa/uncontextual':[{'item_id':'panel:rag:1','question':'Who wrote it?','gold':'A','answer':'','submitted':False}]}}
 scores,_=rescore(review)
 checks+=[set(scores)=={'mc/x'},scores['mc/x']['items']==3,abs(scores['mc/x']['accuracy']-1/3)<1e-12,abs(scores['mc/x']['no_answer']-1/3)<1e-12,scores['mc/x']['choices']=={'D':2}]
 scores,_=rescore(review,[{'id':'p-q01','options':five,'gold':'A','gold_letters':['A'],'hard':True}])
 checks+=[scores['mc/x']['hard']=={'items':1,'accuracy':0.}]
 for a,o,e,got in failures:print(f'FAIL choice({a!r}) = {got!r}, expected {e!r} ({len(o)} options)')
 bad=[i for i,c in enumerate(checks) if not c]
 if bad:print('FAIL checks',bad)
 print(f'selftest: {len(cases)-len(failures)}/{len(cases)} choice cases, {len(checks)-len(bad)}/{len(checks)} checks')
 return not failures and not bad

def main():
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--selftest',action='store_true');p.add_argument('--rescore',type=Path,help='a review_snapshot.py output')
 p.add_argument('--questions',type=Path,help='the benchmark file (a list of questions, or a dict with "questions"): options, gold letters, hard flags by question id')
 p.add_argument('--output',type=Path,help='write the scores and the per-item choices here')
 a=p.parse_args()
 if a.selftest:raise SystemExit(0 if selftest() else 1)
 if not a.rescore:p.error('--selftest or --rescore')
 questions=None
 if a.questions:
  questions=json.loads(a.questions.read_text());questions=questions['questions'] if isinstance(questions,dict) else questions
 scores,rows=rescore(json.loads(a.rescore.read_text()),questions)
 if questions:   # coverage: review_snapshot.py decodes only its first --answers items per group
  for group,entry in scores.items():
   entry['questions_in_file']=len(questions)
   if (group.startswith('mc/') or group.split('|')[0]=='mc') and entry['items']<len(questions):print(f'WARNING: {group} holds {entry["items"]} of {len(questions)} questions (review_snapshot.py --answers too small, or a subsampled panel)',file=sys.stderr)
 if a.output:a.output.write_text(json.dumps({'review':str(a.rescore),'scores':scores,'items':rows},indent=2,ensure_ascii=False)+'\n')
 print(json.dumps(scores,indent=1))

if __name__=='__main__':main()
