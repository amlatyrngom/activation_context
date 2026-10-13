"""Freeze a larger training pool and unused confirmation questions before tuning."""
import collections, hashlib, json, random
from pathlib import Path
from select_bank import read_batch, training_row

ROOT=Path('/workspace/IB/TMP/SYNC/TASK_AWARE_QA')
SHARES={'code':.30,'numeric':.25,'web':.25,'documents':.20}

def main():
    bank=ROOT/'synthesis_bank'; raw=(bank/'manifest.json').read_bytes(); manifest=json.loads(raw)
    rng=random.Random(20260923); groups=collections.defaultdict(list); seen=set()
    pilot=json.loads((ROOT/'preparation/material.json').read_text())
    seen.update(row['example']['query'].strip().casefold() for row in pilot['held'])
    for entry in manifest['batches']:
        file=bank/entry['file']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==entry['sha256']
        for original in read_batch(file):
            row=training_row(original); key=row['example']['query'].strip().casefold()
            if key not in seen: groups[row['domain']].append(row); seen.add(key)
    confirmation=[]
    for domain,share in SHARES.items():
        rng.shuffle(groups[domain]);confirmation.extend(groups[domain][:round(200*share)])
    reserved={r['example']['study_source']['request_id'] for r in confirmation}
    for domain in groups:
        groups[domain]=[r for r in groups[domain] if r['example']['study_source']['request_id'] not in reserved]
    for row in pilot['train']:
        key=row['example']['query'].strip().casefold()
        if row['domain'] in SHARES and key not in seen:
            groups[row['domain']].append(row);seen.add(key)
    capacity=min(int(len(groups[d])/share) for d,share in SHARES.items())
    if capacity<4000:raise ValueError(f'Waiting for next completed bank batch: balanced capacity {capacity}/4000')
    train=[]
    for domain,share in SHARES.items():
        rng.shuffle(groups[domain]);train.extend(groups[domain][:round(4000*share)])
    assert len(train)==4000 and len(confirmation)==200
    assert not reserved.intersection(r['example']['study_source']['request_id'] for r in train)
    train_questions={r['example']['query'].strip().casefold() for r in train}
    assert not train_questions.intersection(r['example']['query'].strip().casefold() for r in confirmation+pilot['held'])
    recipe={'bank_manifest_sha256':hashlib.sha256(raw).hexdigest(),'bank_batches':manifest['batches'],
            'train_questions':4000,'epochs':2,'development_questions':100,'confirmation_questions':200,
            'shares':SHARES,'seed':20260923,'learning_panel':'uniform fixed32, seed20260928',
            'note':'Pilot held set is now development. Confirmation grouped by synthesis request and unused until winner selection. No law/office additions in this first larger wave.'}
    folder=ROOT/'hillclimb_material';folder.mkdir(exist_ok=True)
    for name,value in [('material.json',{'train':train,'held':pilot['held'],'recipe':recipe}),
                       ('confirmation.json',{'held':confirmation,'recipe':recipe})]:
        with (folder/name).open('x') as handle:json.dump(value,handle,ensure_ascii=False)
    (folder/'PRESERVE').write_text('Frozen hill-climbing training/development and untouched confirmation selection.\n')
    print(json.dumps({'train':len(train),'confirmation':len(confirmation),'available':capacity,'sha256':hashlib.sha256((folder/'material.json').read_bytes()).hexdigest()}))

if __name__=='__main__':main()
