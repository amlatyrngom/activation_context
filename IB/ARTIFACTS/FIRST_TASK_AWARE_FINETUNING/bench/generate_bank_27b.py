"""Separate nonthinking 27B QA producer; the deployed 9B producer remains unchanged."""
import argparse, collections, fcntl, hashlib, importlib.util, json, os, re, signal, sys, time, uuid
from pathlib import Path
from select_bank import read_batch, quotas, family_of

SHARES = {'code': .30, 'math_arxiv': .12, 'math_textbook': .08, 'financial': .05,
          'web': .20, 'technical_web': .05, 'documents': .08, 'legal': .06, 'office': .06}
DOMAINS = {'math_arxiv': 'numeric', 'math_textbook': 'numeric', 'financial': 'numeric',
           'technical_web': 'web', 'legal': 'legal', 'office': 'office'}
SEED, STOP = 20260922, False
MODEL_ID = 'Qwen/Qwen3.5-27B'
CHECKER_PROMPT = ('Judge the question and proposed answer using only the supplied evidence. '
    'Check the question\'s factual premises before judging the answer. Reject incorrect premises even when the answer agrees with them. '
    'Recompute numerical differences, their direction and units; trace the exact named code branch rather than borrowing an adjacent branch '
    'or guessing missing dependency definitions. Reject ambiguous, incorrect, or unsupported answers. '
    'Reject any question or answer that depends on temporary numbered source labels such as Source 0 or Source 2: '
    'the question must name the actual entity, document or function, and the answer must stand alone. '
    'supported is true only when both the question\'s premises and the answer are correct without unstated facts. '
    'Return JSON {"supported":true or false}.')
STUDY_CONTEXT = ('Create self-contained grounded questions useful for a terminal assistant. Use only the supplied raw source. '
    'Answers should be concise (typically 2–30 words). Name the entity, function, period, jurisdiction or assumptions needed to remove ambiguity. '
    'Avoid trivia and paraphrases of the same question. Numerical answers require supplied values. For multiple questions, use distinct reasoning or evidence. '
    'For law and office sources, favor applying the stated rules to a short new scenario, exceptions, deadlines, permissions and procedural requirements. '
    'Never infer current legal advice beyond the source. '
    'Check each question\'s premise against the evidence; do not presuppose a decrease when the numbers increase. '
    'Compute numerical differences with their correct sign and units. Trace the exact named code branch, and do not borrow behavior from another branch. '
    'Do not infer meanings of template parameters, capacities or units from missing dependency definitions. '
    'Neither the question nor the answer may refer to temporary numbered source labels such as Source 0 or Source 2; '
    'name the real entity or function instead. Keep evidence indices only in the designated citation metadata.')

def reserved_questions(paths):
    """Only our explicitly reserved questions; raw sources remain reusable."""
    questions = set()
    for path in paths:
        data = json.loads(Path(path).read_text())
        if 'held' in data:
            questions.update(row['example']['query'].strip().casefold() for row in data['held'])
        else:
            questions.update(question.strip().casefold() for question in data['questions'])
    return questions

def current_target(args):
    target = json.loads(args.target_file.read_text())['target_questions'] if args.target_file else args.target
    if type(target) is not int or target < 1000 or target % 1000:
        raise ValueError('Target must be a positive integer multiple of 1000')
    return target

def stop_requested(signum, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGUSR1, stop_requested)
signal.signal(signal.SIGTERM, stop_requested)

def write(path, value):
    """The manifest is the only publication point; incomplete/orphan files are ignored."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)

def immutable_rows(folder, prefix, rows):
    relative = f'{prefix}_{uuid.uuid4().hex}.jsonl'
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open('xb') as handle:
        for row in rows:
            data = (json.dumps(row, ensure_ascii=False) + '\n').encode()
            handle.write(data)
            digest.update(data)
        handle.flush()
        os.fsync(handle.fileno())
    return {'file': relative, 'sha256': digest.hexdigest(), 'questions': len(rows)}

def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def restore(folder, manifest, protected=()):
    """Replay committed data only. Published counts identify the consumed family prefixes."""
    pending = {family: collections.deque() for family in SHARES}
    seen = set(protected)
    skip = dict(manifest.get('by_family', {}))
    for entry in manifest['microchunks']:
        path = folder / entry['file']
        if fingerprint(path) != entry['sha256']:
            raise ValueError('Changed committed microchunk: ' + entry['file'])
        for row in read_batch(path):
            family = family_of(row)
            seen.add(row['question'].strip().casefold())
            if skip.get(family, 0):
                skip[family] -= 1
            else:
                pending[family].append(row)
    if any(skip.values()):
        raise ValueError('Published counts exceed committed questions')
    return pending, seen

def commit_chunk(folder, manifest, pending, pool_id, family, start, stop, rows, recipe_id, source_ids=()):
    cursor = manifest['source_pools'][pool_id]['cursors']
    if cursor[family] != start:
        raise ValueError('Source cursor changed while generating')
    entry = immutable_rows(folder, f'microchunks/chunk_{len(manifest["microchunks"]):06d}_{family}', rows)
    entry.update(source_pool=pool_id, source_family=family, cursor_start=start, cursor_end=stop, recipe_id=recipe_id, source_ids=list(source_ids))
    manifest['microchunks'].append(entry)
    cursor[family] = stop
    manifest['accepted_questions'] += len(rows)
    manifest['accepted_by_family'][family] += len(rows)
    manifest['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    write(folder / 'manifest.json', manifest)
    pending[family].extend(rows)
    return entry

def publish_ready(folder, manifest, pending):
    wanted = quotas(1000, manifest['source_shares'])
    published = []
    while manifest['total_questions'] < manifest['target_questions'] and all(len(pending[d]) >= n for d, n in wanted.items()):
        ordered = [(i / n, family, row) for family, n in wanted.items()
                   for i, row in enumerate(list(pending[family])[:n])]
        rows = [row for _, _, row in sorted(ordered, key=lambda x: (x[0], x[1]))]
        number = len(manifest['batches']) + 1
        entry = immutable_rows(folder, f'batches/batch_{number:05d}', rows)
        counts = dict(collections.Counter(row['domain'] for row in rows))
        total = manifest['total_questions'] + len(rows)
        table = {'batch': number, 'questions': len(rows), 'cumulative': total, **wanted}
        entry.update(format='qa_jsonl_v1', by_family=wanted, by_domain=counts, table=table)
        manifest['batches'].append(entry)
        manifest['total_questions'] = total
        for family, count in wanted.items():
            manifest['by_family'][family] += count
        for domain, count in counts.items():
            manifest['by_domain'][domain] = manifest['by_domain'].get(domain, 0) + count
        manifest['updated_utc'] = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        write(folder / 'manifest.json', manifest)
        for family, count in wanted.items():
            for _ in range(count):
                pending[family].popleft()
        published.append(entry)
    return published

def next_family(pools, cursors, pending):
    wanted = quotas(1000, SHARES)
    deficient = [family for family, count in wanted.items() if len(pending[family]) < count]
    exhausted = [family for family in deficient if cursors[family] >= len(pools[family])]
    if exhausted:
        return None, exhausted
    return max(deficient, key=lambda family: (wanted[family] - len(pending[family])) / wanted[family]), []

def source_block(pool, start, processed, limit):
    """Committed source IDs, including rejects, survive input-file and recipe changes."""
    selected, cursor = [], start
    while cursor < len(pool) and len(selected) < limit:
        row = pool[cursor]
        cursor += 1
        if row['source_id'] not in processed:
            selected.append(row)
    return selected, cursor

def load_extensions(directory, pools, loaded, previous_order=()):
    """Published extension files are immutable and appended in their recorded order."""
    files = {path.name: path for path in Path(directory).glob('*__*.jsonl')}
    order = list(previous_order) + sorted(set(files) - set(previous_order))
    added = 0
    existing = {family: {row['source_id'] for row in values} for family, values in pools.items()}
    for name in order:
        if name in loaded:
            continue
        if name not in files:
            raise FileNotFoundError('Previously consumed source extension is missing: ' + name)
        family = name.split('__', 1)[0]
        if family not in SHARES:
            raise ValueError('Unknown source extension family: ' + family)
        path = files[name]
        digest, count = fingerprint(path), 0
        if isinstance(previous_order, dict) and name in previous_order and digest != previous_order[name]['sha256']:
            raise ValueError('Changed immutable source extension: ' + name)
        for row in read_batch(path):
            if row['source_id'] in existing[family]:
                continue
            row['source_file_sha256'] = digest
            pools[family].append(row)
            existing[family].add(row['source_id'])
            count += 1
        loaded[name] = {'sha256': digest, 'source_family': family, 'added_sources': count}
        added += count
    return added

def portable(example, source_rows, lengths, family, recipe_id, pool_id, tokenizer, model_id=MODEL_ID, model_revision=None):
    source = example.study_source
    units = [{'id': u['doc_id'], 'document_id': source_rows[u['doc_id']]['doc_id'], 'text': u['text'],
              **{key: source_rows[u['doc_id']][key] for key in ('subdomain', 'title', 'provenance') if key in source_rows[u['doc_id']]}}
             for u in source['units']]
    ident = hashlib.sha256(json.dumps([example.query, example.gold_answers[0], source['text']], ensure_ascii=False).encode()).hexdigest()
    return {'schema_version': 1, 'id': ident, 'domain': DOMAINS.get(family, family), 'source_family': family,
            'question': example.query, 'answer': example.gold_answers[0],
            'source': {'text': source['text'], 'units': units, 'mode': source['mode'],
                       'kind': source_rows[units[0]['id']]['source_kind'],
                       **{key: source_rows[units[0]['id']][key] for key in ('subdomain', 'title') if key in source_rows[units[0]['id']] }},
            'supporting_unit_ids': example.positive_doc_ids,
            'source_tokens': sum(lengths[u['id']] for u in units),
            'answer_tokens': len(tokenizer.encode(example.gold_answers[0], add_special_tokens=False)),
            'provenance': {'generator_model': model_id, 'generator_revision': model_revision,
                           'checker_model': model_id, 'enable_thinking': False, 'quantization': 'dynamic FP8', 'recipe_id': recipe_id,
                           'source_pool_sha256': pool_id, 'source_file_sha256': source_rows[units[0]['id']].get('source_file_sha256', pool_id),
                           'source_request_id': source['request_id']}}

def main(args):
    args.target = current_target(args)
    if not 1 <= args.microbatch <= 64:
        raise ValueError('Microbatch must be 1..64')
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.dataset.dataset import LoadedDataset, DatasetDocument
    from vllm.sampling_params import StructuredOutputsParams
    root = Path(os.environ['QA_SOURCE_ROOT'])
    model_path = Path(args.model_path) if args.model_path else None
    if model_path is not None and not (model_path / 'config.json').is_file():
        raise FileNotFoundError('Prefetched model snapshot is incomplete: ' + str(model_path))
    model_revision = model_path.name if model_path else None
    folder = Path(os.environ['QA_REPORT_ROOT']) / args.bank_name
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'writer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (folder / 'PRESERVE').write_text('Reusable synthesized QA, portable independently of generator prompts/code.\n')
    spec = importlib.util.spec_from_file_location('bank_reporting', root / 'activation/common/reporting.py')
    reporting = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporting)
    started = time.monotonic()
    reporter = reporting.HtmlReporter(str(folder), '27B synthesized QA bank — reasoning disabled',
        'Each published 1,000-question release has the declared source mix. Committed surplus survives interruptions; completed releases are usable immediately.', min_render_interval_seconds=2)
    reporter.set_status(phase='loading raw sources', target_questions=args.target)
    reporter.render(force=True)
    source_path = Path(os.environ['QA_BANK_SOURCES'])
    pool_id = fingerprint(source_path)
    raw = json.loads(source_path.read_text())
    pools = raw['pools']
    if set(pools) != set(SHARES):
        raise ValueError('Raw source pool families must match: ' + ', '.join(SHARES))
    recipe = {'generator': args.model_id + ' dynamic FP8, vLLM', 'model_id': args.model_id,
              'model_revision': model_revision, 'model_path': str(model_path) if model_path else None,
              'generator_enable_thinking': False, 'checker_enable_thinking': False,
              'presence_penalty': 0., 'sampling_note': 'Structured QA/checks must freely repeat names, values and boolean decisions.',
              'recipe_version': '27b_grounded_v2', 'checker_prompt': CHECKER_PROMPT, 'generation_study_context': STUDY_CONTEXT,
              'seed': SEED, 'source_sha256': pool_id,
              'source_shares': SHARES, 'source_counts': {k: len(v) for k, v in pools.items()},
              'accepted_questions_per_release': 1000, 'max_requests_per_generation': args.microbatch,
              'questions_per_source': '1 normally; up to 2 math/financial or 3 documents/legal/office for >=1800 tokens',
              'max_source_tokens': 6144, 'validation': 'One batched grounding check; nonempty and exact question deduplication',
              'note': raw.get('note', ''), 'generator_script_sha256': fingerprint(__file__)}
    path = folder / 'manifest.json'
    manifest = json.loads(path.read_text()) if path.exists() else {
        'schema_version': 2, 'source_shares': SHARES, 'recipes': {}, 'source_pools': {}, 'microchunks': [], 'batches': [],
        'generator_model_id': args.model_id,
        'total_questions': 0, 'accepted_questions': 0, 'by_family': {d: 0 for d in SHARES},
        'accepted_by_family': {d: 0 for d in SHARES}, 'by_domain': {}}
    if manifest.get('schema_version') != 2 or manifest['source_shares'] != SHARES:
        raise ValueError('Use a new bank directory for a different publication mix/schema')
    if manifest.get('generator_model_id') != args.model_id:
        raise ValueError('Use a separate bank directory for this generator model; never mix it into the running 9B bank')
    extensions = {}
    extension_dir = Path(os.environ.get('QA_BANK_SOURCE_EXTENSIONS', str(source_path.parent / 'bank_source_extensions')))
    load_extensions(extension_dir, pools, extensions, manifest.get('source_extensions', {}))
    manifest['source_extensions'] = extensions
    recipe['source_counts'] = {k: len(v) for k, v in pools.items()}
    recipe['source_extensions'] = dict(extensions)
    recipe_id = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:20]
    manifest['recipes'][recipe_id] = recipe
    manifest['target_questions'] = args.target
    manifest['target_file'] = str(args.target_file) if args.target_file else None
    manifest.pop('stopped_reason', None)
    manifest['source_pools'].setdefault(pool_id, {'source_counts': recipe['source_counts'], 'cursors': {d: 0 for d in SHARES}})
    write(folder / 'recipes' / f'{recipe_id}.json', recipe)
    write(path, manifest)
    pilot_path = folder.parent / 'preparation/material.json'
    reserved_paths = [pilot_path] if pilot_path.exists() else []
    if os.environ.get('QA_RESERVED_MATERIAL'):
        reserved_paths.append(Path(os.environ['QA_RESERVED_MATERIAL']))
    reserved_paths.extend(args.reserved_material)
    protected = reserved_questions(reserved_paths)
    pending, seen = restore(folder, manifest, protected)
    processed = {family: set() for family in SHARES}
    for entry in manifest['microchunks']:
        processed[entry['source_family']].update(entry.get('source_ids', []))
    reporter.set_text('recipe', 'Source mix and producer recipe', json.dumps(recipe, indent=2))
    reporter.initialize_table('batches', 'Balanced releases', 'Every row is an immutable published batch.', ['batch', 'questions', 'cumulative', *SHARES])
    for entry in manifest['batches']:
        reporter.add_data_point('batches', entry['table'])
    h = None
    reason, exit_code = 'target reached', 0
    while True:
        args.target = current_target(args)
        if manifest['target_questions'] != args.target:
            manifest['target_questions'] = args.target
            write(path, manifest)
        # The owner can atomically extend its reservation while generation continues.
        seen.update(reserved_questions(reserved_paths))
        for entry in publish_ready(folder, manifest, pending):
            reporter.add_data_point('batches', entry['table'])
            print('QA_BANK_PUBLISHED', json.dumps(entry['table']), flush=True)
        reporter.status.clear()
        reporter.set_status(phase='selecting source batch', target_questions=args.target, published_questions=manifest['total_questions'], accepted_questions=manifest['accepted_questions'],
                            buffered_questions=sum(map(len, pending.values())), elapsed_minutes=round((time.monotonic()-started)/60, 1))
        reporter.render(force=True)
        if manifest['total_questions'] >= args.target:
            break
        if STOP or time.monotonic() - started >= args.seconds - 300:
            reason, exit_code = 'segment complete; requeue requested', 75
            break
        cursors = manifest['source_pools'][pool_id]['cursors']
        family, exhausted = next_family(pools, cursors, pending)
        if exhausted:
            if load_extensions(extension_dir, pools, extensions):
                manifest['source_extensions'] = extensions
                manifest['source_pools'][pool_id]['source_counts'] = {k: len(v) for k, v in pools.items()}
                write(path, manifest)
                continue
            reason, exit_code = 'raw sources exhausted: ' + ', '.join(exhausted), 76
            break
        if h is None:
            h = HarnessRuntime(HarnessRuntimeConfig(model_configs={'qa': ModelConfig('qa', str(model_path) if model_path else args.model_id, engine_kwargs={
                'quantization': 'fp8', 'enable_lora': False, 'max_model_len': 8192, 'gpu_memory_utilization': .85,
                'max_num_seqs': 64, 'speculative_config': None})},
                dataset_study_qa_model_name='qa', dataset_study_label_model_name='qa', dataset_study_qa_batch_size=args.microbatch,
                dataset_study_chat_kwargs={'chat_template_kwargs': {'enable_thinking': False}, 'sampling_params': {
                    'max_tokens': 768, 'temperature': .6, 'top_p': .8, 'top_k': 20, 'min_p': 0., 'presence_penalty': 0.}},
                doc_chunk_size_chars=10000000, doc_chunk_overlap_chars=0))
        model, accepted = h.loaded_models['qa'], []
        tokenizer = model.tokenizer
        start = cursors[family]
        selected, cursor_end = source_block(pools[family], start, processed[family], args.microbatch)
        groups = collections.defaultdict(list)
        for row in selected:
            length = len(tokenizer.encode(row['text'], add_special_tokens=False))
            count = (2 if family in ('math_arxiv', 'math_textbook', 'financial') else 3) if length >= 1800 and family in ('math_arxiv', 'math_textbook', 'financial', 'documents', 'legal', 'office') else 1
            groups[count].append((row, length))
        # This entire small source block commits together. A signal finishes at most this block.
        for count, values in sorted(groups.items()):
            dataset = LoadedDataset(dataset_id=f'bank_{pool_id[:12]}_{family}_{start:07d}_q{count}')
            source_rows, lengths = {}, {}
            for row, length in values:
                ident = row['source_id']
                if ident in source_rows:
                    raise ValueError('Duplicate source_id in a generation block: ' + ident)
                source_rows[ident], lengths[ident] = row, length
                dataset.documents[ident] = DatasetDocument(ident, dataset.dataset_id, text=row['text'])
            h.dataset_manager.register_dataset(dataset)
            h.dataset_manager._get_or_create_index(dataset.dataset_id).build_bm25_index()
            reporter.status.clear()
            reporter.set_status(phase='generating QA', source_family=family, source_cursor=start, source_block=len(selected),
                                published_questions=manifest['total_questions'], target_questions=args.target)
            reporter.render(force=True)
            prompt_kind = 'code' if family == 'code' else 'numeric' if family in ('math_arxiv', 'math_textbook', 'financial') else 'constraints' if family in ('legal', 'office') else 'qa'
            examples = h.dataset_manager.synthesize_study_examples_qa(dataset.dataset_id, len(values), base_seed=SEED+start,
                caching_id='qa_bank_27b_no_thinking', source_mode='document', max_source_tokens=6144, max_questions_per_source=count, prompt_kind=prompt_kind,
                study_context=STUDY_CONTEXT)
            schema = {'type': 'object', 'properties': {'supported': {'type': 'boolean'}}, 'required': ['supported'], 'additionalProperties': False}
            for begin in range(0, len(examples), 64):
                chunk = examples[begin:begin+64]
                reporter.status.clear()
                reporter.set_status(phase='checking answers', source_family=family, verification_questions=f'{begin}/{len(examples)}',
                                    published_questions=manifest['total_questions'], target_questions=args.target)
                reporter.render(force=True)
                chats = [[{'role': 'system', 'content': CHECKER_PROMPT},
                          {'role': 'user', 'content': e.study_source['text']+'\nQuestion: '+e.query+'\nAnswer: '+e.gold_answers[0]}] for e in chunk]
                outputs = model.engine_chat_many(chats, chat_kwargs={'chat_template_kwargs': {'enable_thinking': False}, 'sampling_params': {
                    'max_tokens': 48, 'temperature': 0, 'top_p': .8, 'top_k': 20, 'min_p': 0., 'presence_penalty': 0.,
                    'structured_outputs': StructuredOutputsParams(json=schema)}})
                if len(outputs) != len(chunk):
                    raise RuntimeError('Grounding checker returned the wrong batch size')
                for example, output in zip(chunk, outputs):
                    try:
                        good = json.loads(output.text).get('supported') is True
                    except (ValueError, AttributeError):
                        good = False
                    key = example.query.strip().casefold()
                    temporary_labels = re.search(r'\bsources?\s*(?:#\s*)?\d+\b',
                        example.query + '\n' + example.gold_answers[0], flags=re.I)
                    if good and not temporary_labels and key not in seen:
                        accepted.append(portable(example, source_rows, lengths, family, recipe_id, pool_id, tokenizer, args.model_id, model_revision))
                        seen.add(key)
            h.dataset_manager.dataset_study_generators.pop(dataset.dataset_id, None)
            h.dataset_manager.dataset_indexes.pop(dataset.dataset_id, None)
            h.dataset_manager.loaded_datasets.pop(dataset.dataset_id, None)
        source_ids = [row['source_id'] for row in selected]
        commit_chunk(folder, manifest, pending, pool_id, family, start, cursor_end, accepted, recipe_id, source_ids)
        processed[family].update(source_ids)
    manifest['stopped_reason'] = reason
    write(path, manifest)
    reporter.status.clear()
    reporter.set_status(phase='generation complete' if exit_code == 0 else 'waiting for requeue' if exit_code == 75 else 'needs source expansion',
                        reason=reason, published_questions=manifest['total_questions'], accepted_questions=manifest['accepted_questions'], target_questions=args.target)
    reporter.render(force=True)
    if exit_code == 0:
        reporter.finish()
    if h is not None:
        from activation.harness.hf_utils import FREE_DEVICE
        h.loaded_models['qa'].engine_to_device(FREE_DEVICE)
    return exit_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=int, default=32000)
    parser.add_argument('--target-file', type=Path,
                        help='Live JSON {"target_questions": 32000}; reloaded between microbatches, with --target as the no-file fallback.')
    parser.add_argument('--seconds', type=int, default=10200)
    parser.add_argument('--microbatch', type=int, default=32)
    parser.add_argument('--bank-name', default='synthesis_bank_27b')
    parser.add_argument('--model-id', default=MODEL_ID)
    parser.add_argument('--model-path', help='Pinned prefetched snapshot; provenance still records --model-id.')
    parser.add_argument('--reserved-material', type=Path, action='append', default=[],
                        help='Material with held rows, or {"questions": [...]} reservation; repeatable, reloaded between microbatches.')
    sys.exit(main(parser.parse_args()))
