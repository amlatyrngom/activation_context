"""AC reconstruction round: dense, oracle-free compression training on text the model has not memorized.

Commands: `prepare` (chunk pools: train / held novel / Wikipedia panel), `run --run <label>` (an arm from ARMS or arms.json),
`budget` (arm size that fits the anchors' budget from the smoke run's rates), `overview`.
Mixed span reconstruction (arms with `mix`: G4 and its family): `plan --run <label>` (CPU only: the training plan's token table for 8/12/16 examples per update),
`run --run <label>`, `budget --smoke G4s --target G4` (hours of the full run from the smoke's measured rates).
"""
from __future__ import annotations
import argparse
import collections
from contextlib import contextmanager
from dataclasses import asdict, replace
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import threading
import time
import traceback

ROOT = Path(os.environ.get('AC_RECON_SOURCE_ROOT') or Path(__file__).resolve().parents[4])
REPORT_ROOT = Path(os.environ.get('AC_RECON_ROOT', ROOT / 'IB/TMP/SYNC/AC_RECON'))
BENCH = Path(__file__).resolve().parent
MODEL = 'Qwen/Qwen3.5-4B'
NAME, AC, SIDE, READER = 'qwen4b', 'rag_poc', 'rag_side', 'rag_reader'   # the shared initial adapters were saved under these names
SEED = 20260915
TOP_K = 32          # top-k of the base at the no-context reference positions (penalty only; coverage is reported); 256 would cost ~0.5 MB/item of RAM and checkpoint
BUDGET_SECONDS = int(os.environ.get('AC_RECON_BUDGET_SECONDS', 27000))
DEFAULT_ARM = dict(ratio=1 / 16, encoder_lr=1e-4, reader_lr=2e-5, penalty_weight=1.0, penalty_fraction=0.25, continue_share=0.0,
                   train_items=None, epochs=2, reporting_interval=0.125, completion_samples=8, completion_tokens=320, budget_seconds=BUDGET_SECONDS,
                   final_train_subset=512, micro_batch=3, ckpt_min_tokens=1700, material='material.json', seed=None, warmup=50, ckpt_every=None, mixer_layers=None,
                   interventions=0, last_layer_intervention=True, intervention_layers=None, deltas_off=True,
                   side_model=MODEL, reader_model=MODEL, reader_rank=64, held_per_source=None,
                   recursive_keys=False, pooling_stride=0, pooling_window=8,
                   init_ac=None, init_reader=None, note='',
                   mix=None, ratio_range=(0.01, 0.1), span_reads=(1, 5), single_span_share=0.25, span_tokens=(16, 64), span_cue_tokens=(8, 24), long_tokens=(1024, 16384), bm25_fraction=0.5,
                   updates=None, examples_per_update=8, landing=500, landing_fraction=0.01, beat_every=500, panel_ratios=(1 / 10, 1 / 20, 1 / 50, 1 / 100), panel_items=24,
                   continue_from=None, continue_at=None, encode_batch_tokens=None)
# side_model / reader_model: Hub ids of the encoder base and the reader base (equal ids share one loaded base); reader_rank: the reader LoRA rank
# (the side LoRA rank is the AC config's default, 128); held_per_source: cap on held passages per source label (None = the material's whole panel).
# interventions: deep inputs (interior reader layers receiving a residual delta per row; 0 = the embedding channel only; -1 = every full-attention layer);
# intervention_layers: an explicit list of reader layers instead of the frequency rule (with interventions 0); deltas_off: the extra panel without the deltas (deep arms only);
# init_ac / init_reader: start from another run's adapters (the AC modules are ratio-free); seed: the fresh modules' init and the data order (None = SEED).
# mix (None = the single-ratio arms above; every key from here on is read by mixed arms only, but for examples_per_update and encode_batch_tokens): the shares of the example kinds,
# {'whole': reproduce one passage, 'short_spans': 1..N cue-addressed spans of one passage, 'long_spans': the same spans of one passage inside a composed input of long_tokens
# encoder tokens}; span_reads (1, N): one span in single_span_share of the span examples, else 2..N (as many as fit); span_tokens / span_cue_tokens: span and cue lengths;
# ratio_range: the compression ratio of each example, log-uniform; bm25_fraction: the long inputs filled with the target's BM25 neighbours (else random passages).
# updates: the absolute update the arm ends at, examples_per_update examples each (one pass, no passage a target twice); warmup, then constant, then a linear landing to
# landing_fraction of both rates over the last `landing` updates; beat_every: held panel (panel_items items of each kind at each of panel_ratios, and the no-rows floor) with a
# beats.jsonl record and a weights-only snapshot; ckpt_every: full resumable state; the state before the landing is kept (KEEP_ROOT) and continue_from / continue_at name the arm
# and update a continuation starts from (with its optimizer; fresh targets; updates = its own end). encode_batch_tokens: padded encoder tokens per encode batch (None = 65,536).
# The recipe is fixed in the trainer's defaults: SFT with the gold token always among the stored targets, cosine to 10 % after the warm-up, joint clip 1.0,
# the no-context penalty on a stored base reference (off when the reader is frozen), the row head's identity skip.
ARMS = {
    'S': dict(DEFAULT_ARM, train_items=512, epochs=2, reporting_interval=0.25, completion_samples=8, note='smoke: shape and timing'),
    'A16': dict(DEFAULT_ARM, ratio=1 / 16, note='anchor 1/16'),
    'A4': dict(DEFAULT_ARM, ratio=1 / 4, note='anchor 1/4'),
}
G4 = dict(DEFAULT_ARM, side_model='Qwen/Qwen3.5-2B', reader_model='Qwen/Qwen3-4B', reader_rank=128, intervention_layers=[3, 7, 11, 15, 19, 23, 27, 31, 35], deltas_off=False,
          material='material_mix_110m.json', mix={'whole': .2, 'short_spans': .4, 'long_spans': .4}, epochs=1, updates=5000, ckpt_every=250, completion_samples=4,
          note='mixed span reconstruction from scratch: Qwen3.5-2B encoder, Qwen3-4B reader (LoRA 128), rows at the input + deltas at every 4th reader layer')
for size, tag in ((8, ''), (12, '_e12'), (16, '_e16')):   # examples per update: the first 8 / 12 examples of an update are the same in the three sizes
    sized = dict(G4, examples_per_update=size)
    ARMS |= {'G4' + tag: sized,
             'G4s' + tag: dict(sized, updates=30, warmup=5, landing=10, beat_every=10, ckpt_every=10, panel_items=4, completion_samples=2, final_train_subset=32, note='smoke of G4: shape, timing, memory'),
             'G4_in' + tag: dict(sized, intervention_layers=None, note='G4 ablation: rows at the input only, reader LoRA'),
             'G4_in_frozen' + tag: dict(sized, intervention_layers=None, reader_lr=0.0, note='G4 ablation: rows at the input only, reader frozen (no penalty)'),
             'G4_frozen' + tag: dict(sized, reader_lr=0.0, note='G4 ablation: rows + deltas at every 4th layer, reader frozen (no penalty)'),
             'G4c' + tag: dict(sized, continue_from='G4' + tag, continue_at=4500, updates=9500, note="G4 continued from its state before the landing: updates 4,500..9,500 on fresh targets, constant rate, its own landing")}
spec = importlib.util.spec_from_file_location('recon_reporting', ROOT / 'activation/common/reporting.py')
reporting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporting)


def arms():
    extra = BENCH / 'arms.json'
    merged = dict(ARMS)
    if extra.exists():
        for label, values in json.loads(extra.read_text()).items():
            merged[label] = dict(DEFAULT_ARM, **values)
    return merged


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    reporting._write_atomic(path, json.dumps(value, indent=2, allow_nan=False, default=str))


@contextmanager
def heartbeat(folder):
    stop = threading.Event(); started = time.time()
    def beat():
        while True:
            write_json(folder / 'heartbeat.json', {'pid': os.getpid(), 'started': started, 'updated': time.time()})
            if stop.wait(10): break
    thread = threading.Thread(target=beat, daemon=True); thread.start()
    try: yield
    finally: stop.set(); thread.join()


def plain_report(folder, title):
    return reporting.HtmlReporter(str(folder), title, 'AC reconstruction: Qwen3.5-4B side/reader, passages seen only through compressed rows.', min_render_interval_seconds=2)


def phase(reporter, name, **values):
    reporter.set_status(phase=name, **values); reporter.render(force=True)


SIDE_BASE = 'side_base'   # the registered name of a separate encoder base (when side_model differs from reader_model)


def harness(with_ac=False, ac_path=None, reader_path=None, seed=SEED, mixer_layers=None, interventions=0, last_layer_intervention=True, intervention_layers=None,
            side_model=MODEL, reader_model=MODEL, reader_rank=64, recursive_keys=False, pooling_stride=0, pooling_window=8, encode_batch_tokens=None):
    # recursive_keys: a level's read-outs also attend its children's returned final states; pooling_stride/window: content pooling before the side decoder (0 = off)
    # interventions: 0 = input channel only; N = N interior layers (+ last); -1 = every full-attention layer of the reader
    # intervention_layers: an explicit list of reader layers (the config's intervention_layers; needs interventions 0, the frequency rule is replaced)
    # seed: the fresh modules' initialization (replicate arms with their own seed differ in init and data order, not data order only)
    # ac_path/reader_path: explicit adapter folders (another run's epoch, or a resume's latest epoch); otherwise seeded fresh adapters
    import torch
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.ac_model import ActivationContextModelConfig
    if intervention_layers is not None and interventions != 0:
        raise ValueError(f'intervention_layers {intervention_layers} replace the frequency rule: set interventions 0 (got {interventions})')
    torch.set_num_threads(4); torch.manual_seed(seed)
    engine = {'max_model_len': 4096, 'gpu_memory_utilization': .75}
    models = {NAME: ModelConfig(NAME, reader_model, engine_kwargs=engine)}
    side_base = NAME
    if side_model != reader_model:
        side_base = SIDE_BASE; models[SIDE_BASE] = ModelConfig(SIDE_BASE, side_model, engine_kwargs=engine)
    h = HarnessRuntime(HarnessRuntimeConfig(model_configs=models, doc_chunk_size_chars=2048, doc_chunk_overlap_chars=0))
    if with_ac:
        h.module_manager.register_lora(READER, NAME, rank=reader_rank, dropout=0, checkpoint_path=str(reader_path) if reader_path else None)
        h.module_manager.register_ac_model(ActivationContextModelConfig(AC, side_base, SIDE, NAME, READER, min_view_rows=2, default_compression_ratio=1 / 16,
                                                                        checkpoint_path=str(ac_path) if ac_path else None,
                                                                        intervention_frequency=interventions, add_last_layer_intervention=last_layer_intervention,
                                                                        intervention_layers=list(intervention_layers) if intervention_layers is not None else None,
                                                                        recursive_keys=bool(recursive_keys), input_pooling_stride=int(pooling_stride), input_pooling_window=int(pooling_window),
                                                                        **({'mixer_layers': mixer_layers} if mixer_layers is not None else {}),
                                                                        **({'encode_batch_max_tokens': int(encode_batch_tokens)} if encode_batch_tokens else {})))
    return h


# ----------------------------------------------------------------------------------------------------- preparation
def windows(text, rng, lo=400, hi=2000):
    """Character windows of a tool output: whole when short, else consecutive slices of a random length in [lo, hi] cut at line ends."""
    text = text.strip()
    if len(text) < 200: return []
    if len(text) <= hi: return [text]
    out, cursor = [], 0
    while cursor < len(text):
        size = rng.randint(lo, hi)
        end = min(len(text), cursor + size)
        cut = text.rfind('\n', cursor + lo // 2, end) if end < len(text) else end
        if cut <= cursor: cut = end
        piece = text[cursor:cut].strip()
        if len(piece) >= 200: out.append(piece)
        cursor = cut
    return out


def tool_outputs(document):
    for message in document.trajectory or []:
        if message.get('role') == 'tool':
            content = message.get('content')
            if isinstance(content, list):
                content = ''.join(piece.get('text', '') for piece in content if isinstance(piece, dict))
            if isinstance(content, str): yield content


def prepare():
    from activation.dataset.loaders import OpenSweTracesDataset, S1DeepResearchDataset
    from transformers import AutoTokenizer
    folder = REPORT_ROOT / 'preparation'; r = plain_report(folder, 'AC reconstruction — preparation'); start = time.monotonic()
    goal = int(os.environ.get('AC_RECON_TRAIN_GOAL', 40000)); held_goal = 200; wiki_goal = 100
    rng = random.Random(SEED)
    with heartbeat(folder):
        phase(r, 'loading trajectories')
        h = harness()
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        sources = {'swe': OpenSweTracesDataset.load(h, max_examples=int(os.environ.get('AC_RECON_SWE_TRAJECTORIES', 1500))),
                   'research': S1DeepResearchDataset.load(h, max_examples=int(os.environ.get('AC_RECON_RESEARCH_TRAJECTORIES', 800)))}
        pool, seen = [], set()
        for source, dataset in sources.items():
            for doc_id, document in dataset.documents.items():
                for output in tool_outputs(document):
                    for piece in windows(output, rng):
                        key = hashlib.sha1(piece.encode()).hexdigest()
                        if key in seen: continue
                        seen.add(key); pool.append({'text': piece, 'doc_id': doc_id, 'source': source})
            phase(r, 'chunking', source=source, chunks=len(pool))
        rng.shuffle(pool)
        docs = sorted({p['doc_id'] for p in pool}); rng.shuffle(docs)
        held_docs = set(docs[:max(20, len(docs) // 20)])
        held = [p for p in pool if p['doc_id'] in held_docs][:held_goal]
        train = [p for p in pool if p['doc_id'] not in held_docs][:goal]
        phase(r, 'counting tokens', train=len(train), held=len(held))
        for p in train + held:
            p['tokens'] = len(tokenizer(p['text'], add_special_tokens=False)['input_ids'])
        wiki_source = folder / 'wiki_source.json'
        wiki = []
        if wiki_source.exists():
            material = json.loads(wiki_source.read_text())
            for record in material['held'][:wiki_goal]:
                wiki.append({'text': record['passage'], 'doc_id': record['doc_id'], 'source': 'wiki', 'tokens': len(tokenizer(record['passage'], add_special_tokens=False)['input_ids'])})
        by_source = {s: sum(p['source'] == s for p in train) for s in sources}
        material = {'train': train, 'held': held, 'wiki': wiki, 'preparation_seconds': time.monotonic() - start,
                    'train_by_source': by_source, 'token_stats': {'train_mean': statistics.mean(p['tokens'] for p in train), 'held_mean': statistics.mean(p['tokens'] for p in held)}}
        write_json(folder / 'material.json', material)
        phase(r, 'preparation complete', train=len(train), held=len(held), wiki=len(wiki), seconds=f'{material["preparation_seconds"]:.0f}', **by_source)
        r.finish()


def parameter_hash(ac, h, deep=False):
    """Shared parameters (AC modules without the delta heads and the attention blocks, side and reader LoRA): the same across a paired input-only / deep arm. `deep`: the heads and blocks alone."""
    import torch
    from activation.ac_model.ac_model import INTERVENTION_MODULE_PREFIXES
    value = hashlib.sha256()
    modules = [p for name, p in ac.modules.named_parameters() if name.startswith(INTERVENTION_MODULE_PREFIXES) == deep]
    params = modules if deep else modules + h.module_manager.lora_parameters(ac.config.base_side_model_lora_name) + h.module_manager.lora_parameters(READER)
    for p in params:
        value.update(str(tuple(p.shape)).encode()); value.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return value.hexdigest()


# ----------------------------------------------------------------------------------------------------- run
def config(arm, **overrides):
    from activation.ac_model import ActivationContextTrainingConfig
    values = dict(loss_kind='sft', learning_rate_ac=arm['encoder_lr'], learning_rate_target_lora=arm['reader_lr'], examples_per_update=arm.get('examples_per_update', 8),
                  num_epochs=arm['epochs'], warmup_updates=arm['warmup'], checkpoint_every_updates=arm['ckpt_every'], checkpoint_every_epoch=True,
                  completion_samples=arm['completion_samples'], completion_max_new_tokens=arm['completion_tokens'], reporting_interval=arm['reporting_interval'],
                  seed=SEED if arm['seed'] is None else arm['seed'], teacher_cache_top_k=TOP_K, gradient_checkpointing_min_tokens=arm['ckpt_min_tokens'], micro_batch_examples=arm['micro_batch'],
                  no_context_penalty_weight=arm['penalty_weight'], no_context_penalty_fraction=arm['penalty_fraction'], deltas_off_panel=arm['deltas_off'])
    if arm.get('mix'): values.update(mix_config(arm))
    values.update(overrides)
    return ActivationContextTrainingConfig(**values)


def novel_summary(summary):
    """The held panel without wiki: the single-source 'reconstruct' kind, or the token-weighted blend of every non-wiki kind of a mixed material."""
    by_kind = summary.get('by_kind') or {}
    if 'reconstruct' in by_kind: return by_kind['reconstruct']
    rows = [row for row in summary.get('per_item') or [] if row.get('kind') != 'wiki' and row.get('gold_nll') is not None]
    if not rows: return None
    weights = [max(1, row.get('positions') or 1) for row in rows]; total = sum(weights)
    return {'items': len(rows), 'gold_nll': sum(row['gold_nll'] * w for row, w in zip(rows, weights)) / total,
            'token_accuracy': sum((row.get('token_accuracy') or 0.0) * w for row, w in zip(rows, weights)) / total, 'loss': sum((row.get('loss') or 0.0) * w for row, w in zip(rows, weights)) / total}


def similarity(a, b):
    return difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False).ratio()


class RunReporter:
    """The trainer's reporter with the round's reading order, per-panel series and the reconstruction fidelity of the samples."""
    def __new__(cls, folder, title, arm, elapsed):
        from activation.ac_model import ActivationContextTrainingReporter
        class _Reporter(ActivationContextTrainingReporter):
            def __init__(self):
                super().__init__(str(folder), title,
                                 'The reader sees each passage only through its compressed rows and must reproduce it. Novel = held-out tool outputs from '
                                 'unseen trajectories; wiki = NQ passages the base model has likely memorized. No context = the trained reader without the rows.',
                                 set_labels={'reporting': 'AC student, held-out', 'validation': 'no context, current reader, held-out novel', 'deltas_off': 'AC student, deltas off, held-out'})
                self.set_text('summary', 'Run summary', '')
                self.initialize_line_plot('fidelity', 'Greedy reconstruction fidelity (sampled items)',
                    'Word-level similarity (difflib ratio) between the greedy reconstruction and the passage on the fixed sample items.',
                    'epochs in this report', 'ratio', ['AC student'])
                order = ['summary', 'gold_nll', 'token_accuracy', 'penalty', 'fidelity', 'reporting', 'loss', 'completions', 'evals', 'kinds']
                self.widgets = {name: self.widgets[name] for name in order if name in self.widgets} | {n: w for n, w in self.widgets.items() if n not in order}
            def render(self, force=False):
                self.set_status(run_elapsed=f'{(time.monotonic() - elapsed) / 60:.1f} min')
                super().render(force=force)
            def report_eval(self, progress, name, summary):
                super().report_eval(progress, name, summary)
                if not summary['items'] or name not in ('reporting', 'deltas_off'): return
                for kind, values in summary['by_kind'].items():
                    label = {'reconstruct': 'novel', 'continue': 'novel (continue)', 'wiki': 'wiki'}.get(kind, 'novel ' + kind) + (' (deltas off)' if name == 'deltas_off' else '')
                    for widget, key in (('gold_nll', 'gold_nll'), ('token_accuracy', 'token_accuracy')):
                        if values.get(key) is not None and widget in self.widgets:
                            self._ensure_series(widget, label); self.add_data_point(widget, {'x': self.epoch_x, label: values[key]})
                self.render()
            def report_completions(self, progress, rows):
                super().report_completions(progress, rows)
                scores = [similarity(row['reference'], row['student']) for row in rows if row.get('reference')]
                if scores:
                    self.add_data_point('fidelity', {'x': self.epoch_x, 'AC student': sum(scores) / len(scores)})
                    self.render()
        return _Reporter()


# ----------------------------------------------------------------------------------------------------- mixed span reconstruction (arms with `mix`)
MIX_KINDS = ('whole', 'short_1', 'short_n', 'long_1', 'long_n')   # the whole passage; one passage read through 1 / 2+ cue-addressed spans; the same spans of one passage inside a long composed input
MIX_SHARED = ('material', 'seed', 'mix', 'ratio_range', 'span_reads', 'single_span_share', 'span_tokens', 'span_cue_tokens', 'long_tokens', 'bm25_fraction', 'examples_per_update',
              'side_model', 'reader_model', 'reader_rank', 'interventions', 'last_layer_intervention', 'intervention_layers', 'mixer_layers', 'recursive_keys', 'pooling_stride', 'pooling_window')
LANE = 1.25   # target passages reserved per training slot of a column: the slack takes the passages too short for a span and the spans without unambiguous cues
KEEP_ROOT = Path(os.environ.get('AC_RECON_KEEP_ROOT', REPORT_ROOT.parent / 'KEEP_STATE'))   # kept full states (adapters + optimizer) a continuation starts from: <KEEP_ROOT>/<label>/update_NNNNN


def mix_lineage(label, table=None):
    """The blocks of updates leading to a mixed arm, root first: (label, arm, first update, last update). A continuation trains the updates after its parent's kept state."""
    table = table or arms(); chain = []
    while label is not None:
        arm = table[label]; chain.append((label, arm, arm['continue_at'] or 0, arm['updates'])); label = arm['continue_from']
        if len(chain) > 16: raise ValueError('continue_from chain does not end')
    return chain[::-1]


def mix_check(label, table=None):
    """The arm's lineage after checking that its beats, checkpoints, landing and kept state lie on one grid and that a continuation shares its parents' data and model keys."""
    chain = mix_lineage(label, table); _, arm, first, last = chain[-1]; steps = (last or 0) - first
    if not arm['mix'] or steps < 1 or arm['epochs'] != 1: raise ValueError(f'{label}: a mixed arm needs mix, updates past its start and epochs 1')
    if set(arm['mix']) != {'whole', 'short_spans', 'long_spans'} or tuple(arm['span_reads'])[0] != 1 or tuple(arm['span_reads'])[1] < 2:
        raise ValueError(f"{label}: mix needs the shares 'whole', 'short_spans', 'long_spans' and span_reads (1, N >= 2)")
    rules = {'beat_every divides the block': steps % arm['beat_every'] == 0, 'ckpt_every divides beat_every': bool(arm['ckpt_every']) and arm['beat_every'] % arm['ckpt_every'] == 0,
             'the landing starts on a beat inside the block': 0 < arm['landing'] < steps and arm['landing'] % arm['beat_every'] == 0, 'the warm-up ends before the landing': arm['warmup'] <= last - arm['landing']}
    if not all(rules.values()): raise ValueError(f"{label}: {[name for name, ok in rules.items() if not ok]} (updates {first}..{last}, beat {arm['beat_every']}, checkpoint {arm['ckpt_every']}, landing {arm['landing']})")
    for name, other, _, end in chain[:-1]:
        differ = [key for key in MIX_SHARED if json.dumps(other[key]) != json.dumps(arm[key])]
        if differ or not other['mix']: raise ValueError(f'{label} continues {name}: {differ or "not a mixed arm"} differ')
    if len(chain) > 1 and first != chain[-2][3] - chain[-2][1]['landing']:
        raise ValueError(f"{label}: continue_at {first} is not {chain[-2][0]}'s kept state (update {chain[-2][3] - chain[-2][1]['landing']}, before its landing)")
    return chain


def mix_config(arm):
    """Trainer settings of a mixed arm: one pass over the planned examples in plan order, warm-up → constant → linear landing to landing_fraction over the last `landing` updates of
    `updates` (the absolute clock: a continuation or a resume is past the warm-up and on the same curve), a held panel every `beat_every` updates."""
    from activation.ac_model.ac_model_training import reporting_boundaries
    steps = arm['updates'] - (arm['continue_at'] or 0); wanted = list(range(arm['beat_every'], steps + 1, arm['beat_every']))
    interval = next((value for value in (arm['beat_every'] / steps, round(arm['beat_every'] / steps, 6), round(arm['beat_every'] / steps, 4)) if reporting_boundaries(steps, value) == wanted), None)
    if interval is None: raise ValueError(f"no reporting interval puts the panels at every {arm['beat_every']} of {steps} updates")
    return dict(num_epochs=1, shuffle_examples=False, lr_schedule='cooldown', cooldown_updates=arm['landing'], cooldown_shape='linear', lr_final_fraction=arm['landing_fraction'],
                schedule_total_updates=arm['updates'], reporting_interval=interval, resume_optimizer=bool(arm['continue_from']))


def mix_slots(arm, material, block, steps, offset, size=None):
    """Block `block`'s training slots, one list per column of the update (examples_per_update columns; `size` overrides it): per slot the kind (exact shares per column, shuffled),
    the compression ratio (log-uniform in ratio_range), a draw seed and the target passage; with each column, its spare targets. Every column owns a lane of the seeded
    permutation of the training passages, so no passage is a target twice, a column is the same whatever the number of columns, and a later block's lanes follow the earlier
    blocks' (`offset` passages). Spans need a passage of 2 cues + one full span (encoder-tokenizer `tokens`); a shorter one met by a span slot is left unused."""
    seed = SEED if arm['seed'] is None else arm['seed']; train = material['train']; size = size or arm['examples_per_update']
    lane = math.ceil(LANE * steps); need = 2 * arm['span_cue_tokens'][1] + arm['span_tokens'][1]
    if offset + size * lane > len(train):
        raise RuntimeError(f"block {block}: {size} columns x {lane} reserved targets ({steps} updates x {LANE}) after {offset} taken by earlier blocks = {offset + size * lane} passages, "
                           f"{len(train)} in {arm['material']} ({sum(p['tokens'] >= need for p in train)} long enough for a span)")
    order = list(range(len(train))); random.Random(f'{seed}:targets').shuffle(order)
    single = arm['single_span_share']; shares = [arm['mix']['whole']] + [arm['mix'][family] * part for family in ('short_spans', 'long_spans') for part in (single, 1 - single)]
    exact = [steps * share / sum(shares) for share in shares]; counts = [int(value) for value in exact]
    for index in sorted(range(len(counts)), key=lambda i: counts[i] - exact[i])[:steps - sum(counts)]: counts[index] += 1      # largest remainders
    low, high = (math.log(value) for value in arm['ratio_range']); columns = []
    for column in range(size):
        entries = order[offset + column * lane:offset + (column + 1) * lane]; pointer = 0; slots = []
        kinds = [kind for kind, count in zip(MIX_KINDS, counts) for _ in range(count)]; random.Random(f'{seed}:{block}:{column}:kinds').shuffle(kinds)
        for update, kind in enumerate(kinds):
            rng = random.Random(f'{seed}:{block}:{column}:{update}')
            ratio, draw = math.exp(rng.uniform(low, high)), rng.getrandbits(30)
            while kind != 'whole' and pointer < len(entries) and train[entries[pointer]]['tokens'] < need: pointer += 1
            if pointer >= len(entries): raise RuntimeError(f'block {block} column {column}: the lane of {lane} passages ends at update {update} of {steps}')
            slots.append({'block': block, 'update': update, 'column': column, 'kind': kind, 'ratio': ratio, 'seed': draw, 'target': entries[pointer], 'replaced': 0}); pointer += 1
        columns.append((slots, [index for index in entries[pointer:] if train[index]['tokens'] >= need]))
    return columns


def mix_item(study, arm, source, kind, ratio, seed):
    """One item of `kind` at `ratio` on `source` (a passage; for the long kinds a composed input with its target_index): the whole passage, or 1 / 2..N spans of span_tokens tokens
    addressed by span_cue_tokens-token cues (the generator keeps as many reads as fit). None when three draws found no unambiguous cues. The draws do not depend on the ratio."""
    passage = dict(source, ratio=ratio)
    if kind == 'whole':
        items = study.generate_reconstruction_samples([passage], variant='reconstruct', seed=seed)
    else:
        for attempt in range(3):
            items = study.generate_reconstruction_samples([passage], variant='multispan', seed=seed + attempt, cue_tokens=tuple(arm['span_cue_tokens']), target_tokens=tuple(arm['span_tokens']),
                                                          reads=(1, 1) if kind.endswith('_1') else (2, arm['span_reads'][1]), multispan_passages=True)
            if items: break
    return replace(items[0], kind=kind) if items else None


def mix_long(study, arm, pool, targets, seed):
    """A composed long input around each target passage of `pool` (by index): its own draws, so it is the same whatever else is composed."""
    if not targets: return {}
    inputs = study.compose_reconstruction_inputs(pool, len(targets), token_range=tuple(arm['long_tokens']), bm25_fraction=arm['bm25_fraction'], seed=f'{seed}:long', targets=[(index, index) for index in targets])
    return dict(zip(targets, inputs))


def mix_block(study, arm, material, block, steps, offset, size=None):
    """(slot, item) of every training slot of the block, column by column. A span slot whose passage yields no item takes the column's next spare (even spares serve the long
    kinds, odd ones the short kinds), so one slot never moves another's target; slot['target'] is the passage used, slot['context'] the other passages of a long input."""
    train = material['train']; columns = mix_slots(arm, material, block, steps, offset, size); seed = SEED if arm['seed'] is None else arm['seed']
    composed = mix_long(study, arm, train, [index for slots, spares in columns for index in [slot['target'] for slot in slots if slot['kind'].startswith('long')] + spares[::2]], seed)
    for slots, spares in columns:
        pools = {'long': iter(spares[::2]), 'short': iter(spares[1::2])}
        for slot in slots:
            family = slot['kind'].split('_')[0]
            while True:
                item = mix_item(study, arm, composed[slot['target']] if family == 'long' else train[slot['target']], slot['kind'], slot['ratio'], slot['seed'])
                if item is not None: break
                slot['target'] = next(pools[family], None) if family in pools else None; slot['replaced'] += 1
                if slot['target'] is None: raise RuntimeError(f"block {block} column {slot['column']} update {slot['update']}: no spare {family} target left ({len(spares)} spares in the lane)")
            slot['context'] = [index for index in composed[slot['target']]['pool_indices'] if index != slot['target']] if family == 'long' else []
            yield slot, item


def mix_offset(chain, size):
    return sum(size * math.ceil(LANE * (last - first)) for _, _, first, last in chain[:-1])


def mix_training(study, arm, material, label, progress=None):
    """The arm's training examples in plan order (update by update, its columns in order) and the plan's record: a digest of the item ids, counts per kind, replaced targets."""
    chain = mix_check(label); size = arm['examples_per_update']; steps = chain[-1][3] - chain[-1][2]; grid = {}; counts = collections.Counter(); targets = set(); replaced = 0
    for slot, item in mix_block(study, arm, material, len(chain) - 1, steps, mix_offset(chain, size)):
        grid[slot['update'], slot['column']] = item; counts[slot['kind']] += 1; targets.add(slot['target']); replaced += slot['replaced']
        if progress and len(grid) % 2000 == 0: progress(len(grid), steps * size)
    train = [grid[update, column] for update in range(steps) for column in range(size)]
    if len(targets) != len(train): raise RuntimeError(f'{len(train)} examples on {len(targets)} distinct target passages')
    digest = hashlib.sha256('\n'.join(item.item_id for item in train).encode()).hexdigest()
    return train, {'digest': digest, 'examples': len(train), 'updates': steps, 'examples_per_update': size, 'block': len(chain) - 1, 'first_update': chain[-1][2], 'by_kind': dict(counts), 'replaced_targets': replaced}


def mix_panel(study, arm, material):
    """Held panel from held passages only (targets and long contexts): panel_items items of each kind, the SAME items at every panel ratio (kinds 'whole@10', 'long_n@100', ...), the
    kinds on disjoint passages (each its own lane of the held permutation); and the first ratio's items without their rows (kind = the mix kind: the no-rows floor).
    Seeded by the bench SEED: every mixed arm on a material shares the panel, and a smaller panel_items is its prefix."""
    from activation.common.ac_parts import strip_ac_parts
    held = material['held']
    if arm['held_per_source']:
        counts = collections.Counter(); held = [p for p in held if counts.update([p['source']]) or counts[p['source']] <= arm['held_per_source']]
    if {p['text'].strip() for p in held} & {p['text'].strip() for p in material['train']}: raise RuntimeError('held passages also in the training pool')
    need = 2 * arm['span_cue_tokens'][1] + arm['span_tokens'][1]; order = list(range(len(held))); random.Random(f'{SEED}:panel').shuffle(order)
    composed = mix_long(study, arm, held, [index for index in order if held[index]['tokens'] >= need], f'{SEED}:panel')
    first = arm['panel_ratios'][0]; panel, floor = [], []
    for position, kind in enumerate(MIX_KINDS):
        chosen = []
        for index in order[position::len(MIX_KINDS)]:     # a lane of the held permutation per kind
            if kind != 'whole' and held[index]['tokens'] < need: continue
            source = composed[index] if kind.startswith('long') else held[index]; draw = random.Random(f'{SEED}:panel:{index}').getrandbits(30)
            base = mix_item(study, arm, source, kind, first, draw)
            if base is not None: chosen.append((source, draw, base))
            if len(chosen) == arm['panel_items']: break
        if len(chosen) < arm['panel_items']: raise RuntimeError(f"{len(chosen)} of {arm['panel_items']} held {kind} items: {len(held)} held passages do not suffice")
        for ratio in arm['panel_ratios']:
            for source, draw, base in chosen:
                item = base if ratio == first else mix_item(study, arm, source, kind, ratio, draw)
                if item is None or item.assistant_token_ids != base.assistant_token_ids or item.in_context_messages != base.in_context_messages: raise RuntimeError(f'held {kind} item differs at ratio {ratio}')
                panel.append(replace(item, kind=f'{kind}@{round(1 / ratio)}'))
        floor += [replace(base, item_id=base.item_id + ':no_context', ac_messages=strip_ac_parts(base.ac_messages)) for _, _, base in chosen]
    fits = lambda item: item.kind == f'whole@{round(1 / first)}' and item.info['passage_tokens'] <= arm['completion_tokens'] - 32
    panel.sort(key=lambda item: not fits(item))   # stable: the fixed completion samples (first items) are whole passages within the generation budget
    return panel, floor


def mix_snapshot(h, ac, update):
    """Weights only (AC modules + side LoRA, reader LoRA; no optimizer) of the model as it is, under <checkpoint root>/BEATS/update_NNNNN: flagged PRESERVE, outside the trainer's pruning."""
    from activation.common.data_syncing import resolve_path
    root = resolve_path('BEATS', create=False); destination = root / f'update_{update:05d}'
    if destination.exists(): return destination
    manager = h.module_manager; configs = [manager.get_lora_config(name) for name in (ac.config.base_side_model_lora_name, READER)]; paths = [c.checkpoint_path for c in configs]
    temporary = root / f'update_{update:05d}.tmp'; shutil.rmtree(temporary, ignore_errors=True); root.mkdir(parents=True, exist_ok=True)
    if not (root / 'PRESERVE').exists(): (root / 'PRESERVE').write_text('Weights at every held panel of a mixed reconstruction run; kept.\n')
    try:
        ac.save(str(temporary / 'ac')); manager.save_lora(NAME, READER, str(temporary / 'reader'))
    finally:
        for c, path in zip(configs, paths): c.checkpoint_path = path     # a snapshot is not where a re-injection continues from
    temporary.rename(destination)
    return destination


def mix_keep(label, update, step, record):
    """The trainer's full state at `update` (the mid-run checkpoint pair partial_epoch.json publishes at `step`: adapters + optimizer.pt with the update clock) copied to
    KEEP_ROOT/<label>/update_NNNNN/{ac,reader}, flagged PRESERVE, with kept.json: where a continuation starts."""
    from activation.common.data_syncing import resolve_path
    destination = KEEP_ROOT / label / f'update_{update:05d}'
    if (destination / 'kept.json').exists(): return destination
    published = resolve_path(f'AC_MODELS/{AC}/partial_epoch.json', create=False); pair = json.loads(published.read_text()) if published.exists() else {}
    sources = {'ac': resolve_path(pair['ac_checkpoint'], create=False), 'reader': resolve_path(pair['target_lora_checkpoint'], create=False)} if pair.get('step') == step else {}
    if not sources or not (sources['ac'] / 'optimizer.pt').exists():
        raise RuntimeError(f"{label}: no mid-run checkpoint with its optimizer at step {step} (update {update}) to keep; published: {pair or None}")
    temporary = destination.with_name(destination.name + '.tmp'); shutil.rmtree(temporary, ignore_errors=True)
    for name, source in sources.items(): shutil.copytree(source, temporary / name)
    (temporary / 'PRESERVE').write_text('Full state (adapters + optimizer) before the landing of a mixed reconstruction run; a continuation starts here.\n')
    write_json(temporary / 'kept.json', {'label': label, 'update': update, 'step': step, 'source': {name: str(source) for name, source in sources.items()}, 'time': time.time(), **record})
    shutil.rmtree(destination, ignore_errors=True); temporary.rename(destination)
    return destination


def mix_beats(arm, label, folder, trainer, h, ac, floor, reporter, plan, resumed=False):
    """The trainer's reporting callback of a mixed arm (it follows every held panel: the start, each beat_every updates, and the point a resume restarts from). Per beat: one
    record appended to beats.jsonl (update; loss per token of each kind x ratio cell, token-weighted over the cell's items; means per kind, per ratio and over the cells; token
    accuracy; the no-rows floor per kind; seconds) and one aligned console line; the weights-only snapshot; at the update before the landing, the kept full state."""
    first = arm['continue_at'] or 0; keep_at = arm['updates'] - arm['landing']; names = [str(round(1 / ratio)) for ratio in arm['panel_ratios']]; state = {'floor': None, 'beats': 0}
    mean = lambda values: sum(values) / len(values)
    def beat(progress, summary):
        update = progress.global_step; restart = progress.epoch == 0 and update != first
        if progress.epoch == 0 and not (first < update < arm['updates'] if resumed else update == first):      # the update clock comes with the optimizer state
            raise RuntimeError(f"{label} starts at update {update}: expected {'a checkpoint after ' if resumed else ''}{first}")
        measured = state['floor'] is None or trainer.config.learning_rate_target_lora > 0      # a frozen reader's floor is measured once
        if measured:
            state['floor'] = trainer._evaluate(ac, floor, ac.device, None, 'validation'); reporter.report_eval(progress, 'validation', state['floor'])
        cells = summary['by_kind']; values = {key: {kind: {name: cells[f'{kind}@{name}'][key] for name in names} for kind in MIX_KINDS} for key in ('gold_nll_token', 'token_accuracy')}
        loss, accuracy = values['gold_nll_token'], values['token_accuracy']
        record = {'update': update, 'run': label, 'loss': loss, 'loss_by_kind': {kind: mean(list(loss[kind].values())) for kind in MIX_KINDS},
                  'loss_by_ratio': {name: mean([loss[kind][name] for kind in MIX_KINDS]) for name in names}, 'loss_mean': mean([loss[kind][name] for kind in MIX_KINDS for name in names]),
                  'token_accuracy': mean([accuracy[kind][name] for kind in MIX_KINDS for name in names]), 'token_accuracy_cells': accuracy,
                  'floor': {kind: state['floor']['by_kind'][kind]['gold_nll_token'] for kind in MIX_KINDS}, 'floor_measured': measured, 'items_per_cell': arm['panel_items'],
                  'seconds': round(progress.elapsed_s, 1), 'restart': restart, 'plan': plan['digest'][:12], 'time': time.time()}
        with open(folder / 'beats.jsonl', 'a') as stream: stream.write(json.dumps(record) + '\n')
        print(f"BEAT {label} update {update:6d} | loss {record['loss_mean']:7.4f} | " + ' '.join(f'{kind} {record["loss_by_kind"][kind]:7.4f}' for kind in MIX_KINDS) + ' | '
              + ' '.join(f'1/{name} {record["loss_by_ratio"][name]:7.4f}' for name in names) + f" | acc {record['token_accuracy']:6.4f} | floor "
              + ' '.join(f'{record["floor"][kind]:6.3f}' for kind in MIX_KINDS) + f" | {record['seconds']:8.0f} s" + (' | restart' if restart else ''), flush=True)
        if (update - first) % arm['beat_every'] == 0: mix_snapshot(h, ac, update)
        if update == keep_at: mix_keep(label, update, update - first, {'arm': arm, 'plan': plan})
        state['beats'] += 1; reporter.set_status(beats=f"{state['beats']} this attempt, last at update {update}: held loss {record['loss_mean']:.4f}")
    return beat


def mix_rates(arm, stats, plan):
    """Measured rates of a mixed run (this attempt's updates): seconds per update and per example (by kind: a micro-batch's time split evenly over its examples), per encoder
    token, the peak GPU memory and the fixed costs (plan building, reference pass, a panel, a checkpoint). `budget` turns a smoke's rates into hours of the full run."""
    records, steps, kinds = stats.example_records, stats.step_seconds, collections.defaultdict(list)
    for record in records: kinds[record['kind']].append(record['seconds'])
    settled = steps[2:] or steps     # the first updates carry the kernels' warm-up
    panels = sum((update - (arm['continue_at'] or 0)) % arm['beat_every'] == 0 for update in stats.global_steps); reference = stats.no_context_reference or {}
    return {'examples_per_update': plan['examples_per_update'], 'updates': len(steps), 'examples': len(records), 'panel_items': arm['panel_items'],
            'seconds_per_update': {'mean': statistics.mean(settled), 'median': statistics.median(settled), 'max': max(steps)} if steps else None,
            'seconds_per_example': sum(steps) / max(1, len(records)), 'seconds_per_example_by_kind': {kind: statistics.mean(values) for kind, values in kinds.items()},
            'examples_by_kind': {kind: len(values) for kind, values in kinds.items()}, 'side_tokens': stats.side_tokens, 'seconds_per_side_token': sum(steps) / max(1, stats.side_tokens),
            'peak_memory_gb': round(stats.peak_memory_bytes / 2**30, 2), 'reference_seconds_per_example': reference.get('seconds', 0) / max(1, reference.get('examples') or 1),
            'baseline_seconds': stats.baseline_seconds, 'panel_seconds': sum(e.reporting_seconds for e in stats.epochs) / max(1, panels),
            'checkpoint_seconds': max((e.checkpoint_seconds for e in stats.epochs), default=0.0), 'placement_seconds': stats.placement_seconds,
            'build_seconds_per_example': (plan['build_seconds'] + stats.other_seconds) / plan['examples']}      # the plan's items, then the trainer's examples (its unattributed time)


def resume_state(arm, folder):
    """A previous attempt's last completed epoch under this run's checkpoint root (ACTIVATION_SYNC_ROOT), or None for a fresh run.
    The bench reloads that epoch's adapters + optimizer.pt and trains the remaining epochs (data order is seeded per epoch number, so it continues the schedule)."""
    root = os.environ.get('ACTIVATION_SYNC_ROOT')
    if not root: return None
    ac_root = Path(root) / 'AC_MODELS' / AC
    completed, partial = ac_root / 'completed_epoch.json', ac_root / 'partial_epoch.json'
    rec = json.loads(completed.read_text()) if completed.exists() else None
    part = json.loads(partial.read_text()) if partial.exists() else None
    # Epoch numbers are absolute and a warm start inherits the source checkpoint's (an arm initialized from KEEP/<arm>/epoch_001 numbers
    # its own first epoch 2), so this run's progress is the number of epochs completed under its OWN root, never the absolute number.
    finished_here = len([d for d in ac_root.glob('epoch_[0-9]*') if d.is_dir() and '_step_' not in d.name]) if ac_root.exists() else 0
    if part is not None and (rec is None or part['epoch_number'] > rec['epoch_number']):
        rec, skip, epochs_done = part, part['step'], finished_here              # mid-epoch checkpoint: restart that epoch, skip its trained steps
    elif rec is not None:
        skip, epochs_done = 0, finished_here
    else:
        return None
    if epochs_done >= arm['epochs']:
        raise RuntimeError(f"{epochs_done} epochs already completed under {root} (only the final evaluation was lost); move the root aside to rerun, or inspect")
    ac_path, reader_path = Path(root) / rec['ac_checkpoint'], Path(root) / rec['target_lora_checkpoint']
    if not (ac_path / 'optimizer.pt').exists():
        raise RuntimeError(f'{ac_path} has no optimizer.pt (saved by a trainer older than the resume support); move the root aside to rerun')
    attempt = 1 + len(list(folder.glob('report_data.attempt*.json')))
    for name in ('report_data.json', 'report.html', 'training_stats.json'):
        if (folder / name).exists(): (folder / name).rename(folder / f'{Path(name).stem}.attempt{attempt}{Path(name).suffix}')
    return {'epochs_done': epochs_done, 'skip': skip, 'ac': ac_path, 'reader': reader_path, 'attempt': attempt + 1}


def run(label):
    import torch
    from activation.ac_model import ActivationContextTrainer, ActivationContextStudyGenerator
    from activation.common.ac_parts import strip_ac_parts
    arm = arms()[label]
    folder = REPORT_ROOT / label; prep = REPORT_ROOT / 'preparation'; start = time.monotonic()
    resume = resume_state(arm, folder)
    boot = plain_report(folder, f'AC reconstruction — {label} (starting)')
    material = json.loads((prep / arm['material']).read_text())
    if arm['mix']: mix_check(label)
    if arm['continue_from'] and not resume:      # a continuation's first attempt starts from its parent's kept state (a later attempt resumes its own checkpoints)
        kept = KEEP_ROOT / arm['continue_from'] / f"update_{arm['continue_at']:05d}"
        if not (kept / 'kept.json').exists() or not (kept / 'ac' / 'optimizer.pt').exists(): raise RuntimeError(f"{label} continues {arm['continue_from']} at update {arm['continue_at']}: no kept state with its optimizer at {kept}")
        arm = dict(arm, init_ac=str(kept / 'ac'), init_reader=str(kept / 'reader'))
    if arm['train_items'] is None and not arm['mix']:
        calibration = prep / f'calibration_{label}.json'
        if not calibration.exists(): calibration = prep / 'calibration.json'
        if not calibration.exists(): raise RuntimeError('Run a smoke run and `budget` before this arm, or set train_items')
        arm = dict(arm, train_items=json.loads(calibration.read_text())['train_items'])
    if len(material['train']) < (arm['train_items'] or 0):
        raise RuntimeError(f"{arm['train_items']} training passages requested, {len(material['train'])} prepared")
    write_json(folder / 'arm.json', arm)
    with heartbeat(folder):
        phase(boot, 'loading initial checkpoint')
        if resume: phase(boot, f"resuming after epoch {resume['epochs_done']} (attempt {resume['attempt']})")
        h = harness(True, ac_path=resume['ac'] if resume else (Path(arm['init_ac']) if arm['init_ac'] else None),
                    reader_path=resume['reader'] if resume else (Path(arm['init_reader']) if arm['init_reader'] else None), seed=SEED if arm['seed'] is None else arm['seed'],
                    mixer_layers=arm['mixer_layers'], interventions=arm['interventions'], last_layer_intervention=arm['last_layer_intervention'], intervention_layers=arm['intervention_layers'],
                    side_model=arm['side_model'], reader_model=arm['reader_model'], reader_rank=arm['reader_rank'],
                    recursive_keys=arm['recursive_keys'], pooling_stride=arm['pooling_stride'], pooling_window=arm['pooling_window'], encode_batch_tokens=arm['encode_batch_tokens'])
        ac = h.module_manager.get_ac_model(AC); ac.prepare(); h.module_manager.ensure_lora(READER)
        initial_hash = parameter_hash(ac, h)
        deep_hash = parameter_hash(ac, h, deep=True) if ac.is_deep else None
        reader_before = [p.detach().cpu().clone() for p in h.module_manager.lora_parameters(READER)]
        study = ActivationContextStudyGenerator(h, AC)
        plan = None
        if arm['mix']:     # the planned mixed examples (one pass in plan order) and the kind x ratio held panel; the no-rows floor of the panel's items replaces the per-source one
            phase(boot, 'building the mixed plan'); built = time.monotonic()
            train, plan = mix_training(study, arm, material, label, lambda done, total: phase(boot, 'building the mixed plan', examples=f'{done} of {total}'))
            held, no_context = mix_panel(study, arm, material); wiki = []
            plan['build_seconds'] = round(time.monotonic() - built, 1); previous = folder / 'plan.json'
            if resume and previous.exists() and json.loads(previous.read_text())['digest'] != plan['digest']:
                raise RuntimeError(f"{label}: the rebuilt plan differs from the one the checkpoints were trained on ({plan['digest'][:12]} vs {json.loads(previous.read_text())['digest'][:12]}): material, arm or tokenizer changed")
            write_json(previous, plan)
        else:
            rng = random.Random(SEED)
            passages = material['train'][:arm['train_items']]
            n_continue = int(round(arm['continue_share'] * len(passages)))
            chosen = list(range(len(passages))); rng.shuffle(chosen)
            continue_set = set(chosen[:n_continue]) | {i for i, p in enumerate(passages) if p.get('variant_only') == 'continue'}   # nested 1/16·1/16 corner: continue only
            train = (study.generate_reconstruction_samples([p for i, p in enumerate(passages) if i not in continue_set], ratio=arm['ratio'], variant='reconstruct', seed=SEED)
                     + study.generate_reconstruction_samples([p for i, p in enumerate(passages) if i in continue_set], ratio=arm['ratio'], variant='continue', seed=SEED))
            rng.shuffle(train)
            held_passages = material['held']
            if arm['held_per_source']:
                counts = {}; held_passages = []
                for passage in material['held']:
                    if counts.get(passage['source'], 0) < arm['held_per_source']:
                        counts[passage['source']] = counts.get(passage['source'], 0) + 1; held_passages.append(passage)
            held = (study.generate_reconstruction_samples(held_passages, ratio=arm['ratio'], variant='reconstruct', seed=SEED + 1)
                    + study.generate_reconstruction_samples([p for p in held_passages if p.get('variant_only') == 'continue'], ratio=arm['ratio'], variant='continue', seed=SEED + 1))
            sources = sorted({passage['source'] for passage in held_passages})
            if len(sources) > 1:      # mixed material: one panel line per source label ('reconstruct' stays the kind of single-source material); dataset_id holds the source
                held = [replace(item, kind=item.dataset_id + (':continue' if item.info.get('variant') == 'continue' else '')) for item in held]   # the cued corner stays its own line
            held.sort(key=lambda item: item.info['passage_tokens'] > arm['completion_tokens'] - 32)   # stable: the fixed completion samples (first items) fit the generation budget
            wiki = [replace(item, kind='wiki') for item in study.generate_reconstruction_samples(material['wiki'], ratio=arm['ratio'], variant='reconstruct', seed=SEED + 2)]
            no_context = [replace(item, item_id=item.item_id + ':no_context', ac_messages=strip_ac_parts(item.ac_messages)) for item in held]   # kinds kept: per-source floors
        reporting = held + wiki
        r = RunReporter(folder, f'AC reconstruction — {label}: ' + (f'mixed spans, ratios 1/{round(1 / arm["ratio_range"][1])}..1/{round(1 / arm["ratio_range"][0])}' if arm['mix'] else f'ratio 1/{round(1 / arm["ratio"])}')
                        + f', penalty {arm["penalty_weight"]:g}×{arm["penalty_fraction"]:g}, reader LR {arm["reader_lr"]:g}', arm, start)
        r.set_text('summary', 'Run summary', '\n'.join([
            f'Arm {label} · {arm["note"]} · compression 1/{round(1 / arm["ratio"])} · encoder LR {arm["encoder_lr"]:g} · reader LR {arm["reader_lr"]:g}'
            + (f' · no-context penalty λ={arm["penalty_weight"]:g} on {arm["penalty_fraction"]:.0%} of the examples' if arm['reader_lr'] > 0 and arm['penalty_weight'] > 0 else ' · no penalty')
            + (f' · {arm["continue_share"]:.0%} cued-continuation items' if arm['continue_share'] else ''),
            f'Data · {len(train)} training passages (tool outputs from SWE and deep-research trajectories) × {arm["epochs"]} epochs = {math.ceil(len(train) / 8) * arm["epochs"]} updates of 8 · '
            f'held-out panel every {arm["reporting_interval"]:g} epoch: {len(held)} novel + {len(wiki)} Wikipedia passages · no-context floor on the novel panel',
            f'Initial adapters hash {initial_hash[:12]}… · seed {SEED} · SFT on the passage tokens'
            + (f' · deep inputs: residual deltas at the inputs of reader layers {ac.intervention_layers}{" (explicit intervention_layers)" if arm["intervention_layers"] is not None else ""}, '
               f'each read by a zero-initialized row-over-passage attention block over the final side states and a zero-initialized head (heads hash {deep_hash[:12]}…); '
               '"deltas off" panel = the same items with the rows kept and the deltas withheld' if ac.is_deep else ''),
            '',
            'Reading guide · gold NLL is the reconstruction loss per passage token (lower is better); token accuracy is teacher-forced; the no-context line is the same '
            'reader without the rows (the prior; the penalty holds it near the base model, so it may first move toward the base value rather than stay put); wiki above novel means memorized text is being cued rather than compressed.']))
        if arm['mix']: r.set_text('summary', 'Run summary', mix_summary(arm, label, plan, len(held), ac, initial_hash, resume))
        training_config = config(arm, num_epochs=arm['epochs'] - resume['epochs_done'], resume_optimizer=True, resume_skip_steps=resume['skip']) if resume else config(arm)
        if resume: r.epoch_origin = 0     # panels keep absolute epochs across attempts (the reporter would restart the axis at the resumed epoch)
        if resume: r.status['resumed'] = f"attempt {resume['attempt']}: epochs {resume['epochs_done'] + 1}..{arm['epochs']} from {resume['ac'].name}" + (f" (skipping {resume['skip']} trained steps)" if resume['skip'] else '')
        trainer = ActivationContextTrainer(h, training_config)
        # The base model's no-context floor on the held novel panel (no reader adapter): the value the penalty holds the reader to.
        phase(r, 'base no-context floor')
        # Both views stripped: the reference view of an item is its in-context view, which for these items still holds the plain passage.
        base_examples = [trainer.build_example(ac, replace(item, item_id=item.item_id + ':base', in_context_messages=item.ac_messages, in_context_start=item.ac_start), reference=True) for item in no_context]
        trainer._store_teacher_targets(ac, base_examples, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), lora=None, reporting_name='base no-context floor')
        base_floor = float(sum(-trainer.teacher_target_logp[e.teacher_cache_key].float().mean() for e in base_examples) / len(base_examples))
        base_floor_by_source = {}
        for e in base_examples:
            base_floor_by_source.setdefault(e.item.kind, []).append(float(-trainer.teacher_target_logp[e.teacher_cache_key].float().mean()))
        base_floor_by_source = {kind: sum(values) / len(values) for kind, values in base_floor_by_source.items()}
        trainer.teacher_cache.clear(); trainer.teacher_target_logp.clear()
        r.add_data_point('references', {'name': 'base model, no context, held-out novel', 'loss kind': 'sft', 'loss': base_floor, 'measured with': 'no reader adapter; the penalty target'})
        for x in (0, arm['epochs']):
            r._ensure_series('gold_nll', 'base, no context'); r.add_data_point('gold_nll', {'x': x, 'base, no context': base_floor})
        r.status['base no context'] = f'{base_floor:.4f}' + (' · ' + ' '.join(f'{k} {v:.2f}' for k, v in sorted(base_floor_by_source.items())) if len(base_floor_by_source) > 1 else ''); r.render()
        shared_references = prep / 'cache' / f"references_{Path(arm['material']).stem}_r{round(1 / arm['ratio'])}_k{TOP_K}_gold1_n{arm['train_items']}.pt"
        if arm['mix']: shared_references = prep / 'cache' / f"references_{Path(arm['material']).stem}_mix{plan['digest'][:12]}_k{TOP_K}.pt"    # the plan's own references (G4 and its ablations share a plan)
        if training_config.penalty_active and shared_references.exists():
            try: r.status['shared references'] = f"{trainer._merge_teacher_targets(shared_references, None)} loaded"; r.render(force=True)
            except Exception as error: r.status['shared references'] = f'not loaded: {error!r}'
        beats = mix_beats(arm, label, folder, trainer, h, ac, no_context, r, plan, bool(resume)) if arm['mix'] else None      # a mixed arm measures its floor at the beats
        stats = trainer.train(AC, train, reporting_data=reporting, validation_data=() if arm['mix'] else no_context, reporter=r, reporting_callback=beats)
        if stats.no_context_reference.get('examples') and not shared_references.exists():
            shared_references.parent.mkdir(parents=True, exist_ok=True); trainer.save_teacher_targets(shared_references)   # the next arm on this material skips the reference pass
        write_json(folder / 'training_stats.json', stats.summarize())
        if arm['mix']: write_json(folder / 'rates.json', mix_rates(arm, stats, plan))
        phase(r, 'final: seen training subset')
        subset = train[:arm['final_train_subset']]
        seen = trainer.eval(AC, subset, release=False, reporter=r, reporting_name='final_seen_training')
        write_json(folder / 'final_seen.json', {k: v for k, v in seen.items() if k != 'per_item'})
        if arm['reader_lr'] == 0.0:
            assert all(torch.equal(before, p.detach().cpu()) for before, p in zip(reader_before, h.module_manager.lora_parameters(READER))), 'Fixed reader changed'
        last = stats.reporting or {}
        result = {'run': label, 'arm': arm, 'configuration': asdict(config(arm)), 'resumed': {'epochs_done': resume['epochs_done'], 'attempt': resume['attempt']} if resume else None, 'train_items': len(train), 'held_items': len(held), 'wiki_items': len(wiki),
                  'updates': trainer.updates_done.get(AC, 0), 'initial_hash': initial_hash, 'final_hash': parameter_hash(ac, h), 'deep_hash': deep_hash, 'checkpoint': stats.checkpoint_path,
                  'interventions': ac.intervention_summary() or None, 'payload_bytes': stats.payload_bytes,
                  'deltas_off': {k: v for k, v in (stats.deltas_off or {}).items() if k != 'per_item'} if stats.deltas_off else None,
                  'deltas_off_novel': novel_summary(stats.deltas_off or {}), 'deltas_off_wiki': ((stats.deltas_off or {}).get('by_kind') or {}).get('wiki'),
                  'held': {k: v for k, v in last.items() if k not in ('per_item',)},   # blend of both panels; per panel below
                  'held_novel': novel_summary(last), 'held_wiki': (last.get('by_kind') or {}).get('wiki'), 'held_by_source': {k: v for k, v in (last.get('by_kind') or {}).items() if k != 'wiki'}, 'no_context': {k: v for k, v in (stats.epochs[-1].validation or {}).items() if k != 'per_item'} if stats.epochs and stats.epochs[-1].validation else None,
                  'seen_training': {k: v for k, v in seen.items() if k != 'per_item'}, 'no_context_reference': stats.no_context_reference, 'base_floor': base_floor, 'base_floor_by_source': base_floor_by_source,
                  'penalty_last': stats.penalty[-1] if stats.penalty else None,
                  **({'plan': plan, 'beats': str(folder / 'beats.jsonl'), 'kept_state': str(KEEP_ROOT / label / f"update_{arm['updates'] - arm['landing']:05d}")} if arm['mix'] else {}),
                  'duration_seconds': time.monotonic() - start}
        write_json(folder / 'result.json', result)
        r.set_text('result', 'Final result', json.dumps({k: v for k, v in result.items() if k not in ('configuration',)}, indent=2, default=str))
        phase(r, 'complete', run_elapsed=f'{result["duration_seconds"] / 60:.1f} min'); r.finish()


def budget(smoke='S', target='A16', hours=None, scale=None):
    """From a smoke run's rates, the size of `target` (a multiple of 64) whose epochs fit its budget (or `hours`); written to
    preparation/calibration_<target>.json (calibration.json for the A16 anchor, as before). `scale`: multiplies the per-item rates
    (a smoke on shorter passages than the target's material: the ratio of the mean passage tokens; None = the materials' ratio)."""
    if arms()[smoke]['mix']: return mix_budget(smoke, target)
    prep = REPORT_ROOT / 'preparation'; run_folder = REPORT_ROOT / smoke
    stats = json.loads((run_folder / 'training_stats.json').read_text()); result = json.loads((run_folder / 'result.json').read_text())
    arm_s = result['arm']; items = result['train_items']; held = result['held_items'] + result['wiki_items']; epochs = stats['epochs']
    panels = round(1 / arm_s['reporting_interval'])
    train_s = sum(e['training_seconds'] for e in epochs) / (items * len(epochs))
    panel_s = sum(e['reporting_seconds'] for e in epochs) / (panels * held * len(epochs))
    validation_s = sum(e['validation_seconds'] for e in epochs) / (result['held_items'] * len(epochs))
    sample_s = sum(e['sample_seconds'] for e in epochs) / max(1, arm_s['completion_samples'] * len(epochs))
    checkpoint_s = max(e['checkpoint_seconds'] for e in epochs)
    reference = result.get('no_context_reference') or {}
    reference_s = reference.get('seconds', 0) / max(1, reference.get('examples', 1))
    baseline_s = stats['baseline_seconds']; load_s = result['duration_seconds'] - stats['duration_s'] - 60
    anchor = arms()[target]; panels_a = round(1 / anchor['reporting_interval'])
    material = json.loads((prep / anchor['material']).read_text())
    available = len(material['train'])
    if scale is None:
        smoke_material = json.loads((prep / arm_s['material']).read_text())
        scale = statistics.mean(p['tokens'] for p in material['train']) / statistics.mean(p['tokens'] for p in smoke_material['train']) if arm_s['material'] != anchor['material'] else 1.0
    train_s, panel_s, validation_s, reference_s = (x * scale for x in (train_s, panel_s, validation_s, reference_s))
    held_a = len(material['held']) if not anchor['held_per_source'] else sum(min(anchor['held_per_source'], n) for n in collections.Counter(p['source'] for p in material['held']).values())
    wiki_a = len(material['wiki']); budget_seconds = int(hours * 3600) if hours else anchor['budget_seconds']
    def total(n):
        prep_s = load_s + n * reference_s + baseline_s
        epoch_s = n * train_s + panels_a * (held_a + wiki_a) * panel_s + panels_a * held_a * validation_s + anchor['completion_samples'] * sample_s + checkpoint_s
        return 1.15 * (prep_s + anchor['epochs'] * epoch_s + anchor['final_train_subset'] * panel_s + 120)
    fitting = [n for n in range(64, min(available, 400000) + 1, 64) if total(n) <= budget_seconds]
    if not fitting: raise RuntimeError(f'No size fits {budget_seconds} s: 64 items would take {total(64):.0f} s')
    size = fitting[-1]
    record = {'train_items': size, 'prepared_items': available, 'estimated_seconds': round(total(size)), 'budget_seconds': budget_seconds, 'smoke': smoke, 'target': target, 'rate_scale': scale,
              'rates': {'train_s_per_item': train_s, 'panel_s_per_item': panel_s, 'validation_s_per_item': validation_s, 'sample_s_per_item': sample_s,
                        'reference_s_per_item': reference_s, 'checkpoint_s': checkpoint_s, 'load_s': load_s, 'baseline_s': baseline_s}}
    write_json(prep / ('calibration.json' if target == 'A16' else f'calibration_{target}.json'), record); print(json.dumps(record, indent=2))


def mix_summary(arm, label, plan, held, ac, initial_hash, resume):
    first = arm['continue_at'] or 0; share = lambda kind: plan['by_kind'].get(kind, 0) / plan['examples']
    return '\n'.join([
        f'Arm {label} · {arm["note"]} · encoder LR {arm["encoder_lr"]:g} · reader LR {arm["reader_lr"]:g}'
        + (f' · no-context penalty λ={arm["penalty_weight"]:g} on {arm["penalty_fraction"]:.0%} of the examples' if arm['reader_lr'] > 0 and arm['penalty_weight'] > 0 else ' · reader frozen, no penalty' if arm['reader_lr'] == 0 else ' · no penalty'),
        f'Data · updates {first}..{arm["updates"]} of {arm["examples_per_update"]} examples, one pass, no passage a target twice ({plan["examples"]} examples, plan {plan["digest"][:12]}…, block {plan["block"]}) · '
        + ' · '.join(f'{kind} {share(kind):.0%}' for kind in MIX_KINDS) + f' · compression log-uniform in 1/{round(1 / arm["ratio_range"][1])}..1/{round(1 / arm["ratio_range"][0])} · '
        f'spans of {arm["span_tokens"][0]}-{arm["span_tokens"][1]} tokens, long inputs of {arm["long_tokens"][0]}-{arm["long_tokens"][1]} encoder tokens',
        f'Schedule · {arm["warmup"]} warm-up updates, constant, linear landing to {arm["landing_fraction"]:.0%} over the last {arm["landing"]} · held panel every {arm["beat_every"]} updates: '
        f'{arm["panel_items"]} items of each kind at 1/' + ', 1/'.join(str(round(1 / ratio)) for ratio in arm['panel_ratios']) + f' ({held} items; beats.jsonl) · full state every {arm["ckpt_every"]} updates, '
        f'kept at update {arm["updates"] - arm["landing"]} under {KEEP_ROOT / label}' + (f' · continues {arm["continue_from"]} from update {first} with its optimizer' if arm['continue_from'] else ''),
        f'Initial adapters hash {initial_hash[:12]}… · seed {SEED if arm["seed"] is None else arm["seed"]} · SFT on the target tokens'
        + (f' · deep inputs: residual deltas at the inputs of reader layers {ac.intervention_layers}' if ac.is_deep else ' · rows at the input only'),
        '',
        'Reading guide · panel kinds are <kind>@<1/ratio>; loss is per target token; the no-context line is the same reader without the rows on the same held items.'])


def token_study(tokenizer):
    """The study generator on a tokenizer alone (no harness, no model): the reconstruction generators use nothing else."""
    from activation.ac_model import ActivationContextStudyGenerator
    study = ActivationContextStudyGenerator.__new__(ActivationContextStudyGenerator)
    study.tokenizer, study.ac_name, study._token_cache = tokenizer, AC, {}
    return study


def mix_tokens(item, side, reader):
    """(encoder tokens of the input as the encoder reads it, loss-bearing target tokens, rows, reader input tokens) of an item, as the trainer builds it under the bench's AC config
    (ceil(tokens x ratio) rows, 2 at least, 4,096 at most)."""
    from activation.ac_model.ac_model_utils import normalize_messages, prepare_training_history, tokenize_with_parts
    from activation.common.ac_parts import direct_parts
    (part,) = direct_parts(item.ac_messages)
    seen = len(tokenize_with_parts(side, normalize_messages(part['messages']), [], side.pad_token_id or 0)[0])
    rows = min(4096, max(2, math.ceil(seen * part['compression_target'])))
    return seen, sum(len(ids) for ids in item.assistant_token_ids), rows, len(prepare_training_history(reader, item.ac_messages, item.ac_start, item.assistant_token_ids, [rows])[0])


def plan(label, sizes=(8, 12, 16), reader_tokenizer=None):
    """CPU only (tokenizers, no model): the training plan of a mixed arm's block at each number of examples per update in `sizes`, as a token table per kind — examples, encoder
    tokens seen, loss-bearing tokens reconstructed, rows, reader input tokens — with the distinct target and context passages; written to preparation/plan_<label>.json and
    printed. A first block is built once at the widest size (a narrower plan is its first columns). `reader_tokenizer`: a stand-in for the reader's tokenizer when it is not on
    this machine (the encoder-side columns do not depend on it)."""
    from transformers import AutoTokenizer
    table = arms(); arm = table[label]; chain = mix_check(label, table); prep = REPORT_ROOT / 'preparation'; started = time.monotonic()
    material = json.loads((prep / arm['material']).read_text()); steps = chain[-1][3] - chain[-1][2]; block = len(chain) - 1
    side = AutoTokenizer.from_pretrained(arm['side_model'])
    try: reader = AutoTokenizer.from_pretrained(reader_tokenizer or arm['reader_model'])
    except Exception as error: raise RuntimeError(f"no tokenizer of {reader_tokenizer or arm['reader_model']} on this machine ({error!r}); pass --reader-tokenizer <a model of the same family>") from error
    study = token_study(reader); fields = ('examples', 'seen_tokens', 'reconstructed_tokens', 'rows', 'reader_input_tokens'); results = {}
    fitting = [size for size in sizes if mix_offset(chain, size) + size * math.ceil(LANE * steps) <= len(material['train'])]
    for size in sizes:
        if size not in fitting: results[str(size)] = {'error': f"{mix_offset(chain, size) + size * math.ceil(LANE * steps)} target passages reserved, {len(material['train'])} in {arm['material']}"}
    for build in ([max(fitting)] if block == 0 and fitting else fitting):     # a later block's lanes start after the earlier blocks', which depends on the size: one build per size
        covered = [size for size in fitting if size <= build] if block == 0 else [build]
        tallies = {size: {'kinds': {kind: dict.fromkeys(fields, 0) for kind in MIX_KINDS}, 'targets': set(), 'context': collections.Counter(), 'replaced': 0, 'ids': {}, 'long': [], 'bm25': 0} for size in covered}
        for done, (slot, item) in enumerate(mix_block(study, arm, material, block, steps, mix_offset(chain, build), build), 1):
            counts = (1, *mix_tokens(item, side, reader))
            if done % 5000 == 0: print(f'{label}: {done} of {steps * build} examples planned, {time.monotonic() - started:.0f} s', flush=True)
            for size in covered:
                if slot['column'] >= size: continue
                tally = tallies[size]; row = tally['kinds'][slot['kind']]
                for field, value in zip(fields, counts): row[field] += value
                tally['targets'].add(slot['target']); tally['context'].update(slot['context']); tally['replaced'] += slot['replaced']; tally['ids'][slot['update'], slot['column']] = item.item_id
                if slot['kind'].startswith('long'): tally['long'].append(counts[1]); tally['bm25'] += item.info['composition'] == 'bm25'
        for size, tally in tallies.items():
            total = {field: sum(row[field] for row in tally['kinds'].values()) for field in fields}; lengths = sorted(tally['long']); uses = tally['context']
            results[str(size)] = {
                'digest': hashlib.sha256('\n'.join(tally['ids'][update, column] for update in range(steps) for column in range(size)).encode()).hexdigest(),
                'by_kind': tally['kinds'], 'total': total, 'per_update': {field: total[field] / steps for field in fields},
                'target_passages': len(tally['targets']), 'context_passages': len(uses), 'context_uses': sum(uses.values()), 'context_mean_uses': sum(uses.values()) / max(1, len(uses)),
                'context_max_uses': max(uses.values(), default=0), 'passages_target_or_context': len(tally['targets'] | set(uses)), 'replaced_targets': tally['replaced'],
                'long_input_tokens': {'min': lengths[0], 'p10': lengths[len(lengths) // 10], 'median': lengths[len(lengths) // 2], 'mean': sum(lengths) / len(lengths), 'p90': lengths[len(lengths) * 9 // 10], 'max': lengths[-1]} if lengths else None,
                'long_bm25_share': tally['bm25'] / max(1, len(lengths))}
    record = {'run': label, 'block': block, 'first_update': chain[-1][2], 'updates': steps, 'material': arm['material'], 'train_passages': len(material['train']), 'held_passages': len(material['held']),
              'side_tokenizer': arm['side_model'], 'reader_tokenizer': reader_tokenizer or arm['reader_model'],
              'reader_tokenizer_stand_in': bool(reader_tokenizer and reader_tokenizer != arm['reader_model']), 'seconds': round(time.monotonic() - started, 1), 'sizes': results}
    write_json(prep / f'plan_{label}.json', record)
    for size, result in results.items():
        print(f"\n{label} · {size} examples per update · {steps} updates" + (f" · {result['error']}" if 'error' in result else f" · plan {result['digest'][:12]}"))
        if 'error' in result: continue
        print(f"{'kind':<10}{'examples':>10}{'seen tokens':>16}{'mean':>9}{'reconstructed':>16}{'mean':>8}{'rows':>14}{'mean':>8}{'reader input':>16}{'mean':>8}")
        for kind, row in [*result['by_kind'].items(), ('total', result['total'])]:
            n = max(1, row['examples'])
            print(f"{kind:<10}{row['examples']:>10,}{row['seen_tokens']:>16,}{row['seen_tokens'] / n:>9.0f}{row['reconstructed_tokens']:>16,}{row['reconstructed_tokens'] / n:>8.1f}"
                  f"{row['rows']:>14,}{row['rows'] / n:>8.1f}{row['reader_input_tokens']:>16,}{row['reader_input_tokens'] / n:>8.1f}")
        print(f"target passages {result['target_passages']:,} (distinct; {result['replaced_targets']} slots took a spare) · context passages {result['context_passages']:,} distinct, used {result['context_uses']:,} times "
              f"(mean {result['context_mean_uses']:.2f}, max {result['context_max_uses']}) · long inputs {result['long_input_tokens']} · BM25-filled {result['long_bm25_share']:.0%}")
    print(f"\nreader tokenizer: {record['reader_tokenizer']}" + (' (stand-in: the reconstructed / reader-input columns and the span cuts follow it)' if record['reader_tokenizer_stand_in'] else '') + f" · {record['seconds']:.0f} s · {prep / f'plan_{label}.json'}")
    return record


def mix_budget(smoke, target, sizes=(8, 12, 16)):
    """Hours of the mixed arm `target` at each number of examples per update, from the measured rates (rates.json) of the smoke run `smoke` — and of its `_e12` / `_e16` twins
    where they ran (else the smoke's seconds per example times the size; with preparation/plan_<target>.json also its seconds per encoder token times the planned tokens).
    Written to preparation/budget_<target>.json."""
    prep = REPORT_ROOT / 'preparation'; arm = arms()[target]; steps = arm['updates'] - (arm['continue_at'] or 0); frozen = arm['reader_lr'] == 0 or arm['penalty_weight'] == 0
    planned = json.loads((prep / f'plan_{target}.json').read_text())['sizes'] if (prep / f'plan_{target}.json').exists() else {}
    base = json.loads((REPORT_ROOT / smoke / 'rates.json').read_text()); record = {'smoke': smoke, 'target': target, 'updates': steps, 'peak_memory_gb': {}, 'sizes': {}}
    for size in sizes:
        twin = next((folder for folder in (smoke, f'{smoke}_e{size}') if (REPORT_ROOT / folder / 'rates.json').exists()
                     and json.loads((REPORT_ROOT / folder / 'rates.json').read_text())['examples_per_update'] == size), None)
        rates = json.loads((REPORT_ROOT / twin / 'rates.json').read_text()) if twin else base
        update_s = rates['seconds_per_update']['mean'] if twin else base['seconds_per_example'] * size
        by_tokens = base['seconds_per_side_token'] * planned[str(size)]['per_update']['seen_tokens'] if 'per_update' in planned.get(str(size), {}) else None
        fixed = (steps * size * (rates['build_seconds_per_example'] + (0 if frozen else rates['reference_seconds_per_example'])) + rates['placement_seconds']
                 + (steps // arm['beat_every'] + 1) * max(rates['panel_seconds'], rates['baseline_seconds']) * arm['panel_items'] / rates['panel_items'] + (steps // arm['ckpt_every']) * rates['checkpoint_seconds'])
        record['sizes'][str(size)] = {'measured_by': twin or f'{smoke} (seconds per example x {size})', 'seconds_per_update': update_s, 'seconds_per_example': update_s / size,
                                      'training_hours': steps * update_s / 3600, 'fixed_hours': fixed / 3600, 'hours': (steps * update_s + fixed) / 3600,
                                      **({'hours_by_planned_tokens': (steps * by_tokens + fixed) / 3600} if by_tokens else {})}
        record['peak_memory_gb'][twin or smoke] = rates['peak_memory_gb']
    record['smoke_rates'] = base
    write_json(prep / f'budget_{target}.json', record); print(json.dumps(record, indent=2))
    for size, row in record['sizes'].items():
        print(f"{target} at {size:>2} examples per update: {row['seconds_per_update']:6.1f} s per update ({row['measured_by']}) · {steps} updates {row['training_hours']:5.1f} h + panels, checkpoints, start-up {row['fixed_hours']:4.1f} h = {row['hours']:5.1f} h"
              + (f" ({row['hours_by_planned_tokens']:.1f} h by the plan's tokens)" if 'hours_by_planned_tokens' in row else ''))
    print('peak GPU memory: ' + ', '.join(f'{name} {value} GB' for name, value in record['peak_memory_gb'].items()))
    return record


def overview():
    r = plain_report(REPORT_ROOT, 'AC reconstruction — overview')
    r.initialize_table('runs', 'Runs', 'One row per run folder.', ['run', 'phase', 'epoch', 'updates', 'held gold NLL', 'token acc', 'updated'])
    for folder in sorted(REPORT_ROOT.iterdir()):
        data = folder / 'report_data.json'
        if folder.name == 'preparation' or not data.exists(): continue
        d = json.loads(data.read_text()); s = d['status']
        r.add_data_point('runs', {'run': folder.name, 'phase': s.get('phase'), 'epoch': s.get('epoch'), 'updates': s.get('global updates'), 'held gold NLL': '', 'token acc': '', 'updated': d.get('updated_at')})
    r.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'budget', 'overview', 'plan']); parser.add_argument('--run')
    parser.add_argument('--smoke', default='S'); parser.add_argument('--target', default='A16'); parser.add_argument('--hours', type=float, default=None); parser.add_argument('--scale', type=float, default=None)
    parser.add_argument('--sizes', default='8,12,16'); parser.add_argument('--reader-tokenizer', default=None)
    args = parser.parse_args()
    try:
        if args.command == 'run': run(args.run)
        elif args.command == 'budget': budget(args.smoke, args.target, args.hours, args.scale)
        elif args.command == 'plan': plan(args.run, tuple(int(size) for size in args.sizes.split(',')), args.reader_tokenizer)
        else: globals()[args.command]()
    except BaseException as error:
        folder = REPORT_ROOT / (args.run if args.command == 'run' else 'preparation')
        write_json(folder / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'time': time.time()})
        try:
            r = plain_report(folder, f'AC reconstruction — {args.run or args.command} (failed)'); r.set_status(phase='error', error=repr(error)); r.set_text('traceback', 'Traceback', traceback.format_exc()); r.finish()
        finally:
            raise
