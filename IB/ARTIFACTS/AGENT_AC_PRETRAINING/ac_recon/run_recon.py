"""AC reconstruction round: dense, oracle-free compression training on text the model has not memorized.

Commands: `prepare` (chunk pools: train / held novel / Wikipedia panel), `run --run <label>` (an arm from ARMS or arms.json),
`budget` (arm size that fits the anchors' budget from the smoke run's rates), `overview`.
"""
from __future__ import annotations
import argparse
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
                   final_train_subset=512, micro_batch=1, ckpt_min_tokens=2048, gold_targets=False, reader_frozen_updates=0, material='material.json', gate_lr=None, fresh_initial=False, row_head_skip=False, lora_targets_extra=[], seed=None, lr_schedule='constant', warmup=10, clip_per_group=False, penalty_exact=False, ckpt_every=None, mixer_layers=None, row_scale_init=1.0, loss_weighting='example', gates_scope='all', penalty_ref='stored',
                   interventions=0, last_layer_intervention=True, prefer_attention=True, intervention_source='summary', intervention_target='residual', intervention_attention_depth='final', scale_fraction=0.05, delta_scale_lr=1e-3, heads_frozen=False, detach_source=False, deltas_off=True, head_style='scaled', calibration='no_rows', heads_only_updates=0, transfer_init=1.0, transfer_correction=False, intervention_layers=None,
                   distill_weight=0.0, distill_layers=None, distill_source='attn_out', distill_hold=0.25, distill_decay_end=0.75, distill_floor=0.05,
                   encoder_passes=1, rare_weight=0.0, pool_surprisal=0.0, refine_lr=None,
                   init_ac=None, init_reader=None, note='')   # encoder_passes: iterative encoder (side-decoder passes over [content; summaries]; pass k >= 2 re-reads with summaries_1 + refine_adapter(rows of pass k - 1), zero-init; rows and K/V from the last pass; 1 = today); rare_weight: surprisal-weighted terminal loss (per-token weight 1 + rare_weight x s_t / mean(s), s_t = the base's no-context surprisal from the stored reference; metric unweighted; 0 = off); pool_surprisal: surprisal-weighted summary pooling (bins by cumulative mass 1 + pool_surprisal x s_t / mean(s) + a zero-init per-head logit bias; rows per part unchanged; 0 = today's bins); intervention_layers: an explicit list of reader layers instead of the frequency rule (with interventions 0); distill_*: dense supervision - the base reader's full-context attention outputs distilled into the row-fed reader at the target positions (weight 0 = off; layers: list, -1 = every full-attention layer, null = the intervention layers of a deep arm else every full-attention layer; the hold / decay-end fractions and the floor share of the λ schedule)
                   # interventions: deep inputs (interior reader layers with a delta per row; 0 = embedding channel only; -1 = every full-attention layer); transfer_init / transfer_correction: kv_transfer's takeover init and optional zero-init correction head; deltas_off: the extra panel without the deltas (deep arms only); head_style: 'scaled' (unit rows x calibrated x learned relative scale) or 'lora' (zero-initialized down projection, no scale); calibration: 'no_rows' (stripped histories) or 'row_positions' (real sequences, at the rows); heads_only_updates: first N updates train the intervention modules only
ARMS = {
    'S': dict(DEFAULT_ARM, train_items=512, epochs=2, reporting_interval=0.25, completion_samples=8, note='smoke: shape and timing'),
    'A16': dict(DEFAULT_ARM, ratio=1 / 16, note='anchor 1/16'),
    'A4': dict(DEFAULT_ARM, ratio=1 / 4, note='anchor 1/4'),
}
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


def harness(with_ac=False, initial=None, row_head_skip=False, lora_targets_extra=(), ac_path=None, reader_path=None, seed=SEED, mixer_layers=None, row_scale_init=1.0,
            interventions=0, last_layer_intervention=True, prefer_attention=True, intervention_source='summary', intervention_target='residual',
            scale_fraction=0.05, heads_frozen=False, detach_source=False, intervention_attention_depth='final', head_style='scaled', calibration='no_rows',
            transfer_init=1.0, transfer_correction=False, intervention_layers=None, encoder_passes=1, pool_surprisal=0.0):
    # encoder_passes / pool_surprisal: the round-8 write-side settings of the AC model config (the iterative encoder's passes; the surprisal-weighted summary pooling's strength)
    # intervention_layers: an explicit list of reader layers (the config's intervention_layers; needs interventions 0, the frequency rule is replaced)
    # head_style: 'scaled' (today's unit rows x calibrated RMS x learned relative scale) or 'lora' (down(silu(up(z))), down zero-initialized: zero delta at init, no scale group)
    # calibration: 'no_rows' (median residual RMS of the stripped histories) or 'row_positions' (mean RMS at the row positions of real sequences with the rows present)
    # intervention_attention_depth: for intervention_source 'passage_attention', 'final' (z queries the passage's final states) or 'matched' (side layer s(l) states)
    # scale_fraction: the heads' relative scales start at this share of the calibrated RMS; heads_frozen / detach_source: the NDIR / NDID controls
    # intervention_source: what the delta heads read ('summary' = the final side state, 'side_layers' = the side state at the matching layer);
    # intervention_target: 'residual' (layer input), 'kv' (pre-RoPE K/V deltas added at the row positions, full-attention layers only), 'kv_slots' (the row
    #   positions' pre-RoPE K/V REPLACED by keep x the reader's own + a side-computed slot content; zero-init head and takeover 0 at init, so bit-identical to no
    #   intervention; needs head_style 'lora'), or 'cross_attention' (a MEASUREMENT, not a candidate: every reader position adds a zero-init cross-attention over the
    #   side model's final states of the whole passage; zero compression, not engine-compatible; needs intervention_source 'summary'), 'kv_transfer' (the row positions'
    #   pre-RoPE K/V REPLACED by (1 - t) x the reader's own + t x the SIDE MODEL'S OWN K/V at the same layer, t = transfer_init, optional zero-init correction head;
    #   needs 'side_layers' and head_style 'lora'; the engine mapping is the kv_slots one) or 'kv_transfer_full' (CEILING: the same splice over the whole passage run)
    # interventions: 0 = input channel only; N = N interior layers (+ last); -1 = every full-attention layer of the reader
    # seed: the fresh modules' initialization (replicate arms with their own seed differ in init and data order, not data order only)
    # ac_path/reader_path: explicit adapter folders (a resume's latest epoch); otherwise the shared `initial` pair
    import torch
    from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
    from activation.harness.module_manager import DEFAULT_LORA_TARGET_MODULES
    from activation.ac_model import ActivationContextModelConfig
    targets = [*DEFAULT_LORA_TARGET_MODULES, *lora_targets_extra]
    if intervention_layers is not None and interventions != 0:
        raise ValueError(f'intervention_layers {intervention_layers} replace the frequency rule: set interventions 0 (got {interventions})')
    torch.set_num_threads(4); torch.manual_seed(seed)
    h = HarnessRuntime(HarnessRuntimeConfig(
        model_configs={NAME: ModelConfig(NAME, MODEL, engine_kwargs={'max_model_len': 4096, 'gpu_memory_utilization': .75})},
        doc_chunk_size_chars=2048, doc_chunk_overlap_chars=0))
    if with_ac:
        h.module_manager.register_lora(READER, NAME, rank=64, dropout=0, checkpoint_path=str(reader_path) if reader_path else (str(initial / 'reader') if initial else None), target_modules=targets)
        h.module_manager.register_ac_model(ActivationContextModelConfig(AC, NAME, SIDE, NAME, READER, min_view_rows=2, default_compression_ratio=1 / 16,
                                                                        checkpoint_path=str(ac_path) if ac_path else (str(initial / 'ac') if initial else None),
                                                                        row_head_skip=row_head_skip, side_lora_target_modules=targets, row_scale_init=row_scale_init,
                                                                        intervention_frequency=interventions, add_last_layer_intervention=last_layer_intervention, prefer_attention_interventions=prefer_attention,
                                                                        intervention_source=intervention_source, intervention_target=intervention_target,
                                                                        intervention_scale_fraction=scale_fraction, intervention_heads_frozen=heads_frozen, intervention_detach_source=detach_source,
                                                                        intervention_attention_depth=intervention_attention_depth,
                                                                        intervention_head_style=head_style, intervention_calibration=calibration,
                                                                        intervention_transfer_init=transfer_init, intervention_transfer_correction=transfer_correction,
                                                                        intervention_layers=list(intervention_layers) if intervention_layers is not None else None,
                                                                        encoder_passes=int(encoder_passes), pool_surprisal=float(pool_surprisal),
                                                                        **({'mixer_layers': mixer_layers} if mixer_layers is not None else {})))
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
    values = dict(loss_kind='sft', learning_rate_ac=arm['encoder_lr'], learning_rate_target_lora=arm['reader_lr'], examples_per_update=8,
                  num_epochs=arm['epochs'], warmup_updates=arm['warmup'], lr_schedule=arm['lr_schedule'], clip_per_group=arm['clip_per_group'], penalty_count_exact=arm['penalty_exact'], checkpoint_every_updates=arm['ckpt_every'], persist_no_context_reference=True, loss_weighting=arm['loss_weighting'], learning_rate_gates_scope=arm['gates_scope'], no_context_penalty_reference=arm['penalty_ref'], completion_samples=arm['completion_samples'], completion_max_new_tokens=arm['completion_tokens'],
                  reporting_interval=arm['reporting_interval'], seed=SEED if arm['seed'] is None else arm['seed'], teacher_cache_top_k=TOP_K, gradient_checkpointing_min_tokens=arm['ckpt_min_tokens'], micro_batch_examples=arm['micro_batch'], teacher_targets_include_gold=arm['gold_targets'], target_lora_frozen_updates=arm['reader_frozen_updates'], learning_rate_gates=arm['gate_lr'], deltas_off_panel=arm['deltas_off'], learning_rate_delta_scales=arm['delta_scale_lr'], intervention_heads_only_updates=arm['heads_only_updates'],
                  no_context_penalty_weight=arm['penalty_weight'] if arm['reader_lr'] > 0 else 0.0, no_context_penalty_fraction=arm['penalty_fraction'],
                  distill_weight=arm['distill_weight'], distill_layers=arm['distill_layers'], distill_source=arm['distill_source'],
                  distill_hold=arm['distill_hold'], distill_decay_end=arm['distill_decay_end'], distill_floor=arm['distill_floor'],
                  rare_weight=arm['rare_weight'], learning_rate_refine=arm['refine_lr'],
                  checkpoint_every_epoch=True)
    values.update(overrides)
    return ActivationContextTrainingConfig(**values)


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
                                 set_labels={'reporting': 'AC student, held-out', 'validation': 'no context, current reader, held-out novel', 'deltas_off': 'AC student, deltas off, held-out',
                                             'passes_off': 'AC student, passes off (pass-1 rows), held-out'})
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
                if not summary['items'] or name not in ('reporting', 'deltas_off', 'passes_off'): return
                for kind, values in summary['by_kind'].items():
                    label = {'reconstruct': 'novel', 'continue': 'novel (continue)', 'wiki': 'wiki'}.get(kind, kind) + {'deltas_off': ' (deltas off)', 'passes_off': ' (passes off)'}.get(name, '')
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


def resume_state(arm, folder):
    """A previous attempt's last completed epoch under this run's checkpoint root (ACTIVATION_SYNC_ROOT), or None for a fresh run.
    The bench reloads that epoch's adapters + optimizer.pt and trains the remaining epochs (data order is seeded per epoch number, so it continues the schedule)."""
    root = os.environ.get('ACTIVATION_SYNC_ROOT')
    if not root: return None
    completed, partial = Path(root) / 'AC_MODELS' / AC / 'completed_epoch.json', Path(root) / 'AC_MODELS' / AC / 'partial_epoch.json'
    rec = json.loads(completed.read_text()) if completed.exists() else None
    part = json.loads(partial.read_text()) if partial.exists() else None
    if part is not None and (rec is None or part['epoch_number'] > rec['epoch_number']):
        rec, skip, epochs_done = part, part['step'], part['epoch_number'] - 1    # mid-epoch checkpoint: restart that epoch, skip its trained steps
    elif rec is not None:
        skip, epochs_done = 0, rec['epoch_number']
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
    if arm['train_items'] is None:
        calibration = prep / 'calibration.json'
        if not calibration.exists(): raise RuntimeError('Run the smoke run (S) and `budget` before an anchor arm, or set train_items')
        arm = dict(arm, train_items=json.loads(calibration.read_text())['train_items'])
    if len(material['train']) < arm['train_items']:
        raise RuntimeError(f"{arm['train_items']} training passages requested, {len(material['train'])} prepared")
    write_json(folder / 'arm.json', arm)
    with heartbeat(folder):
        phase(boot, 'loading initial checkpoint')
        if resume: phase(boot, f"resuming after epoch {resume['epochs_done']} (attempt {resume['attempt']})")
        h = harness(True, None if arm['fresh_initial'] else prep / 'initial', arm['row_head_skip'], arm['lora_targets_extra'],   # fresh: seeded new adapters (architecture arms cannot load the shared initial)
                    ac_path=resume['ac'] if resume else (Path(arm['init_ac']) if arm['init_ac'] else None),   # init_*: start from another run's adapters (the AC modules are ratio-free)
                    reader_path=resume['reader'] if resume else (Path(arm['init_reader']) if arm['init_reader'] else None), seed=SEED if arm['seed'] is None else arm['seed'],
                    mixer_layers=arm['mixer_layers'], row_scale_init=arm['row_scale_init'],
                    interventions=arm['interventions'], last_layer_intervention=arm['last_layer_intervention'], prefer_attention=arm['prefer_attention'],
                    intervention_source=arm['intervention_source'], intervention_target=arm['intervention_target'],
                    scale_fraction=arm['scale_fraction'], heads_frozen=arm['heads_frozen'], detach_source=arm['detach_source'],
                    intervention_attention_depth=arm['intervention_attention_depth'], head_style=arm['head_style'], calibration=arm['calibration'],
                    transfer_init=arm['transfer_init'], transfer_correction=arm['transfer_correction'], intervention_layers=arm['intervention_layers'],
                    encoder_passes=arm['encoder_passes'], pool_surprisal=arm['pool_surprisal'])
        ac = h.module_manager.get_ac_model(AC); ac.prepare(); h.module_manager.ensure_lora(READER)
        initial_hash = parameter_hash(ac, h)
        deep_hash = parameter_hash(ac, h, deep=True) if ac.is_deep else None
        reader_before = [p.detach().cpu().clone() for p in h.module_manager.lora_parameters(READER)]
        study = ActivationContextStudyGenerator(h, AC)
        rng = random.Random(SEED)
        passages = material['train'][:arm['train_items']]
        n_continue = int(round(arm['continue_share'] * len(passages)))
        chosen = list(range(len(passages))); rng.shuffle(chosen); continue_set = set(chosen[:n_continue])
        train = (study.generate_reconstruction_samples([p for i, p in enumerate(passages) if i not in continue_set], ratio=arm['ratio'], variant='reconstruct', seed=SEED)
                 + study.generate_reconstruction_samples([p for i, p in enumerate(passages) if i in continue_set], ratio=arm['ratio'], variant='continue', seed=SEED))
        rng.shuffle(train)
        held = study.generate_reconstruction_samples(material['held'], ratio=arm['ratio'], variant='reconstruct', seed=SEED + 1)
        held.sort(key=lambda item: item.info['passage_tokens'] > arm['completion_tokens'] - 32)   # stable: the fixed completion samples (first items) fit the generation budget
        wiki = [replace(item, kind='wiki') for item in study.generate_reconstruction_samples(material['wiki'], ratio=arm['ratio'], variant='reconstruct', seed=SEED + 2)]
        no_context = [replace(item, item_id=item.item_id + ':no_context', ac_messages=strip_ac_parts(item.ac_messages)) for item in held]
        reporting = held + wiki
        r = RunReporter(folder, f'AC reconstruction — {label}: ratio 1/{round(1 / arm["ratio"])}, penalty {arm["penalty_weight"]:g}×{arm["penalty_fraction"]:g}, reader LR {arm["reader_lr"]:g}', arm, start)
        r.set_text('summary', 'Run summary', '\n'.join([
            f'Arm {label} · {arm["note"]} · compression 1/{round(1 / arm["ratio"])} · encoder LR {arm["encoder_lr"]:g} · reader LR {arm["reader_lr"]:g}'
            + (f' · no-context penalty λ={arm["penalty_weight"]:g} on {arm["penalty_fraction"]:.0%} of the examples' if arm['reader_lr'] > 0 and arm['penalty_weight'] > 0 else ' · no penalty')
            + (f' · {arm["continue_share"]:.0%} cued-continuation items' if arm['continue_share'] else '')
            + (f' · dense supervision: the base reader\'s full-context attention outputs (after o_proj, no adapters, no rows) distilled into the row-fed reader at the target positions of '
               f'{"reader layers " + str(arm["distill_layers"]) if isinstance(arm["distill_layers"], list) else "every full-attention reader layer" if arm["distill_layers"] == -1 or not ac.is_deep else "the intervention layers"}, '
               f'λ {arm["distill_weight"]:g} held for {arm["distill_hold"]:.0%} of the run then decaying linearly to {arm["distill_floor"]:g} × λ at {arm["distill_decay_end"]:.0%} '
               f'(the term per layer is mean ||A_student - A_teacher||² / the teacher\'s mean squared norm; panels score it with the rows on and off and the rows\' attention share)' if arm['distill_weight'] > 0 else '')
            + (f' · iterative encoder: {arm["encoder_passes"]} side-decoder passes (pass k ≥ 2 re-reads [content; summaries_1 + refine_adapter(rows of pass k − 1)], zero-init adapter; rows and side K/V from the last pass; '
               f'side cost × {arm["encoder_passes"]}; "passes off" panel = the same items with the rows of pass 1)' if arm['encoder_passes'] > 1 else '')
            + (f' · surprisal-weighted terminal loss: per-token weight 1 + {arm["rare_weight"]:g} × s_t / mean(s) (s_t = the base\'s no-context surprisal of the target token, from the stored reference; '
               f'weights average 1 per example; the reported losses stay unweighted)' if arm['rare_weight'] > 0 else '')
            + (f' · surprisal-weighted summary pooling: bins by cumulative mass 1 + {arm["pool_surprisal"]:g} × s_t / mean(s) (s_t = the side base\'s no-context surprisal of the content token) '
               f'plus a zero-init per-head logit bias × {arm["pool_surprisal"]:g}; rows per part unchanged' if arm['pool_surprisal'] > 0 else '')
            + (' · rare-token panel: the held-out loss / accuracy on the target tokens in each item\'s top 20 % of base surprisal' if arm['rare_weight'] > 0 or arm['pool_surprisal'] > 0 else ''),
            f'Data · {len(train)} training passages (tool outputs from SWE and deep-research trajectories) × {arm["epochs"]} epochs = {math.ceil(len(train) / 8) * arm["epochs"]} updates of 8 · '
            f'held-out panel every {arm["reporting_interval"]:g} epoch: {len(held)} novel + {len(wiki)} Wikipedia passages · no-context floor on the novel panel',
            f'Initial adapters hash {initial_hash[:12]}… · seed {SEED} · SFT on the passage tokens'
            + ((f' · deep inputs: {"pre-RoPE K/V deltas at the row positions of" if ac.kv_interventions else "pre-RoPE K/V slots REPLACING the row positions of (keep x the reader's own + side-computed content; takeover 0 at init)" if ac.kv_slots else "CEILING MEASUREMENT (not a candidate): the SIDE MODEL'S OWN pre-RoPE K/V of the WHOLE passage run REPLACING the reader's at" if ac.kv_transfer_full else f"the SIDE MODEL'S OWN pre-RoPE K/V REPLACING the row positions' at ((1 - t) x the reader's own + t x the side's, t init {arm['transfer_init']:g}{', zero-init correction head' if arm['transfer_correction'] else ''})" if ac.kv_transfer else "CEILING MEASUREMENT (not a candidate): cross-attention of every reader position over the whole passage at" if ac.cross_attention else "deltas at the inputs of"} reader layers {ac.intervention_layers}{" (explicit intervention_layers)" if arm["intervention_layers"] is not None else ""}'
                + (' read from the side model\'s final states of all passage tokens (zero compression, not engine-compatible)' if ac.cross_attention else
                   f' read from {"the side state at layers " + str(ac.intervention_source_layers) if ac.config.intervention_source == "side_layers" else ("row-over-passage attention at " + ("side layers " + str(ac.intervention_source_layers) if ac.config.intervention_attention_depth == "matched" else "the final side states") + ", zero-initialized out projection") if ac.config.intervention_source == "passage_attention" else "the final side state"}')
                + f' (heads hash {deep_hash[:12]}…; ' + ('zero-initialized cross-attention blocks, no heads' if ac.cross_attention else f'the side model\'s K/V captured in the side pass, takeovers at {arm["delta_scale_lr"]:g}' if ac.kv_transfer else f'zero-initialized slot heads around the reader\'s own K/V, takeovers at {arm["delta_scale_lr"]:g}' if ac.kv_slots else f'relative scales from {arm["scale_fraction"]:g} x RMS at {arm["delta_scale_lr"]:g}' if arm['head_style'] == 'scaled' else 'lora-style heads: zero-initialized down projection, no scale group')
                + f'; calibration on {"the row positions of real sequences" if arm["calibration"] == "row_positions" else "stripped histories (no rows)"}'
                + (f'; first {arm["heads_only_updates"]} updates train the intervention modules only' if arm['heads_only_updates'] else '')
                + f'{"; heads frozen" if arm["heads_frozen"] else ""}{"; source detached" if arm["detach_source"] else ""}), "deltas off" panel = the same items with the rows kept and the deltas withheld') if ac.is_deep else ''),
            '',
            'Reading guide · gold NLL is the reconstruction loss per passage token (lower is better); token accuracy is teacher-forced; the no-context line is the same '
            'reader without the rows (the prior; the penalty holds it near the base model, so it may first move toward the base value rather than stay put); wiki above novel means memorized text is being cued rather than compressed.']))
        training_config = config(arm, num_epochs=arm['epochs'] - resume['epochs_done'], resume_optimizer=True, resume_skip_steps=resume['skip']) if resume else config(arm)
        if resume: r.epoch_origin = 0     # panels keep absolute epochs across attempts (the reporter would restart the axis at the resumed epoch)
        if resume: r.status['resumed'] = f"attempt {resume['attempt']}: epochs {resume['epochs_done'] + 1}..{arm['epochs']} from {resume['ac'].name}" + (f" (skipping {resume['skip']} trained steps)" if resume['skip'] else '')
        trainer = ActivationContextTrainer(h, training_config)
        # The base model's no-context floor on the held novel panel (no reader adapter): the value the penalty holds the reader to.
        phase(r, 'base no-context floor')
        # Both views stripped: the reference view of an item is its in-context view, which for these items still holds the plain passage.
        base_examples = [trainer.build_example(ac, replace(item, item_id=item.item_id + ':base', in_context_messages=item.ac_messages, in_context_start=item.ac_start), reference=True) for item in no_context]
        trainer._store_teacher_targets(ac, base_examples, torch.device('cuda'), lora=None, reporting_name='base no-context floor')
        base_floor = float(sum(-trainer.teacher_target_logp[e.teacher_cache_key].float().mean() for e in base_examples) / len(base_examples))
        trainer.teacher_cache.clear(); trainer.teacher_target_logp.clear()
        r.add_data_point('references', {'name': 'base model, no context, held-out novel', 'loss kind': 'sft', 'loss': base_floor, 'measured with': 'no reader adapter; the penalty target'})
        for x in (0, arm['epochs']):
            r._ensure_series('gold_nll', 'base, no context'); r.add_data_point('gold_nll', {'x': x, 'base, no context': base_floor})
        r.status['base no context'] = f'{base_floor:.4f}'; r.render()
        shared_references = prep / 'cache' / f"references_{Path(arm['material']).stem}_r{round(1 / arm['ratio'])}_k{TOP_K}_gold{int(arm['gold_targets'])}_n{arm['train_items']}.pt"
        if arm['penalty_ref'] == 'stored' and arm['reader_lr'] > 0 and arm['penalty_weight'] > 0 and shared_references.exists():
            try: r.status['shared references'] = f"{trainer._merge_teacher_targets(shared_references, None)} loaded"; r.render(force=True)
            except Exception as error: r.status['shared references'] = f'not loaded: {error!r}'
        stats = trainer.train(AC, train, reporting_data=reporting, validation_data=no_context, reporter=r)
        if arm['penalty_ref'] == 'stored' and stats.no_context_reference.get('examples') and not shared_references.exists():
            shared_references.parent.mkdir(parents=True, exist_ok=True); trainer.save_teacher_targets(shared_references)   # the next arm on this material skips the reference pass
        write_json(folder / 'training_stats.json', stats.summarize())
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
                  'deltas_off_novel': ((stats.deltas_off or {}).get('by_kind') or {}).get('reconstruct'), 'deltas_off_wiki': ((stats.deltas_off or {}).get('by_kind') or {}).get('wiki'),
                  'held': {k: v for k, v in last.items() if k not in ('per_item',)},   # blend of both panels; per panel below
                  'held_novel': (last.get('by_kind') or {}).get('reconstruct'), 'held_wiki': (last.get('by_kind') or {}).get('wiki'), 'no_context': {k: v for k, v in (stats.epochs[-1].validation or {}).items() if k != 'per_item'} if stats.epochs and stats.epochs[-1].validation else None,
                  'seen_training': {k: v for k, v in seen.items() if k != 'per_item'}, 'no_context_reference': stats.no_context_reference, 'base_floor': base_floor,
                  'penalty_last': stats.penalty[-1] if stats.penalty else None,
                  'distill': {'layers': stats.distill_layers, 'lambda_last': stats.distill_lambda[-1], 'loss_last': stats.distill_loss[-1], 'panel_last': stats.distill_panel[-1] if stats.distill_panel else None} if stats.distill_lambda else None,
                  'encoder_passes': arm['encoder_passes'], 'rare_weight': arm['rare_weight'], 'pool_surprisal': arm['pool_surprisal'],
                  'passes_off': {k: v for k, v in (stats.passes_off or {}).items() if k != 'per_item'} if stats.passes_off else None,
                  'passes_off_novel': ((stats.passes_off or {}).get('by_kind') or {}).get('reconstruct'), 'passes_off_wiki': ((stats.passes_off or {}).get('by_kind') or {}).get('wiki'),
                  'rare_token': {'held_loss': last.get('rare_token_loss'), 'held_accuracy': last.get('rare_token_accuracy'),
                                 'novel': {k: (last.get('by_kind') or {}).get('reconstruct', {}).get(k) for k in ('rare_token_loss', 'rare_token_accuracy')},
                                 'wiki': {k: (last.get('by_kind') or {}).get('wiki', {}).get(k) for k in ('rare_token_loss', 'rare_token_accuracy')}} if last.get('rare_token_loss') is not None else None,
                  'encoder_trace_last': stats.encoder_trace[-1] if stats.encoder_trace else None, 'encoder_panel_last': stats.encoder_panel[-1] if stats.encoder_panel else None,
                  'duration_seconds': time.monotonic() - start}
        write_json(folder / 'result.json', result)
        r.set_text('result', 'Final result', json.dumps({k: v for k, v in result.items() if k not in ('configuration',)}, indent=2, default=str))
        phase(r, 'complete', run_elapsed=f'{result["duration_seconds"] / 60:.1f} min'); r.finish()


def budget():
    """From the smoke run's rates, the anchor size (multiple of 64) whose epochs fit the anchors' budget."""
    prep = REPORT_ROOT / 'preparation'; smoke = REPORT_ROOT / 'S'
    stats = json.loads((smoke / 'training_stats.json').read_text()); result = json.loads((smoke / 'result.json').read_text())
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
    available = len(json.loads((prep / 'material.json').read_text())['train'])
    anchor = ARMS['A16']; panels_a = round(1 / anchor['reporting_interval'])
    def total(n):
        prep_s = load_s + n * reference_s + baseline_s
        epoch_s = n * train_s + panels_a * held * panel_s + panels_a * result['held_items'] * validation_s + anchor['completion_samples'] * sample_s + checkpoint_s
        return 1.15 * (prep_s + anchor['epochs'] * epoch_s + anchor['final_train_subset'] * panel_s + 120)
    fitting = [n for n in range(64, min(available, 200000) + 1, 64) if total(n) <= anchor['budget_seconds']]
    if not fitting: raise RuntimeError(f'No size fits {anchor["budget_seconds"]} s: 64 items would take {total(64):.0f} s')
    size = fitting[-1]
    record = {'train_items': size, 'prepared_items': available, 'estimated_seconds': round(total(size)), 'budget_seconds': anchor['budget_seconds'],
              'rates': {'train_s_per_item': train_s, 'panel_s_per_item': panel_s, 'validation_s_per_item': validation_s, 'sample_s_per_item': sample_s,
                        'reference_s_per_item': reference_s, 'checkpoint_s': checkpoint_s, 'load_s': load_s, 'baseline_s': baseline_s}}
    write_json(prep / 'calibration.json', record); print(json.dumps(record, indent=2))


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
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'run', 'budget', 'overview']); parser.add_argument('--run')
    args = parser.parse_args()
    try:
        if args.command == 'run': run(args.run)
        else: globals()[args.command]()
    except BaseException as error:
        folder = REPORT_ROOT / (args.run if args.command == 'run' else 'preparation')
        write_json(folder / 'error.json', {'error': repr(error), 'traceback': traceback.format_exc(), 'time': time.time()})
        try:
            r = plain_report(folder, f'AC reconstruction — {args.run or args.command} (failed)'); r.set_status(phase='error', error=repr(error)); r.set_text('traceback', 'Traceback', traceback.format_exc()); r.finish()
        finally:
            raise
