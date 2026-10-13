"""Nested (two-level) training corpus for the AC reconstruction round: material_nested.json from material_mix.json.

Every training passage becomes a parent that holds part of itself as a compressed child, so the encoder learns to read its own
rows next to plain text (the reader still sees the parent's rows only, and reproduces / continues the whole text).

Passage format (read by `ActivationContextStudyGenerator.generate_reconstruction_samples`):
  {"segments": [{"kind": "text", "text": ...}, {"kind": "child", "text": ..., "ratio": <child ratio>}, ...],   # in order
   "ratio": <the parent's own ratio>, "variant_only": "continue" (optional; the passage is rendered in that variant alone),
   "source", "doc_id", "tokens" (= the whole text's count, as in material_mix)}
  The concatenation of the segment texts is the original (stripped) passage text; a flat passage (no `segments`, `text` instead)
  renders as before. Each segment also carries `tokens` (its own count, informative). Exactly one child per passage here.

Transformation (seeded per passage, deterministic): the child is a contiguous 25-75 % of the text cut at line ends or sentence ends
(whitespace, then any character, when a passage has neither), placed first / middle / last with equal probability; with probability
WHOLE_CHILD (0.15) the whole text is the child and the parent's text is empty or a one-line lead-in (half each). Ratio pairs
(child, parent) drawn with the weights in RATIO_PAIRS: (1/4, 1/4) 50 %, (1/16, 1/4) 20 %, (1/4, 1/16) 20 %, (1/16, 1/16) 10 %,
the last marked `variant_only: "continue"` (at 1/256 overall, only the cued continuation is a fair target).
Splits: `train` = the first TRAIN_PREFIX (34,048) passages of material_mix's training pool (the prefix the short arms train on);
`held` = the whole held panel, transformed with a fixed seed; `wiki` = copied flat, unchanged.

Run: .venv/bin/python IB/ARTIFACTS/AGENT_AC_PRETRAINING/ac_recon/prepare_nested.py   (CPU, ~1 min; tokenizer from the Hub cache)
"""
from __future__ import annotations
import bisect
import collections
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time

ROOT = Path(os.environ.get('AC_RECON_SOURCE_ROOT') or Path(__file__).resolve().parents[4])
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')

PREP = Path(os.environ.get('AC_RECON_ROOT', ROOT / 'IB/TMP/SYNC/AC_RECON')) / 'preparation'
SOURCE = PREP / 'material_mix.json'
TARGET = PREP / 'material_nested.json'
REPORT = PREP / 'material_nested_report.md'
MODEL = 'Qwen/Qwen3.5-4B'                                   # the tokenizer that counted material_mix's tokens
TRAIN_PREFIX = 34_048                                       # the short arms' prefix of the mixed corpus
SEED_TRAIN, SEED_HELD = 20260918, 20260919
CHILD_SHARE = (0.25, 0.75)                                  # of the text's characters
WHOLE_CHILD = 0.15                                          # the whole text is the child ...
LEAD_IN_SHARE = 0.5                                         # ... and half of those get a one-line lead-in as the parent's text
LEAD_INS = ['Excerpt:\n', 'The text reads as follows.\n', 'Quoted in full:\n', 'Source text:\n', 'Content:\n', 'Reproduced below.\n']
RATIO_PAIRS = [((1 / 4, 1 / 4), 0.5), ((1 / 16, 1 / 4), 0.2), ((1 / 4, 1 / 16), 0.2), ((1 / 16, 1 / 16), 0.1)]   # (child, parent): weight
CONTINUE_ONLY = (1 / 16, 1 / 16)
MIN_VIEW_ROWS = 2                                           # the bench harness's AC config (run_recon.harness)
SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def ratio_label(ratio):
    return f'1/{round(1 / ratio)}'


def boundaries(text):
    """Cut positions (the next piece starts there) after line ends and sentence ends, strictly inside the text; whitespace
    boundaries when there are none; every character when there is no whitespace either."""
    cuts = {index + 1 for index, char in enumerate(text) if char == '\n'} | {match.end() for match in SENTENCE_END.finditer(text)}
    cuts = sorted(cut for cut in cuts if 0 < cut < len(text))
    if not cuts:
        cuts = sorted({index + 1 for index, char in enumerate(text) if char.isspace()} - {len(text)})
    if not cuts:
        cuts = list(range(1, len(text)))
    return cuts


def nearest(cuts, target, lo, hi):
    """The boundary closest to `target` within [lo, hi] (None when the window holds none)."""
    inside = cuts[bisect.bisect_left(cuts, lo):bisect.bisect_right(cuts, hi)]
    return min(inside, key=lambda cut: abs(cut - target)) if inside else None


def split_passage(text, rng, crossing=False):
    """(before, child, after) texts whose concatenation is `text`; `before` / `after` may be empty. With `crossing` (the continue-only
    corner) the child is never last or whole, so a continuation cut inside the child always crosses into parent text."""
    if not crossing and rng.random() < WHOLE_CHILD:
        lead = rng.choice(LEAD_INS) if rng.random() < LEAD_IN_SHARE else ''
        return lead, text, '', 'whole'
    cuts = boundaries(text)
    n = len(text)
    share = rng.uniform(*CHILD_SHARE)
    position = rng.choice(['first', 'middle'] if crossing else ['first', 'middle', 'last'])
    if position == 'first':
        end = nearest(cuts, share * n, 1, n - 1) or cuts[0]
        return '', text[:end], text[end:], position
    if position == 'last':
        start = nearest(cuts, (1 - share) * n, 1, n - 1) or cuts[-1]
        return text[:start], text[start:], '', position
    start = nearest(cuts, rng.uniform(0.05, 0.95) * (1 - share) * n, 1, n - 2)
    end = nearest(cuts, (start or 0) + share * n, (start or 0) + 1, n - 1) if start is not None else None
    if start is None or end is None or end <= start or not text[start:end].strip():   # too few boundaries (or a blank chunk) for a middle chunk: fall back to a first chunk
        end = nearest(cuts, share * n, 1, n - 1) or cuts[0]
        return '', text[:end], text[end:], 'first'
    return text[:start], text[start:end], text[end:], position


def draw_ratios(rng):
    pairs, weights = zip(*RATIO_PAIRS)
    return rng.choices(pairs, weights=weights, k=1)[0]


def nest(passage, seed, index):
    """The nested passage and its bookkeeping (position, ratios); `text` is replaced by `segments`."""
    rng = random.Random(f'{seed}:{index}')
    text = passage['text'].strip()
    child_ratio, parent_ratio = draw_ratios(rng)                              # drawn first: the continue-only corner needs a crossing split
    before, child_text, after, position = split_passage(text, rng, crossing=(child_ratio, parent_ratio) == CONTINUE_ONLY)
    segments = ([{'kind': 'text', 'text': before}] if before else []) + [{'kind': 'child', 'text': child_text, 'ratio': child_ratio}] + ([{'kind': 'text', 'text': after}] if after else [])
    assert ''.join(segment['text'] for segment in segments) == before + child_text + after and child_text.strip()
    assert (child_ratio, parent_ratio) != CONTINUE_ONLY or after.strip(), 'the continue-only corner must have parent text after the child'
    nested = {key: value for key, value in passage.items() if key != 'text'}
    nested.update(segments=segments, ratio=parent_ratio)
    if (child_ratio, parent_ratio) == CONTINUE_ONLY:
        nested['variant_only'] = 'continue'
    return nested, position


def count_segment_tokens(passages, tokenizer):
    texts = [segment['text'] for passage in passages for segment in passage['segments']]
    counts = iter(len(ids) for ids in tokenizer(texts, add_special_tokens=False, return_attention_mask=False)['input_ids'])
    for passage in passages:
        for segment in passage['segments']:
            segment['tokens'] = next(counts)


def rows(tokens, ratio):
    return max(MIN_VIEW_ROWS, math.ceil(tokens * ratio))


def expected_rows(passage):
    """(child rows, parent rows): the child's V, then the parent's V over its text tokens plus the child's rows (template tokens aside)."""
    child = next(segment for segment in passage['segments'] if segment['kind'] == 'child')
    child_rows = rows(child['tokens'], child['ratio'])
    text_tokens = sum(segment['tokens'] for segment in passage['segments'] if segment['kind'] == 'text')
    return child_rows, rows(text_tokens + child_rows, passage['ratio'])


def summarize(passages, positions):
    """Counts, mean tokens, the ratio-pair histogram and the expected rows per pair."""
    pairs = collections.Counter()
    by_pair = collections.defaultdict(lambda: {'tokens': 0, 'child_tokens': 0, 'child_rows': 0, 'parent_rows': 0, 'n': 0})
    for passage in passages:
        child = next(segment for segment in passage['segments'] if segment['kind'] == 'child')
        pair = (ratio_label(child['ratio']), ratio_label(passage['ratio']))
        pairs[pair] += 1
        child_rows, parent_rows = expected_rows(passage)
        entry = by_pair[pair]
        entry['n'] += 1; entry['tokens'] += passage['tokens']; entry['child_tokens'] += child['tokens']; entry['child_rows'] += child_rows; entry['parent_rows'] += parent_rows
    total_rows = sum(entry['parent_rows'] for entry in by_pair.values())
    total_tokens = sum(passage['tokens'] for passage in passages)
    return {'passages': len(passages), 'tokens': total_tokens, 'mean_tokens': statistics.mean(p['tokens'] for p in passages) if passages else 0,
            'child_share_chars': statistics.mean(len(next(s['text'] for s in p['segments'] if s['kind'] == 'child')) / max(1, sum(len(s['text']) for s in p['segments'])) for p in passages) if passages else 0,
            'positions': dict(collections.Counter(positions)), 'continue_only': sum(1 for p in passages if p.get('variant_only')),
            'ratio_pairs': {f'child {c}, parent {p}': n for (c, p), n in sorted(pairs.items())},
            'rows_by_pair': {f'child {c}, parent {p}': {'passages': e['n'], 'mean_tokens': e['tokens'] / e['n'], 'mean_child_tokens': e['child_tokens'] / e['n'],
                                                        'mean_child_rows': e['child_rows'] / e['n'], 'mean_parent_rows': e['parent_rows'] / e['n'],
                                                        'effective_ratio': e['parent_rows'] / e['tokens']} for (c, p), e in sorted(by_pair.items())},
            'parent_rows_total': total_rows, 'effective_ratio': total_rows / max(1, total_tokens)}


def main():
    started = time.time()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    log(f'reading {SOURCE}')
    material = json.loads(SOURCE.read_text())
    if len(material['train']) < TRAIN_PREFIX:
        raise RuntimeError(f'{TRAIN_PREFIX} training passages needed, {len(material["train"])} in {SOURCE}')
    splits, stats = {}, {}
    for split, seed, passages in (('train', SEED_TRAIN, material['train'][:TRAIN_PREFIX]), ('held', SEED_HELD, material['held'])):
        nested, positions = zip(*(nest(passage, seed, index) for index, passage in enumerate(passages)))
        nested = list(nested)
        count_segment_tokens(nested, tokenizer)
        splits[split] = nested
        stats[split] = summarize(nested, positions)
        log(f'{split}: {len(nested):,} nested passages, mean {stats[split]["mean_tokens"]:.0f} tokens, '
            f'{stats[split]["continue_only"]} continue-only, positions {stats[split]["positions"]}')
    splits['wiki'] = material['wiki']
    output = {'train': splits['train'], 'held': splits['held'], 'wiki': splits['wiki'], 'source_material': SOURCE.name, 'train_prefix': TRAIN_PREFIX,
              'seeds': {'train': SEED_TRAIN, 'held': SEED_HELD}, 'ratio_pairs': [[ratio_label(c), ratio_label(p), w] for (c, p), w in RATIO_PAIRS],
              'stats': stats, 'preparation_seconds': time.time() - started,
              'token_stats': {'train_mean': stats['train']['mean_tokens'], 'held_mean': stats['held']['mean_tokens'], 'train_tokens': stats['train']['tokens']}}
    tmp = TARGET.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(output)); tmp.replace(TARGET)
    log(f'wrote {TARGET} ({TARGET.stat().st_size / 1e6:.0f} MB)')

    lines = ['# material_nested.json — preparation report', '',
             f'Written {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} by `prepare_nested.py` from `{SOURCE.name}` in {time.time() - started:.0f} s.', '',
             f'Two-level passages: the first {TRAIN_PREFIX:,} training passages of the mixed corpus (the short arms\' prefix) and the whole held panel ({len(splits["held"])}); '
             f'wiki ({len(splits["wiki"])}) copied flat. Each passage holds one child: a contiguous {CHILD_SHARE[0]:.0%}-{CHILD_SHARE[1]:.0%} chunk cut at line or sentence ends, '
             f'first / middle / last with equal probability; with probability {WHOLE_CHILD:.0%} the whole text is the child (half of those with a one-line lead-in as the parent\'s text). '
             f'Seeds: train {SEED_TRAIN}, held {SEED_HELD} (one stream per passage index). Tokens counted with the {MODEL} tokenizer without special tokens; '
             f'the expected rows use V = max({MIN_VIEW_ROWS}, ceil(tokens × ratio)) on content tokens (the side chat template adds a few tokens per level).', '']
    for split in ('train', 'held'):
        s = stats[split]
        lines += [f'## {split}', '',
                  f'{s["passages"]:,} passages, {s["tokens"]:,} tokens (mean {s["mean_tokens"]:.0f}); child share of the characters {s["child_share_chars"]:.0%} on average; '
                  f'positions {s["positions"]}; {s["continue_only"]:,} passages rendered only as cued continuations (child 1/16, parent 1/16).', '',
                  f'Expected parent rows in total: {s["parent_rows_total"]:,} = 1/{1 / s["effective_ratio"]:.1f} of the tokens.', '',
                  '| ratio pair (child, parent) | passages | share | mean tokens | mean child tokens | mean child rows | mean parent rows | effective ratio |',
                  '|---|---|---|---|---|---|---|---|']
        for pair, e in s['rows_by_pair'].items():
            lines.append(f'| {pair} | {e["passages"]:,} | {e["passages"] / s["passages"]:.1%} | {e["mean_tokens"]:.0f} | {e["mean_child_tokens"]:.0f} | '
                         f'{e["mean_child_rows"]:.1f} | {e["mean_parent_rows"]:.1f} | 1/{1 / e["effective_ratio"]:.1f} |')
        lines.append('')
    lines += ['## Rendering', '',
              'The reader sees the parent\'s rows only. The parent encoder\'s input is the segments in order: text verbatim, the child as its rows '
              '(encoded first at the child\'s ratio). `reconstruct` targets the whole text; `continue` gives the text up to a point 20 % into the child as a '
              'plain cue and targets the rest, so the continuation crosses from the child\'s rows into the parent\'s text. '
              'A `variant_only: "continue"` passage must be routed to the continue variant by the bench (it is skipped by the reconstruct call).', '']
    REPORT.write_text('\n'.join(lines))
    log(f'wrote {REPORT}')
    for split in ('train', 'held'):
        s = stats[split]
        print(f'{split}: {s["passages"]:,} passages, mean {s["mean_tokens"]:.0f} tokens, expected parent rows {s["parent_rows_total"]:,} (1/{1 / s["effective_ratio"]:.1f})')
        for pair, n in s['ratio_pairs'].items():
            e = s['rows_by_pair'][pair]
            print(f'  {pair}: {n:,} ({n / s["passages"]:.1%}); mean child rows {e["mean_child_rows"]:.1f}, mean parent rows {e["mean_parent_rows"]:.1f}')
    print(f'wiki: {len(splits["wiki"])} flat passages (unchanged)')


if __name__ == '__main__':
    main()
