"""§3.1 reference lines (2026-11-15, user: put the gold reference back in the figure): text_lora_eval_hf.py unchanged, with the reader's
view swapped before main() runs, so decode, scorer and output format are the text rows' exactly (greedy, cap --max-new, rq_common.score).
  gold  the gold-view teacher's prompt (rq_common.teacher_messages): READ over the row's two gold passages only, in q["golds"] order
  none  rq_common.NO_PASSAGES: the question alone (closed book)
  python ref_view_eval.py gold|none --data DIR --split test1000 --base --out OUT.jsonl [...]   (needs $C/text_lora, $C/kit, $C/rq_kit on PYTHONPATH)
"""
import sys

VIEW = sys.argv.pop(1)
assert VIEW in ("gold", "none"), VIEW
import text_lora_eval_hf as M  # noqa: E402

if VIEW == "gold":
    M.row_passages = lambda r: [int(x) for x in r["golds"]]
else:
    M.row_passages = lambda r: []
    M.C.READ = M.C.NO_PASSAGES          # .format(passages="", question=...) ignores the unused passages field
M.main()
