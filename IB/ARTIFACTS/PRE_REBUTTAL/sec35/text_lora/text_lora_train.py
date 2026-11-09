"""Text + reader-LoRA baseline for section 4.5 / Fig. 7b-c (user, Oct 27 ~20:00Z): "a pure reader LoRA ... trained on the same pairs as
the task tuning runs", WITHOUT the no-context penalty ("more what people will do in the wild"). The 4B reader reads the 20 passages
AS TEXT and learns the answer with a plain LoRA; no encoder, no AC.

Matched to the T runs (jobs/generated_v4/T_<mode>_v4.sh -> kit/sec45_train.py, ActivationContextTrainingConfig defaults):
  items     the HotpotQA train pool shuffled with random.Random(seed) (seed 0), the first steps x epu = 4,000 x 8 = 32,000 rows,
            trained in the trainer's epoch order random.Random(seed + 1).shuffle(range(n)) -> the same questions in the same order
  prompt    rq_common.READ with the row's top20 (both golds injected) as text, thinking off = the user turn the AC student sees with
            its parts replaced by text, and the prompt the 53.0 text line (text4b_sec45.py) was scored with
  target    "Final answer: <answer>" + the end-of-turn token (rq_common.answer_replies; SFT on the target tokens only)
  loss      per example: mean cross-entropy over its target tokens; per update: the mean over the epu examples
  LoRA      rank 128, alpha 256, dropout 0, q/k/v/o/gate/up/down (= the AC runs' reader adapter_config.json)
  optimizer AdamW lr 2e-5, betas (0.9, 0.95), eps 1e-8, weight decay 0, grad-norm clip 1.0; 50 linear warm-up updates, then
            cosine to 0.1 x the rate over all updates
  held      dev200 target NLL every 250 updates and before the first (mean over items of the per-item mean NLL = sec45 held_nll;
            the token-weighted mean as held_nll_token)
  saves     the adapter every --ckpt-every updates and at the end (ckpt/step_NNNNN; default: 25/50/75/100% as in the paper run),
            evaluated by text_lora_eval_hf.py (the pre-rebuttal single scorer) while or after training

Pre-rebuttal (2026-11-09): RESUMABLE. With --ckpt-every K every K updates one recovery slot OUT/resume/ (adapter + AdamW state +
update count; written to resume.tmp, then renamed over the old one) is published; a rerun with the same arguments continues from it
exactly (same rows, same order: both are pure functions of --seed; the LR is a pure function of the update; no dropout, no RNG).
The slot is deleted when the run completes. --name names the run's files (default T_text_lora).

  python text_lora_train.py --data HOTPOT_TASKS_DIR --out OUT [--steps 4000 --epu 8 --ckpt-every 800 --name X]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("RQ_THINKING", "0")
sys.path.insert(0, os.environ.get("RQ_KIT", str(Path(__file__).resolve().parent)))
import rq_common as C  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kit"))
from sec45_common import row_passages, target_replies  # noqa: E402  (§3.5 top20; retrieval top200[:k] with SEC45_K; SEC45_TARGETS)

MODEL_ID = os.environ.get("TEXT_MODEL_ID", "Qwen/Qwen3-4B-Instruct-2507")   # retrieval: Qwen/Qwen3.5-9B


def load_model(torch):
    """The reader base: a causal LM, or (Qwen3.5) the multimodal class whose text model is used alone, as the harness loads it."""
    import transformers
    config = transformers.AutoConfig.from_pretrained(MODEL_ID)
    multimodal = hasattr(config, "vision_config")
    cls = transformers.AutoModelForMultimodalLM if multimodal else transformers.AutoModelForCausalLM   # as activation/harness/runtime.py
    return cls.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda")
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TARGET_REGEX = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"   # the language model only (Qwen3.5 has a vision tower)


TARGETS_BY_ID = {}


def encode(tok, corpus, row):
    """(prompt ids, target ids) for one row: the chat-templated READ prompt and 'Final answer: X' + end of turn."""
    user = C.READ.format(passages=C.passages_block(corpus, row_passages(row)), question=row["question"])
    prompt = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template([{"role": "user", "content": user}, {"role": "assistant", "content": TARGETS_BY_ID.get(row["id"], f"Final answer: {row['answer']}")}],
                                   tokenize=False, add_generation_prompt=False, enable_thinking=False)
    assert full.startswith(prompt), "the assistant turn does not extend the generation prompt"
    target = full[len(prompt):].rstrip("\n")                       # "Final answer: X<|im_end|>" (no trailing newline, as assistant_target_ids)
    return tok.encode(prompt, add_special_tokens=False), tok.encode(target, add_special_tokens=False)


def example_nll(model, torch, prompt, target, device):
    ids = torch.tensor([prompt + target], device=device)
    logits = model(input_ids=ids, logits_to_keep=len(target) + 1).logits[0, :-1].float()   # only the target positions (a 24k-token prompt x 248k vocab would not fit)
    return torch.nn.functional.cross_entropy(logits, torch.tensor(target, device=device), reduction="none")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4000); p.add_argument("--epu", type=int, default=8); p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=50); p.add_argument("--final-fraction", type=float, default=0.1)
    p.add_argument("--report-every", type=int, default=250); p.add_argument("--held", type=int, default=200)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--rank", type=int, default=128); p.add_argument("--alpha", type=int, default=256)
    p.add_argument("--ckpt-every", type=int, help="save the adapter (and the recovery slot) every K updates; default: at 25/50/75/100%%")
    p.add_argument("--name", default="T_text_lora")
    a = p.parse_args()
    N = a.name
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    a.out.mkdir(parents=True, exist_ok=True)
    needed = a.steps * a.epu
    pool = C.load_tasks(a.data, "train") if hasattr(C, "load_tasks") else [json.loads(l) for l in open(a.data / "train.jsonl")]
    random.Random(a.seed).shuffle(pool)                            # sec45_train: the same seed -> the same questions as S and T
    if needed > len(pool):
        raise SystemExit(f"{needed} items needed, the pool holds {len(pool)}")
    rows = pool[:needed]
    order = list(range(len(rows)))
    random.Random(a.seed + 1).shuffle(order)                       # the trainer's epoch-1 order (seed + epoch number)
    held_rows = [json.loads(l) for l in open(a.data / "dev200.jsonl")][:a.held]
    TARGETS_BY_ID.update(target_replies(rows + held_rows))
    corpus = C.load_corpus(a.data)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    held = [encode(tok, corpus, r) for r in held_rows]
    run = {"name": N, "model": MODEL_ID, "args": {k: str(v) for k, v in vars(a).items()}, "penalty": None,
           "first_items": [rows[i]["id"] for i in order[:32]], "items": needed, "lora": {"r": a.rank, "alpha": a.alpha, "targets": TARGETS}}
    resume_dir = a.out / "resume"
    resume = json.loads((resume_dir / "state.json").read_text()) if (resume_dir / "state.json").exists() else None
    if resume:
        first = json.loads((a.out / f"{N}.run.json").read_text())
        if first["args"] != run["args"] or first["first_items"] != run["first_items"]:
            raise SystemExit(f"resume refused: the arguments differ from the first attempt ({a.out / f'{N}.run.json'})")
        print(f"text_lora_train: RESUME from update {resume['update']} ({resume_dir})", flush=True)
    else:
        (a.out / f"{N}.run.json").write_text(json.dumps(run, indent=1))

    device = "cuda"
    model = load_model(torch)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules=TARGET_REGEX, task_type="CAUSAL_LM"))
    params = [q for q in model.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    first_update = 0
    if resume:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(resume_dir / "adapter" / "adapter_model.safetensors"), device=device))
        opt.load_state_dict(torch.load(resume_dir / "optimizer.pt", map_location=device))
        first_update = resume["update"]

    def rate(update):                                              # warm-up, then cosine to final_fraction (ac_model_training "cosine")
        if update < a.warmup:
            return a.lr * (update + 1) / a.warmup
        progress = min(1.0, (update - a.warmup) / max(1, a.steps - a.warmup))
        return a.lr * (a.final_fraction + (1 - a.final_fraction) * 0.5 * (1 + math.cos(math.pi * progress)))

    def report(update):
        model.eval()
        per_item, total, count = [], 0.0, 0
        with torch.no_grad():
            for prompt, target in held:
                ce = example_nll(model, torch, prompt, target, device)
                per_item.append(float(ce.mean())); total += float(ce.sum()); count += ce.numel()
        model.train()
        row = {"update": update, "held_nll": sum(per_item) / len(per_item), "held_nll_token": total / count, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(a.out / f"{N}.held.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    saves = set(range(a.ckpt_every, a.steps + 1, a.ckpt_every)) | {a.steps} if a.ckpt_every else {a.steps * q // 4 for q in (1, 2, 3, 4)}
    stats = {"loss": [], "seconds": [], "grad_norm": []}
    if resume:
        stats = json.loads((resume_dir / "stats.json").read_text())
    model.train()
    if not resume:
        report(0)
    t0 = time.time() - (stats.get("elapsed", 0) if resume else 0)

    def publish_slot(done):                                        # adapter + moments + update; the old slot goes only after the new is complete
        import shutil
        tmp = a.out / "resume.tmp"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir()
        model.save_pretrained(tmp / "adapter")
        torch.save(opt.state_dict(), tmp / "optimizer.pt")
        (tmp / "stats.json").write_text(json.dumps({**stats, "elapsed": round(time.time() - t0)}))
        (tmp / "state.json").write_text(json.dumps({"update": done, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
        old = a.out / "resume.old"
        shutil.rmtree(old, ignore_errors=True)
        if resume_dir.exists():
            resume_dir.rename(old)
        tmp.rename(resume_dir)
        shutil.rmtree(old, ignore_errors=True)

    for update in range(first_update, a.steps):
        start = time.time()
        for g in opt.param_groups:
            g["lr"] = rate(update)
        losses = []
        for j in order[update * a.epu:(update + 1) * a.epu]:
            prompt, target = encode(tok, corpus, rows[j])
            loss = example_nll(model, torch, prompt, target, device).mean() / a.epu
            loss.backward(); losses.append(float(loss) * a.epu)
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step(); opt.zero_grad(set_to_none=True)
        stats["loss"].append(round(sum(losses) / len(losses), 5)); stats["seconds"].append(round(time.time() - start, 3)); stats["grad_norm"].append(round(gn, 4))
        done = update + 1
        if done % 25 == 0:
            print(f"update {done}/{a.steps} loss {sum(stats['loss'][-25:]) / 25:.4f} lr {rate(update):.2e} {sum(stats['seconds'][-25:]) / 25:.2f}s/update", flush=True)
        if done % a.report_every == 0 or done == a.steps:
            report(done)
        if done in saves:
            folder = a.out / "ckpt" / f"step_{done:05d}"
            model.save_pretrained(folder.with_name(folder.name + ".tmp"))
            folder.with_name(folder.name + ".tmp").rename(folder)        # complete adapters only (the eval watcher picks up ckpt/step_*)
            (a.out / f"{N}.stats.json").write_text(json.dumps({**stats, "updates": done, "elapsed": round(time.time() - t0)}))
            if a.ckpt_every and done < a.steps:
                publish_slot(done)
    (a.out / f"{N}.stats.json").write_text(json.dumps({**stats, "updates": a.steps, "elapsed": round(time.time() - t0), "stop_reason": "complete"}))
    import shutil
    shutil.rmtree(resume_dir, ignore_errors=True)
    (a.out / "TRAINED").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


if __name__ == "__main__":
    main()
