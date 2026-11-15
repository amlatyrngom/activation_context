"""
HF reference for the vLLM Hidden Residuals check. Builds dev1000 items exactly as sec45_eval.py does for the pair (rq_eval_job.sh
environment: 2B encoder -> 9B reader, res at every full-attention layer, linear-skip head), assembles each item's reader prefix as
ActivationContextTrainer._sample_completions_batched does (one item, no padding), and runs the HF reader through the intervention
hooks twice: with the deltas (res) and without them (rows only). Saves per item: the prompt token ids, the AC row positions, the
rows and the per-layer deltas (bf16, the reader boundary), and per mode the first-position log-probabilities (fp32, full vocab)
and the greedy continuation (stopped at the first end token, max --max-new).

  python hf_reference.py --code BUNDLE_DIR --data TASK_DIR --pair AC_DIR READER_DIR --out OUT_DIR [--per-k 20:5,100:6,200:5]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True); p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pair", nargs=2, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-k", default="20:5,100:6,200:5"); p.add_argument("--max-new", type=int, default=64)
    a = p.parse_args()
    for sub in ("kit", "rq_kit", "bench"):
        sys.path.insert(0, str(a.code / sub))
    import sec45_common as S
    S.setup_environment("res")
    import rq_common as C
    import torch
    import torch.nn.functional as F
    from activation.ac_model import ActivationContextStudyGenerator, ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.ac_model.ac_model import TRAINING
    from activation.autotuners import configure_fla_runtime
    from activation.harness.hf_utils import generate_with_interventions
    from run_qa import AC, harness

    want = {int(k): int(n) for k, n in (x.split(":") for x in a.per_k.split(","))}
    chosen = []
    for row in S.load_tasks(a.data, "dev1000"):
        k = S.row_k(row)
        if want.get(k, 0) > 0:
            want[k] -= 1
            chosen.append(row)
    print(f"hf_reference: {len(chosen)} questions, k = {[S.row_k(r) for r in chosen]}", flush=True)

    S.check_pair_mode(a.pair[0], a.pair[1], "res")
    h = harness(a.pair[0], a.pair[1], summary_layout=S.LAYOUT)
    study = ActivationContextStudyGenerator(h, AC)
    ac = study.ac_model
    items = S.task_items(study, C.load_corpus(a.data), chosen, AC)
    trainer = ActivationContextTrainer(h, ActivationContextTrainingConfig(loss_kind="sft", distributed=False, completion_sampling=None,
                                                                          completion_max_new_tokens=a.max_new, max_example_tokens=72000))
    target, manager = ac.target, h.module_manager
    lora = ac.config.target_model_lora_name
    (a.out / "items").mkdir(parents=True, exist_ok=True)
    manifest = {"pair": a.pair, "layers": list(ac.intervention_layers), "max_new": a.max_new, "items": []}
    with configure_fla_runtime(trainer._autotuning_variables(ac)["fla_config"]):
        ac.prepare()
        peft_model = manager.ensure_lora(lora)
        lora_b = 0.0
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                lora_b = max(lora_b, float(parameter.detach().abs().max()))
        for parameter in manager.lora_parameters(lora):            # as sec45_eval.py: no fp32 GEMMs in the decode
            parameter.data = parameter.data.to(torch.bfloat16)
        manifest["reader_lora_B_max_abs"] = lora_b
        print(f"hf_reference: reader LoRA max |B| = {lora_b} (0: the reader is the untuned base)", flush=True)
        ac.set_mode(TRAINING)
        for base in dict.fromkeys([target.model, ac.side.model]):
            base.eval()
            if base.is_gradient_checkpointing:
                base.gradient_checkpointing_disable()
        embedding = target.model.get_input_embeddings()
        eos_ids = peft_model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
        eot = target.model_config.model_description.eot_token
        if eot is not None and target.tokenizer.convert_tokens_to_ids(eot) not in eos_ids:
            eos_ids.append(target.tokenizer.convert_tokens_to_ids(eot))
        pad = target.tokenizer.pad_token_id if target.tokenizer.pad_token_id is not None else eos_ids[0]
        manifest["eos_ids"] = eos_ids
        generation = dict(max_new_tokens=a.max_new, do_sample=False, use_cache=True, eos_token_id=eos_ids, pad_token_id=pad,
                          return_dict_in_generate=True, output_logits=True)
        for index, (row, item) in enumerate(zip(chosen, items)):
            started = time.monotonic()
            with torch.no_grad():
                example = trainer.build_example(ac, item)
                prefix_length = example.student_positions[0] + 1
                ids = list(example.student_ids[:prefix_length])
                embeds = embedding(torch.tensor(ids, device=ac.device))
                present = [(span, request) for span, request in zip(example.spans, example.part_requests) if span[1] <= prefix_length]
                outputs, _ = trainer._encode_parts_cached(ac, [request for _, request in present], {}, 0)
                embeds, payload = trainer._assemble_student(embeds, [span for span, _ in present], outputs, embeds.dtype)
                record = {"id": row["id"], "k": S.row_k(row), "ids": ids, "positions": list(payload["positions"]),
                          "row_embeds": embeds[payload["positions"]].to(torch.bfloat16).cpu(),
                          "deltas": {int(layer): rows.to(torch.bfloat16).cpu() for layer, rows in payload["layer_inputs"].items()}}
                with manager.lora_context(target.model_config.model_name, lora):
                    for mode in ("res", "rows"):
                        if mode == "res":
                            out = generate_with_interventions(peft_model, inputs_embeds=embeds[None], interventions=[payload], **generation)
                        else:
                            out = peft_model.generate(inputs_embeds=embeds[None], **generation)
                        tokens = out.sequences[0].tolist()
                        stop = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
                        record[mode] = {"tokens": tokens if stop is None else tokens[:stop + 1],
                                        "logprobs": F.log_softmax(out.logits[0][0].float(), dim=-1).cpu()}
            torch.save(record, a.out / "items" / f"{index:02d}.pt")
            entry = {"index": index, "id": row["id"], "k": record["k"], "prompt_tokens": len(ids), "rows": len(record["positions"]),
                     "res_text": target.tokenizer.decode(record["res"]["tokens"], skip_special_tokens=True),
                     "rows_text": target.tokenizer.decode(record["rows"]["tokens"], skip_special_tokens=True),
                     "seconds": round(time.monotonic() - started, 1)}
            manifest["items"].append(entry)
            print(f"hf_reference: {json.dumps(entry)}", flush=True)
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("hf_reference: done", flush=True)


if __name__ == "__main__":
    main()
