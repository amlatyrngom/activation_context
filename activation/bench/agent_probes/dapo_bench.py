"""
DAPO-Math success-rate probe: N problems x a group of G rollouts with one model, scored exactly.
Answers two questions: how often does the agent solve a problem, and is there a learning signal (groups
with mixed outcomes are what group-relative training learns from). The live page and the uncut
trajectories go to the synced folder (`DAPO_BENCH/<tag>`), the rollouts to the cache
(`ROLLOUTS/dapo_bench_<tag>`), and a summary JSON next to the page.

  uv run sky exec --sync <node> -- uv run python -m activation.bench.agent_probes.dapo_bench --model Qwen/Qwen3.5-9B --problems 100 --group 2
"""
import argparse
import collections
import json
import random
import time
from dataclasses import replace

from activation.agent import AgentConfig, RolloutReporter
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import DapoMathDataset
from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig

SYSTEM_PROMPT = (
    "You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
    "rather than guessing. When you are done, call submit_answer exactly once with the final answer."
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="HF model id")
    parser.add_argument("--problems", type=int, default=100, help="distinct problems")
    parser.add_argument("--group", type=int, default=2, help="rollouts per problem")
    parser.add_argument("--pool", type=int, default=1000, help="unique problems loaded before sampling")
    parser.add_argument("--seed", type=int, default=0, help="problem sampling seed and rollout base seed")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048, help="per turn")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrent", type=int, default=64, help="agents in flight")
    parser.add_argument("--tag", default=None, help="report / cache tag (default: from the model id)")
    args = parser.parse_args()

    tag = args.tag or args.model.split("/")[-1].lower()
    model_name = tag
    harness = HarnessRuntime(HarnessRuntimeConfig(
        model_configs={model_name: ModelConfig(model_name, args.model)},
        agent_max_concurrent=args.concurrent,
    ))
    dataset = DapoMathDataset.load(harness, max_examples=args.pool)
    harness.dataset_manager.register_dataset(dataset)
    tasks = list(dataset.scorable_tasks.values())
    random.Random(args.seed).shuffle(tasks)
    tasks = tasks[:args.problems]

    base = AgentConfig(
        system_prompt=SYSTEM_PROMPT, model_name=model_name,
        call_kwargs={"sampling_params": {"max_tokens": args.max_tokens, "temperature": args.temperature}},
        max_turns=args.max_turns, max_tool_errors=3, max_duration=600,
    )
    configs = [replace(base, user_prompt=task.agent_prompt, dataset_task=task) for task in tasks]
    report_folder = resolve_path(f"DAPO_BENCH/{tag}")
    reporter = RolloutReporter(
        str(report_folder), title=f"DAPO-Math probe: {args.model}",
        description=f"{len(tasks)} problems x group of {args.group}, {args.max_turns} turns max, temperature {args.temperature}, default tools.",
    )
    started = time.time()
    groups = harness.rollout_manager.perform_grouped_rollouts(
        configs, group_count=args.group, base_seed=args.seed, perform_scoring=True,
        caching_id=f"dapo_bench_{tag}", reporter=reporter,
    )
    elapsed = time.time() - started

    # ------------------------------------------------------------------------------- summary
    rollouts = [result for group in groups for result in group]
    per_problem = [sum(result.score for result in group) / len(group) for group in groups]
    solved_any = sum(1 for rate in per_problem if rate > 0)
    solved_all = sum(1 for rate in per_problem if rate == 1.0)
    mixed = sum(1 for rate in per_problem if 0 < rate < 1)
    finish = collections.Counter(result.finish_reason for result in rollouts)
    summary = {
        "model": args.model, "problems": len(tasks), "group": args.group, "rollouts": len(rollouts),
        "accuracy": sum(result.score for result in rollouts) / max(1, len(rollouts)),
        "problems_solved_by_any_rollout": solved_any, "problems_solved_by_every_rollout": solved_all,
        "problems_with_mixed_outcomes": mixed,   # the learning signal for group-relative training
        "problems_never_solved": len(tasks) - solved_any,
        "finish_reasons": dict(finish),
        "mean_turns": sum(result.num_turns for result in rollouts) / max(1, len(rollouts)),
        "mean_output_tokens": sum(result.num_output_tokens for result in rollouts) / max(1, len(rollouts)),
        "mean_input_tokens": sum(result.num_input_tokens for result in rollouts) / max(1, len(rollouts)),
        "mean_duration_s": sum(result.duration for result in rollouts) / max(1, len(rollouts)),
        "wall_clock_s": elapsed,
        "per_problem": [
            {"task_id": task.task_id, "gold": task.gold_answer, "scores": [result.score for result in group],
             "answers": [result.answer for result in group], "finish": [result.finish_reason for result in group]}
            for task, group in zip(tasks, groups)
        ],
    }
    (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({key: value for key, value in summary.items() if key != "per_problem"}, indent=1))
    print(f"summary: {report_folder / 'summary.json'}")


if __name__ == "__main__":
    main()
