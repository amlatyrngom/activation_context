"""
Live page for rollouts, a bit richer than the training reporter: overall progress and statistics plus
the rendered trajectories, the running ones first and then the last `finished_trajectories_shown`
complete ones; every text on the page is cut to size, the uncut trajectory of each agent is written
next to the page as `trajectories/<agent>.json` (linked from its header). The page is written once
and polls `report_data.json`, which is rewritten on every render (see common.reporting), so the file does not flicker. Simple hooks that keep agent code clean:
report_agent_start / step / finish / scored; the statistics come from the agent's run results.
"""
from __future__ import annotations

import json
import threading
import time
import typing as t
from collections import deque
from pathlib import Path

from activation.common.reporting import HtmlReporter, _write_atomic, format_seconds

if t.TYPE_CHECKING:
    from .agent import Agent

PROMPT_CHARS = 1500       # per trajectory
CONTENT_CHARS = 1200      # per assistant step
ARGUMENT_CHARS = 1500     # per tool call
RESULT_CHARS = 1000       # per tool result
ROLLOUT_COLUMNS = ["agent", "task", "seed", "finish", "turns", "tool calls", "tokens in", "cached", "tokens out", "duration", "score"]


class RolloutReporter(HtmlReporter):
    def __init__(self, report_folder: str, title: str, description: str = "", finished_trajectories_shown: int = 100):
        super().__init__(report_folder, title, description, eyebrow="Rollouts", refresh_seconds=10, min_render_interval_seconds=2.0)
        self.lock = threading.Lock()
        self.started_at = time.time()
        self.total = 0
        self.cached = 0
        self.running: dict[str, "Agent"] = {}
        self.finished_items: deque[dict] = deque(maxlen=max(1, min(finished_trajectories_shown, 100)))
        self.finished_count = 0
        self.scores: list[float] = []
        self.output_tokens = 0
        self.initialize_line_plot(
            "progress", "Rollouts over time",
            "Finished rollouts and the running mean score (scored rollouts only) against wall-clock seconds.",
            "seconds", "value", ["finished", "mean score"],
        )
        self.initialize_table("rollouts", "Finished rollouts", "One row per finished agent (subagents included).", ROLLOUT_COLUMNS)
        self.set_trajectories(
            "trajectories", "Trajectories",
            f"Running rollouts first (open), then the last {self.finished_items.maxlen} finished ones, newest first. Long texts are cut.", [],
        )
        self._update_status()
        self.render(force=True)

    @property
    def report_folder(self) -> str:
        """Where report.tressoir.html, report_data.json and trajectories/ live."""
        return str(self.folder)

    # ----------------------------------------------------------------------------- manager hooks
    def begin_rollouts(self, total: int, cached: int) -> None:
        with self.lock:
            self.total, self.cached = total, cached
            self._update_status()
        self.render(force=True)

    # ----------------------------------------------------------------------------- agent hooks
    def report_agent_start(self, agent: "Agent") -> None:
        with self.lock:
            self.running[agent.agent_id] = agent
            self._update_trajectories()
            self._update_status()
        self.render()

    def report_agent_step(self, agent: "Agent") -> None:
        with self.lock:
            self._update_trajectories()
            self._update_status()
        self.render()

    def report_agent_finish(self, agent: "Agent") -> None:
        results = agent.run_results
        with self.lock:
            self.running.pop(agent.agent_id, None)
            self.finished_items.appendleft(trajectory_item(agent, "finished", self.folder))
            self.finished_count += 1
            self.output_tokens += results.num_output_tokens
            task = agent.agent_config.dataset_task
            self.add_data_point("rollouts", {
                "agent": _agent_label(agent),
                "task": task.task_id[:12] if task is not None else "-", "seed": results.seed, "finish": results.finish_reason,
                "turns": results.num_turns, "tool calls": results.num_tool_calls, "tokens in": results.num_input_tokens,
                "cached": results.num_cached_input_tokens, "tokens out": results.num_output_tokens,
                "duration": format_seconds(results.duration), "score": "" if task is None else f"{results.score:.2f}",
            })
            self._add_progress_point()
            self._update_trajectories()
            self._update_status()
        self.render(force=True)

    def report_agent_scored(self, agent: "Agent") -> None:
        """Scoring happens after finish; the table row, the trajectory header and the mean are updated."""
        results = agent.run_results
        with self.lock:
            self.scores.append(results.score)
            label = _agent_label(agent)
            for row in reversed(self.widgets["rollouts"]["rows"]):
                if row["agent"] == label:
                    row["score"] = f"{results.score:.2f}"
                    break
            for item in self.finished_items:
                if item["id"] == agent.agent_id:
                    item["stats"] = _stats(agent)
                    break
            self._add_progress_point()
            self._update_trajectories()
            self._update_status()
        self.render(force=True)

    # ----------------------------------------------------------------------------- internals
    def _update_trajectories(self) -> None:
        items = [trajectory_item(agent, "running", self.folder) for agent in self.running.values()] + list(self.finished_items)
        self.widgets["trajectories"]["items"] = items

    def _add_progress_point(self) -> None:
        point = {"x": round(time.time() - self.started_at, 1), "finished": self.finished_count}
        if self.scores:
            point["mean score"] = sum(self.scores) / len(self.scores)
        self.add_data_point("progress", point)

    def _update_status(self) -> None:
        elapsed = time.time() - self.started_at
        self.set_status(
            total=str(self.total), cached=str(self.cached), running=str(len(self.running)), finished=str(self.finished_count),
            mean_score=f"{sum(self.scores) / len(self.scores):.3f}" if self.scores else "-",
            output_tokens_per_s=f"{self.output_tokens / elapsed:.1f}" if elapsed > 0 else "-",
            elapsed=format_seconds(elapsed),
        )


# --------------------------------------------------------------------------------- rendering items
def trajectory_item(agent: "Agent", state: str, folder: Path | None = None) -> dict:
    """
    The page's view of one agent: header, prompt, and every step with calls and results cut to size.
    With a folder, the uncut trajectory goes to `<folder>/trajectories/<agent>.json` and the item links it.
    """
    results = agent.run_results
    file = None
    if folder is not None:
        file = f"trajectories/{agent.agent_id[:8]}.json"
        (folder / "trajectories").mkdir(parents=True, exist_ok=True)
        _write_atomic(folder / file, json.dumps({
            "agent_id": agent.agent_id, "state": state, "system_prompt": agent.agent_config.system_prompt,
            "user_prompt": agent.agent_config.user_prompt,
            "trajectory": [{key: value for key, value in step.items() if key not in ("token_ids", "logprobs", "messages")} for step in results.trajectory],
            "compactions": len(results.compactions), "prompt_ac_spans": results.prompt_ac_spans,
            "answer": results.answer, "finish_reason": results.finish_reason, "score": results.score,
        }, indent=1, default=str))
    steps = []
    turn = 0
    for step in results.trajectory:
        if step["role"] == "assistant":
            turn += 1
            steps.append({
                "role": "assistant", "turn": turn, "content": _cut(step["content"], CONTENT_CHARS),
                "calls": [{"name": call["name"], "arguments": _cut(_arguments(call["arguments"]), ARGUMENT_CHARS)} for call in step["tool_calls"]],
            })
        elif step["role"] == "tool":
            steps.append({
                "role": "tool", "turn": turn, "content": "",
                "results": [{"name": call["name"], "output": _cut(output, RESULT_CHARS) + (_parts_note(step) if index == 0 else "")}
                            for index, (call, output) in enumerate(zip(step["tool_calls"], step["tool_call_results"]))],
            })
        else:                                                                       # the nudge
            steps.append({"role": "tool", "turn": turn, "content": "", "results": [{"name": "user", "output": _cut(step["content"], RESULT_CHARS)}]})
    return {
        "id": agent.agent_id, "title": _title(agent), "state": state, "stats": _stats(agent),
        "prompt": _cut(agent.agent_config.user_prompt, PROMPT_CHARS), "steps": steps, "file": file,
    }


def _parts_note(step: dict) -> str:
    """One line per activation-context part of the step: its rows, so the page shows what the model read as rows."""
    spans = step.get("ac_spans") or []
    if not spans:
        return ""
    return "\n" + "\n".join(f"[activation context part {index + 1}: {span['length']} rows]" for index, span in enumerate(spans))


def _title(agent: "Agent") -> str:
    task = agent.agent_config.dataset_task
    return _agent_label(agent) + (f" · task {task.task_id[:12]}" if task is not None else "") + f" · seed {agent.run_results.seed}"


def _stats(agent: "Agent") -> str:
    results = agent.run_results
    parts = [f"turn {results.num_turns}", f"{results.num_input_tokens} in ({results.num_cached_input_tokens} cached) / {results.num_output_tokens} out"]
    if results.compactions:
        parts.append(f"{len(results.compactions)} compactions")
    if results.num_ac_parts:
        parts.append(f"{results.num_ac_parts} AC parts / {results.num_ac_rows} rows")
    if agent.finished:
        parts.append(results.finish_reason)
        parts.append(f"answer {str(results.answer)[:40]!r}")
        if agent.agent_config.dataset_task is not None and results.score_feedback != "" or results.score:
            parts.append(f"score {results.score:.2f}")
    return " · ".join(parts)


def _agent_label(agent: "Agent") -> str:
    return agent.agent_id[:8] + (" (sub)" if agent.parent_agent is not None else "")


def _arguments(arguments: dict) -> str:
    """Multi-line string arguments (code, scripts) verbatim; everything else as JSON."""
    lines = []
    for key, value in arguments.items():
        if isinstance(value, str) and ("\n" in value or len(value) > 60):
            lines.append(f"{key}:\n{value}")
        else:
            lines.append(f"{key}: {json.dumps(value, default=str)}")
    return "\n".join(lines)


def _cut(text: str, limit: int) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [{len(text) - limit} more chars; the uncut text is in the trajectory file]"
