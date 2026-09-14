"""
Static throughput tuning of the rollout stack (slice 4a tuning subplan). One probing phase under the defaults produces
`ProbeMetrics`; `Autotuner.tune` turns them into one `RolloutTuningConfig` by a short table of rules; the manager saves
it as `TUNING/<autotune_id>.json` and every later pass with the same autotune id applies the file as is (pool or node
changes do not re-probe; a new id is a new tuning). Only throughput knobs are tuned: agents per GPU, the engine's
batched-token chunk, its memory utilization and speculative decoding. Budgets that shape the data are not touched.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import typing as t
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from activation.common.data_syncing import resolve_path

if t.TYPE_CHECKING:
    from activation.harness import HarnessRuntime
    from activation.harness.loaded_model import LoadedModel
    from .agent_config import AgentRunResult

TUNING_FOLDER = "TUNING"
ALIGN_BLOCK = 1056                    # prefix-cache block of the hybrid 27B with the fp8 cache: the trailing partial block always misses


@dataclass
class RolloutTuningConfig:
    model_id: str
    gpu_name: str
    gpu_count: int
    agents_per_gpu: int = 40
    max_num_batched_tokens: int | None = None      # None: the engine's default (8,192 in vLLM 0.28)
    gpu_memory_utilization: float = 0.9
    speculative: bool = True                       # keep the model's MTP speculative config
    notes: list[str] = field(default_factory=list) # which rules fired, with the numbers

    @property
    def rollout_config_id(self) -> str:
        payload = [self.model_id, self.gpu_name, self.gpu_count, self.agents_per_gpu, self.max_num_batched_tokens,
                   self.gpu_memory_utilization, self.speculative]
        return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:12]

    @classmethod
    def defaults(cls, loaded_model: "LoadedModel", harness: "HarnessRuntime") -> "RolloutTuningConfig":
        import torch
        engine_kwargs = loaded_model.model_config.engine_kwargs or {}
        return cls(model_id=loaded_model.model_config.model_id,
                   gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   gpu_count=loaded_model.vllm_model.world_size if loaded_model.vllm_model is not None else 1,
                   agents_per_gpu=harness.harness_config.agent_max_concurrent_per_gpu,
                   max_num_batched_tokens=engine_kwargs.get("max_num_batched_tokens"),
                   gpu_memory_utilization=float(engine_kwargs.get("gpu_memory_utilization", 0.9)),
                   speculative=bool(engine_kwargs.get("speculative_config")))

    def engine_kwargs_delta(self, loaded_model: "LoadedModel") -> dict:
        """Engine construction arguments that differ from the resident engine's (empty when no reload is needed)."""
        current = dict(loaded_model.model_config.engine_kwargs or {})
        wanted = dict(current)
        if self.max_num_batched_tokens is not None:
            wanted["max_num_batched_tokens"] = self.max_num_batched_tokens
        wanted["gpu_memory_utilization"] = self.gpu_memory_utilization
        if not self.speculative:
            wanted.pop("speculative_config", None)
        return {key: value for key, value in wanted.items() if current.get(key) != value} | \
               {key: None for key in current if key not in wanted}

    def apply(self, harness: "HarnessRuntime", loaded_model: "LoadedModel", reload: bool = True) -> bool:
        """
        The pool bound now; the engine arguments through one reload when they differ. Returns whether the engine reloaded.
        With `reload=False` (agents in flight) the engine arguments are left for the next run, which applies the saved
        configuration before its threads start.
        """
        harness.harness_config.agent_max_concurrent_per_gpu = int(self.agents_per_gpu)
        harness.harness_config.agent_max_concurrent = None
        delta = self.engine_kwargs_delta(loaded_model)
        if not delta or not reload:
            return False
        kwargs = dict(loaded_model.model_config.engine_kwargs or {})
        for key, value in delta.items():
            if value is None:
                kwargs.pop(key, None)
            else:
                kwargs[key] = value
        loaded_model.model_config.engine_kwargs = kwargs
        loaded_model.reload_engine()
        return True

    def save(self, autotune_id: str) -> Path:
        path = resolve_path(TUNING_FOLDER) / f"{autotune_id}.json"
        path.write_text(json.dumps(asdict(self) | {"rollout_config_id": self.rollout_config_id, "autotune_id": autotune_id}, indent=1))
        return path

    @classmethod
    def load(cls, autotune_id: str) -> "RolloutTuningConfig | None":
        path = resolve_path(TUNING_FOLDER, create=False) / f"{autotune_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(**{key: value for key, value in data.items() if key in cls.__dataclass_fields__})


@dataclass
class ProbeMetrics:
    """What the rules read, summarized from the engine series, the host series and the phase's run results."""
    label: str
    wall_seconds: float
    runs: int
    gpu_count: int
    kv_tokens: int | None
    p50_kv_usage: float
    p95_kv_usage: float
    waiting_share: float                 # share of engine samples with a waiting queue
    preempted: int
    prefill_share: float                 # prompt tokens / (prompt + generation tokens) processed by the engine
    prefix_recompute_share: float        # missed prompt tokens beyond the alignment bound / prompt tokens
    prefix_hit_share: float
    p50_live_prefix: int
    p90_live_prefix: int
    p50_turn_seconds: float
    p90_turn_seconds: float
    sandbox_share: float                 # tool seconds / (tool + turn seconds) over every run
    p50_gpu_utilization: float
    p95_host_memory: float
    p95_host_cpu: float
    tasks_per_hour_per_gpu: float
    generated_tokens_per_hour_per_gpu: float
    generation_tokens_per_second: float  # engine-side, per replica
    p95_run_seconds: float
    mean_score: float | None

    @classmethod
    def summarize(cls, label: str, engine_metrics: list[dict], host_samples: list[dict], results: list["AgentRunResult"],
                  wall_seconds: float) -> "ProbeMetrics":
        samples = [sample for replica in engine_metrics for sample in replica["samples"]]
        kv = [s["kv_usage"] for s in samples]
        prompt = sum(s["prompt_tokens"] for s in samples); generation = sum(s["generation_tokens"] for s in samples)
        queries = sum(s["prefix_queries"] for s in samples); hits = sum(s["prefix_hits"] for s in samples)
        gpu_count = max(1, len(engine_metrics))
        kv_tokens = next((replica["kv_tokens"] for replica in engine_metrics if replica.get("kv_tokens")), None)
        turn_seconds, tool_seconds, prefixes = [], [], []
        alignment = 0
        for result in results:
            for segment in list(result.compactions) + [result]:
                alignment += len(segment.prompt_token_ids)
                for step in segment.trajectory:
                    timing = step.get("timing") or {}
                    if step.get("role") == "assistant":
                        if "turn_seconds" in timing:
                            turn_seconds.append(float(timing["turn_seconds"])); prefixes.append(int(timing.get("prefix_tokens", 0)))
                        alignment += ALIGN_BLOCK
                    elif "tool_seconds" in timing:
                        tool_seconds.append(float(timing["tool_seconds"]))
        prompt_tokens = sum(r.num_input_tokens for r in results); cached = sum(r.num_cached_input_tokens for r in results)
        durations = [float(r.duration) for r in results]
        scored = [r.score for r in results if r.score is not None and r.agent_config.dataset_task is not None and r.agent_config.dataset_task.reference_metrics_kind.value != "judge"]
        hours = max(wall_seconds, 1.0) / 3600.0
        replica_seconds = max(1.0, (samples[-1]["t"] - samples[0]["t"])) if len(samples) > 1 else max(wall_seconds, 1.0)
        return cls(
            label=label, wall_seconds=round(wall_seconds, 1), runs=len(results), gpu_count=gpu_count, kv_tokens=kv_tokens,
            p50_kv_usage=_pct(kv, 50), p95_kv_usage=_pct(kv, 95),
            waiting_share=(sum(1 for s in samples if s["waiting"] > 0) / len(samples)) if samples else 0.0,
            preempted=sum(s["preempted"] for s in samples),
            prefill_share=prompt / max(prompt + generation, 1),
            prefix_recompute_share=max(0.0, (prompt_tokens - cached - alignment)) / max(prompt_tokens, 1),
            prefix_hit_share=(hits / queries) if queries else (cached / max(prompt_tokens, 1)),
            p50_live_prefix=int(_pct(prefixes, 50)), p90_live_prefix=int(_pct(prefixes, 90)),
            p50_turn_seconds=_pct(turn_seconds, 50), p90_turn_seconds=_pct(turn_seconds, 90),
            sandbox_share=sum(tool_seconds) / max(sum(tool_seconds) + sum(turn_seconds), 1e-9),
            p50_gpu_utilization=_pct([s["gpu_utilization"] for s in host_samples], 50),
            p95_host_memory=_pct([s["host_memory"] for s in host_samples], 95),
            p95_host_cpu=_pct([s["host_cpu"] for s in host_samples], 95),
            tasks_per_hour_per_gpu=len(results) / hours / gpu_count,
            generated_tokens_per_hour_per_gpu=sum(r.num_output_tokens for r in results) / hours / gpu_count,
            generation_tokens_per_second=generation / replica_seconds / gpu_count,
            p95_run_seconds=_pct(durations, 95),
            mean_score=(sum(scored) / len(scored)) if scored else None,
        )


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((q / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


class Autotuner:
    """
    The rules, in order. Common sense over search: each rule is a threshold anyone can read and override.
    1. Agents per GPU from the KV budget: 0.8 x capacity / p90 live prefix, clamped to [16, 96].
    2. Starved engine (p95 KV usage < 60% and sandbox share > 25%): x 1.25.
    3. Evicting (recompute share > 20% or a waiting queue in > 10% of samples or any preemption): x 0.75.
    4. Prefill-bound (prompt share of engine tokens > 70%): batched tokens 16,384.
    5. KV headroom (host memory p95 < 80%): gpu_memory_utilization 0.92.
    6. Speculation off when the engine is saturated (p95 KV usage > 80%) and decodes below 40 tokens/s per replica.
    """
    min_agents, max_agents = 16, 96

    def tune(self, metrics: ProbeMetrics, defaults: RolloutTuningConfig) -> RolloutTuningConfig:
        config = replace(defaults, notes=[])
        if metrics.kv_tokens and metrics.p90_live_prefix > 0:
            budget = int(0.8 * metrics.kv_tokens / metrics.p90_live_prefix)
            config.agents_per_gpu = max(self.min_agents, min(self.max_agents, budget))
            config.notes.append(f"kv budget: 0.8 x {metrics.kv_tokens:,} / p90 prefix {metrics.p90_live_prefix:,} = {budget} -> {config.agents_per_gpu} agents per GPU")
        else:
            config.notes.append(f"kv budget unknown (kv_tokens={metrics.kv_tokens}, p90 prefix={metrics.p90_live_prefix}); agents per GPU kept at {config.agents_per_gpu}")
        if metrics.p95_kv_usage < 0.6 and metrics.sandbox_share > 0.25:
            config.agents_per_gpu = min(self.max_agents, int(config.agents_per_gpu * 1.25))
            config.notes.append(f"starved engine: p95 kv {metrics.p95_kv_usage:.0%}, sandbox share {metrics.sandbox_share:.0%} -> {config.agents_per_gpu}")
        if metrics.prefix_recompute_share > 0.2 or metrics.waiting_share > 0.1 or metrics.preempted > 0:
            config.agents_per_gpu = max(self.min_agents, int(config.agents_per_gpu * 0.75))
            config.notes.append(f"evicting: recompute {metrics.prefix_recompute_share:.0%}, waiting {metrics.waiting_share:.0%}, preempted {metrics.preempted} -> {config.agents_per_gpu}")
        if metrics.prefill_share > 0.7:
            config.max_num_batched_tokens = 16_384
            config.notes.append(f"prefill-bound: prompt share {metrics.prefill_share:.0%} -> max_num_batched_tokens 16,384")
        if metrics.p95_host_memory < 0.8:
            config.gpu_memory_utilization = 0.92
            config.notes.append(f"kv headroom: host memory p95 {metrics.p95_host_memory:.0%} -> gpu_memory_utilization 0.92")
        if metrics.p95_kv_usage > 0.8 and metrics.generation_tokens_per_second < 40:
            config.speculative = False
            config.notes.append(f"saturated: p95 kv {metrics.p95_kv_usage:.0%}, {metrics.generation_tokens_per_second:.0f} tok/s per replica -> speculative off")
        return config


def render_tuning_section(report: dict) -> list[str]:
    """Markdown lines for a rollout report: the configuration, its notes and the phases side by side."""
    lines = ["## Tuning", ""]
    config = report.get("config") or {}
    if config:
        lines.append(f"Autotune id `{report.get('autotune_id')}` (rollout config `{config.get('rollout_config_id', '')}`, "
                     f"{'saved configuration applied' if report.get('from_saved') else 'tuned in this pass'}): "
                     f"{config['agents_per_gpu']} agents per GPU, batched tokens {config.get('max_num_batched_tokens') or 'engine default'}, "
                     f"gpu_memory_utilization {config['gpu_memory_utilization']}, speculative {'on' if config['speculative'] else 'off'}"
                     f"{'; engine reloaded' if report.get('reloaded') else '; engine reload deferred to the next run' if report.get('reload_deferred') else ''}.")
        for note in config.get("notes") or []:
            lines.append(f"- {note}")
        lines.append("")
    phases = report.get("phases") or []
    if phases:
        keys = ["runs", "wall_seconds", "tasks_per_hour_per_gpu", "generated_tokens_per_hour_per_gpu", "generation_tokens_per_second", "p50_kv_usage", "p95_kv_usage",
                "waiting_share", "preempted", "prefix_hit_share", "prefix_recompute_share", "prefill_share", "p50_live_prefix", "p90_live_prefix",
                "p50_turn_seconds", "p90_turn_seconds", "sandbox_share", "p50_gpu_utilization", "p95_host_cpu", "p95_host_memory", "p95_run_seconds", "mean_score"]
        lines.append("| Metric | " + " | ".join(phase["label"] for phase in phases) + " |")
        lines.append("| --- |" + " --- |" * len(phases))
        for key in keys:
            cells = []
            for phase in phases:
                value = phase.get(key)
                cells.append("-" if value is None else (f"{value:.1%}" if key.endswith("share") or "usage" in key or "utilization" in key or key == "p95_host_cpu" or key == "p95_host_memory"
                                                         else f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"))
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines
