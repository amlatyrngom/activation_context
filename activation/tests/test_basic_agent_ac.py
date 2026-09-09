"""
Slice 3b key test: activation context in the agent loop, on a GPU node.

    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent_ac.py --gpu --slow -s

Untrained AC rows are junk, so the simulated scenarios assert mechanics (parts, spans, rows, records, the engine
accepting mixed prompts and hitting its prefix cache), not answers. `AC_CHECKPOINT=AC_MODELS/ac_dev/epoch_002` loads a
3a1 checkpoint instead. The last two tests are real rollouts with a low compaction threshold: with AC rows, and text-only.
"""
import json
import os
from dataclasses import replace

import pytest
import torch

from activation.ac_model import ActivationContextModelConfig, ActivationContextStudyGenerator
from activation.common.ac_parts import direct_parts
from activation.agent import Agent, AgentConfig, AgentRunResult, RolloutReporter, SemanticSearchTool, SubagentTool, SyntheticTurn, synthesize_agent
from activation.agent.agent_tools import SEARCH_EXTRA_MAX
from activation.common.data_syncing import resolve_path
from activation.dataset.loaders import BrightDataset
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig

MODEL_NAME, MODEL_ID = "qwen3.5-4b", "Qwen/Qwen3.5-4B"
SIDE_NAME, SIDE_ID = "qwen3.5-0.8b", "Qwen/Qwen3.5-0.8B"
AC_NAME, SIDE_LORA, TARGET_LORA = "ac_dev", "ac_side", "ac_target"
ENGINE_CACHE_BLOCK_TOKENS = 1056   # prefix-cache block of the hybrid 4B engine with the fp8 KV cache (mamba-aligned); prompts at or below it cannot hit
SYSTEM = ("You are a careful problem solver working in a sandbox. Use the python or shell tools to compute rather than "
          "guessing. When you are done, call submit_answer exactly once with the final answer.")
PRIMES = "Compute the sum of the 140th, 141st and 142nd prime numbers (2 is the 1st prime)."
BASE = AgentConfig(system_prompt=SYSTEM, model_name=MODEL_NAME, ac_model_name=AC_NAME, enable_ac_communication=True,
                   call_kwargs={"sampling_params": {"max_tokens": 1024, "temperature": 0.7}}, max_turns=12, max_tool_errors=3, max_duration=300)
LONG = "0123456789" * 120   # 1,200 chars per synthetic tool output


def _harness(with_ac: bool = True) -> HarnessRuntime:
    configs = {MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)}
    if with_ac:
        configs[SIDE_NAME] = ModelConfig(SIDE_NAME, SIDE_ID)
    harness = HarnessRuntime(HarnessRuntimeConfig(model_configs=configs, agent_max_concurrent=8))
    if with_ac:
        harness.module_manager.register_lora(TARGET_LORA, MODEL_NAME, rank=64)
        harness.module_manager.register_ac_model(ActivationContextModelConfig(
            AC_NAME, SIDE_NAME, SIDE_LORA, MODEL_NAME, TARGET_LORA, checkpoint_path=os.environ.get("AC_CHECKPOINT")))
    return harness


def _release(harness: HarnessRuntime) -> None:
    for loaded in harness.loaded_models.values():
        loaded.engine_to_device(FREE_DEVICE)
    if AC_NAME in harness.module_manager.ac_models:
        ac_model = harness.module_manager.get_ac_model(AC_NAME)
        ac_model.release()
        for model_name, lora_name in ((ac_model.config.target_model_name, ac_model.config.target_model_lora_name),
                                      (ac_model.config.base_side_model_name, ac_model.config.base_side_model_lora_name)):
            if harness.module_manager.has_loras(model_name):
                harness.module_manager.free_lora(model_name, lora_name)       # adapters first: a base with adapters cannot be freed
    for loaded in harness.loaded_models.values():
        loaded.model_to_device(FREE_DEVICE)


def _probe_engine(agent: Agent, label: str) -> None:
    """One real mixed-prompt request, then the same again: the engine accepts the rows and serves the prefix from its cache."""
    prefix, embeds, mask = agent.prepare_request()
    first = agent.submit_probe()
    second = agent.submit_probe()
    print(f"  [{label}] prompt {len(prefix)} tokens, rows at {0 if mask is None else mask.count(False)} positions, "
          f"cached {first.cached_prompt_token_count} -> {second.cached_prompt_token_count}, sampled {first.output_token_count} tokens: "
          f"{first.text[:80]!r}")
    assert first.prompt_token_count == len(prefix)
    if embeds is not None:
        assert tuple(embeds.shape) == (len(prefix), agent.loaded_model.model_config.model_description.d_model)
        assert embeds.dtype == agent.row_dtype
    # Prefix hits come in cache-aligned blocks: hybrid Qwen3.5 with the fp8 KV cache aligns at 1,056 tokens (528 with bf16), and a prompt
    # shorter than one block can never hit (verified with token-only prompts on the node: 909 tokens -> 0, 1,254 tokens -> 1,056).
    if len(prefix) > ENGINE_CACHE_BLOCK_TOKENS:
        assert second.cached_prompt_token_count > 0, "the second identical request did not hit the prefix cache"


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_agent_channels_simulated():
    """Compaction, subagent, large output and search through simulate_step; a resume round trip; one engine submit per scenario."""
    harness = _harness()
    bright = BrightDataset.load(harness, max_examples=20, domain="biology", max_corpus_documents=200)
    harness.dataset_manager.register_dataset(bright)
    example = next(iter(bright.labeled_qa_examples.values()))           # BRIGHT is a retrieval benchmark: labeled queries, no scorable tasks
    ac_model = harness.module_manager.get_ac_model(AC_NAME)
    harness.loaded_models[MODEL_NAME].ensure_engine_loaded()          # after the AC registration: enable_prompt_embeds and the memory reservation apply
    tokenizer = harness.loaded_models[MODEL_NAME].tokenizer
    try:
        # --- compaction: three long tool outputs cross a 600-token delta; a python call is refused; a lone compact restarts the segment.
        config = replace(BASE, user_prompt=PRIMES, compaction_threshold_tokens=600)
        turns = [SyntheticTurn(f"Step {i}.", [("python", {"code": f"print({i})"})], [LONG]) for i in range(3)]
        agent = synthesize_agent(harness, config, turns)
        assert agent.compaction_due and not agent.run_results.compactions
        agent.record_assistant_turn("More work.", [{"id": "r1", "name": "python", "arguments": {"code": "print(4)"}}], tokenizer.encode("More work."), [])
        state = agent.simulate_step()
        assert state["results"][0].output.startswith("Refused python") and agent.compaction_refusals == 1
        agent.record_assistant_turn("", [{"id": "c1", "name": "compact", "arguments": {"summary": "Printed 0..3; next sum the primes."}}],
                                    tokenizer.encode("compact"), [])
        state = agent.simulate_step()
        results = agent.run_results
        assert state["compactions"] == 1 and results.compactions[0].finish_reason == "compacted"
        assert results.compactions[0].num_turns == 5 and len(results.compactions[0].trajectory) == 9   # the compact call has no tool step
        assert results.trajectory == [] and not agent.compaction_due and agent.segment_start == len(agent.prefix)
        assert [m["role"] for m in results.prompt_messages] == ["system", "user", "user"]
        tree = direct_parts(results.prompt_messages)[0]
        assert tree["compression_target"] == config.ac_compaction_ratio and tree["messages"][0]["content"] == config.user_prompt
        assert len(tree["messages"]) == 1 + 9 and tree["messages"][-1]["tool_calls"][0]["function"]["name"] == "compact"
        assert results.prompt_messages[2]["content"][1]["text"].endswith("Printed 0..3; next sum the primes.")
        assert len(agent.spans) == 1 and agent.rows[0].shape[0] == ac_model.part_view_rows(tree["messages"], tree["compression_target"])
        assert state["embeds_shape"] == (len(agent.prefix), agent.rows[0].shape[1]) and state["row_positions"] == agent.rows[0].shape[0]
        _probe_engine(agent, "compaction")
        # A second compaction nests the first tree verbatim.
        agent.record_assistant_turn("", [{"id": "c2", "name": "compact", "arguments": {"summary": "again"}}], tokenizer.encode("compact"), [])
        agent.compaction_due = True
        agent.simulate_step()
        outer = direct_parts(agent.run_results.prompt_messages)[0]
        assert direct_parts(outer["messages"])[0] == tree and len(agent.run_results.compactions) == 2
        _probe_engine(agent, "compaction x2")

        # --- subagent in step mode: the child's prompt carries the parent segment; the parent's result carries the child's segment.
        child_base = replace(BASE, user_prompt="", tools={})
        config = replace(BASE, user_prompt="Delegate the prime search.", tools={"subagent": (SubagentTool, {"base_config": child_base})})
        agent = synthesize_agent(harness, config, [SyntheticTurn("Delegating.", [("subagent", {"task": "Find the 140th prime."})])])
        state = agent.simulate_step()
        child = agent.run_results.subagent_results[0]
        assert child.finish_reason == "simulated" and child.answer.endswith("simulated run")
        child_user = child.prompt_messages[1]
        assert [p.get("type") for p in child_user["content"]] == ["text", "activation_context", "text"]
        assert child_user["content"][2]["text"] == "Find the 140th prime." and child_user["content"][1]["messages"][0]["content"] == config.user_prompt
        assert child.prompt_ac_spans and child.prompt_ac_spans[0]["length"] == ac_model.part_view_rows(child_user["content"][1]["messages"], BASE.ac_subagent_ratio)
        tool_step = agent.run_results.trajectory[-1]
        assert [p.get("type") for p in tool_step["messages"][0]["content"]] == ["activation_context", "text"]
        assert tool_step["messages"][0]["content"][0]["messages"][0]["content"][1]["type"] == "activation_context"   # the child's segment starts with its part
        assert len(tool_step["ac_spans"]) == 1 and len(agent.spans) == 1
        _probe_engine(agent, "subagent")

        # --- large tool output: a real python call prints 100k chars; the part holds the full text nested with the parent segment.
        config = replace(BASE, user_prompt="Print a lot.")
        agent = synthesize_agent(harness, config, [SyntheticTurn("Printing.", [("python", {"code": "print('x' * 100000)"})])])
        state = agent.simulate_step()
        result = state["results"][0]
        assert len(result.output) < 21_000 and "truncated" in result.output and result.content is not None
        part, text = result.content
        assert part["type"] == "activation_context" and text["text"] == result.output
        assert part["compression_target"] == BASE.ac_tool_output_ratio and part["messages"][1]["role"] == "tool"
        assert len(part["messages"][1]["content"]) >= 80_000 and direct_parts(part["messages"])[0]["messages"][0]["content"] == config.user_prompt
        assert agent.run_results.trajectory[-1]["ac_spans"][0]["length"] == agent.rows[0].shape[0]
        _probe_engine(agent, "tool output")
        agent.shutdown()

        # --- semantic search: top_k visible, min(20, 2 top_k) extra as a part after the text.
        config = replace(BASE, user_prompt=example.query, tools={"semantic_search": (SemanticSearchTool, {"top_k": 3, "dataset_id": bright.dataset_id})})
        agent = synthesize_agent(harness, config, [SyntheticTurn("Searching.", [("semantic_search", {"query": example.query[:200]})])])
        state = agent.simulate_step()
        result = state["results"][0]
        assert result.output.count("\n[") + result.output.startswith("[") == 3 and result.content[0]["type"] == "text" and result.content[1]["type"] == "activation_context"
        extra = result.content[1]["messages"][1]["content"]
        assert extra.count("\n[") + extra.startswith("[") == min(SEARCH_EXTRA_MAX, 6) and result.content[1]["compression_target"] == BASE.ac_search_ratio
        _probe_engine(agent, "search")

        # --- resume: the serialized record rebuilds the same prefix, spans and rows.
        data = json.loads(json.dumps(agent.run_results.serialize()))
        resumed = Agent.resume_from_run_result(harness, AgentRunResult.deserialize(data, harness))
        assert resumed.prefix == agent.prefix and resumed.spans == agent.spans
        assert [tuple(r.shape) for r in resumed.rows] == [tuple(r.shape) for r in agent.rows]
        assert all(torch.equal(a, b) for a, b in zip(resumed.rows, agent.rows))
        assert resumed.messages == agent.messages and resumed.compaction_due == agent.compaction_due
        print("\n=== AC model stats ===\n" + json.dumps(ac_model.stats.summarize(), indent=1))
    finally:
        _release(harness)


@pytest.mark.gpu
@pytest.mark.slow
def test_ac_agent_rollout_with_ac():
    """A real rollout with AC rows and a low threshold: mechanics only (segments recorded, rows shipped, cache hits), the answer is printed. Ends with the harvest of AC training items from the runs."""
    harness = _harness()
    config = replace(BASE, user_prompt=PRIMES, compaction_threshold_tokens=300)          # a few hundred tokens per turn: compaction after ~2 turns
    reporter = RolloutReporter(str(resolve_path("AGENT_AC_TEST/with_ac")), title="Agent AC test: primes with compaction")
    try:
        results = harness.rollout_manager.perform_grouped_rollouts([config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter)[0]
        generator = ActivationContextStudyGenerator(harness, AC_NAME)
        items = generator.items_from_run_results(results, weight_of=lambda run: 1.0)
        print(f"\n=== harvested {len(items)} AC training items from {len(results)} runs: {generator.harvest_report}")
    finally:
        _release(harness)
    for result in results:
        print(f"\n=== seed {result.seed}: {result.finish_reason}, {result.num_turns} turns, {len(result.compactions)} compactions, "
              f"{result.num_ac_parts} parts / {result.num_ac_rows} rows, cached {result.num_cached_input_tokens}, answer {result.answer!r}")
        assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call", "compaction_refused", "trajectory_cap")
        assert result.ac_model_name == AC_NAME and result.ac_model_version is not None
        for segment in result.compactions:
            assert segment.finish_reason == "compacted" and segment.prompt_token_ids and segment.trajectory
        if result.compactions:
            assert result.prompt_ac_spans and result.num_ac_rows > 0
    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))


@pytest.mark.gpu
@pytest.mark.slow
def test_agent_compaction_text_only():
    """No AC model: the same protocol with the summary alone; at least one rollout compacts. Finish reasons are printed, not asserted beyond the allowed set."""
    harness = _harness(with_ac=False)
    config = replace(BASE, ac_model_name=None, enable_ac_communication=False, compaction_threshold_tokens=300, user_prompt=PRIMES)
    reporter = RolloutReporter(str(resolve_path("AGENT_AC_TEST/text_only")), title="Agent AC test: text-only compaction")
    try:
        results = harness.rollout_manager.perform_grouped_rollouts([config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter)[0]
    finally:
        _release(harness)
    for result in results:
        print(f"\n=== seed {result.seed}: {result.finish_reason}, {result.num_turns} turns, {len(result.compactions)} compactions, answer {result.answer!r}")
        assert result.num_ac_parts == 0 and result.prompt_ac_spans == []
        for segment in result.compactions:
            assert segment.finish_reason == "compacted"
            assert isinstance(result.prompt_messages[-1]["content"], str)
        assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call", "compaction_refused", "trajectory_cap")
    assert any(result.compactions for result in results), "no rollout crossed the 300-token delta"
