"""Slice 3b diagnostic (IB-only): prefix-cache behavior of the agent's compaction prompts, replaying the node test's sequence."""
import sys
sys.modules.setdefault("fla", None)
from dataclasses import replace

from activation.ac_model import ActivationContextModelConfig
from activation.agent import AgentConfig, SubagentTool
from activation.agent.agent_utils import SyntheticTurn, synthesize_agent
from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
from activation.tests.test_basic_agent_ac import BASE, LONG, MODEL_NAME, PRIMES, AC_NAME, _harness, _release

def main():
    global harness, loaded, tokenizer
    harness = _harness()
    loaded = harness.loaded_models[MODEL_NAME]
    loaded.ensure_engine_loaded()
    tokenizer = loaded.tokenizer


    def probe(agent, label, n=3, max_tokens=None):
        prefix, embeds, mask = agent.prepare_request()
        outs = [agent.submit_probe(max_tokens=max_tokens) for _ in range(n)]
        print(f"[{label}] prompt {len(prefix)} spans {agent.spans} cached {[o.cached_prompt_token_count for o in outs]} sampled {[o.output_token_count for o in outs]}", flush=True)
        return prefix


    def submit_tokens_only(prefix, label, n=2):
        outs = [loaded.engine_submit_tokens(list(prefix), seed=7, agent_id=f"diag-{label}", chat_kwargs={"sampling_params": {"max_tokens": 1, "temperature": 0.0}}) for _ in range(n)]
        print(f"[{label} tokens-only] prompt {len(prefix)} cached {[o.cached_prompt_token_count for o in outs]}", flush=True)


    try:
        config = replace(BASE, user_prompt=PRIMES, compaction_threshold_tokens=600)
        turns = [SyntheticTurn(f"Step {i}.", [("python", {"code": f"print({i})"})], [LONG]) for i in range(3)]
        agent = synthesize_agent(harness, config, turns)
        agent.record_assistant_turn("More work.", [{"id": "r1", "name": "python", "arguments": {"code": "print(4)"}}], tokenizer.encode("More work."), [])
        agent.simulate_step()
        agent.record_assistant_turn("", [{"id": "c1", "name": "compact", "arguments": {"summary": "Printed 0..3."}}], tokenizer.encode("compact"), [])
        agent.simulate_step()
        p1 = probe(agent, "compaction 1")
        p1b = probe(agent, "compaction 1, max_tokens=1", max_tokens=1)
        agent.record_assistant_turn("", [{"id": "c2", "name": "compact", "arguments": {"summary": "again"}}], tokenizer.encode("compact"), [])
        agent.compaction_due = True
        agent.simulate_step()
        p2 = probe(agent, "compaction 2")
        p2b = probe(agent, "compaction 2, max_tokens=1", max_tokens=1)
        common = next((i for i, (a, b) in enumerate(zip(p1, p2)) if a != b), min(len(p1), len(p2)))
        print(f"prompts share the first {common} tokens; p1 {len(p1)} p2 {len(p2)}", flush=True)
        submit_tokens_only(p2, "compaction 2")
        submit_tokens_only(p1, "compaction 1")
        # A fresh agent straight to one compaction: same prompt as compaction 1?
        agent2 = synthesize_agent(harness, config, turns)
        agent2.record_assistant_turn("More work.", [{"id": "r1", "name": "python", "arguments": {"code": "print(4)"}}], tokenizer.encode("More work."), [])
        agent2.simulate_step()
        agent2.record_assistant_turn("", [{"id": "c1", "name": "compact", "arguments": {"summary": "Printed 0..3."}}], tokenizer.encode("compact"), [])
        agent2.simulate_step()
        p3 = probe(agent2, "fresh agent compaction 1")
        print(f"fresh prompt equals p1: {p3 == p1}", flush=True)
        print("AC stats:", harness.module_manager.get_ac_model(AC_NAME).stats if hasattr(harness.module_manager.get_ac_model(AC_NAME), 'stats') else '', flush=True)
    finally:
        _release(harness)


if __name__ == "__main__":
    main()
