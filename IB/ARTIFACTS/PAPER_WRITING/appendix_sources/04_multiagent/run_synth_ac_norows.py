"""
The rows-removed control of run_synth_ac.py (synthesis fork, Oct 14 night): the same reader adapter, the same prompt with the three
AC parts taken out and nothing else changed ("Agent N\\nIts run:\\n" then directly "\\n\\nFinal answer: <result>"), so the only
difference from the AC condition is the rows. Same arguments and outputs as run_synth_ac.py (the pair is still registered: its
reader adapter answers; the encoder is never called).

    CUDA_VISIBLE_DEVICES=0 python run_synth_ac_norows.py --tag tqa_synth_norows_sft_v1 --pair PAIR --at3-id tqa_reserve_solve_v1u ...
"""
import run_synth_ac as base
from run_synth1call import FINAL_RULE


class SynthesizeWithoutRows(base.SynthesizeFromRows):
    name = "synthesize_without_rows"

    def install(self) -> dict:
        agent = self.agent
        agent.begin()
        content = [{"type": "text", "text": self.task_body + "\n\n" + base.INSTRUCTION + "\n\n"}]
        for position, result in enumerate(self.results, start=1):
            content += [{"type": "text", "text": f"Agent {position}\nIts run:\n"}, {"type": "text", "text": f"\n\nFinal answer: {result}\n\n"}]
        content.append({"type": "text", "text": FINAL_RULE[self.benchmark]})
        agent._begin_segment([{"role": "user", "content": content}])
        return {"task_id": agent.agent_config.dataset_task.task_id, "benchmark": self.benchmark, "prompt_tokens": len(agent.prefix),
                "rows": [], "encoder_tokens": [], "results": self.results, "control": "rows removed"}


base.SynthesizeFromRows = SynthesizeWithoutRows      # main() reads the module global when it builds each task's config

if __name__ == "__main__":
    raise SystemExit(base.main())
