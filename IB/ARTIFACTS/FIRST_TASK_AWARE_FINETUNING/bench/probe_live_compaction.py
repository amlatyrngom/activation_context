"""Private serial HF live-tool probe; not a production vLLM bridge or benchmark."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import time
import traceback

import torch

from activation.agent.agent import Agent
from activation.agent.agent_config import AgentConfig
from activation.agent.agent_tools import AgentTool, DEFAULT_TOOLS, ToolCallResult
from activation.ac_model.ac_model_utils import EncodeRequest, assistant_target_ids
from activation.common.ac_parts import direct_parts
from activation.harness.loaded_model import EngineChatOutput
from activation.harness.hf_utils import generate_with_interventions
from run_qa import AC, NAME, READER, harness, write
from evaluate_trajectory_cohort import file_hash

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


class DeepHFAgent(Agent):
    """Retain Agent's actual loop; replace only encoding and serial inference."""

    def _outputs(self, messages):
        parts = direct_parts(messages)
        if not parts:
            return []
        assert self.ac_model is not None and self.ac_model.mode == 'rollout'
        requests = [EncodeRequest(part['messages'], float(part.get('compression_target')
                     or self.ac_model.config.default_compression_ratio), False, part.get('tools'))
                    for part in parts]
        with torch.no_grad():
            outputs = self.ac_model.encode_batch(requests, with_layers=True)
        for output in outputs:
            assert set(output.layer_inputs) == set(self.ac_model.intervention_layers)
            assert output.has_interventions
        return outputs

    def _encode_parts(self, messages):
        rows = [output.input_embeds.detach().to(dtype=self.row_dtype).cpu().contiguous()
                for output in self._outputs(messages)]
        self.run_results.num_ac_parts += len(rows)
        self.run_results.num_ac_rows += sum(int(row.shape[0]) for row in rows)
        return rows

    def _submit(self, chat_kwargs=None):
        assert not self.agent_config.record_sampling
        target = self.loaded_model
        manager = self.harness.module_manager
        model = manager.ensure_lora(self.agent_config.lora_name)
        kwargs = self.agent_config.call_kwargs if chat_kwargs is None else chat_kwargs
        sampling = (kwargs or {}).get('sampling_params', {})
        assert sampling.get('temperature', 0) == 0
        budget = int(sampling.get('max_tokens', 512))
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        eot = target.model_config.model_description.eot_token
        if eot is not None:
            eos.append(target.tokenizer.convert_tokens_to_ids(eot))
        eos = sorted(set(eos))
        with torch.no_grad(), manager.lora_context(NAME, self.agent_config.lora_name):
            ids = torch.tensor(self.prefix, device=next(model.parameters()).device)
            embeds = model.get_input_embeddings()(ids)
            outputs = self._outputs(self.messages)
            assert len(outputs) == len(self.spans) == len(self.rows)
            for output, row in zip(outputs, self.rows):
                assert torch.equal(output.input_embeds.to(row).cpu(), row)
            embeds, payload = self.probe_trainer._assemble_student(embeds, self.spans, outputs, embeds.dtype)
            if outputs:
                assert set(payload['layer_inputs']) == set(self.ac_model.intervention_layers)
            generation = dict(max_new_tokens=budget, do_sample=False, use_cache=True,
                              pad_token_id=self._pad_id(), eos_token_id=eos)
            if payload is None:
                generated = model.generate(inputs_embeds=embeds[None], **generation)
            else:
                generated = generate_with_interventions(model, inputs_embeds=embeds[None],
                                                        interventions=[payload], **generation)
            sampled = generated[0].tolist()
        # Inputs were embeddings: HF returns generated IDs only. Keep tool delimiters.
        assert len(sampled) <= budget
        return EngineChatOutput(text=target.tokenizer.decode(sampled, skip_special_tokens=False),
                                prompt_token_count=len(self.prefix), output_token_count=len(sampled),
                                finish_reason='stop' if sampled and sampled[-1] in eos else 'length',
                                token_ids=sampled, lora_name=self.agent_config.lora_name)


class ReadPacket(AgentTool):
    name = 'read_packet'
    description = ('Read one numbered part of the release packet. Each part can be delivered only once; '
                   'after delivery its contents cannot be fetched again. Parts contain service settings.')
    parameters = {'type': 'object', 'properties': {'part': {'type': 'integer'}}, 'required': ['part']}

    def __init__(self, harness, agent, folder):
        super().__init__(harness, agent)
        self.folder = Path(folder)
        self.delivered = set()

    def execute(self, part):
        if not isinstance(part, int) or part < 0 or not (self.folder / f'{part}.json').exists():
            return ToolCallResult('No such packet part.', is_error=True)
        if part in self.delivered:
            return ToolCallResult('This part was already delivered and is no longer available.', is_error=True)
        value = (self.folder / f'{part}.json').read_text()
        self.delivered.add(part)
        return ToolCallResult(value)


class CommitRelease(AgentTool):
    name = 'commit_release'
    description = ('Commit the requested service with its exact version, rollback tag and worker count from '
                   'the release packet. Returns a receipt on success. Incorrect settings are rejected.')
    parameters = {'type': 'object', 'properties': {
        'service': {'type': 'string'}, 'version': {'type': 'string'},
        'rollback': {'type': 'string'}, 'workers': {'type': 'integer'}},
        'required': ['service', 'version', 'rollback', 'workers']}

    def __init__(self, harness, agent, expected, receipt):
        super().__init__(harness, agent)
        self.expected, self.receipt, self.committed = expected, receipt, False

    def execute(self, service, version, rollback, workers):
        actual = dict(service=service, version=version, rollback=rollback, workers=workers)
        if actual != self.expected:
            return ToolCallResult('Rejected: the service or settings do not match the release packet.', is_error=True)
        self.committed = True
        return ToolCallResult('Release committed. Receipt: ' + self.receipt)


def prepare_fixture(tokenizer, folder, target_tokens, position):
    """Deterministic fictional packets, fixed before any model response is seen."""
    rng = random.Random(20260924 + target_tokens + position)
    rows, count = [], 0
    # Leave room for array indentation, tool schemas and turn wrappers. The final
    # Agent prefix is checked against the requested cap, not inferred from this sum.
    while count < (target_tokens - 900) * .89:
        row = {'service': f'service-{len(rows):04d}', 'version': f'v{rng.randrange(1,10)}.{rng.randrange(1,30)}.{rng.randrange(10,90)}',
               'rollback': f'rel-{rng.randrange(100000,999999)}', 'workers': rng.randrange(10,96),
               'owner': rng.choice(['platform', 'billing', 'ingestion', 'support', 'analytics']),
               'region': rng.choice(['us-east', 'us-west', 'eu-west', 'eu-central']),
               'note': 'Routine release. Apply only to the service explicitly requested by the operator.'}
        rows.append(row)
        count += len(tokenizer.encode(json.dumps(row, indent=1), add_special_tokens=False)) + 3
    selected = rows[len(rows) * position // 100]
    expected = {key: selected[key] for key in ('service', 'version', 'rollback', 'workers')}
    receipt = 'receipt-' + hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:14]
    folder.mkdir(parents=True, exist_ok=True)
    parts = [rows[start:start + 16] for start in range(0, len(rows), 16)]
    for index, values in enumerate(parts):
        content = json.dumps(values, indent=1)
        assert len(content) < 20_000
        path = folder / f'{index}.json'
        if path.exists():
            assert path.read_text() == content
        else:
            path.write_text(content)
    return {'case': f'{target_tokens}_{position}', 'folder': str(folder), 'expected': expected,
            'receipt': receipt, 'parts': len(parts), 'nominal_history_tokens': target_tokens,
            'target_position_percent': position,
            'packet_hashes': [file_hash(folder / f'{i}.json') for i in range(len(parts))]}


def seed_agent(h, trainer, fixture, use_ac):
    tools = {name: None for name in DEFAULT_TOOLS if name != 'submit_answer'}
    tools.update(read_packet=(ReadPacket, {'folder': fixture['folder']}),
                 commit_release=(CommitRelease, {'expected': fixture['expected'], 'receipt': fixture['receipt']}))
    config = AgentConfig(model_name=NAME, lora_name=READER, ac_model_name=AC if use_ac else None,
        tools=tools, system_prompt='You are a release operator. Use the provided tools, then submit the receipt. Do not invent settings.',
        user_prompt=f"Read the release packet, commit {fixture['expected']['service']} with its exact settings, then submit only the receipt.",
        call_kwargs={'sampling_params': {'temperature': 0, 'max_tokens': 512}, 'chat_template_kwargs': {'enable_thinking': False}},
        record_sampling=False, max_turns=fixture['parts'] + 6, max_duration=900,
        compaction_threshold_tokens=100_000, absolute_trajectory_cap=40_000, ac_compaction_ratio=1/16)
    agent = DeepHFAgent(h, config)
    agent.probe_trainer = trainer
    agent.begin()
    for index in range(fixture['parts']):
        call = {'id': f'packet_{index}', 'name': 'read_packet', 'arguments': {'part': index}}
        message = agent.dialect.rendering.assistant_message('', [call])
        ids = assistant_target_ids(agent.loaded_model.tokenizer, message, agent.template_tools, agent.template_kwargs)
        agent.record_assistant_turn('', [call], ids, [])
        result = agent.execute_tool_call(call)
        assert not result.is_error and not result.activation_content
        agent.append_tool_results([call], [result])
        agent.run_results.num_turns += 1
    assert not agent.spans and not agent.run_results.finish_reason
    # Every condition receives this same cut, after actual one-time reads, with no summary.
    assert agent.tools['read_packet'].delivered == set(range(fixture['parts']))
    return agent


def main(args):
    from activation.ac_model import ActivationContextTrainer, ActivationContextTrainingConfig
    from activation.autotuners import configure_fla_runtime
    from activation.common.reporting import HtmlReporter
    started = time.monotonic()
    pair = json.loads(args.pair.read_text())
    identity = {'worker': file_hash(__file__), 'pair': {name: file_hash(path) for name, path in {
        'heads': Path(pair['ac_checkpoint']) / 'ac_modules.pt',
        'encoder': Path(pair['ac_checkpoint']) / 'side_lora/adapter_model.safetensors',
        'reader': Path(pair['reader_checkpoint']) / 'adapter_model.safetensors'}.items()}, 'condition': args.condition}
    report = args.output / 'report'
    result_path = report / 'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'identity': identity, 'rows': [], 'status': 'running'}
    assert result['identity'] == identity
    if result['status'] == 'complete':
        return
    if result['status'] == 'failed':
        raise RuntimeError('A previous condition failed; retain this evidence and use a new output for a repaired probe')
    reporter = HtmlReporter(str(report), 'HF live-tool compaction probe: ' + args.condition,
        'Six fixed fictional release packets. Scripted real reads precede the cut; model-generated tool calls follow it. '
        'This is an integration probe, not a natural benchmark or production vLLM validation.', page_rewrite_seconds=math.inf)
    reporter.set_trajectories('cases', 'Actual continuations', 'All failures are retained.', [])
    reporter.set_status(phase='loading paired checkpoint'); reporter.render(force=True)
    h = harness(Path(pair['ac_checkpoint']), Path(pair['reader_checkpoint']))
    ac = h.module_manager.get_ac_model(AC)
    trainer = ActivationContextTrainer(h, ActivationContextTrainingConfig(loss_kind='sft', no_context_penalty_weight=0))
    variables = trainer._autotuning_variables(ac)
    h.module_manager.ensure_lora(READER)
    ac.set_mode('rollout'); ac.prepare()
    for model in (ac.side.model, ac.target.model):
        model.eval()
        if model.is_gradient_checkpointing:
            model.gradient_checkpointing_disable()
    views = ['full history', 'no history', 'initial AC'] if args.condition == 'initial' else ['trained AC', 'trained reader without history']
    with configure_fla_runtime(variables['fla_config']), torch.no_grad():
        for length in (8192, 16384, 32768):
            for position in (20, 80):
                fixture = prepare_fixture(ac.target.tokenizer, args.fixtures / f'{length}_{position}', length, position)
                for view in views:
                    if any(row['case'] == fixture['case'] and row['view'] == view for row in result['rows']):
                        continue
                    if STOP or time.monotonic() - started > args.seconds - 1000:
                        result['status'] = 'paused'; write(result_path, result); return 75
                    reporter.status.clear(); reporter.set_status(phase='running actual tools', case=fixture['case'], view=view)
                    reporter.render(force=True)
                    agent = None
                    try:
                        agent = seed_agent(h, trainer, fixture, use_ac=view.endswith('AC'))
                        before = len(agent.prefix)
                        prefix_hash = hashlib.sha256(json.dumps(agent.prefix).encode()).hexdigest()
                        assert before <= length, (before, length)
                        for previous in result['rows']:
                            if previous['case'] == fixture['case']:
                                assert previous['seed_prefix_hash'] == prefix_hash
                        if view != 'full history':
                            agent.compact('')
                            assert agent.run_results.total_turns() == fixture['parts']
                        after = len(agent.prefix)
                        output = agent.run()
                    except Exception:
                        result['status'] = 'failed'
                        result['rows'].append({'case': fixture['case'], 'view': view, 'fixture': fixture,
                            'success': False, 'finish_reason': 'setup_or_compaction_error',
                            'error': traceback.format_exc(), 'prefix_at_failure': len(agent.prefix) if agent else None})
                        write(result_path, result)
                        reporter.status.clear(); reporter.set_status(phase='failed; evidence retained', case=fixture['case'], view=view)
                        reporter.set_text('failure', 'Probe failure', result['rows'][-1]['error'])
                        reporter.render(force=True)
                        raise
                    committed = agent.tools['commit_release'].committed
                    success = committed and str(output.answer).strip() == fixture['receipt'] and output.finish_reason == 'submitted'
                    record_dir = args.output / 'records'
                    record_dir.mkdir(exist_ok=True)
                    record_path = record_dir / f"{fixture['case']}_{view.replace(' ', '_')}.json"
                    write(record_path, output.serialize(base_dir=record_dir))
                    row = {'case': fixture['case'], 'view': view, 'fixture': fixture, 'seed_prefix_hash': prefix_hash,
                           'prefix_before': before, 'prefix_after': after, 'rows': output.num_ac_rows,
                           'layers': ac.intervention_layers if view.endswith('AC') else [],
                           'success': success, 'committed': committed, 'answer': output.answer,
                           'finish_reason': output.finish_reason, 'error': output.score_feedback,
                           'generated_turns': output.total_turns() - fixture['parts'], 'record': str(record_path),
                           'steps': [{'role': step['role'], 'turn': i, 'content': step['content'],
                                      'calls': step.get('tool_calls', []), 'results': step.get('tool_call_results', [])}
                                     for i, step in enumerate(output.trajectory) if view != 'full history' or i >= fixture['parts'] * 2]}
                    result['rows'].append(row); write(result_path, result)
                    reporter.widgets['cases']['items'] = [dict(title=x['case'] + ': ' + x['view'], state='finished',
                        stats={key: x[key] for key in ('success', 'committed', 'prefix_before', 'prefix_after', 'generated_turns', 'finish_reason')},
                        prompt='Commit ' + x['fixture']['expected']['service'], steps=x['steps']) for x in result['rows']]
                    reporter.render(force=True)
                    print(json.dumps({key: row[key] for key in ('case', 'view', 'success', 'finish_reason', 'prefix_before', 'prefix_after')}), flush=True)
                    del agent
                    ac.cache.clear()
    result['status'] = 'complete'; write(result_path, result)
    reporter.status.clear(); reporter.set_status(phase='complete'); reporter.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', type=Path, required=True)
    parser.add_argument('--condition', choices=['initial', 'trained'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=6600)
    args = parser.parse_args()
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    raise SystemExit(main(args))
