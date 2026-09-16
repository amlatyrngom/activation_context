# Correctness review v2

Verdict: pass

Reviewed the revised paired `PLAN.md` and `PLAN.tressoir.md` against the pinned vLLM source and accepted user contract. No unresolved correctness findings.

- C1 fixed: PLAN.md V2 explicitly adds the mixed token/embedding mask to every mixed-prompt block's embedding identity, including blocks before the first delta and input-only mixed prompts; ordinary token-only keys remain unchanged. The human projection describes this narrow existing-cache correction and both acceptance tables include changed-mask misses. This fixes the collision rather than merely salting intervention-containing blocks.
- C2 fixed: V1 now names `offline_utils._add_completion_requests`, `_render_and_add_requests`, `_add_request`, and `_run_completion` where applicable, with batch order and sampling-child association preserved.
- The revised install plan is more concrete: full immutable uv Git source matching the gitlink, compatible pinned native wheel reuse or source-build fallback, clean-install SHA/capability proof and preserved IB cloud-upload exclusion. None of these planned build checks is represented as already run.

The prior positive assessments stand: explicit request/IPC/scheduler/worker/model seams; correct addition to the fused residual input; complete immutable replay; per-block delta hashing and suffix invalidation for hybrid state; sparse mixed-request staging; HF gradients/checkpoint/adapter binding and generation; captured-record lifecycle and legacy migration; actual-source M0 reconciliation; separate mechanics and quality gates. Human projection appropriately emphasizes the non-vLLM interfaces while the agent source spells out engine details.

This pass approves the plan's correctness and scope, not an implementation. GPU parity, preemption, caching, derivative, serialization and deployment gates remain required future work. Training-friendly recipe findings have their own independent review.
