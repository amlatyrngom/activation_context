# Agent rollouts, slice 1: the handoff

Round 4 (2026-09-07). Your round-3 note on the memory limit is applied (see the application handoff and the drifts). Round 3 text follows.

Round 3 (2026-09-07). Your two round-2 changes are in: `AgentConfig.tools` maps a name to `(tool class, constructor kwargs)`, and the DAPO loader pattern-replaces the dataset's "Answer:" instruction with one that requires `submit_answer`. Your three mid-run notes are in too: the report renders the trajectories (running ones open, then the last N finished, uncut copies as files next to the page), model-specific parsing and rendering live in `agent/agent_utils.py` and adapt to the chat template, and a tool-less turn gets the nudge "Call one of the provided tools or submit_answer when done." Both key tests passed on a fresh RTX PRO 6000 node (`ac-agent`, us-east-2, torn down afterwards). The M0 revert comes first below; the cards are exact deltas against the tree you have after it, and the staged copies sit next to this document (`activation/`, `README.md`).

## Application handoff (for your workspace agent)

Order of work, in your checkout of `/source`:

1. **Revert first**, with the block below (eight `git checkout`s, five deletions, one folder, `data_syncing.py` removed). Then `git status` should show only `README.md`, `pyproject.toml`, `uv.lock`, the `bench/agent_probes/` move and your agent stubs as modified or untracked. Do not touch `pyproject.toml` or `uv.lock`.
2. **Copy the staged tree** rather than applying 27 cards by hand: `activation/` and `README.md` next to this document are byte-for-byte what the cards produce. `cp -r IB/ARTIFACTS/AGENT_ROLLOUTS/activation/. activation/` and the README section by hand (the card shows the exact replacement of your two-line podman block). The cards are there to review, and as the fallback if a file of yours drifted from `/source` as I saw it (then apply that card with `git apply`, or reconcile by hand).
3. **Check the import surface** without a GPU: `uv run python -c "import activation.agent, activation.bench.agent_probes.dapo_bench, activation.dataset.loaders"`; `uv run python -m compileall -q activation`. The private CPU check (`IB/TMP/AGENT_ROLLOUTS/cpu_check.py`, fake env + scripted engine) also runs anywhere: `uv run python IB/TMP/AGENT_ROLLOUTS/cpu_check.py` ends with `CPU CHECK OK`.
4. **Podman on your host** (rootless): the README section; `podman info` and the hello-world must work as your user before any local agent run. The agent code never needs `sudo`.
5. **Rebuild the Sky image** once: `uv run sky setup --name <node> --gpu rtx6000pro` without `--no-image-rebuild` publishes `sky:uv-<lock>-df<dockerfile>` with podman baked in; until then the setup block installs it on the fallback image at every fresh node (about two minutes).
6. **Run the two tests on a node**: `uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent.py --gpu --slow -s -x`. Expected: `2 passed` in about four minutes after the engine load; the pages appear in `IB/TMP/SYNC/AGENT_TEST/{primes,dapo}/report.tressoir.html` while it runs. The DAPO test's second call must print `RolloutCache - 4 cached rollouts`.
7. **Then the study test once** (`test_basic_dataset_study.py --gpu --slow`) as the regression gate for `chat()` on the unified `AsyncLLM` engine; it was not rerun today.

Things that look wrong but are not: `env_args` resource flags are ignored inside Sky nodes (no cgroups; the soft limit carries); the report page's `cached` count stays at 0 for a fully cached call because nothing ran; `EngineDeadError` lines at process exit are vLLM's shutdown noise after a `SUCCEEDED` job; `rsync exit 24` during a sync is a report file rewritten mid-transfer.

## Test results

`uv run sky exec --sync ac-agent -- uv run pytest activation/tests/test_basic_agent.py --gpu --slow -s -x` on `Qwen/Qwen3.5-9B`, thinking off, default tools, `python:3.12-slim` sandboxes under podman inside the node's container. Fourth run; the three before it are in the drifts section. Full pytest output: `IB/TMP/AGENT_ROLLOUTS/test_report_run4.txt`; the live pages and the uncut trajectories synced home to `IB/TMP/SYNC/AGENT_TEST/{primes,dapo}/`; the cache to `IB/TMP/SYNC/ROLLOUTS/agent_test_dapo/rollouts.jsonl`.

`activation/tests/test_basic_agent.py` · pytest output, run 4 (engine logs removed)

```
============================= test session starts ==============================
activation/tests/test_basic_agent.py
=== rollout seed=0 finish=submitted turns=2 tokens in/cached/out=1806/0/427 duration=6.6s answer='2441'
=== rollout seed=1 finish=submitted turns=2 tokens in/cached/out=1807/0/428 duration=6.6s answer='2441'
answers [2441, 2441]; expected 2441 (809 + 811 + 821)
.
RolloutCache - 4 cached rollouts in /root/activation_artifacts/SYNC/ROLLOUTS/agent_test_dapo/rollouts.jsonl.
=== rollout seed=0 finish=submitted turns=4 tokens in/cached/out=7455/4224/1939 duration=28.1s score=1.00 answer='30'
=== rollout seed=1 finish=submitted turns=3 tokens in/cached/out=6758/2112/3837 duration=54.1s score=1.00 answer='30'
=== rollout seed=0 finish=submitted turns=5 tokens in/cached/out=17786/13728/4343 duration=61.2s score=1.00 answer='3'
=== rollout seed=1 finish=submitted turns=4 tokens in/cached/out=14589/9504/5596 duration=77.0s score=1.00 answer='3'
.
======================== 2 passed in 255.82s (0:04:15) =========================
```

| test | rollouts | finish | score | notes |
| --- | --- | --- | --- | --- |
| `test_basic_agent_primes` | 2 (group of 2, seeds 0 and 1) | both `submitted` in 2 turns, ~6.6 s each | not scored (no dataset task); both answered 2441 | one `python` call each (a sieve), then `submit_answer`; the report page and `trajectories/*.json` were written and synced |
| `test_basic_agent_dapo` | 4 (2 tasks × group of 2) | all `submitted`, 3–5 turns, 28–77 s | mean 1.00 | the sandbox has no `sympy`; the model read the traceback and rewrote in plain Python; one `NameError` recovered the same way. The second `perform_grouped_rollouts` call was served from the cache: 4 hits, no engine load |

What the two runs say about the harness: with the prompt shared across a group the second agent's prefix hits the cache (2,112–13,728 cached tokens on the DAPO rollouts); per-turn latency is dominated by generation, not by the sandbox (a `podman exec` round trip is well under a second). The `(sub)` rows and the parallel tool were exercised on the CPU check only (`IB/TMP/AGENT_ROLLOUTS/cpu_check.py`, scripted engine, local fake env; passes).

## DAPO probe: 100 problems × group of 2, Qwen3.5-4B vs 9B (same 100 problems, seed 0)

`activation/bench/agent_probes/dapo_bench.py` · one RTX PRO 6000 node per model, 64 agents in flight, 10 turns / 2048 output tokens per turn / 600 s per rollout, temperature 0.7, thinking off, default tools in `python:3.12-slim`. Pages and uncut trajectories: `IB/TMP/SYNC/DAPO_BENCH/qwen3.5-{4b,9b}/`; 4B summary `summary.json` and its 200 cached rollouts `IB/TMP/SYNC/ROLLOUTS/dapo_bench_qwen3.5-4b/`; the 9B numbers are from its report data (`summary_from_report.json`; its run predates the cache-folder fix, so no cache and no `summary.json`).

| | Qwen3.5-4B | Qwen3.5-9B |
| --- | --- | --- |
| accuracy over 200 rollouts | 0.725 | 0.795 |
| problems solved by both rollouts | 64 | 71 |
| problems with mixed outcomes (the group-relative signal) | 17 | 17 |
| problems never solved | 19 | 12 |
| finish reasons | submitted 154 (9 wrong), max_tool_errors 24, max_turns 22 | submitted 169 (10 wrong), max_turns 14, max_tool_errors 13, max_duration 4 |
| mean turns / output tokens / input tokens | 4.8 / 4,155 / 22,558 | 4.1 / 3,623 / 15,698 |
| wall clock (after engine load) | 11.9 min | 18.0 min |

Paired per problem (rows 4B, columns 9B; all = both rollouts solved, none = neither):

| | 9B all | 9B mixed | 9B none |
| --- | --- | --- | --- |
| 4B all | 58 | 6 | 0 |
| 4B mixed | 10 | 5 | 2 |
| 4B none | 3 | 6 | 10 |

Reading: 58 of the 100 problems are solved every time by both models and 10 by neither; 29 are mixed for at least one model, and that is the set a group-relative method learns from at group size 2 (a larger group would turn some of the 58/10 into mixed as well). Failures are mostly budget exhaustion, not wrong answers: 9–10 wrong submissions per model against 38–46 rollouts that ran out of turns, tool errors or time. Failed rollouts are long (9B: 6.5 turns and 36k input tokens on average vs 3.5 turns and 10k when solved), and their tool errors are dominated by tracebacks from the model's own code and by `import sympy` in an image without it (10 of the 9B `max_tool_errors` rollouts); two cheap harness changes would move the numbers: a sandbox image with `sympy`/`numpy`, and a stricter turn budget or context compaction. One 4B rollout exceeded the 40,960-token context (now `context_exceeded`), and one earlier 4B rollout grew a 47 GB Python process before the memory limit existed.

## What changed on the way (drifts from the plan)

- **Tool-call format (run 2, both primes rollouts `max_tool_errors`).** Qwen3.5's chat template teaches `<tool_call><function=python><parameter=code>…</parameter></function></tool_call>`, not Hermes JSON; the parser only knew JSON, so every call was a `parse_error`. Now `agent/agent_utils.py` holds every model-specific detail: `ToolCallFormat`s (XML function blocks; Hermes JSON) tried per block, text parameter values coerced by the tool schema, unterminated blocks (max_tokens) still parsed, `MessageRendering` (structured messages through the template) vs `InlineRendering` (templates without tool support), and `ModelDialect.from_chat_template` picking them off the tokenizer's template. The agent loop holds no format knowledge.
- **Report (your note).** The `active-<id>` text blocks became a `trajectories` widget in `common/reporting.py`: running rollouts first and open, then the last `finished_trajectories_shown` (default 100) complete ones newest first, each with the prompt, every assistant step (content, calls with arguments), every tool result; texts are cut on the page and the uncut trajectory of each agent is written to `<report>/trajectories/<agent>.json`, linked from its header. Tool calls are counted once (the tool step used to double them).
- **Nudge (your note).** "Call one of the provided tools or submit_answer when done." after every tool-less turn; the third in a row ends the run with `no_tool_call` (was: one nudge, then stop).
- **Podman inside the node (setup runs 1–3).** Three fixes to the setup block and the Dockerfile: apt runs non-interactively keeping the image's `/etc/fuse.conf` (dpkg was waiting at a conffile prompt), the storage config names `runroot`/`graphroot` (podman 4.9 refuses a partial `storage.conf`), and `[containers] cgroups = "disabled"` because docker delegates no cgroup controllers to the node's container (crun: "controller `pids` is not available"). Consequence: `env_args` resource limits such as `--memory` do not work inside nodes; the example in `AgentConfig` says so.
- **Sandbox memory limit (DAPO probe, Qwen3.5-4B; reshaped per your round-3 note).** One agent's Python grew to 47 GB and Ray's out-of-memory killer took the whole node job. The limit is now an env start setting: `AgentConfig.env_memory_limit_mb` (None = the harness default `agent_env_memory_limit_mb`, 8192, which most runs will use). `AgentEnv` passes it as `podman run --memory <mb>m` (the hard kill, where cgroups exist: your rootless host) and wraps every exec in `ulimit -v` at 85% of it (the soft per-process kill: `MemoryError` the model reads, before the container limit strikes). Inside Sky nodes podman has no cgroups; the first `--memory` refusal clears a class-wide flag and only the soft limit applies there. `env_args` still accepts any `podman run` flag; a `--memory` given there wins.
- **Cache folder for dotted ids (DAPO probe).** `resolve_path` takes a name with a dot (`dapo_bench_qwen3.5-9b`) for a file and creates only its parent, so the probe's first `cache.append` would have failed after the rollouts. `RolloutCache` now creates its folder itself. The 9B probe ran on the old code; its numbers below come from its report data (`summary_from_report.json`), not from a `summary.json`.
- **Context overflow is a finish reason (DAPO probe).** A 4B agent's conversation reached 42,945 prompt tokens against the 40,960 limit; vLLM's validation error was caught as `finish_reason="error"` with a traceback. It is now `context_exceeded` with the message as feedback (slice 2's compaction is the real fix; `compaction_threshold_tokens` is carried for it).
- **`reporter.report_folder` (run 3).** The approved test reads it; the reporter had only `folder`. Added as a property rather than editing the test.
- **The Sky image was not rebuilt here** (no docker in this container). The fallback image plus the setup block carried the runs; the new tag `uv-<lock>-df<dockerfile>` is picked up by your next `sky setup` without `--no-image-rebuild`.
- **Run 1 never ran the tests:** my launcher was bound to a tool timeout and I killed it, which orphaned the node job; cancelled and relaunched detached.

## The revert (run first, in your checkout)

`activation/` · the reverted and deleted files, from your repository root

```bash
git checkout -- activation/dataset/dataset_manager.py \
  activation/dataset/dataset_study.py \
  activation/retrieval/retrieval_ac.py \
  activation/retrieval/retrieval_model.py \
  activation/retrieval/retrieval_trainer.py \
  activation/harness/runtime_config.py \
  activation/harness/vllm_wrapper.py \
  activation/tests/test_basic_dataset_study.py
rm activation/dataset/dataset_caching.py activation/dataset/dataset_study_prompts.py activation/dataset/dataset_bm25_index.py activation/dataset/dataset_study_interface.py activation/tests/test_programmatic_dataset_study.py
rm -r activation/bench/bright_econ_ablation/
rm activation/common/data_syncing.py   # replaced by the card below
```

Your `pyproject.toml`, `uv.lock`, `bench/agent_probes/` move and the untouched agent stubs stay as they are; the cards below then apply cleanly, or copy the staged files from `activation/` and `README.md` next to this document over your tree.

## Diffs to apply

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">README.md</span>
    <span class="card-oneliner">Rootless podman install for the agent environments.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `README.md`:

```diff
--- a/README.md
+++ b/README.md
@@ -20,10 +20,19 @@ newgrp docker
 docker run hello-world
 ```
 
-Installing `podman`.
+Installing `podman` in rootless (non-privileged) mode. Agent environments are `podman run --rm --network none`
+containers created by the harness as your normal user; nothing needs `sudo` after this block.
 ```bash
-sudo apt update && sudo apt install podman -y
-podman run docker.io/hello-world
+sudo apt update && sudo apt install -y podman uidmap slirp4netns fuse-overlayfs
+# Rootless podman maps container users onto a range of sub-UIDs/GIDs owned by you.
+grep -q "^$USER:" /etc/subuid || sudo usermod --add-subuids 100000-165535 --add-subgids 100000-165535 "$USER"
+podman system migrate
+# Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor. Podman ships a profile, but if
+# `podman run` fails with a user-namespace or "cannot clone" error, relax the restriction:
+#   sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0
+#   echo 'kernel.apparmor_restrict_unprivileged_userns=0' | sudo tee /etc/sysctl.d/60-podman.conf
+podman info --format 'rootless={{.Host.Security.Rootless}} driver={{.Store.GraphDriverName}}'  # rootless=true driver=overlay
+podman run --rm docker.io/hello-world
 ```
 
 Installing `node` and `npm`.
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/cloud/sky.py</span>
    <span class="card-oneliner">exec --sync/--watch/--interval, the sync command, the pull thread; image tag follows the Dockerfile; privileged run option and podman in setup.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/cloud/sky.py`:

```diff
--- a/activation/cloud/sky.py
+++ b/activation/cloud/sky.py
@@ -14,6 +14,7 @@ import os
 from pathlib import Path
 import re
 import shlex
+import threading
 import shutil
 import socket
 import subprocess
@@ -60,7 +61,8 @@ DEFAULT_IMAGE_REPOSITORY = (
     "814218043106.dkr.ecr.us-east-1.amazonaws.com/activation-context/sky"
 )
 LOCK_DIGEST = hashlib.sha256((PROJECT_ROOT / "uv.lock").read_bytes()).hexdigest()[:16]
-DEFAULT_IMAGE_ID = f"docker:{DEFAULT_IMAGE_REPOSITORY}:uv-{LOCK_DIGEST}"
+DOCKERFILE_DIGEST = hashlib.sha256((PROJECT_ROOT / "activation" / "cloud" / "Dockerfile").read_bytes()).hexdigest()[:8]
+DEFAULT_IMAGE_ID = f"docker:{DEFAULT_IMAGE_REPOSITORY}:uv-{LOCK_DIGEST}-df{DOCKERFILE_DIGEST}"
 IMAGE_ID = os.environ.get("ACTIVATION_SKY_IMAGE", DEFAULT_IMAGE_ID)
 CONFIG_VERSION = "2"
 LABEL_PREFIX = "activation-sky"
@@ -68,6 +70,9 @@ REMOTE_WORKDIR = "/root/sky_workdir"
 REMOTE_ARTIFACTS = "/root/activation_artifacts"
 PUBLIC_REMOTE_ARTIFACTS = "~/activation_artifacts"
 LOCAL_DOWNLOAD_ROOT = PROJECT_ROOT / "IB" / "TMP"
+LOCAL_SYNC_ROOT = LOCAL_DOWNLOAD_ROOT / "SYNC"
+REMOTE_SYNC_ROOT = f"{REMOTE_ARTIFACTS}/SYNC"
+SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
 ECR_REGISTRY = re.compile(
     r"^[0-9]+\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com$"
 )
@@ -433,9 +438,21 @@ resources:
   labels:
 {labels}
 {secrets_yaml}workdir: {workdir}
+config:
+  docker:
+    run_options: ["--privileged"]   # podman inside the node's container (agent environments)
 setup: |
   test -x /usr/local/cuda/bin/nvcc
   mkdir -p {REMOTE_ARTIFACTS} /root/.cache/huggingface /root/.cache/flashinfer
+  # Podman for agent environments; a no-op on an image that already has it (fallback images do not).
+  if ! dpkg -s fuse-overlayfs 2>/dev/null | grep -q 'Status: install ok installed'; then
+    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \\
+      -o Dpkg::Options::=--force-confold podman fuse-overlayfs && rm -rf /var/lib/apt/lists/*
+  fi
+  mkdir -p /etc/containers
+  printf '[storage]\\ndriver = "overlay"\\nrunroot = "/run/containers/storage"\\ngraphroot = "/var/lib/containers/storage"\\n[storage.options.overlay]\\nmount_program = "/usr/bin/fuse-overlayfs"\\n' > /etc/containers/storage.conf
+  printf '[containers]\\ncgroups = "disabled"\\n[engine]\\ncgroup_manager = "cgroupfs"\\nevents_logger = "file"\\n' > /etc/containers/containers.conf
+  podman run --rm docker.io/library/python:3.12-slim python3 -c 'print("podman ok")'
 """
 
 
@@ -614,7 +631,50 @@ def download(
     return _run_rsync(*args).returncode
 
 
-# @AI: Double-check this works with directories too.
+def _sync_rsync_args() -> list[str]:
+    return ["-azu", "--itemize-changes", "--protect-args", "--no-owner", "--no-group", "--chmod=D755,F644"]
+
+
+def _push_sync(record: dict) -> None:
+    """Local IB/TMP/SYNC -> the node's SYNC folder; update-only, so files the node wrote more recently survive."""
+    LOCAL_SYNC_ROOT.mkdir(parents=True, exist_ok=True)
+    result = subprocess.run(["rsync", *_sync_rsync_args(), "--rsync-path", f"mkdir -p {REMOTE_SYNC_ROOT} && rsync",
+                             f"{LOCAL_SYNC_ROOT}/", f"{record['name']}:{REMOTE_SYNC_ROOT}/"], text=True, capture_output=True)
+    if result.returncode != 0:
+        raise RuntimeError(f"sync push failed: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else result.returncode}")
+    print(f"Synced {LOCAL_SYNC_ROOT} -> {record['name']}:{REMOTE_SYNC_ROOT}", flush=True)
+
+
+def _pull_once(record: dict, pairs: list[tuple[str, Path]], quiet: bool = False) -> None:
+    """Each (remote folder, local folder) pair pulled once; changed files replaced, nothing deleted locally."""
+    for source, destination in pairs:
+        destination.mkdir(parents=True, exist_ok=True)
+        result = subprocess.run(["rsync", *_sync_rsync_args(), f"{record['name']}:{source}", str(destination)], text=True, capture_output=True)
+        stamp = time.strftime("%H:%M:%S")
+        changed = [line for line in result.stdout.splitlines() if line[:1] in ("<", ">", "c")]
+        if result.returncode != 0:
+            detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
+            if "No such file" not in detail and not quiet:
+                print(f"[{stamp}] rsync exit {result.returncode}: {detail}", flush=True)
+        elif changed and not quiet:
+            names = ", ".join(line.split()[-1] for line in changed[:6]) + (" ..." if len(changed) > 6 else "")
+            print(f"[{stamp}] {len(changed)} file(s) updated: {names}", flush=True)
+
+
+def _pull_loop(record: dict, pairs: list[tuple[str, Path]], interval_seconds: float, stop: threading.Event) -> None:
+    while not stop.wait(interval_seconds):
+        _pull_once(record, pairs)
+
+
+def sync(name: str) -> int:
+    """One push and one pull of IB/TMP/SYNC, e.g. after a watcher died with its session."""
+    record, _, _ = _managed_record(name)
+    _start_if_stopped(record)
+    _push_sync(record)
+    _pull_once(record, [(f"{REMOTE_SYNC_ROOT}/", LOCAL_SYNC_ROOT)])
+    return 0
+
+
 def watch(
     name: str,
     remote_path: str,
@@ -668,8 +728,21 @@ def watch(
         return 0
 
 
-# @AI: make this more practical by allow --watch  etc.
-def exec_cmd(name: str, cmd: str | list[str]) -> int:
+def exec_cmd(
+    name: str,
+    cmd: str | list[str],
+    *,
+    sync: bool = False,
+    watch_paths: tuple[str, str] | None = None,
+    interval_seconds: float = 15.0,
+) -> int:
+    """
+    Upload the mirror and run the command. With sync, IB/TMP/SYNC is pushed to the node first
+    (update-only), pulled back every interval while the command runs and once more when it ends,
+    and the command sees ACTIVATION_SYNC_ROOT so `data_syncing.resolve_path` lands in that folder.
+    watch_paths (remote folder under ~/activation_artifacts, local folder under IB/TMP) adds one
+    more pulled pair, replacing a separate `watch` session.
+    """
     command = [cmd] if isinstance(cmd, str) else cmd.copy()
     if command[:1] == ["--"]:
         command = command[1:]
@@ -679,11 +752,21 @@ def exec_cmd(name: str, cmd: str | list[str]) -> int:
     record, gpu_type, gpu_count = _managed_record(name)
     _start_if_stopped(record)
     _upload_record(record)
+    pairs: list[tuple[str, Path]] = []
+    if sync:
+        _push_sync(record)
+        pairs.append((f"{REMOTE_SYNC_ROOT}/", LOCAL_SYNC_ROOT))
+    if watch_paths:
+        remote_folder = _remote_artifact_path(watch_paths[0])
+        if not remote_folder.endswith("/"):
+            raise ValueError("watch needs a remote folder (end the remote path with '/')")
+        pairs.append((remote_folder, _local_download_path(watch_paths[1])))
 
     remote_command = shlex.join(
         [
             "env",
             "VLLM_WORKER_MULTIPROC_METHOD=spawn",
+            f"{SYNC_ROOT_ENV}={REMOTE_SYNC_ROOT}",
             # The baked env is a warm cache: sync it to the uploaded lock
             # (old images set UV_NO_SYNC=1) without re-resolving on the node.
             "UV_NO_SYNC=0",
@@ -695,15 +778,27 @@ def exec_cmd(name: str, cmd: str | list[str]) -> int:
     secret_file_args = (
         ["--secret-file", str(ENV_FILE)] if ENV_FILE.is_file() else []
     )
-    return _run_skypilot(
-        "exec",
-        *secret_file_args,
-        "--gpus",
-        f"{GPU_TYPES[gpu_type]}:{gpu_count}",
-        name,
-        "--",
-        remote_command,
-    ).returncode
+    stop = threading.Event()
+    puller = None
+    if pairs:
+        puller = threading.Thread(target=_pull_loop, args=(record, pairs, interval_seconds, stop), daemon=True)
+        puller.start()
+        print(f"Pulling {', '.join(source for source, _ in pairs)} every {interval_seconds:g}s while the command runs.", flush=True)
+    try:
+        return _run_skypilot(
+            "exec",
+            *secret_file_args,
+            "--gpus",
+            f"{GPU_TYPES[gpu_type]}:{gpu_count}",
+            name,
+            "--",
+            remote_command,
+        ).returncode
+    finally:
+        if puller is not None:
+            stop.set()
+            puller.join()
+            _pull_once(record, pairs)
 
 
 def _positive_int(value: str) -> int:
@@ -755,9 +850,17 @@ def _build_parser() -> argparse.ArgumentParser:
         "exec",
         help="upload current code and run a command on a node",
     )
+    exec_parser.add_argument("--sync", action="store_true",
+                             help="mirror IB/TMP/SYNC to the node before, during and after the command (flags go before the name)")
+    exec_parser.add_argument("--watch", nargs=2, metavar=("REMOTE_FOLDER", "LOCAL_FOLDER"), default=None,
+                             help="also pull a ~/activation_artifacts folder to a local IB/TMP folder while the command runs")
+    exec_parser.add_argument("--interval", type=float, default=15.0, help="seconds between pulls (default 15)")
     exec_parser.add_argument("name", help="cluster name")
     exec_parser.add_argument("command", nargs=argparse.REMAINDER)
 
+    sync_parser = commands.add_parser("sync", help="push and pull IB/TMP/SYNC once (e.g. after a watcher died)")
+    sync_parser.add_argument("name", help="cluster name")
+
     upload_parser = commands.add_parser(
         "upload",
         help="exactly mirror local code into the remote workdir",
@@ -832,7 +935,10 @@ def main() -> int:
                 rebuild_image=not args.no_image_rebuild,
             )
         if args.action == "exec":
-            return exec_cmd(args.name, args.command)
+            return exec_cmd(args.name, args.command, sync=args.sync, watch_paths=tuple(args.watch) if args.watch else None,
+                            interval_seconds=args.interval)
+        if args.action == "sync":
+            return sync(args.name)
         if args.action == "upload":
             return upload(args.name, dry_run=args.dry_run)
         if args.action == "download":
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/cloud/Dockerfile</span>
    <span class="card-oneliner">podman and fuse-overlayfs inside the node image, with the storage and engine configs.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/cloud/Dockerfile`:

```diff
--- a/activation/cloud/Dockerfile
+++ b/activation/cloud/Dockerfile
@@ -17,9 +17,17 @@ RUN apt-get update \
         ca-certificates \
         curl \
         openssh-server \
+        podman \
+        fuse-overlayfs \
         rsync \
     && rm -rf /var/lib/apt/lists/*
 
+# Podman inside the node's (privileged) container: fuse-overlayfs on top of docker's overlay root,
+# cgroups disabled (docker delegates no controllers) and cgroupfs because there is no systemd. Agent environments are `podman run --rm --network none`.
+RUN mkdir -p /etc/containers \
+    && printf '[storage]\ndriver = "overlay"\nrunroot = "/run/containers/storage"\ngraphroot = "/var/lib/containers/storage"\n[storage.options.overlay]\nmount_program = "/usr/bin/fuse-overlayfs"\n' > /etc/containers/storage.conf \
+    && printf '[containers]\ncgroups = "disabled"\n[engine]\ncgroup_manager = "cgroupfs"\nevents_logger = "file"\n' > /etc/containers/containers.conf
+
 WORKDIR /opt/activation/image-lock
 COPY pyproject.toml uv.lock ./
 RUN uv python install 3.14 \
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/common/data_syncing.py</span>
    <span class="card-oneliner">sync_root and resolve_path: one path that is IB/TMP/SYNC here and the node&#x27;s SYNC folder there.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/common/data_syncing.py`:

```diff
--- a/activation/common/data_syncing.py
+++ b/activation/common/data_syncing.py
@@ -1,16 +1,36 @@
 """
-Helper class for coarse-grained but effective data syncing. Uses env vars to detect where it's running.
-Simple:
-- Always on for exec_cmd --sync
-- bidirectionally rsyncs IB/TMP/SYNC/, even if it contains large jsonl files.
-    - The remote folder can be the existing ~/activation_artifacts, it does not matter much.
-    - Final rsync when command is done.
-- When, for example, the oracle caching requires:
-    - data_sync.resolve_path(BRIGHT_ECON_ABLATION/CACHE/), writes to it, and those are propagated here to my machine.
-    - a new exec_cmd in a new node gets it too. Not just setup.
-- Same with reports.
-- When a run is fully local, resolves to the same path as given, and has no rsync watcher.
-- Replace watch with this: a simple sync to potentially restart orphaned syncing.
+Coarse-grained data syncing between this machine and a Sky node.
+
+One folder, `IB/TMP/SYNC/`, is the synced working area: `uv run sky exec --sync` pushes it to the
+node before the job (as `~/activation_artifacts/SYNC/`), pulls it back every few seconds while the
+job runs and once more when the job ends, so caches and reports written on the node land here as
+they are written and a later job on a fresh node starts with everything a previous one wrote.
+Code never cares where it runs: `resolve_path("BRIGHT_ECON_ABLATION/CACHE")` is the local folder on
+this machine and the remote folder on a node (the Sky wrapper exports `ACTIVATION_SYNC_ROOT`).
+A fully local run resolves to the local folder and has no watcher.
+"""
+import os
+from pathlib import Path
+
+SYNC_ROOT_ENV = "ACTIVATION_SYNC_ROOT"
+"""Set by the Sky wrapper on the node; unset here."""
+
+PROJECT_ROOT = Path(__file__).resolve().parents[2]
+LOCAL_SYNC_ROOT = PROJECT_ROOT / "IB" / "TMP" / "SYNC"
+REMOTE_SYNC_ROOT = "/root/activation_artifacts/SYNC"
+
+
+def sync_root() -> Path:
+    """The synced folder on this machine: the local root here, the artifacts folder on a node."""
+    return Path(os.environ.get(SYNC_ROOT_ENV) or LOCAL_SYNC_ROOT)
 
 
-"""
\ No newline at end of file
+def resolve_path(relative: str, create: bool = True) -> Path:
+    """A folder (or file path) under the synced area, created on request; `relative` may not escape it."""
+    relative_path = Path(relative)
+    if relative_path.is_absolute() or ".." in relative_path.parts:
+        raise ValueError(f"sync paths are relative to the synced folder: {relative!r}")
+    resolved = sync_root() / relative_path
+    if create:
+        (resolved if not resolved.suffix else resolved.parent).mkdir(parents=True, exist_ok=True)
+    return resolved
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/common/reporting.py</span>
    <span class="card-oneliner">The page is written once and polls report_data.json; remove_widget; the trajectories widget.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/common/reporting.py`:

```diff
--- a/activation/common/reporting.py
+++ b/activation/common/reporting.py
@@ -2,13 +2,16 @@
 Live HTML report for long-running jobs.
 
 One self-describing JSON document (status, named widgets, their points) rendered into a linear
-Tressoir page with Plotly.js, rewritten atomically as the job progresses. Every
-number the page shows is in the embedded data block, mirrored to report_data.json beside it. The
-page renders inside the Tressoir VS Code webview (which injects the markup after load, morphs it in
-place on file changes and fires `tressoir:render`) and in a plain browser (timed reload).
+Tressoir page with Plotly.js. `report_data.json` is rewritten atomically on every render; the page
+polls it (every `poll_seconds`) and redraws in place, so the HTML file itself is rewritten only
+every `page_rewrite_seconds` (and at the end) and the viewer does not flicker on file changes. Every
+number the page shows is also in the embedded data block, which is the fallback when the poll cannot
+fetch (e.g. a file:// page), so the finished page still works alone. The page renders inside the
+Tressoir VS Code webview (which injects the markup after load, morphs it in place on file changes
+and fires `tressoir:render`) and in a plain browser (timed reload when polling fails).
 
 `HtmlReporter` is job-agnostic: widgets are created by name (line plots, bar plots, tables, text
-blocks) and fed through `add_data_point`. Job-specific reporters subclass it and own the widget
+blocks, rendered trajectories) and fed through `add_data_point`. Job-specific reporters subclass it and own the widget
 layout, e.g. `retrieval.RetrievalReporter`.
 
 The page is self-contained: the data is embedded, and the Tressoir linear page assets, CodeMirror
@@ -52,6 +55,19 @@ _PAGE_TEMPLATE = """<!doctype html>
     .report-plot.tall { height: 26rem; }
     .report-footer { color: var(--muted); font-size: 0.85rem; }
     .report-table td { font-variant-numeric: tabular-nums; white-space: nowrap; }
+    .trajectory { border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); padding: var(--space-2) var(--space-3); margin-block: var(--space-2); }
+    .trajectory > summary { cursor: pointer; font-weight: 600; }
+    .trajectory > summary .traj-stats { color: var(--muted); font-weight: 400; font-size: 0.85rem; margin-left: 0.5rem; }
+    .trajectory > summary .traj-state { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; margin-left: 0.5rem; color: var(--accent); }
+    .trajectory > summary .traj-state.finished { color: var(--muted); }
+    .traj-step { margin-block: var(--space-2); padding-left: var(--space-3); border-left: 3px solid var(--line); }
+    .traj-step.assistant { border-color: var(--accent); }
+    .traj-step.tool { border-color: var(--positive); }
+    .traj-step.prompt { border-color: var(--warning); }
+    .traj-role { display: block; color: var(--muted); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.04em; }
+    .traj-step p { white-space: pre-wrap; margin-block: 0.25rem; }
+    .traj-step pre { max-height: 20rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; margin-block: 0.25rem; }
+    .traj-call { color: var(--muted); font-size: 0.85rem; margin-block: 0.25rem 0; }
   </style>
 </head>
 <body>
@@ -102,6 +118,8 @@ _PAGE_TEMPLATE = """<!doctype html>
   (function () {
     var report = null;
     var plots = [];
+    var polling = false;
+    var embedded = JSON.parse(document.getElementById("report-data").textContent);
 
     function cssColor(token, fallback) {
       var probe = document.createElement("span");
@@ -186,14 +204,54 @@ _PAGE_TEMPLATE = """<!doctype html>
       if (text !== undefined) node.textContent = text;
       return node;
     }
+    function codeBlock(text) {
+      var pre = el("pre", "code-block");
+      pre.appendChild(el("code", "", text));
+      return pre;
+    }
+    function trajectory(item) {
+      // One rollout: a collapsible block, open while running, with the prompt then every step.
+      var details = el("details", "trajectory");
+      if (item.state === "running") details.open = true;
+      var summary = el("summary", "", item.title);
+      summary.appendChild(el("span", "traj-state " + item.state, item.state));
+      summary.appendChild(el("span", "traj-stats", item.stats));
+      if (item.file) {
+        var link = el("a", "traj-stats", "full trajectory");
+        link.href = item.file;
+        summary.appendChild(link);
+      }
+      details.appendChild(summary);
+      if (item.prompt) {
+        var prompt = el("div", "traj-step prompt");
+        prompt.appendChild(el("span", "traj-role", "prompt"));
+        prompt.appendChild(codeBlock(item.prompt));
+        details.appendChild(prompt);
+      }
+      item.steps.forEach(function (step) {
+        var div = el("div", "traj-step " + step.role);
+        div.appendChild(el("span", "traj-role", step.role + (step.turn ? " · turn " + step.turn : "")));
+        if (step.content) div.appendChild(el("p", "", step.content));
+        (step.calls || []).forEach(function (call) {
+          div.appendChild(el("div", "traj-call", "→ " + call.name));
+          if (call.arguments) div.appendChild(codeBlock(call.arguments));
+        });
+        (step.results || []).forEach(function (result) {
+          div.appendChild(el("div", "traj-call", "← " + result.name));
+          div.appendChild(codeBlock(result.output));
+        });
+        details.appendChild(div);
+      });
+      return details;
+    }
     function build() {
-      report = JSON.parse(document.getElementById("report-data").textContent);
+      if (!report) report = embedded;
       if (window.Plotly) plots.forEach(function (p) { try { Plotly.purge(p.div); } catch (_) {} });
       plots = [];
       document.title = report.title;
       var meta = document.getElementById("report-meta");
       meta.replaceChildren();
-      ["Updated " + report.updated_at, "Refreshes every " + report.refresh_seconds + " s", "Self-contained: data embedded, libraries from pinned HTTPS URLs"]
+      ["Updated " + report.updated_at, (polling ? "Polling report_data.json every " + report.poll_seconds + " s" : "Refreshes every " + report.refresh_seconds + " s"), "Self-contained: data embedded, libraries from pinned HTTPS URLs"]
         .forEach(function (text) { meta.appendChild(el("li", "", text)); });
       var status = document.getElementById("report-status");
       status.replaceChildren();
@@ -230,6 +288,9 @@ _PAGE_TEMPLATE = """<!doctype html>
           var pre = el("pre", "code-block");
           pre.appendChild(el("code", "", w.text));
           section.appendChild(pre);
+        } else if (w.type === "trajectories") {
+          w.items.forEach(function (item) { section.appendChild(trajectory(item)); });
+          if (!w.items.length) section.appendChild(el("p", "", "No trajectories yet."));
         }
         root.appendChild(section);
       });
@@ -244,15 +305,35 @@ _PAGE_TEMPLATE = """<!doctype html>
         if (tries++ < 600) setTimeout(whenPlotly, 50);   // up to 30 s for the network
       })();
     }
+    function poll() {
+      // report_data.json beside the page is rewritten on every render; the page itself only rarely.
+      if (report && report.finished && polling) return;
+      fetch("report_data.json?t=" + Date.now(), { cache: "no-store" }).then(function (response) {
+        if (!response.ok) throw new Error("status " + response.status);
+        return response.json();
+      }).then(function (fresh) {
+        polling = true;
+        if (JSON.stringify(fresh) !== JSON.stringify(report)) { report = fresh; render(); }
+        setTimeout(poll, (report.poll_seconds || 2) * 1000);
+      }).catch(function () {
+        // No fetch here (file://, or a viewer without the folder as a resource root): the embedded
+        // data and the occasional file rewrite carry the page instead.
+        if (!polling && /^(https?|file):$/.test(location.protocol) && !window.tressoirNotebook && !report.finished) {
+          setTimeout(function () { location.reload(); }, report.refresh_seconds * 1000);
+        }
+      });
+    }
     function boot() {
       // The Tressoir extension injects this page after load and fires tressoir:render once the
       // scripts have run, and again after it morphs the file in place; a plain browser renders
-      // now and gets a timed reload instead.
-      document.addEventListener("tressoir:render", render);
+      // now. Either way the page then polls report_data.json.
+      document.addEventListener("tressoir:render", function () {
+        var fresh = JSON.parse(document.getElementById("report-data").textContent);
+        if (!report || fresh.updated_at > report.updated_at) report = fresh;
+        render();
+      });
       if (!window.tressoirNotebook) render();
-      if (/^(https?|file):$/.test(location.protocol) && !window.tressoirNotebook) {
-        setTimeout(function () { location.reload(); }, report.refresh_seconds * 1000);
-      }
+      poll();
       if (window.matchMedia) {
         var mq = window.matchMedia("(prefers-color-scheme: dark)");
         (mq.addEventListener ? mq.addEventListener("change", drawAll) : mq.addListener(drawAll));
@@ -316,6 +397,8 @@ class HtmlReporter:
         eyebrow: str = "Report",
         refresh_seconds: int = 30,
         min_render_interval_seconds: float = 5.0,
+        poll_seconds: float = 2.0,
+        page_rewrite_seconds: float = 60.0,
     ):
         self.folder = Path(folder)
         self.title = title
@@ -325,8 +408,11 @@ class HtmlReporter:
         self.min_render_interval_seconds = min_render_interval_seconds
         self.status: dict[str, str] = {}
         self.widgets: dict[str, dict] = {}
+        self.poll_seconds = poll_seconds
+        self.page_rewrite_seconds = page_rewrite_seconds
         self.finished = False
         self.last_render_time = 0.0
+        self.last_page_time = 0.0
 
     # ----------------------------------------------------------------------------- widgets
     def set_status(self, **fields: str) -> None:
@@ -368,6 +454,14 @@ class HtmlReporter:
         """A text block; calling again with the same name replaces it."""
         self.widgets[name] = {"type": "text", "name": name, "title": title, "description": "", "text": text}
 
+    def set_trajectories(self, name: str, title: str, description: str, items: list[dict]) -> None:
+        """
+        Rendered agent trajectories; calling again replaces the list. Each item: {"title", "state"
+        ("running" | "finished"), "stats", "prompt", "file" (relative link or None), "steps": [{"role", "turn", "content", "calls":
+        [{"name", "arguments"}], "results": [{"name", "output"}]}]} with every text already cut to size.
+        """
+        self.widgets[name] = {"type": "trajectories", "name": name, "title": title, "description": description, "items": items}
+
     def add_data_point(self, name: str, data: dict) -> None:
         """
         line: {"x": float, "<series>": float, ...}; bar: {"label": str, "<series>": float} or {"label", "value"};
@@ -396,7 +490,7 @@ class HtmlReporter:
                 row["running"] = True
             rows.append(row)
         else:
-            raise AssertionError(f"Widget {name!r} is a text block; use set_text.")
+            raise AssertionError(f"Widget {name!r} is a {widget['type']} block; use set_text / set_trajectories.")
 
     # ----------------------------------------------------------------------------- rendering
     def document(self) -> dict:
@@ -411,21 +505,31 @@ class HtmlReporter:
             "description": self.description,
             "eyebrow": self.eyebrow,
             "refresh_seconds": self.refresh_seconds,
+            "poll_seconds": self.poll_seconds,
             "finished": self.finished,
             "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "status": dict(self.status),
             "widgets": widgets,
         }
 
+    def remove_widget(self, name: str) -> None:
+        self.widgets.pop(name, None)
+
     def render(self, force: bool = False) -> None:
-        """Write report.tressoir.html and report_data.json atomically; throttled unless forced."""
+        """
+        Write report_data.json atomically (throttled unless forced); the page itself is written on the
+        first render, then at most every page_rewrite_seconds, and when the run finishes.
+        """
         now = time.time()
         if not force and now - self.last_render_time < self.min_render_interval_seconds:
             return
         self.folder.mkdir(parents=True, exist_ok=True)
         report = self.document()
         _write_atomic(self.folder / DATA_FILENAME, json.dumps(report, indent=1))
-        _write_atomic(self.folder / REPORT_FILENAME, render_report_page(report))
+        page = self.folder / REPORT_FILENAME
+        if self.finished or not page.exists() or now - self.last_page_time >= self.page_rewrite_seconds:
+            _write_atomic(page, render_report_page(report))
+            self.last_page_time = now
         self.last_render_time = now
 
     def finish(self) -> None:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/retrieval/retrieval_batching.py</span>
    <span class="card-oneliner">Bug fix: the embedding input limit may be None.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/retrieval/retrieval_batching.py`:

```diff
--- a/activation/retrieval/retrieval_batching.py
+++ b/activation/retrieval/retrieval_batching.py
@@ -133,7 +133,8 @@ def token_budget_groups(lengths: list[int], budget_tokens: int, num_views: int)
 def embedded_text_length(retrieval_model: "RetrievalModel", text: str, is_query: bool) -> int:
     """Characters embed_batch will actually see: cut to the input limit, plus the query instruction."""
     prefix = len(retrieval_model.query_instruction) if is_query else 0
-    return min(len(text), retrieval_model.input_limit_chars) + prefix
+    limit = retrieval_model.input_limit_chars
+    return (len(text) if limit is None else min(len(text), limit)) + prefix
 
 
 def embed_in_length_groups(
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset_study.py</span>
    <span class="card-oneliner">Bug fix (on the committed file): a label pool over the engine context skips its prompts instead of failing the run.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs the committed file (after the revert) for `activation/dataset/dataset_study.py`:

```diff
--- a/activation/dataset/dataset_study.py
+++ b/activation/dataset/dataset_study.py
@@ -15,6 +15,7 @@ import random
 import time
 import typing as t
 
+from vllm.exceptions import VLLMValidationError
 from vllm.sampling_params import StructuredOutputsParams
 
 from .dataset import (
@@ -25,6 +26,8 @@ from .dataset import (
     DatasetDocumentChunk,
 )
 from .dataset_utils import safe_truncate_embedding_chunk, shuffle_fill_truncate
+
+LABEL_PROMPT_TOKEN_LIMIT = 18_000   # under the engine's 20k context, with room for the answer
 from ..harness.vllm_wrapper import RECOMMENDED_BATCH_SIZE
 
 if t.TYPE_CHECKING:
@@ -401,7 +404,19 @@ class DatasetStudyGenerator:
                     {"role": "user", "content": [{"type": "text", "text": user_text}]},
                 ])
             batch_start_time = time.time()
-            outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
+            try:
+                outputs = loaded_model.engine_chat_many(conversations, chat_kwargs=chat_kwargs)
+            except VLLMValidationError:
+                # One pool over the engine's context fails the whole batch (glyph-dense chunks tokenize
+                # near one token per character): skip those prompts, unlabeled, and run the rest.
+                lengths = [len(loaded_model.tokenizer("\n".join(part["text"] for message in conversation for part in message["content"])).input_ids)
+                           for conversation in conversations]
+                kept = [index for index, length in enumerate(lengths) if length <= LABEL_PROMPT_TOKEN_LIMIT]
+                skipped = len(conversations) - len(kept)
+                stats.study_num_label_parse_failures += skipped
+                print(f"{self.dataset_id} - Study labels: {skipped} prompts over {LABEL_PROMPT_TOKEN_LIMIT} tokens skipped (max {max(lengths)}).")
+                batch = [batch[index] for index in kept]
+                outputs = loaded_model.engine_chat_many([conversations[index] for index in kept], chat_kwargs=chat_kwargs) if kept else []
             elapsed = time.time() - batch_start_time
             total_label_time += elapsed
             stats.study_label_batch_latencies.append(elapsed)
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/dataset.py</span>
    <span class="card-oneliner">DataOrigin back to SYNTHETIC; DatasetTask.agent_prompt; NUMERIC_EXACT dispatch.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/dataset/dataset.py`:

```diff
--- a/activation/dataset/dataset.py
+++ b/activation/dataset/dataset.py
@@ -7,9 +7,7 @@ class DataOrigin(StrEnum):
     """Who authored one registerd labeled example"""
     NATIVE = auto()
     EXTERNAL = auto()
-    SYNTHETIC_QA = auto()
-    SYNTHETIC_DESCRIPTION = auto()
-    PROGRAMMATIC = auto()
+    SYNTHETIC = auto()
 
 
 class DataSplit(StrEnum):
@@ -107,6 +105,7 @@ class DatasetTask:
     reference_metrics_kind: DatasetTaskMetricsKind
     gold_answer: str = ""
     gold_answer_aliases: list[str] = field(default_factory=list)
+    agent_prompt: str = "" # What an agent is asked to do; the loader fills it.
 
     def score(self, run_result: t.Any, force_metric: DatasetTaskMetricsKind | None = None) -> float:
         """
@@ -122,6 +121,8 @@ class DatasetTask:
             return scoring.token_f1(answer, golds)
         if metric == DatasetTaskMetricsKind.NUMERIC_APPROX:
             return scoring.numeric_approx(answer, golds)
+        if metric == DatasetTaskMetricsKind.NUMERIC_EXACT:
+            return scoring.numeric_exact(answer, golds)
         if metric == DatasetTaskMetricsKind.FINQA:
             return scoring.finqa_match(answer, golds)
         if metric == DatasetTaskMetricsKind.JUDGE:
@@ -243,10 +244,7 @@ class LoadedDataset:
     """List of documents."""
 
     labeled_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
-    """List of labeled retrieval examples."""
-
-    programmatic_retrieval_examples: dict[str, LabeledRetrievalQAExample] = field(default_factory=dict)
-    """List of programmatic retrieval examples. Separated due to potentially large volume."""
+    """List of labeled retrieval examples"""
 
     labeled_messages_examples: dict[str, LabeledMessagesExample] = field(default_factory=dict)
     """List of labeled messages."""
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/__init__.py</span>
    <span class="card-oneliner">DatasetTaskMetricsKind exported.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/dataset/__init__.py`:

```diff
--- a/activation/dataset/__init__.py
+++ b/activation/dataset/__init__.py
@@ -3,6 +3,7 @@ from .dataset import (
     DatasetDocument,
     DatasetDocumentChunk,
     DatasetTask,
+    DatasetTaskMetricsKind,
 )
 
 from .dataset_index import (
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/scoring.py</span>
    <span class="card-oneliner">numeric_exact: the last number of the prediction as an exact fraction.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/dataset/scoring.py`:

```diff
--- a/activation/dataset/scoring.py
+++ b/activation/dataset/scoring.py
@@ -98,9 +98,43 @@ def numeric_approx(pred, golds, rel_tol: float = 0.01) -> float:
             return 1.0
     return 0.0
 
+_EXACT_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?")
+
+
+def _last_exact_number(text):
+    """
+    The last number in the text as an exact Fraction: \boxed{} unwrapped, commas, dollar signs and
+    spaces removed; "a/b" fractions and decimals accepted. None when the text has no number.
+    """
+    from fractions import Fraction
+
+    value = str(text or "")
+    boxed = re.findall(r"\\boxed\{([^{}]*)\}", value)
+    if boxed:
+        value = boxed[-1]
+    value = value.replace("$", "")
+    matches = _EXACT_NUMBER.findall(value)
+    if not matches:
+        return None
+    token = matches[-1].replace(",", "").replace(" ", "")
+    try:
+        return Fraction(token)
+    except (ValueError, ZeroDivisionError):
+        return None
+
+
 def numeric_exact(pred, golds) -> float:
-    """@AI: exact numetic check. Returns 0.0/1.0"""
-    pass
+    """
+    Exact numeric check: the last number of the prediction equals a gold exactly. Returns 0.0/1.0.
+    """
+    predicted = _last_exact_number(pred)
+    if predicted is None:
+        return 0.0
+    for gold in golds:
+        expected = _last_exact_number(gold)
+        if expected is not None and expected == predicted:
+            return 1.0
+    return 0.0
 
 def _finqa_boolean(text) -> bool | None:
     """
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/loaders/dapo_math.py</span>
    <span class="card-oneliner">DAPO-Math-17k as scorable tasks, deduped, with the Answer: instruction pattern-replaced by submit_answer.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/dataset/loaders/dapo_math.py`:

```diff
--- a/activation/dataset/loaders/dapo_math.py
+++ b/activation/dataset/loaders/dapo_math.py
@@ -1,4 +1,70 @@
 """
-@AI: Load DAPO Math 17k.
-Unlike our retrieval tasks, this populates the DatasetTask set.
-"""
\ No newline at end of file
+DAPO-Math-17k loader (`BytedTsinghua-SIA/DAPO-Math-17k`). Unlike the retrieval loaders this populates
+`scorable_tasks`: one DatasetTask per unique problem, scored by exact numeric match against
+`reward_model.ground_truth`. The Hugging Face file holds each of the 17,917 problems 100 times, so
+the loader streams and dedupes on `extra_info.index` until `max_examples` unique problems.
+"""
+from __future__ import annotations
+
+import re
+import time
+import typing as t
+
+from datasets import load_dataset
+
+from ..dataset import DatasetTask, DatasetTaskMetricsKind, LoadedDataset
+from ..dataset_utils import initialize_dataset_stats
+
+if t.TYPE_CHECKING:
+    from activation.harness import HarnessRuntime
+
+HF_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
+DATASET_ID = "dapo_math"
+
+# The dataset's own answer-format instructions, replaced so the agent uses submit_answer instead.
+_HEAD_INSTRUCTION = re.compile(
+    r"Solve the following math problem step by step\.\s*The last line of your response should be of the form"
+    r"\s*Answer:\s*\$Answer\s*\(without quotes\)\s*where\s*\$Answer is the answer to the problem\.\s*",
+    re.S,
+)
+_TAIL_INSTRUCTION = re.compile(r"\s*Remember to put your answer on its own line after\s*\"Answer:\"\.?\s*$", re.S)
+AGENT_HEAD = ("Solve the following math problem. Use the python tool for any computation you are not certain "
+              "about, then call submit_answer with the final answer as a single number.\n\n")
+AGENT_TAIL = "\n\nWhen you are done, call submit_answer with only the final number."
+
+
+def agent_prompt_from_dapo(prompt_text: str) -> str:
+    """The problem with the dataset's "Answer:" instructions pattern-replaced by submit_answer instructions."""
+    problem = _TAIL_INSTRUCTION.sub("", _HEAD_INSTRUCTION.sub("", prompt_text)).strip()
+    return AGENT_HEAD + problem + AGENT_TAIL
+
+
+class DapoMathDataset:
+    """
+    Loader for DAPO-Math-17k problems as scorable agent tasks.
+    """
+
+    @classmethod
+    def load(cls, harness: "HarnessRuntime", max_examples: int | None) -> LoadedDataset:
+        start = time.time()
+        rows = load_dataset(HF_DATASET, split="train", streaming=True)
+        dataset_id = f"{DATASET_ID}_{max_examples if max_examples is not None else 'all'}"
+        tasks: dict[str, DatasetTask] = {}
+        for row in rows:
+            index = row["extra_info"]["index"]
+            if index in tasks:
+                continue
+            prompt_text = row["prompt"][0]["content"]
+            tasks[index] = DatasetTask(
+                task_id=index,
+                dataset_id=dataset_id,
+                task_datum=row,
+                reference_metrics_kind=DatasetTaskMetricsKind.NUMERIC_EXACT,
+                gold_answer=str(row["reward_model"]["ground_truth"]),
+                agent_prompt=agent_prompt_from_dapo(prompt_text),
+            )
+            if max_examples is not None and len(tasks) >= max_examples:
+                break
+        loaded = LoadedDataset(dataset_id=dataset_id, scorable_tasks=tasks)
+        loaded.stats = initialize_dataset_stats(loaded, load_time=time.time() - start)
+        return loaded
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/dataset/loaders/__init__.py</span>
    <span class="card-oneliner">DapoMathDataset exported.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/dataset/loaders/__init__.py`:

```diff
--- a/activation/dataset/loaders/__init__.py
+++ b/activation/dataset/loaders/__init__.py
@@ -3,3 +3,4 @@ from .msmarco import MsMarcoDataset
 from .nq import NqDataset
 from .scifact import SciFactDataset
 from .sciq import SciQDataset
+from .dapo_math import DapoMathDataset
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/vllm_wrapper.py</span>
    <span class="card-oneliner">One AsyncLLM per GPU; submit for agents; chat rebuilt on top; Qwen3.5-9B entries.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs the committed file (after the revert) for `activation/harness/vllm_wrapper.py`:

```diff
--- a/activation/harness/vllm_wrapper.py
+++ b/activation/harness/vllm_wrapper.py
@@ -1,15 +1,26 @@
 """
-Independent vLLM engine replicas: one `vllm.LLM` per visible GPU, a batch split by world size,
-results reassembled in input order. No scheduler on our side; a shard that finishes early idles
-until the others do. Each replica is data-parallel size one, so vLLM never enters its lockstep
-MoE path and single-GPU behavior is exactly today's.
+Independent vLLM engine replicas: one `AsyncLLM` per visible GPU, each driven by its own event loop
+on a background thread. Requests are submitted one at a time from any thread (`submit`) and the
+engine batches whatever is in flight (continuous batching), which is what agent rollouts need: agents
+finish their turns at different times and cannot be batched synchronously. `chat` keeps the
+`vllm.LLM.chat` signature on top of that: every conversation is templated with the tokenizer,
+submitted with strided replica assignment, and the results come back in input order. No scheduler
+on our side beyond the replica choice; each replica is data-parallel size one, so vLLM never enters
+its lockstep MoE path and single-GPU behavior is exactly one engine.
 """
 
+import asyncio
+import concurrent.futures
 import os
 import threading
+import uuid
+import zlib
 
 import torch
 import vllm
+from vllm.engine.arg_utils import AsyncEngineArgs
+from vllm.sampling_params import RequestOutputKind
+from vllm.v1.engine.async_llm import AsyncLLM
 
 WORLD_SIZE = max(1, torch.cuda.device_count())
 ENGINE_MAX_NUM_SEQS = 128
@@ -17,28 +28,77 @@ ENGINE_CONCURRENCY = WORLD_SIZE * ENGINE_MAX_NUM_SEQS
 RECOMMENDED_BATCH_SIZE = 4 * ENGINE_CONCURRENCY # Decode-heavy passes (study): backlog amortizes the tail.
 
 
+class _Replica:
+    """One AsyncLLM and the event loop thread that drives it."""
+
+    def __init__(self, engine_kwargs: dict):
+        self.loop = asyncio.new_event_loop()
+        self.thread = threading.Thread(target=self.loop.run_forever, name="vllm-replica-loop", daemon=True)
+        self.thread.start()
+
+        async def construct():
+            # Constructed inside the loop so anything AsyncLLM binds to "the running loop" binds to this one.
+            return AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_kwargs))
+
+        self.engine: AsyncLLM = asyncio.run_coroutine_threadsafe(construct(), self.loop).result()
+
+    async def _collect(self, prompt_token_ids: list[int], sampling_params, request_id: str, lora_request) -> vllm.RequestOutput:
+        sampling_params = sampling_params.clone()
+        sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
+        final = None
+        async for output in self.engine.generate(
+            vllm.TokensPrompt(prompt_token_ids=prompt_token_ids), sampling_params, request_id, lora_request=lora_request,
+        ):
+            final = output
+        assert final is not None, f"request {request_id} produced no output"
+        return final
+
+    def submit(self, prompt_token_ids, sampling_params, request_id, lora_request=None) -> concurrent.futures.Future:
+        return asyncio.run_coroutine_threadsafe(
+            self._collect(prompt_token_ids, sampling_params, request_id, lora_request), self.loop,
+        )
+
+    def shutdown(self):
+        try:
+            self.engine.shutdown()
+        except Exception:
+            pass
+        self.loop.call_soon_threadsafe(self.loop.stop)
+        self.thread.join(timeout=10)
+
+
 class VLLMWrapper:
-    def __init__(self, **engine_kwargs):
+    def __init__(self, tokenizer=None, **engine_kwargs):
+        """
+        `tokenizer` templates conversations for `chat`; without one the engine's own tokenizer is used.
+        Every other keyword goes to vLLM's engine arguments.
+        """
         engine_kwargs = dict(engine_kwargs)
         if engine_kwargs.setdefault("max_num_seqs", ENGINE_MAX_NUM_SEQS) != ENGINE_MAX_NUM_SEQS:
             print(
                 f"Engine max_num_seqs={engine_kwargs['max_num_seqs']} overrides {ENGINE_MAX_NUM_SEQS}; "
                 "batch constants assume the latter."
             )
+        engine_kwargs.setdefault("disable_log_stats", True)
         visible = os.environ.get("CUDA_VISIBLE_DEVICES")
         devices = visible.split(",") if visible else [str(i) for i in range(WORLD_SIZE)]
-        self.replicas: list[vllm.LLM] = []
+        self.replicas: list[_Replica] = []
         try:
             # Sequential: CUDA_VISIBLE_DEVICES is process-global and inherited by each spawned
             # engine core. The first replica pays the kernel JIT; later ones load warm.
             for device in devices:
                 os.environ["CUDA_VISIBLE_DEVICES"] = device
-                self.replicas.append(vllm.LLM(**engine_kwargs))
+                self.replicas.append(_Replica(engine_kwargs))
         finally:
             if visible is None:
                 os.environ.pop("CUDA_VISIBLE_DEVICES", None)
             else:
                 os.environ["CUDA_VISIBLE_DEVICES"] = visible
+        self.tokenizer = tokenizer if tokenizer is not None else self.replicas[0].engine.get_tokenizer()
+
+    @property
+    def world_size(self) -> int:
+        return len(self.replicas)
 
     @staticmethod
     def recommended_engine_kwargs(model_id: str, is_for_training: bool = False) -> dict:
@@ -59,6 +119,9 @@ class VLLMWrapper:
             # A3B MoE labeler: MTP taxes prefill, and labeling is prefill-bound.
             "nvidia/Qwen3.6-35B-A3B-NVFP4": base,
             "Qwen/Qwen3.6-35B-A3B-FP8": base,
+            # Agent rollouts: room for a 32k compaction threshold plus the answer.
+            "Qwen/Qwen3.5-9B": base | {"max_model_len": 40_960},
+            "Qwen/Qwen3.5-4B": base | {"max_model_len": 40_960},
         }
         if model_id not in known:
             print(f"{model_id} - No recommended engine kwargs; using the base formula without speculative decoding.")
@@ -87,43 +150,62 @@ class VLLMWrapper:
             "Qwen/Qwen3.8-27B-FP8": qwen_non_thinking,
             "nvidia/Qwen3.6-35B-A3B-NVFP4": qwen_non_thinking,
             "Qwen/Qwen3.6-35B-A3B-FP8": qwen_non_thinking,
+            # Qwen3.5 thinks by default; agents and study prompts want the direct answer.
+            "Qwen/Qwen3.5-9B": qwen_non_thinking | {"chat_template_kwargs": {"enable_thinking": False}},
+            "Qwen/Qwen3.5-4B": qwen_non_thinking | {"chat_template_kwargs": {"enable_thinking": False}},
         }
         if model_id not in known:
             print(f"{model_id} - No recommended chat kwargs; using engine defaults.")
         return {key: dict(value) for key, value in known.get(model_id, {}).items()}
 
-        
+    # ----------------------------------------------------------------------------- requests
+    @staticmethod
+    def replica_for(agent_id: str, world_size: int) -> int:
+        """Stable agent -> replica assignment, so an agent's prefix cache serves its next turn."""
+        return zlib.crc32(agent_id.encode()) % world_size
+
+    def template(self, conversation: list[dict], tools: list[dict] | None = None,
+                 add_generation_prompt: bool = True, chat_template_kwargs: dict | None = None) -> list[int]:
+        """Token ids of one conversation through the tokenizer's chat template (text-only content)."""
+        flattened = []
+        for message in conversation:
+            message = dict(message)
+            content = message.get("content")
+            if isinstance(content, list):
+                message["content"] = "".join(part.get("text", "") for part in content if isinstance(part, dict))
+            flattened.append(message)
+        ids = self.tokenizer.apply_chat_template(
+            flattened, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=True,
+            **(chat_template_kwargs or {}),
+        )
+        if hasattr(ids, "keys"):                                                      # BatchEncoding / dict
+            ids = ids["input_ids"]
+        return list(ids)
+
+    def submit(self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams, request_id: str | None = None,
+               replica: int = 0, lora_request=None) -> concurrent.futures.Future:
+        """One request on one replica; the future resolves to the final vllm.RequestOutput."""
+        request_id = request_id or uuid.uuid4().hex
+        return self.replicas[replica % len(self.replicas)].submit(prompt_token_ids, sampling_params, request_id, lora_request)
 
     def chat(self, conversations: list, **chat_kwargs) -> list[vllm.RequestOutput]:
-        """Same signature as vllm.LLM.chat; strided shards, one thread per replica, in-order results."""
-        num_replicas = len(self.replicas)
-        if num_replicas == 1:
-            return self.replicas[0].chat(conversations, **chat_kwargs)
-        shards = [conversations[rank::num_replicas] for rank in range(num_replicas)]
-        results: list = [None] * num_replicas
-        errors: list = [None] * num_replicas
-
-        def work(rank: int):
-            try:
-                results[rank] = self.replicas[rank].chat(shards[rank], **chat_kwargs) if shards[rank] else []
-            except BaseException as error:
-                errors[rank] = error
-
-        threads = [threading.Thread(target=work, args=(rank,), daemon=True) for rank in range(num_replicas)]
-        for thread in threads:
-            thread.start()
-        for thread in threads:
-            thread.join()
-        for error in errors:
-            if error is not None:
-                raise error
-        return [results[i % num_replicas][i // num_replicas] for i in range(len(conversations))]
+        """Same signature as vllm.LLM.chat: template every conversation, submit strided over the replicas, return in order."""
+        sampling_params = chat_kwargs.pop("sampling_params", None) or vllm.SamplingParams()
+        tools = chat_kwargs.pop("tools", None)
+        chat_template_kwargs = chat_kwargs.pop("chat_template_kwargs", None)
+        add_generation_prompt = chat_kwargs.pop("add_generation_prompt", True)
+        lora_request = chat_kwargs.pop("lora_request", None)
+        chat_kwargs.pop("use_tqdm", None)
+        if chat_kwargs:
+            print(f"VLLMWrapper.chat ignores {sorted(chat_kwargs)}")
+        futures = []
+        for index, conversation in enumerate(conversations):
+            ids = self.template(conversation, tools, add_generation_prompt, chat_template_kwargs)
+            futures.append(self.submit(ids, sampling_params, replica=index % len(self.replicas), lora_request=lora_request))
+        return [future.result() for future in futures]
 
     def shutdown(self):
         """Best-effort shutdown of every replica's engine core."""
         for replica in self.replicas:
-            try:
-                replica.llm_engine.engine_core.shutdown()
-            except Exception:
-                pass
+            replica.shutdown()
         self.replicas = []
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/loaded_model.py</span>
    <span class="card-oneliner">engine_submit, EngineChatOutput with cached tokens and finish reason, the kwargs merge shared.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/harness/loaded_model.py`:

```diff
--- a/activation/harness/loaded_model.py
+++ b/activation/harness/loaded_model.py
@@ -36,6 +36,19 @@ class EngineChatOutput:
     text: str
     prompt_token_count: int
     output_token_count: int
+    cached_prompt_token_count: int = 0   # prefix-cache hits reported by the engine
+    finish_reason: str = ""              # "stop" or "length"
+
+    @staticmethod
+    def from_request_output(output: "vllm.RequestOutput") -> "EngineChatOutput":
+        completion = output.outputs[0]
+        return EngineChatOutput(
+            text=completion.text,
+            prompt_token_count=len(output.prompt_token_ids or ()),
+            output_token_count=len(completion.token_ids),
+            cached_prompt_token_count=output.num_cached_tokens or 0,
+            finish_reason=completion.finish_reason or "",
+        )
 
 class LoadedModel:
     """
@@ -207,7 +220,7 @@ class LoadedModel:
             engine_kwargs.update(self.model_config.engine_kwargs)
             engine_kwargs.setdefault("trust_remote_code", self.model_config.trust_remote_code)
             self.model_config.engine_kwargs = engine_kwargs
-            self.vllm_model = VLLMWrapper(**engine_kwargs)
+            self.vllm_model = VLLMWrapper(tokenizer=self.tokenizer, **engine_kwargs)
             self.current_engine_device = device
             elapsed = time.time() - device_change_start_time
             print(f"{self.model_config.model_name} - Engine moved to target in {elapsed:.2f}s")
@@ -254,17 +267,8 @@ class LoadedModel:
         )[0].text
 
 
-    def engine_chat_many(
-        self,
-        conversations: list[list[dict]],
-        lora_name: str|None = None,
-        chat_kwargs: dict|None = None,
-    ) -> list[EngineChatOutput]:
-        """
-        Batched engine chat. Returns texts with token counts for stats.
-        """
-        self.engine_to_device(TARGET_DEVICE)
-        # Harness defaults < recommended chat kwargs for this model < explicit chat kwargs.
+    def _merged_chat_kwargs(self, chat_kwargs: dict|None, seed: int|None = None) -> tuple["vllm.SamplingParams", dict]:
+        """Harness defaults < recommended chat kwargs for this model < explicit chat kwargs; the seed on top."""
         chat_kwargs = dict(chat_kwargs or {})
         recommended_chat_kwargs = VLLMWrapper.recommended_chat_kwargs(self.model_config.model_id)
         default_sampling_params_kwargs = dict(
@@ -273,26 +277,55 @@ class LoadedModel:
         )
         recommended_sampling_params_kwargs = recommended_chat_kwargs.pop("sampling_params", None) or dict()
         chat_sampling_params_kwargs = chat_kwargs.pop("sampling_params", None) or dict()
-        sampling_params = vllm.SamplingParams(
-            **(default_sampling_params_kwargs | recommended_sampling_params_kwargs | chat_sampling_params_kwargs)
-        )
+        sampling_kwargs = default_sampling_params_kwargs | recommended_sampling_params_kwargs | chat_sampling_params_kwargs
+        if seed is not None:
+            sampling_kwargs["seed"] = seed
+        sampling_params = vllm.SamplingParams(**sampling_kwargs)
         default_extra_kwargs = dict(
             chat_template_kwargs={"enable_thinking": False},
         )
+        return sampling_params, default_extra_kwargs | recommended_chat_kwargs | chat_kwargs
+
+    def engine_chat_many(
+        self,
+        conversations: list[list[dict]],
+        lora_name: str|None = None,
+        chat_kwargs: dict|None = None,
+    ) -> list[EngineChatOutput]:
+        """
+        Batched engine chat. Returns texts with token counts for stats.
+        """
+        self.engine_to_device(TARGET_DEVICE)
+        sampling_params, extra_kwargs = self._merged_chat_kwargs(chat_kwargs)
         request_outputs = self.vllm_model.chat(
             conversations,
             sampling_params=sampling_params,
             use_tqdm=False,
-            **(default_extra_kwargs | recommended_chat_kwargs | chat_kwargs)
+            **extra_kwargs,
         )
-        return [
-            EngineChatOutput(
-                text=output.outputs[0].text,
-                prompt_token_count=len(output.prompt_token_ids or ()),
-                output_token_count=len(output.outputs[0].token_ids),
-            )
-            for output in request_outputs
-        ]
+        return [EngineChatOutput.from_request_output(output) for output in request_outputs]
+
+    def engine_submit(
+        self,
+        messages: list[dict],
+        tools: list[dict]|None = None,
+        seed: int|None = None,
+        agent_id: str = "",
+        lora_name: str|None = None,
+        chat_kwargs: dict|None = None,
+    ) -> EngineChatOutput:
+        """
+        One conversation for one agent. Same kwargs merge as engine_chat_many, a per-request seed,
+        and the agent pinned to one replica (by agent_id) so its prefix cache serves the next turn.
+        Blocks the calling thread while the engine batches this request with everything in flight.
+        """
+        self.engine_to_device(TARGET_DEVICE)
+        assert lora_name is None, "LoRA on the serving engine is not wired yet (no adapter to test)."
+        sampling_params, extra_kwargs = self._merged_chat_kwargs(chat_kwargs, seed)
+        engine = self.vllm_model
+        token_ids = engine.template(messages, tools, True, extra_kwargs.get("chat_template_kwargs"))
+        future = engine.submit(token_ids, sampling_params, replica=VLLMWrapper.replica_for(agent_id, engine.world_size))
+        return EngineChatOutput.from_request_output(future.result())
 
 
     def simple_chat(self, user_msg: str) -> str:
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/runtime_config.py</span>
    <span class="card-oneliner">agent_max_concurrent, agent_env_default_image.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs the committed file (after the revert) for `activation/harness/runtime_config.py`:

```diff
--- a/activation/harness/runtime_config.py
+++ b/activation/harness/runtime_config.py
@@ -36,6 +36,11 @@ class HarnessRuntimeConfig:
     dataset_study_label_batch_size: int|None = None # None = recommended value.
     dataset_study_label_model_name: str|None = None
 
+    # Agents
+    agent_max_concurrent: int = 64 # Thread pool bound for rollouts; the engine batches across them.
+    agent_env_default_image: str = "docker.io/library/python:3.12-slim" # Sandbox image when a config names no dockerfile.
+    agent_env_memory_limit_mb: int | None = 8192 # Address space per sandboxed process (ulimit -v); None = unlimited.
+
 @dataclass
 class HarnessStats:
     # (model_name, seconds) records for actual state changes (no-ops unrecorded).
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/harness/runtime.py</span>
    <span class="card-oneliner">harness.rollout_manager.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/harness/runtime.py`:

```diff
--- a/activation/harness/runtime.py
+++ b/activation/harness/runtime.py
@@ -30,12 +30,14 @@ class HarnessRuntime:
     """
     def __init__(self, harness_config: HarnessRuntimeConfig):
         from ..dataset import DatasetManager
+        from ..agent import RolloutManager
         self.harness_config = harness_config
         self.harness_stats = HarnessStats()
         self.loaded_models: dict[str, LoadedModel] = dict() # Maps from name.
         self._load_models()
         self.dataset_manager = DatasetManager(self)
         self.module_manager = ModuleManager(self)
+        self.rollout_manager = RolloutManager(self)
 
 
     def _load_models(self):
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/__init__.py</span>
    <span class="card-oneliner">Package exports.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/__init__.py`:

```diff
--- a/activation/agent/__init__.py
+++ b/activation/agent/__init__.py
@@ -0,0 +1,16 @@
+from .agent_config import AgentConfig, AgentRunResult, TrajectoryStep
+from .agent_env import AgentEnv
+from .agent_tools import (
+    AgentTool,
+    ToolCallResult,
+    ShellTool,
+    PythonTool,
+    SubmitAnswerTool,
+    ParallelCallTool,
+    SemanticSearchTool,
+    SubagentTool,
+)
+from .agent import Agent
+from .rollout_caching import RolloutCache
+from .rollout_manager import RolloutManager
+from .rollout_reporter import RolloutReporter
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_config.py</span>
    <span class="card-oneliner">AgentConfig, TrajectoryStep, AgentRunResult with serialization.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent_config.py`:

```diff
--- a/activation/agent/agent_config.py
+++ b/activation/agent/agent_config.py
@@ -1,70 +1,149 @@
+"""
+Agent configuration, trajectory steps and run results: plain dataclasses that serialize to JSON so a
+rollout can be cached and read back without the engine.
+"""
+from __future__ import annotations
+
+import importlib
+import json
 import typing as t
+from dataclasses import dataclass, field, replace
 
 if t.TYPE_CHECKING:
     from .agent_tools import AgentTool
-    from .agent import Agent
     from activation.harness import HarnessRuntime
-    from dataset import DatasetTask
-
-
+    from activation.dataset import DatasetTask
 
 
+@dataclass
 class AgentConfig:
     # Inputs
-    system_prompt: str = "" # System prompt.
-    user_prompt: str = "" # User prompt.
-    ac_inputs: dict[str, object] = dict # activation context inputs.
-    dataset_task: "DatasetTask"|None = None # Set if this corresponds to a specific dataset task.
+    system_prompt: str = ""                                   # System prompt.
+    user_prompt: str = ""                                     # User prompt (the task).
+    ac_inputs: dict[str, object] = field(default_factory=dict)  # Activation-context inputs; carried as data in slice 1.
+    dataset_task: "DatasetTask | None" = None                 # Set if this corresponds to a specific dataset task.
 
     # Model calls.
-    model_name: str | None = None
-    lora_name: str | None = None
-    call_kwargs: dict | None = None
+    model_name: str | None = None                             # A harness model name; its engine serves the rollout.
+    lora_name: str | None = None                              # Adapter on the serving engine (plumbed, untested).
+    call_kwargs: dict | None = None                           # Engine chat kwargs (sampling_params inside), merged like engine_chat_many.
 
-    # Environment
-    # Note that all agents get the shell/python/parallel_tool_call/submit_answer tool,
-    env_dockerfile_path: str|None = None
-    env_args: dict[str, str]|None = None # args to pass into docker/podman run.
-    tools: dict[str, type[AgentTool]] = dict()
-    enable_ac_communication: bool = False
+    # Environment. Every agent gets the shell / python / parallel_tool_call / submit_answer tools.
+    env_dockerfile_path: str | None = None                    # None: the harness default image.
+    env_args: dict[str, str] | None = None                    # Extra `podman run` flags, e.g. {"--env": "PYTHONHASHSEED=0"}.
+    env_memory_limit_mb: int | None = None                    # Sandbox memory; None = the harness default (agent_env_memory_limit_mb). Hard container limit where cgroups exist, ulimit -v at 85% per process always.
+    tools: dict[str, tuple[type["AgentTool"], dict]] = field(default_factory=dict)  # name -> (class, constructor kwargs), added to the defaults.
+    enable_ac_communication: bool = False                     # Subagents exchange trajectories as activation context.
 
     # Budgets
     max_turns: int = 20
     max_tool_errors: int = 5
-    max_duration: float = 600
-    compaction_threshold_tokens: int = 32768
+    max_duration: float = 600                                 # seconds
+    compaction_threshold_tokens: int = 32768                  # carried; compaction is slice 2
+
+    def serialize(self) -> dict:
+        data = {key: value for key, value in self.__dict__.items() if key not in ("dataset_task", "tools")}
+        data["tools"] = {
+            name: {"class": f"{cls.__module__}:{cls.__qualname__}", "kwargs": _jsonable(kwargs)}
+            for name, (cls, kwargs) in self.tools.items()
+        }
+        data["ac_inputs"] = _jsonable(self.ac_inputs)
+        task = self.dataset_task
+        data["dataset_task"] = None if task is None else {
+            "task_id": task.task_id, "dataset_id": task.dataset_id, "reference_metrics_kind": str(task.reference_metrics_kind),
+            "gold_answer": task.gold_answer, "gold_answer_aliases": list(task.gold_answer_aliases), "agent_prompt": task.agent_prompt,
+        }
+        return data
+
+    @staticmethod
+    def deserialize(data: dict, harness: "HarnessRuntime | None" = None) -> "AgentConfig":
+        """The dataset task is looked up in the harness when it holds the dataset, else rebuilt from the stored fields."""
+        from activation.dataset import DatasetTask, DatasetTaskMetricsKind
+        data = dict(data)
+        tools = {}
+        for name, spec in (data.pop("tools", None) or {}).items():
+            module_name, _, qualname = spec["class"].partition(":")
+            cls = importlib.import_module(module_name)
+            for part in qualname.split("."):
+                cls = getattr(cls, part)
+            tools[name] = (cls, dict(spec.get("kwargs") or {}))
+        task_data = data.pop("dataset_task", None)
+        task = None
+        if task_data is not None:
+            loaded = harness.dataset_manager.loaded_datasets.get(task_data["dataset_id"]) if harness is not None else None
+            task = loaded.scorable_tasks.get(task_data["task_id"]) if loaded is not None else None
+            if task is None:
+                task = DatasetTask(
+                    task_id=task_data["task_id"], dataset_id=task_data["dataset_id"], task_datum={},
+                    reference_metrics_kind=DatasetTaskMetricsKind(task_data["reference_metrics_kind"]),
+                    gold_answer=task_data["gold_answer"], gold_answer_aliases=list(task_data["gold_answer_aliases"]),
+                    agent_prompt=task_data.get("agent_prompt", ""),
+                )
+        return AgentConfig(tools=tools, dataset_task=task, **data)
 
 
+@dataclass
 class TrajectoryStep:
     """
     Contains simple formats the model expects (e.g., raw dicts rather than our objects).
     """
-    role: str
-    content: dict|str
-    tool_calls: list[dict]
-    tool_call_results: list[str]
-    activations: dict[str, object]
+    role: str                                                  # "assistant" or "tool"
+    content: str                                               # assistant text (tool-call blocks removed) or the joined tool outputs
+    tool_calls: list[dict] = field(default_factory=list)       # [{"id", "name", "arguments"}]
+    tool_call_results: list[str] = field(default_factory=list) # truncated outputs, same order as tool_calls
+    activations: dict[str, object] = field(default_factory=dict)  # ac_outputs of this step's tools, keyed by call id
+
 
+@dataclass
 class AgentRunResult:
-    agent_config: "AgentConfig"
-    answer: t.Any = None
+    agent_config: AgentConfig
+    answer: t.Any = None                                       # the submit_answer argument; None when never submitted
     num_turns: int = 0
     num_input_tokens: int = 0
     num_cached_input_tokens: int = 0
     num_output_tokens: int = 0
     num_ac_input_bytes: int = 0
     num_ac_input_tokens: int = 0
-    duration: float = 0
-    trajectory: list[dict] = [] # Simple trajectory, with a new activation type.
+    duration: float = 0.0
+    trajectory: list[dict] = field(default_factory=list)       # TrajectoryStep dicts, in order (a new activation type per step)
     score: float = 0.0
-    score_feedback: str|None = None
-    subagent_results: list[AgentRunResult] = []
+    score_feedback: str | None = None
+    subagent_results: list["AgentRunResult"] = field(default_factory=list)
+    finish_reason: str = ""                                    # submitted | max_turns | max_tool_errors | max_duration | no_tool_call | error
+    seed: int = 0
 
+    @property
+    def num_tool_calls(self) -> int:
+        # Assistant steps only: the tool step that follows repeats the same calls next to their results.
+        return sum(len(step.get("tool_calls", [])) for step in self.trajectory if step.get("role") == "assistant")
 
     def serialize(self) -> dict:
-        # Should handle the nested config and dataset task btw.
-        pass
+        data = {key: value for key, value in self.__dict__.items() if key not in ("agent_config", "subagent_results", "trajectory", "answer")}
+        data["agent_config"] = self.agent_config.serialize()
+        data["answer"] = _jsonable(self.answer)
+        data["trajectory"] = _jsonable(self.trajectory)
+        data["subagent_results"] = [result.serialize() for result in self.subagent_results]
+        return data
 
     @staticmethod
-    def deserialize(self, d: dict) -> AgentRunResult:
-        pass
\ No newline at end of file
+    def deserialize(data: dict, harness: "HarnessRuntime | None" = None, agent_config: AgentConfig | None = None) -> "AgentRunResult":
+        """With agent_config given (the caller's live config), the stored config is not rebuilt."""
+        fields = {field_.name for field_ in AgentRunResult.__dataclass_fields__.values()}
+        data = {key: value for key, value in data.items() if key in fields}          # cache rows carry extra keys
+        config_data = data.pop("agent_config")
+        config = agent_config if agent_config is not None else AgentConfig.deserialize(config_data, harness)
+        children = [AgentRunResult.deserialize(child, harness) for child in data.pop("subagent_results", [])]
+        return AgentRunResult(agent_config=config, subagent_results=children, **data)
+
+
+def _jsonable(value):
+    """Values that json.dumps accepts; anything else becomes its repr."""
+    try:
+        json.dumps(value)
+        return value
+    except (TypeError, ValueError):
+        if isinstance(value, dict):
+            return {str(key): _jsonable(item) for key, item in value.items()}
+        if isinstance(value, (list, tuple)):
+            return [_jsonable(item) for item in value]
+        return repr(value)
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_env.py</span>
    <span class="card-oneliner">Podman environment: python, shell, files, timeouts with partial output, idempotent shutdown.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent_env.py`:

```diff
--- a/activation/agent/agent_env.py
+++ b/activation/agent/agent_env.py
@@ -1,35 +1,129 @@
+"""
+Simple, podman-based agent environment: one `--rm` container per env, every operation a `podman exec`
+with the code or script on stdin. Rootless podman on a workstation (see README), rootful podman inside a
+privileged Sky node container. Truncation of outputs is the tool layer's job, not the env's.
+"""
+from __future__ import annotations
+
+import hashlib
+import shutil
+import subprocess
+import uuid
+from pathlib import Path
+
+DEFAULT_IMAGE = "docker.io/library/python:3.12-slim"
+DEFAULT_MEMORY_LIMIT_MB = 8192   # sandbox memory: `podman run --memory` where cgroups exist (hard kill), ulimit -v at 85% per process (MemoryError first)
+SOFT_LIMIT_FRACTION = 0.85
+EXEC_GRACE_SECONDS = 10   # how long the client waits past the in-container `timeout` before killing itself
+
+
+def podman_available() -> bool:
+    return shutil.which("podman") is not None
+
+
 class AgentEnv:
     """
     Simple, podman-based environment.
     """
-    def __init__(self, dockerfile_path: str|None=None):
-        # Always start in --rm mode for auto-clears.
-        pass
+    hard_limits_available = True   # class-wide: cleared the first time `podman run --memory` is refused for lack of cgroups
+
+    def __init__(self, dockerfile_path: str | None = None, env_args: dict[str, str] | None = None,
+                 default_image: str = DEFAULT_IMAGE, memory_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB):
+        if not podman_available():
+            raise RuntimeError("podman is not installed; see README.md (rootless podman) or the Sky image.")
+        self.env_id = uuid.uuid4().hex[:12]
+        self.memory_limit_mb = memory_limit_mb   # None disables both limits
+        self.image = self._build_image(dockerfile_path) if dockerfile_path else default_image
+        run_args = []
+        for key, value in (env_args or {}).items():
+            run_args.append(key)
+            if value not in (None, ""):
+                run_args.append(str(value))
+        # Always --rm so a lost handle still leaves nothing behind; no network unless env_args ask for it.
+        argv = ["podman", "run", "-d", "--rm", "--name", f"agent-env-{self.env_id}"]
+        if not any(arg.startswith("--network") or arg == "--net" for arg in run_args):
+            argv += ["--network", "none"]
+        # The hard limit is a container cgroup; inside Sky nodes podman has no cgroups, so the first failure
+        # mentioning them turns the flag off for every later env in this process and the soft limit carries.
+        hard_limit = []
+        if self.memory_limit_mb and AgentEnv.hard_limits_available and not any(arg.startswith("--memory") or arg == "-m" for arg in run_args):
+            hard_limit = ["--memory", f"{int(self.memory_limit_mb)}m"]
+        tail = run_args + [self.image, "sleep", "infinity"]
+        completed = subprocess.run(argv + hard_limit + tail, capture_output=True, text=True)
+        if completed.returncode != 0 and hard_limit and "cgroup" in completed.stderr.lower():
+            AgentEnv.hard_limits_available = False
+            completed = subprocess.run(argv + tail, capture_output=True, text=True)
+        if completed.returncode != 0:
+            raise RuntimeError(f"podman run failed: {completed.stderr.strip()}")
+        self.container_id: str | None = completed.stdout.strip()
+
+    @staticmethod
+    def _build_image(dockerfile_path: str) -> str:
+        """Built once per Dockerfile content; the tag is the content hash."""
+        path = Path(dockerfile_path).resolve()
+        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
+        tag = f"localhost/activation-env:{digest}"
+        exists = subprocess.run(["podman", "image", "exists", tag], capture_output=True)
+        if exists.returncode != 0:
+            print(f"AgentEnv - Building {tag} from {path}.", flush=True)
+            build = subprocess.run(["podman", "build", "-t", tag, "-f", str(path), str(path.parent)], capture_output=True, text=True)
+            if build.returncode != 0:
+                raise RuntimeError(f"podman build failed: {build.stderr.strip()[-2000:]}")
+        return tag
 
-    def run_python_code(self, code: str, timeout: int|None) -> tuple[str, bool]:
-        # Runs python code. No need for session persistence.
-        # Returns combined stdout/stderr, and an error indicating whether an error occured.
-        # In case of timeout, the partial stdout/stderr should still be returned.
-        # Docker exec python3 should be enough.
-        pass
+    # ----------------------------------------------------------------------------- operations
+    def _exec(self, argv: list[str], stdin: str | None, timeout: int | None) -> tuple[str, bool]:
+        """
+        Run inside the container. Combined stdout/stderr and ok. On timeout the in-container `timeout`
+        kills the command first (so partial output is still read), then the client gives up.
+        """
+        assert self.container_id, "environment was shut down"
+        command = ["podman", "exec", "-i", self.container_id]
+        inner: list[str] = []
+        if timeout is not None:
+            inner += ["timeout", "-s", "KILL", str(int(timeout))]
+        inner += argv
+        if self.memory_limit_mb:
+            # Soft limit at 85% of the sandbox memory: the process gets MemoryError (which the model reads)
+            # before the container's hard limit, where there is one, kills it without a word.
+            soft_kb = int(self.memory_limit_mb * SOFT_LIMIT_FRACTION) * 1024
+            command += ["sh", "-c", f'ulimit -v {soft_kb} && exec "$@"', "sh"]
+        command += inner
+        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
+        try:
+            output, _ = process.communicate(stdin, timeout=None if timeout is None else timeout + EXEC_GRACE_SECONDS)
+        except subprocess.TimeoutExpired:
+            process.kill()
+            output, _ = process.communicate()
+            return (output or "") + f"\n[timed out after {timeout} s]", False
+        if process.returncode == 137 and timeout is not None:
+            return (output or "") + f"\n[timed out after {timeout} s]", False
+        return output or "", process.returncode == 0
 
-    def run_shell(self, script: str) -> tuple[str, bool]:
-        # Direct docker exec.
-        # Returns combined stdout/stderr, and an error indicating whether an error occured.
-        # In case of timeout, the partial stdout/stderr should still be returned.
-        # Returns [output, ok].
-        pass
+    def run_python_code(self, code: str, timeout: int | None = None) -> tuple[str, bool]:
+        """Runs python code (no session persistence). Returns combined stdout/stderr and ok; partial output on timeout."""
+        return self._exec(["python3", "-"], code, timeout)
 
+    def run_shell(self, script: str, timeout: int | None = None) -> tuple[str, bool]:
+        """Runs a bash script. Returns combined stdout/stderr and ok; partial output on timeout."""
+        return self._exec(["bash", "-s"], script, timeout)
 
     def write_file(self, path: str, content: str) -> bool:
-        pass
+        _, ok = self._exec(["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", path], content, 60)
+        return ok
 
     def read_file(self, path: str) -> tuple[str, bool]:
-        pass
+        return self._exec(["cat", path], None, 60)
 
-    def shutdown(self):
-        pass # make idempotent.
+    def shutdown(self) -> None:
+        """Idempotent; the container is --rm so `rm -f` both stops and removes it."""
+        container_id, self.container_id = self.container_id, None
+        if container_id:
+            subprocess.run(["podman", "rm", "-f", container_id], capture_output=True)
 
     def __del__(self):
         """Auto-delete container"""
-        self.shutdown()
+        try:
+            self.shutdown()
+        except Exception:
+            pass
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_utils.py</span>
    <span class="card-oneliner">Model-specific details: tool-call formats (XML function blocks, Hermes JSON) parsed adaptively, rendering back to the model, the dialect read off the chat template.</span>
    <span class="card-badge">Diff</span>
  </summary>

New file vs `/source` for `activation/agent/agent_utils.py`:

```diff
new file mode 100644
--- /dev/null
+++ b/activation/agent/agent_utils.py
@@ -0,0 +1,245 @@
+"""
+Model-specific details of talking to an agent model, kept out of the agent loop: how tool calls
+appear in a reply (parsing), how a turn is rendered back to the model (assistant turns with calls,
+tool results, the tool list), and the nudge. `ModelDialect.from_chat_template` reads them off the
+model's chat template, so the same loop drives a Qwen3.5 (XML function blocks), a Hermes-style
+model (JSON in <tool_call>) or a template without tool support (everything inlined as text).
+Parsing is adaptive per block: every registered format is tried, the dialect only decides which
+one is shown in error messages and used when rendering inline.
+"""
+from __future__ import annotations
+
+import json
+import re
+import typing as t
+import uuid
+from dataclasses import dataclass, field
+
+if t.TYPE_CHECKING:
+    from .agent_tools import ToolCallResult
+
+PARSE_ERROR_TOOL = "parse_error"
+TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
+TOOL_CALL_OPEN = re.compile(r"<tool_call>(?!.*</tool_call>)(.*)$", re.S)   # an unterminated block (max_tokens hit)
+ParameterTypes = dict[str, dict[str, str]]   # tool name -> parameter name -> JSON schema type
+
+
+# --------------------------------------------------------------------------------------- formats
+class ToolCallFormat:
+    """One way of writing a tool call inside a <tool_call> block."""
+    name: str = ""
+    reminder: str = ""   # one line telling the model the shape, used in parse errors
+
+    def matches(self, block: str) -> bool:
+        raise NotImplementedError
+
+    def parse(self, block: str, parameter_types: ParameterTypes | None) -> tuple[str, dict]:
+        """(tool name, arguments) or raise ValueError."""
+        raise NotImplementedError
+
+    def render(self, name: str, arguments: dict) -> str:
+        """The block as the model would write it (inline rendering, examples)."""
+        raise NotImplementedError
+
+
+class XmlFunctionFormat(ToolCallFormat):
+    """Qwen3.5 / Qwen3-Coder: <function=name><parameter=key>value</parameter></function>."""
+    name = "xml_function"
+    reminder = ("<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>, "
+                "one <parameter> block per argument, values as plain text")
+    FUNCTION = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
+    PARAMETER = re.compile(r"<parameter=([^>\s]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", re.S)
+
+    def matches(self, block: str) -> bool:
+        return "<function=" in block
+
+    def parse(self, block: str, parameter_types: ParameterTypes | None) -> tuple[str, dict]:
+        match = self.FUNCTION.search(block)
+        if match is None:
+            raise ValueError("no <function=NAME> block")
+        name, body = match.group(1).strip(), match.group(2)
+        arguments = {}
+        for parameter in self.PARAMETER.finditer(body):
+            value = parameter.group(2)
+            value = value[1:] if value.startswith("\n") else value
+            value = value[:-1] if value.endswith("\n") else value
+            arguments[parameter.group(1).strip()] = value
+        return name, coerce_arguments(name, arguments, parameter_types)
+
+    def render(self, name: str, arguments: dict) -> str:
+        parts = [f"<function={name}>"]
+        for key, value in arguments.items():
+            text = value if isinstance(value, str) else json.dumps(value)
+            parts.append(f"<parameter={key}>\n{text}\n</parameter>")
+        parts.append("</function>")
+        return "<tool_call>\n" + "\n".join(parts) + "\n</tool_call>"
+
+
+class HermesJsonFormat(ToolCallFormat):
+    """Qwen3 / Qwen2.5 / Hermes: one JSON object {"name": ..., "arguments": {...}} per block."""
+    name = "hermes_json"
+    reminder = '<tool_call>{"name": NAME, "arguments": {KEY: VALUE, ...}}</tool_call>, one JSON object per block'
+
+    def matches(self, block: str) -> bool:
+        return block.lstrip().startswith("{")
+
+    def parse(self, block: str, parameter_types: ParameterTypes | None) -> tuple[str, dict]:
+        data = json.loads(block)
+        arguments = data.get("arguments", {})
+        if isinstance(arguments, str):
+            arguments = json.loads(arguments) if arguments.strip() else {}
+        if not isinstance(data.get("name"), str) or not isinstance(arguments, dict):
+            raise ValueError("a tool call needs a string 'name' and an object 'arguments'")
+        return data["name"], arguments
+
+    def render(self, name: str, arguments: dict) -> str:
+        return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"
+
+
+FORMATS: list[ToolCallFormat] = [XmlFunctionFormat(), HermesJsonFormat()]
+
+
+def coerce_arguments(tool_name: str, arguments: dict, parameter_types: ParameterTypes | None) -> dict:
+    """Text values (XML format) become what the schema declares: integer, number, boolean, object, array."""
+    types = (parameter_types or {}).get(tool_name, {})
+    out = {}
+    for key, value in arguments.items():
+        declared = types.get(key)
+        if isinstance(value, str) and declared and declared != "string":
+            try:
+                value = json.loads(value)
+            except json.JSONDecodeError:
+                pass   # the tool's own TypeError/ValueError reaches the model
+        out[key] = value
+    return out
+
+
+def parse_tool_calls(
+    text: str, parameter_types: ParameterTypes | None = None, formats: list[ToolCallFormat] | None = None,
+    preferred: ToolCallFormat | None = None,
+) -> tuple[str, list[dict]]:
+    """
+    The assistant text without the <tool_call> blocks, and the calls as [{"id", "name", "arguments"}].
+    Each block is parsed by the first format that recognises it; a block no format accepts becomes a
+    `parse_error` call whose arguments carry the raw text and a reminder of the preferred shape.
+    """
+    formats = formats or FORMATS
+    preferred = preferred or formats[0]
+    calls = []
+    blocks = [match.group(1) for match in TOOL_CALL_BLOCK.finditer(text)]
+    tail = TOOL_CALL_OPEN.search(TOOL_CALL_BLOCK.sub("", text))
+    if tail is not None and tail.group(1).strip():
+        blocks.append(tail.group(1))   # an unterminated block still gets a readable error
+    for block in blocks:
+        call_id = uuid.uuid4().hex[:8]
+        error = None
+        for fmt in [f for f in formats if f.matches(block)] or [preferred]:
+            try:
+                name, arguments = fmt.parse(block, parameter_types)
+                calls.append({"id": call_id, "name": name, "arguments": arguments})
+                error = None
+                break
+            except (json.JSONDecodeError, ValueError, AttributeError) as failure:
+                error = f"{fmt.name}: {failure}"
+        if error is not None:
+            calls.append({"id": call_id, "name": PARSE_ERROR_TOOL,
+                          "arguments": {"raw": block[:2000], "error": error, "expected": preferred.reminder}})
+    content = TOOL_CALL_OPEN.sub("", TOOL_CALL_BLOCK.sub("", text)).strip()
+    return content, calls
+
+
+def parameter_types_of(tool_definitions: list[dict]) -> ParameterTypes:
+    """Tool name -> parameter -> declared type, from the function schemas."""
+    out = {}
+    for definition in tool_definitions:
+        function = definition.get("function", definition)
+        properties = (function.get("parameters") or {}).get("properties") or {}
+        out[function["name"]] = {key: str(spec.get("type", "")) for key, spec in properties.items() if isinstance(spec, dict)}
+    return out
+
+
+# --------------------------------------------------------------------------------------- rendering
+class MessageRendering:
+    """How a turn goes back to the model. The chat template gets structured messages by default."""
+
+    def system_prompt(self, system_prompt: str, tool_definitions: list[dict]) -> str:
+        return system_prompt
+
+    def template_tools(self, tool_definitions: list[dict]) -> list[dict] | None:
+        return tool_definitions
+
+    def assistant_message(self, content: str, calls: list[dict]) -> dict:
+        message: dict = {"role": "assistant", "content": content}
+        if calls:
+            message["tool_calls"] = [
+                {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}
+                for call in calls
+            ]
+        return message
+
+    def tool_messages(self, calls: list[dict], results: list["ToolCallResult"]) -> list[dict]:
+        return [{"role": "tool", "content": result.output} for result in results]
+
+
+class InlineRendering(MessageRendering):
+    """
+    For templates without tool support: the tool list goes into the system prompt, calls are written
+    into the assistant text in the preferred format and results come back as one user message of
+    <tool_response> blocks (the shape Qwen's own templates produce).
+    """
+    def __init__(self, fmt: ToolCallFormat):
+        self.format = fmt
+
+    def system_prompt(self, system_prompt: str, tool_definitions: list[dict]) -> str:
+        listing = "\n".join(json.dumps(definition) for definition in tool_definitions)
+        return (f"{system_prompt}\n\n# Tools\n\nYou may call these functions:\n<tools>\n{listing}\n</tools>\n\n"
+                f"To call one, reply in this exact shape: {self.format.reminder}. Results come back in <tool_response> blocks.")
+
+    def template_tools(self, tool_definitions: list[dict]) -> list[dict] | None:
+        return None
+
+    def assistant_message(self, content: str, calls: list[dict]) -> dict:
+        blocks = [self.format.render(call["name"], call["arguments"]) for call in calls]
+        return {"role": "assistant", "content": "\n\n".join([content] + blocks if content else blocks)}
+
+    def tool_messages(self, calls: list[dict], results: list["ToolCallResult"]) -> list[dict]:
+        body = "\n".join(f"<tool_response>\n{result.output}\n</tool_response>" for result in results)
+        return [{"role": "user", "content": body}]
+
+
+# --------------------------------------------------------------------------------------- dialect
+@dataclass
+class ModelDialect:
+    preferred_format: ToolCallFormat = field(default_factory=XmlFunctionFormat)
+    formats: list[ToolCallFormat] = field(default_factory=lambda: list(FORMATS))
+    rendering: MessageRendering = field(default_factory=MessageRendering)
+    nudge_message: str = "Call one of the provided tools or submit_answer when done."
+
+    @classmethod
+    def from_chat_template(cls, chat_template: str | None) -> "ModelDialect":
+        """
+        Read the model's habits off its chat template: the call shape it teaches (XML function blocks
+        vs JSON), and whether it renders `tools` / `tool_calls` at all (else everything is inlined).
+        """
+        template = chat_template or ""
+        if "<function=" in template:
+            preferred: ToolCallFormat = XmlFunctionFormat()
+        elif "<tool_call>" in template or '"arguments"' in template:
+            preferred = HermesJsonFormat()
+        else:
+            preferred = FORMATS[0]
+        formats = [preferred] + [fmt for fmt in FORMATS if fmt.name != preferred.name]
+        supports_tools = "tools" in template and "tool_calls" in template
+        rendering = MessageRendering() if supports_tools else InlineRendering(preferred)
+        return cls(preferred_format=preferred, formats=formats, rendering=rendering)
+
+    @classmethod
+    def for_tokenizer(cls, tokenizer) -> "ModelDialect":
+        return cls.from_chat_template(getattr(tokenizer, "chat_template", None))
+
+    def parse(self, text: str, parameter_types: ParameterTypes | None = None) -> tuple[str, list[dict]]:
+        return parse_tool_calls(text, parameter_types, self.formats, self.preferred_format)
+
+    def parse_error_message(self, arguments: dict) -> str:
+        return (f"Could not parse the tool call ({arguments.get('error')}). Expected shape: "
+                f"{arguments.get('expected') or self.preferred_format.reminder}.")
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent_tools.py</span>
    <span class="card-oneliner">Tool schemas, truncation, the default tools, semantic search on BM25, the subagent tool.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent_tools.py`:

```diff
--- a/activation/agent/agent_tools.py
+++ b/activation/agent/agent_tools.py
@@ -1,70 +1,225 @@
 """
 Defines tool calls.
-All tool results should be truncated to ~20000 chars.
-as head[:10000] ... [truncated and written to /tmp/agent_outputs/<id>.txt] ... tail[:10000]
-(use env to write these files).
-When truncated, we may pass the full output as AC.
+All tool results are truncated to ~20000 chars as head[:10000] ... [truncated and written to
+/tmp/agent_outputs/<id>.txt] ... tail[-10000:] (the env writes these files). When truncated, the
+full output travels as an activation-context output (`ac_outputs["full_output"]`).
 """
+from __future__ import annotations
 
+import re
 import typing as t
+import uuid
+from concurrent.futures import ThreadPoolExecutor
 from copy import deepcopy
+from dataclasses import dataclass, field
 
 if t.TYPE_CHECKING:
     from .agent_config import AgentConfig
+    from .agent_env import AgentEnv
     from activation.harness import HarnessRuntime
     from .agent import Agent
 
+TOOL_OUTPUT_LIMIT_CHARS = 20_000
+TOOL_OUTPUT_DIR = "/tmp/agent_outputs"
+@dataclass
 class ToolCallResult:
-    output: str
-    is_error: bool
-    ac_outputs: dict[str, t.Any] # activation context outputs (ac name -> ac input).
+    output: str                                          # what the model sees (already truncated)
+    is_error: bool = False
+    ac_outputs: dict[str, t.Any] = field(default_factory=dict)  # activation context outputs (ac name -> ac input)
+    is_final: bool = False                               # submit_answer sets it
+
+
+def truncate_output(env: "AgentEnv | None", text: str, call_id: str, limit: int = TOOL_OUTPUT_LIMIT_CHARS) -> tuple[str, dict]:
+    """Head + tail of a long output; the full text goes to a file in the env and out as activation context."""
+    if len(text) <= limit:
+        return text, {}
+    path = f"{TOOL_OUTPUT_DIR}/{call_id}.txt"
+    written = env.write_file(path, text) if env is not None else False
+    half = limit // 2
+    note = f"full output written to {path}" if written else "full output kept as activation context"
+    truncated = f"{text[:half]}\n... [truncated {len(text) - limit} chars; {note}] ...\n{text[-half:]}"
+    return truncated, {"full_output": text}
+
 
 class AgentTool:
+    name: str = ""
+    description: str = ""
+    parameters: dict = {"type": "object", "properties": {}}
+
     def __init__(self, harness: "HarnessRuntime", agent: "Agent", **kwargs):
-        pass
+        self.harness = harness
+        self.agent = agent
+        if kwargs:
+            raise TypeError(f"{type(self).__name__} got unexpected kwargs {sorted(kwargs)}")
 
     def tool_definition(self) -> dict:
-        pass # return the tool definition placed in prompt templates.
+        """The function schema handed to the chat template's `tools`."""
+        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}
 
-    def execute(self, **kwargs) -> ToolCallResult:
-        pass
+    def execute(self, **arguments) -> ToolCallResult:
+        raise NotImplementedError
 
 
 class ShellTool(AgentTool):
-    pass
+    name = "shell"
+    description = "Run a bash script in your sandbox (no network). Returns combined stdout and stderr."
+    parameters = {
+        "type": "object",
+        "properties": {
+            "script": {"type": "string", "description": "The bash script to run."},
+            "timeout": {"type": "integer", "description": "Seconds before the script is killed (default 120)."},
+        },
+        "required": ["script"],
+    }
+
+    def execute(self, script: str, timeout: int = 120) -> ToolCallResult:
+        output, ok = self.agent.agent_env.run_shell(script, timeout=int(timeout))
+        return ToolCallResult(output=output or "(no output)", is_error=not ok)
 
 
 class PythonTool(AgentTool):
-    pass
+    name = "python"
+    description = "Run a Python 3 program in your sandbox (no network, no state kept between calls). Print what you need to see."
+    parameters = {
+        "type": "object",
+        "properties": {
+            "code": {"type": "string", "description": "The Python source to run."},
+            "timeout": {"type": "integer", "description": "Seconds before the program is killed (default 120)."},
+        },
+        "required": ["code"],
+    }
+
+    def execute(self, code: str, timeout: int = 120) -> ToolCallResult:
+        output, ok = self.agent.agent_env.run_python_code(code, timeout=int(timeout))
+        return ToolCallResult(output=output or "(no output)", is_error=not ok)
+
+
+class SubmitAnswerTool(AgentTool):
+    name = "submit_answer"
+    description = "Submit your final answer and end the task. Call it exactly once, when you are done."
+    parameters = {
+        "type": "object",
+        "properties": {"answer": {"type": "string", "description": "The final answer, as concise as possible."}},
+        "required": ["answer"],
+    }
+
+    def execute(self, answer: t.Any = None) -> ToolCallResult:
+        self.agent.run_results.answer = answer
+        return ToolCallResult(output=f"Answer submitted: {answer}", is_final=True)
+
 
 class ParallelCallTool(AgentTool):
-    pass
+    name = "parallel_tool_call"
+    description = "Run several tool calls at once. Each call names a tool and its arguments; the results come back in the same order."
+    parameters = {
+        "type": "object",
+        "properties": {
+            "calls": {
+                "type": "array",
+                "items": {
+                    "type": "object",
+                    "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}},
+                    "required": ["name", "arguments"],
+                },
+            },
+        },
+        "required": ["calls"],
+    }
+
+    def execute(self, calls: list[dict]) -> ToolCallResult:
+        if not isinstance(calls, list) or not calls:
+            return ToolCallResult(output="parallel_tool_call needs a non-empty list of {name, arguments}.", is_error=True)
+        nested = [{"id": uuid.uuid4().hex[:8], "name": str(call.get("name")), "arguments": call.get("arguments") or {}} for call in calls]
+        with ThreadPoolExecutor(max_workers=min(8, len(nested))) as pool:
+            results = list(pool.map(self.agent.execute_tool_call, nested))
+        outputs = [f"[{call['name']}] {result.output}" for call, result in zip(nested, results)]
+        ac_outputs = {call["id"]: result.ac_outputs for call, result in zip(nested, results) if result.ac_outputs}
+        return ToolCallResult(
+            output="\n\n".join(outputs), is_error=any(result.is_error for result in results),
+            ac_outputs=ac_outputs, is_final=any(result.is_final for result in results),
+        )
+
 
 class SemanticSearchTool(AgentTool):
-    pass
+    """Search over a registered dataset's chunks. BM25 for now; the dense path returns with the retrieval rethink."""
+    name = "semantic_search"
+    description = "Search the task's corpus for passages relevant to a query. Returns the best matching passages with their ids."
+    parameters = {
+        "type": "object",
+        "properties": {
+            "query": {"type": "string", "description": "What to look for."},
+            "top_k": {"type": "integer", "description": "How many passages to return."},
+        },
+        "required": ["query"],
+    }
+
+    def __init__(self, harness, agent, dataset_id: str | None = None, top_k: int = 5, max_chars: int = 2000):
+        super().__init__(harness, agent)
+        task = agent.agent_config.dataset_task
+        self.dataset_id = dataset_id or (task.dataset_id if task is not None else None)
+        self.top_k = top_k
+        self.max_chars = max_chars
+
+    def execute(self, query: str, top_k: int | None = None) -> ToolCallResult:
+        if self.dataset_id is None or self.dataset_id not in self.harness.dataset_manager.loaded_datasets:
+            return ToolCallResult(output=f"No searchable corpus is registered for this task ({self.dataset_id}).", is_error=True)
+        index = self.harness.dataset_manager._get_or_create_index(self.dataset_id)
+        index.build_bm25_index()
+        chunks = index.bm25_query_many_frozen([query], top_k=int(top_k or self.top_k))[0]
+        if not chunks:
+            return ToolCallResult(output="No passage matched the query.")
+        blocks = [f"[{chunk.chunk_id}]\n{index.get_chunk_section(chunk)[:self.max_chars]}" for chunk in chunks]
+        return ToolCallResult(output="\n\n".join(blocks), ac_outputs={"chunk_ids": [chunk.chunk_id for chunk in chunks]})
 
 
 class SubagentTool(AgentTool):
-    ... # Simple in/out without a persistent handle for now. Runs in same env as caller.
-    ... # Gets caller current traj as AC. Returns traj as AC. Registers in subtrajectories.
-    ... # Unless AC communication disabled.
-    ... # Some subagents are general-purpose (name=general_subagent_tool()).
-    ... # Others are specific (past_problem_search_subagent(...)).
-    ... # all registered as tools though.
-    def __init__(self, shared_args, base_config: AgentConfig, name: str, extra_description: str):
-        pass
-
+    """
+    Simple in/out without a persistent handle. Runs in the same env as the caller. Gets the caller's
+    current trajectory as activation context and returns its own trajectory as activation context
+    (unless AC communication is disabled); its result is registered in the caller's subagent results.
+    Some subagents are general-purpose, others specific: all are registered as tools under their own
+    name, with `base_config` and `extra_description` as constructor kwargs.
+    """
+    parameters = {
+        "type": "object",
+        "properties": {"task": {"type": "string", "description": "A self-contained description of what the subagent should do and return."}},
+        "required": ["task"],
+    }
+
+    def __init__(self, harness, agent, base_config: "AgentConfig", extra_description: str = ""):
+        super().__init__(harness, agent)
+        self.base_config = base_config
+        self.description = ("Delegate a self-contained task to a subagent that shares your sandbox and returns its answer. "
+                            + extra_description).strip()
 
     def execute(self, task: str) -> ToolCallResult:
         from .agent import Agent
+        parent = self.agent
         subagent_config = deepcopy(self.base_config)
-        subagent_config.task = task
+        subagent_config.user_prompt = task
+        share_ac = parent.agent_config.enable_ac_communication
+        if share_ac:
+            subagent_config.ac_inputs = dict(subagent_config.ac_inputs) | {"caller_trajectory": deepcopy(parent.run_results.trajectory)}
         subagent = Agent(
-            harness = self.harness,
-            agent_env = self.agent.agent_env,
-            agent_config = subagent_config,
+            harness=self.harness,
+            agent_config=subagent_config,
+            agent_env=parent.agent_env,
+            parent_agent=parent,
+            reporter=parent.reporter,
+            seed=parent.seed,
         )
-        subagent.run()
-        with self.agent.lock:
-            self.agent.subagent_results.append(subagent.run_results)
-        pass
\ No newline at end of file
+        result = subagent.run()
+        with parent.lock:
+            parent.run_results.subagent_results.append(result)
+        output = (f"Subagent finished ({result.finish_reason}, {result.num_turns} turns). "
+                  f"Answer: {result.answer if result.answer is not None else '(none)'}")
+        ac_outputs = {"trajectory": result.trajectory} if share_ac else {}
+        return ToolCallResult(output=output, is_error=result.finish_reason == "error", ac_outputs=ac_outputs)
+
+
+DEFAULT_TOOLS: dict[str, tuple[type[AgentTool], dict]] = {
+    ShellTool.name: (ShellTool, {}),
+    PythonTool.name: (PythonTool, {}),
+    ParallelCallTool.name: (ParallelCallTool, {}),
+    SubmitAnswerTool.name: (SubmitAnswerTool, {}),
+}
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/agent.py</span>
    <span class="card-oneliner">The turn loop with budgets and error capture.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/agent.py`:

```diff
--- a/activation/agent/agent.py
+++ b/activation/agent/agent.py
@@ -1,56 +1,237 @@
+"""
+The agent turn loop: messages -> one engine request -> tool calls -> tool results -> next turn, until
+the answer is submitted or a budget runs out. One Agent owns one AgentRunResult; the rollout manager
+runs many agents on threads and the engine batches their requests.
+"""
+from __future__ import annotations
+
+import json
+import threading
+import time
+import traceback
 import typing as t
-from .agent_config import AgentConfig, AgentRunResult
+import uuid
+from dataclasses import asdict
+
+from .agent_config import AgentConfig, AgentRunResult, TrajectoryStep
 from .agent_env import AgentEnv
-from .agent_tools import AgentTool
+from .agent_tools import DEFAULT_TOOLS, AgentTool, ToolCallResult, truncate_output
+from .agent_utils import PARSE_ERROR_TOOL, ModelDialect, parameter_types_of
 
+MAX_CONSECUTIVE_NO_TOOL_TURNS = 3   # nudged after each; the run ends with no_tool_call at the third in a row
 
 if t.TYPE_CHECKING:
     from activation.harness import HarnessRuntime
     from .rollout_reporter import RolloutReporter
 
+
+
 class Agent:
     def __init__(
         self,
-        harness: HarnessRuntime,
+        harness: "HarnessRuntime",
         agent_config: AgentConfig,
-        agent_env: "AgentEnv|None" = None,
-        parent_agent: "Agent|None" = None,
-        reporter: "RolloutReporter|None" = None,
+        agent_env: AgentEnv | None = None,
+        parent_agent: "Agent | None" = None,
+        reporter: "RolloutReporter | None" = None,
+        seed: int = 0,
     ):
         self.harness = harness
         self.agent_config = agent_config
+        self.parent_agent = parent_agent
+        self.reporter = reporter
+        self.dialect = ModelDialect()   # replaced by the model's own dialect when run() starts
+        self.seed = seed
         self.owns_env = agent_env is None
-        self.agent_env = agent_env or AgentEnv(agent_config.dockerfile_path, agent_config.dockerfile_arg)
-        self.run_results = AgentRunResult(config=agent_config)
-        self.agent_id = ... # uuid. Allows speaking to the same vllm instance for kv caching.
-        self.lock = ... # to support parallel subtraj appends from the subagent tool
-        self.tools: dict[str, AgentTool] = dict()
+        self._env = agent_env
+        self.run_results = AgentRunResult(agent_config=agent_config, seed=seed)
+        self.agent_id = uuid.uuid4().hex  # Pins the agent to one engine replica, so its prefix cache serves the next turn.
+        self.lock = threading.Lock()      # Parallel subagent result appends.
+        self.messages: list[dict] = []
+        self.tools: dict[str, AgentTool] = {}
+        self.finished = False
         self._initialize_tools()
 
-    def run(self):
-        # Simple in/out run.
-        # Populates run_reuslts.
-        # Compaction is for later.
-        pass
+    # ----------------------------------------------------------------------------- setup
+    @property
+    def agent_env(self) -> AgentEnv:
+        """Created on first use, so a cached or failed-early run never starts a container."""
+        if self._env is None:
+            self._env = AgentEnv(self.agent_config.env_dockerfile_path, self.agent_config.env_args,
+                                 default_image=self.harness.harness_config.agent_env_default_image,
+                                 memory_limit_mb=self.agent_config.env_memory_limit_mb
+                                 if self.agent_config.env_memory_limit_mb is not None
+                                 else self.harness.harness_config.agent_env_memory_limit_mb)
+        return self._env
 
+    def _initialize_tools(self):
+        specs = dict(DEFAULT_TOOLS) | dict(self.agent_config.tools)
+        for name, (tool_class, kwargs) in specs.items():
+            tool = tool_class(self.harness, self, **kwargs)
+            if not tool.name:
+                tool.name = name
+            self.tools[name] = tool
 
-    def score(self):
-        pass
+    def _tool_definitions(self) -> list[dict]:
+        return [tool.tool_definition() for tool in self.tools.values()]
 
+    @property
+    def loaded_model(self):
+        config = self.agent_config
+        assert config.model_name in self.harness.loaded_models, f"unknown model {config.model_name!r}"
+        return self.harness.loaded_models[config.model_name]
 
-    def _initialize_tools(self):
-        pass
+    # ----------------------------------------------------------------------------- run
+    def run(self) -> AgentRunResult:
+        """
+        Simple in/out run. Populates run_results. Never raises for a model or tool failure: the run
+        ends with finish_reason "error" and the traceback in score_feedback. Compaction is for later.
+        """
+        config, results = self.agent_config, self.run_results
+        start = time.time()
+        self.dialect = ModelDialect.for_tokenizer(self.loaded_model.tokenizer)
+        rendering = self.dialect.rendering
+        tool_definitions = self._tool_definitions()
+        parameter_types = parameter_types_of(tool_definitions)
+        template_tools = rendering.template_tools(tool_definitions)
+        self.messages = []
+        system_prompt = rendering.system_prompt(config.system_prompt or "", tool_definitions)
+        if system_prompt:
+            self.messages.append({"role": "system", "content": system_prompt})
+        self.messages.append({"role": "user", "content": config.user_prompt})
+        self._count_ac_inputs()
+        if self.reporter is not None:
+            self.reporter.report_agent_start(self)
+        no_tool_turns = 0
+        errors = 0
+        try:
+            while True:
+                if results.num_turns >= config.max_turns:
+                    results.finish_reason = "max_turns"
+                    break
+                if time.time() - start > config.max_duration:
+                    results.finish_reason = "max_duration"
+                    break
+                output = self.loaded_model.engine_submit(
+                    self.messages, tools=template_tools, seed=self.seed + results.num_turns,
+                    agent_id=self.agent_id, lora_name=config.lora_name, chat_kwargs=config.call_kwargs,
+                )
+                results.num_turns += 1
+                results.num_input_tokens += output.prompt_token_count
+                results.num_cached_input_tokens += output.cached_prompt_token_count
+                results.num_output_tokens += output.output_token_count
+                content, calls = self.dialect.parse(output.text, parameter_types)
+                self.messages.append(rendering.assistant_message(content, calls))
+                step = TrajectoryStep(role="assistant", content=content, tool_calls=calls)
+                if not calls:
+                    # A turn without a call gets the nudge; several in a row end the run.
+                    results.trajectory.append(asdict(step))
+                    no_tool_turns += 1
+                    if no_tool_turns >= MAX_CONSECUTIVE_NO_TOOL_TURNS:
+                        results.finish_reason = "no_tool_call"
+                        break
+                    self.messages.append({"role": "user", "content": self.dialect.nudge_message})
+                    self._report_step()
+                    continue
+                no_tool_turns = 0
+                call_results = self._execute_tool_calls(calls)
+                errors += sum(1 for result in call_results if result.is_error)
+                tool_step = TrajectoryStep(
+                    role="tool",
+                    content="\n\n".join(result.output for result in call_results),
+                    tool_calls=calls,
+                    tool_call_results=[result.output for result in call_results],
+                    activations={call["id"]: result.ac_outputs for call, result in zip(calls, call_results) if result.ac_outputs},
+                )
+                results.trajectory.extend([asdict(step), asdict(tool_step)])
+                self.messages.extend(rendering.tool_messages(calls, call_results))
+                self._report_step()
+                if any(result.is_final for result in call_results):
+                    results.finish_reason = "submitted"
+                    break
+                if errors > config.max_tool_errors:
+                    results.finish_reason = "max_tool_errors"
+                    break
+        except Exception as error:
+            if _is_context_overflow(error):
+                # The conversation outgrew the model's context (no compaction in slice 1): a budget, not a bug.
+                results.finish_reason = "context_exceeded"
+                results.score_feedback = str(error)[:500]
+            else:
+                results.finish_reason = "error"
+                results.score_feedback = traceback.format_exc()
+                print(f"Agent {self.agent_id[:8]} - error:\n{results.score_feedback}", flush=True)
+        finally:
+            results.duration = time.time() - start
+            self.finished = True
+            if self.reporter is not None:
+                self.reporter.report_agent_finish(self)
+        return results
 
-    def _execute_tool_calls(self):
-        pass
-    
+    def _report_step(self):
+        if self.reporter is not None:
+            self.reporter.report_agent_step(self)
 
-    def shutdown(self):
-        if not self.owns_env:
+    def _count_ac_inputs(self):
+        ac_inputs = self.agent_config.ac_inputs
+        if not ac_inputs:
             return
-        self.agent_env.shutdown()
+        text = json.dumps(ac_inputs, default=repr)
+        self.run_results.num_ac_input_bytes = len(text.encode("utf-8"))
+        try:
+            self.run_results.num_ac_input_tokens = len(self.loaded_model.tokenizer(text, add_special_tokens=False)["input_ids"])
+        except Exception:
+            self.run_results.num_ac_input_tokens = len(text) // 4
 
+    # ----------------------------------------------------------------------------- tools
+    def execute_tool_call(self, call: dict) -> ToolCallResult:
+        """One call -> one truncated result. Unknown tools and bad arguments are errors the model reads."""
+        name, arguments = call["name"], call.get("arguments") or {}
+        if name == PARSE_ERROR_TOOL:
+            result = ToolCallResult(output=self.dialect.parse_error_message(arguments), is_error=True)
+        elif name not in self.tools:
+            result = ToolCallResult(output=f"Unknown tool {name!r}. Available: {', '.join(self.tools)}.", is_error=True)
+        else:
+            try:
+                result = self.tools[name].execute(**arguments)
+            except TypeError as error:
+                result = ToolCallResult(output=f"Bad arguments for {name}: {error}", is_error=True)
+            except Exception as error:
+                result = ToolCallResult(output=f"{name} failed: {type(error).__name__}: {error}", is_error=True)
+        output, ac_outputs = truncate_output(self._env if self._env is not None else None, result.output, call["id"])
+        result.output = output
+        if ac_outputs:
+            result.ac_outputs = dict(result.ac_outputs) | ac_outputs
+        return result
+
+    def _execute_tool_calls(self, calls: list[dict]) -> list[ToolCallResult]:
+        return [self.execute_tool_call(call) for call in calls]
+
+    # ----------------------------------------------------------------------------- scoring and teardown
+    def score(self) -> float:
+        task = self.agent_config.dataset_task
+        if task is None:
+            return 0.0
+        try:
+            self.run_results.score = float(task.score(self.run_results))
+        except Exception as error:
+            self.run_results.score = 0.0
+            self.run_results.score_feedback = f"scoring failed: {type(error).__name__}: {error}"
+        return self.run_results.score
+
+    def shutdown(self):
+        if not self.owns_env or self._env is None:
+            return
+        self._env.shutdown()
 
     def __del__(self):
-        self.shutdown()
+        try:
+            self.shutdown()
+        except Exception:
+            pass
+
 
+def _is_context_overflow(error: BaseException) -> bool:
+    """vLLM's validation error for a prompt over max_model_len (its class differs across versions)."""
+    text = str(error)
+    return "maximum context length" in text or "max_model_len" in text or "longer than the maximum" in text
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/rollout_manager.py</span>
    <span class="card-oneliner">Single and grouped rollouts on a bounded pool, seeds, scoring, caching.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/rollout_manager.py`:

```diff
--- a/activation/agent/rollout_manager.py
+++ b/activation/agent/rollout_manager.py
@@ -1,30 +1,39 @@
 """
-Allows perform single and grouped rollouts with varying seeds.
-Rollouts always occur through the engine.
-@AI: One thing we have to slightly relax is different model co-habitation, if that's not already the case.
-- We can, I think, still auto-free other engine models.
-- But engin models must certainly co-habitate with raw hf models (e.g., for embeddings, etc).
+Single and grouped rollouts with varying seeds. Rollouts always occur through the engine: every agent
+runs on its own thread and submits one request per turn; the engine batches across agents (no
+synchronous batching of rollouts). One engine is resident at a time, as before, and raw HF models of
+other names stay loaded next to it.
 
-@AI: Here, parallelism through batching-only becomes unrealistic: confirm that by simply submitting in parallel, vllm will handle the rest.
-
-Slice 1 is about getting full e2e rollouts + caching (without any rejection sampling).
-Slice 2 adds trajectory harvesting and sft.
+Slice 1 is full end-to-end rollouts plus caching (no rejection sampling). Slice 2 adds trajectory
+harvesting and SFT.
 """
+from __future__ import annotations
 
-class RolloutManager:
-    def __init__(self, harness: HarnessRuntime):
-        pass
+import typing as t
+from concurrent.futures import ThreadPoolExecutor, as_completed
+
+from .agent import Agent
+from .agent_config import AgentConfig, AgentRunResult
+from .rollout_caching import RolloutCache, config_key
 
+if t.TYPE_CHECKING:
+    from activation.harness import HarnessRuntime
+    from .rollout_reporter import RolloutReporter
+
+
+class RolloutManager:
+    def __init__(self, harness: "HarnessRuntime"):
+        self.harness = harness
 
     def perform_single_rollouts(
         self,
         agent_configs: list[AgentConfig],
         seed: int = 0,
-        caching_id: str|None = None, # Used to save runs in a safe way.
+        caching_id: str | None = None,   # Used to save runs in a safe way.
         perform_scoring: bool = True,
-    ) -> list[AgentRunResults]:
-        pass
-
+        reporter: "RolloutReporter | None" = None,
+    ) -> list[AgentRunResult]:
+        return self._run_jobs([(config, seed) for config in agent_configs], caching_id, perform_scoring, reporter)
 
     def perform_grouped_rollouts(
         self,
@@ -32,6 +41,45 @@ class RolloutManager:
         group_count: int,
         base_seed: int = 0,
         perform_scoring: bool = True,
-        caching_id: str|None = None, # Used to save runs. group increasing from 4 to 6 should save 4 runs.
-    ) -> list[list[AgentRunResults]]:
-        pass
+        caching_id: str | None = None,   # Used to save runs: a group growing from 4 to 6 reuses the 4.
+        reporter: "RolloutReporter | None" = None,
+    ) -> list[list[AgentRunResult]]:
+        jobs = [(config, base_seed + member) for config in agent_configs for member in range(group_count)]
+        flat = self._run_jobs(jobs, caching_id, perform_scoring, reporter)
+        return [flat[index:index + group_count] for index in range(0, len(flat), group_count)]
+
+    # ----------------------------------------------------------------------------- internals
+    def _run_jobs(self, jobs: list[tuple[AgentConfig, int]], caching_id, perform_scoring, reporter) -> list[AgentRunResult]:
+        cache = RolloutCache(caching_id)
+        results: list[AgentRunResult | None] = []
+        for config, seed in jobs:
+            row = cache.get(config_key(config), seed)
+            results.append(None if row is None else AgentRunResult.deserialize(row, self.harness, agent_config=config))
+        pending = [index for index, result in enumerate(results) if result is None]
+        if reporter is not None:
+            reporter.begin_rollouts(total=len(jobs), cached=len(jobs) - len(pending))
+        if pending:
+            model_names = {jobs[index][0].model_name for index in pending}
+            assert len(model_names) == 1, f"one engine is resident at a time; configs name {sorted(map(str, model_names))}"
+            self.harness.loaded_models[model_names.pop()].ensure_engine_loaded()   # once, before the threads
+            with ThreadPoolExecutor(max_workers=self.harness.harness_config.agent_max_concurrent) as pool:
+                futures = {pool.submit(self._run_one, jobs[index][0], jobs[index][1], perform_scoring, reporter): index for index in pending}
+                for future in as_completed(futures):
+                    result = future.result()
+                    results[futures[future]] = result
+                    cache.append(result)
+        if reporter is not None:
+            reporter.finish()
+        return t.cast(list[AgentRunResult], results)
+
+    def _run_one(self, config: AgentConfig, seed: int, perform_scoring: bool, reporter) -> AgentRunResult:
+        agent = Agent(self.harness, config, reporter=reporter, seed=seed)
+        try:
+            result = agent.run()
+            if perform_scoring and config.dataset_task is not None:
+                agent.score()
+                if reporter is not None:
+                    reporter.report_agent_scored(agent)
+        finally:
+            agent.shutdown()
+        return result
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/rollout_caching.py</span>
    <span class="card-oneliner">JSONL cache under the synced folder, keyed by task and seed.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/rollout_caching.py`:

```diff
--- a/activation/agent/rollout_caching.py
+++ b/activation/agent/rollout_caching.py
@@ -0,0 +1,63 @@
+"""
+Rollout results as JSONL under the synced folder (`resolve_path("ROLLOUTS/<caching_id>")`), one
+serialized AgentRunResult per line, flushed per append. Keyed by (config key, seed): a dataset task's
+key is "<dataset_id>/<task_id>", a free-form config hashes its prompts and model. A killed run keeps
+every finished agent; a group of 6 after a cached group of 4 runs members 4 and 5 only. Rows written
+on a Sky node land here through `sky exec --sync` and a later node starts with them.
+"""
+from __future__ import annotations
+
+import hashlib
+import json
+import threading
+
+from activation.common.data_syncing import resolve_path
+
+from .agent_config import AgentConfig, AgentRunResult
+
+CACHE_FILENAME = "rollouts.jsonl"
+
+
+def config_key(config: AgentConfig) -> str:
+    task = config.dataset_task
+    if task is not None:
+        return f"{task.dataset_id}/{task.task_id}"
+    digest = hashlib.sha256(json.dumps([config.system_prompt, config.user_prompt, config.model_name, config.lora_name]).encode()).hexdigest()
+    return f"prompt/{digest[:16]}"
+
+
+class RolloutCache:
+    """`None` caching id disables the cache (reads return None, appends are no-ops)."""
+
+    def __init__(self, caching_id: str | None):
+        self.path = None if caching_id is None else resolve_path(f"ROLLOUTS/{caching_id}", create=False) / CACHE_FILENAME
+        if self.path is not None:
+            self.path.parent.mkdir(parents=True, exist_ok=True)   # resolve_path would take a dotted id for a file name
+        self.lock = threading.Lock()
+        self.rows: dict[tuple[str, int], dict] = {}
+        if self.path is not None and self.path.exists():
+            with open(self.path) as handle:
+                for line in handle:
+                    line = line.strip()
+                    if not line:
+                        continue
+                    try:
+                        row = json.loads(line)
+                    except json.JSONDecodeError:
+                        break                                                      # a torn last line from a killed run
+                    self.rows[(row["config_key"], int(row["seed"]))] = row
+            if self.rows:
+                print(f"RolloutCache - {len(self.rows)} cached rollouts in {self.path}.", flush=True)
+
+    def get(self, key: str, seed: int) -> dict | None:
+        return self.rows.get((key, seed))
+
+    def append(self, result: AgentRunResult) -> None:
+        if self.path is None:
+            return
+        row = result.serialize() | {"config_key": config_key(result.agent_config)}
+        with self.lock:
+            self.rows[(row["config_key"], int(result.seed))] = row
+            with open(self.path, "a") as handle:
+                handle.write(json.dumps(row, ensure_ascii=False, default=repr) + "\n")
+                handle.flush()
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/agent/rollout_reporter.py</span>
    <span class="card-oneliner">Live page: progress, finished rollouts, rendered trajectories (running first, then the last N), uncut copies as files.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/agent/rollout_reporter.py`:

```diff
--- a/activation/agent/rollout_reporter.py
+++ b/activation/agent/rollout_reporter.py
@@ -1,13 +1,213 @@
 """
-This is a bit more sophisticated than the training reporter.
-In addition to general rollout progress, I would love to have the option of viewing active trajectories.
-If possible, do not consistently rewrite the html files, but write to a local json that is periodically refetched.
-To avoid file flickers.
-(Update training to be like that too btw; slightl violates self-containment, but fine).
-
-Likely, simple functions (that don't pollute the agent code) like:
-- report_agent_start(agent)
-- report_agent_step(agent)
-- report_agent_finish(agent)
-should be enough. step can update the statistics/trajs, since the the run results should contain them.
-"""
\ No newline at end of file
+Live page for rollouts, a bit richer than the training reporter: overall progress and statistics plus
+the rendered trajectories, the running ones first and then the last `finished_trajectories_shown`
+complete ones; every text on the page is cut to size, the uncut trajectory of each agent is written
+next to the page as `trajectories/<agent>.json` (linked from its header). The page is written once
+and polls `report_data.json`, which is rewritten on every render (see common.reporting), so the file does not flicker. Simple hooks that keep agent code clean:
+report_agent_start / step / finish / scored; the statistics come from the agent's run results.
+"""
+from __future__ import annotations
+
+import json
+import threading
+import time
+import typing as t
+from collections import deque
+from pathlib import Path
+
+from activation.common.reporting import HtmlReporter, _write_atomic, format_seconds
+
+if t.TYPE_CHECKING:
+    from .agent import Agent
+
+PROMPT_CHARS = 1500       # per trajectory
+CONTENT_CHARS = 1200      # per assistant step
+ARGUMENT_CHARS = 1500     # per tool call
+RESULT_CHARS = 1000       # per tool result
+ROLLOUT_COLUMNS = ["agent", "task", "seed", "finish", "turns", "tool calls", "tokens in", "cached", "tokens out", "duration", "score"]
+
+
+class RolloutReporter(HtmlReporter):
+    def __init__(self, report_folder: str, title: str, description: str = "", finished_trajectories_shown: int = 100):
+        super().__init__(report_folder, title, description, eyebrow="Rollouts", refresh_seconds=10, min_render_interval_seconds=2.0)
+        self.lock = threading.Lock()
+        self.started_at = time.time()
+        self.total = 0
+        self.cached = 0
+        self.running: dict[str, "Agent"] = {}
+        self.finished_items: deque[dict] = deque(maxlen=max(1, min(finished_trajectories_shown, 100)))
+        self.finished_count = 0
+        self.scores: list[float] = []
+        self.output_tokens = 0
+        self.initialize_line_plot(
+            "progress", "Rollouts over time",
+            "Finished rollouts and the running mean score (scored rollouts only) against wall-clock seconds.",
+            "seconds", "value", ["finished", "mean score"],
+        )
+        self.initialize_table("rollouts", "Finished rollouts", "One row per finished agent (subagents included).", ROLLOUT_COLUMNS)
+        self.set_trajectories(
+            "trajectories", "Trajectories",
+            f"Running rollouts first (open), then the last {self.finished_items.maxlen} finished ones, newest first. Long texts are cut.", [],
+        )
+        self._update_status()
+        self.render(force=True)
+
+    @property
+    def report_folder(self) -> str:
+        """Where report.tressoir.html, report_data.json and trajectories/ live."""
+        return str(self.folder)
+
+    # ----------------------------------------------------------------------------- manager hooks
+    def begin_rollouts(self, total: int, cached: int) -> None:
+        with self.lock:
+            self.total, self.cached = total, cached
+            self._update_status()
+        self.render(force=True)
+
+    # ----------------------------------------------------------------------------- agent hooks
+    def report_agent_start(self, agent: "Agent") -> None:
+        with self.lock:
+            self.running[agent.agent_id] = agent
+            self._update_trajectories()
+            self._update_status()
+        self.render()
+
+    def report_agent_step(self, agent: "Agent") -> None:
+        with self.lock:
+            self._update_trajectories()
+            self._update_status()
+        self.render()
+
+    def report_agent_finish(self, agent: "Agent") -> None:
+        results = agent.run_results
+        with self.lock:
+            self.running.pop(agent.agent_id, None)
+            self.finished_items.appendleft(trajectory_item(agent, "finished", self.folder))
+            self.finished_count += 1
+            self.output_tokens += results.num_output_tokens
+            task = agent.agent_config.dataset_task
+            self.add_data_point("rollouts", {
+                "agent": _agent_label(agent),
+                "task": task.task_id[:12] if task is not None else "-", "seed": results.seed, "finish": results.finish_reason,
+                "turns": results.num_turns, "tool calls": results.num_tool_calls, "tokens in": results.num_input_tokens,
+                "cached": results.num_cached_input_tokens, "tokens out": results.num_output_tokens,
+                "duration": format_seconds(results.duration), "score": "" if task is None else f"{results.score:.2f}",
+            })
+            self._add_progress_point()
+            self._update_trajectories()
+            self._update_status()
+        self.render(force=True)
+
+    def report_agent_scored(self, agent: "Agent") -> None:
+        """Scoring happens after finish; the table row, the trajectory header and the mean are updated."""
+        results = agent.run_results
+        with self.lock:
+            self.scores.append(results.score)
+            label = _agent_label(agent)
+            for row in reversed(self.widgets["rollouts"]["rows"]):
+                if row["agent"] == label:
+                    row["score"] = f"{results.score:.2f}"
+                    break
+            for item in self.finished_items:
+                if item["id"] == agent.agent_id:
+                    item["stats"] = _stats(agent)
+                    break
+            self._add_progress_point()
+            self._update_trajectories()
+            self._update_status()
+        self.render(force=True)
+
+    # ----------------------------------------------------------------------------- internals
+    def _update_trajectories(self) -> None:
+        items = [trajectory_item(agent, "running", self.folder) for agent in self.running.values()] + list(self.finished_items)
+        self.widgets["trajectories"]["items"] = items
+
+    def _add_progress_point(self) -> None:
+        point = {"x": round(time.time() - self.started_at, 1), "finished": self.finished_count}
+        if self.scores:
+            point["mean score"] = sum(self.scores) / len(self.scores)
+        self.add_data_point("progress", point)
+
+    def _update_status(self) -> None:
+        elapsed = time.time() - self.started_at
+        self.set_status(
+            total=str(self.total), cached=str(self.cached), running=str(len(self.running)), finished=str(self.finished_count),
+            mean_score=f"{sum(self.scores) / len(self.scores):.3f}" if self.scores else "-",
+            output_tokens_per_s=f"{self.output_tokens / elapsed:.1f}" if elapsed > 0 else "-",
+            elapsed=format_seconds(elapsed),
+        )
+
+
+# --------------------------------------------------------------------------------- rendering items
+def trajectory_item(agent: "Agent", state: str, folder: Path | None = None) -> dict:
+    """
+    The page's view of one agent: header, prompt, and every step with calls and results cut to size.
+    With a folder, the uncut trajectory goes to `<folder>/trajectories/<agent>.json` and the item links it.
+    """
+    results = agent.run_results
+    file = None
+    if folder is not None:
+        file = f"trajectories/{agent.agent_id[:8]}.json"
+        (folder / "trajectories").mkdir(parents=True, exist_ok=True)
+        _write_atomic(folder / file, json.dumps({
+            "agent_id": agent.agent_id, "state": state, "system_prompt": agent.agent_config.system_prompt,
+            "user_prompt": agent.agent_config.user_prompt, "trajectory": results.trajectory,
+            "answer": results.answer, "finish_reason": results.finish_reason, "score": results.score,
+        }, indent=1, default=str))
+    steps = []
+    turn = 0
+    for step in results.trajectory:
+        if step["role"] == "assistant":
+            turn += 1
+            steps.append({
+                "role": "assistant", "turn": turn, "content": _cut(step["content"], CONTENT_CHARS),
+                "calls": [{"name": call["name"], "arguments": _cut(_arguments(call["arguments"]), ARGUMENT_CHARS)} for call in step["tool_calls"]],
+            })
+        else:
+            steps.append({
+                "role": "tool", "turn": turn, "content": "",
+                "results": [{"name": call["name"], "output": _cut(output, RESULT_CHARS)}
+                            for call, output in zip(step["tool_calls"], step["tool_call_results"])],
+            })
+    return {
+        "id": agent.agent_id, "title": _title(agent), "state": state, "stats": _stats(agent),
+        "prompt": _cut(agent.agent_config.user_prompt, PROMPT_CHARS), "steps": steps, "file": file,
+    }
+
+
+def _title(agent: "Agent") -> str:
+    task = agent.agent_config.dataset_task
+    return _agent_label(agent) + (f" · task {task.task_id[:12]}" if task is not None else "") + f" · seed {agent.run_results.seed}"
+
+
+def _stats(agent: "Agent") -> str:
+    results = agent.run_results
+    parts = [f"turn {results.num_turns}", f"{results.num_input_tokens} in ({results.num_cached_input_tokens} cached) / {results.num_output_tokens} out"]
+    if agent.finished:
+        parts.append(results.finish_reason)
+        parts.append(f"answer {str(results.answer)[:40]!r}")
+        if agent.agent_config.dataset_task is not None and results.score_feedback != "" or results.score:
+            parts.append(f"score {results.score:.2f}")
+    return " · ".join(parts)
+
+
+def _agent_label(agent: "Agent") -> str:
+    return agent.agent_id[:8] + (" (sub)" if agent.parent_agent is not None else "")
+
+
+def _arguments(arguments: dict) -> str:
+    """Multi-line string arguments (code, scripts) verbatim; everything else as JSON."""
+    lines = []
+    for key, value in arguments.items():
+        if isinstance(value, str) and ("\n" in value or len(value) > 60):
+            lines.append(f"{key}:\n{value}")
+        else:
+            lines.append(f"{key}: {json.dumps(value, default=str)}")
+    return "\n".join(lines)
+
+
+def _cut(text: str, limit: int) -> str:
+    text = str(text)
+    if len(text) <= limit:
+        return text
+    return text[:limit] + f"\n… [{len(text) - limit} more chars; the uncut text is in the trajectory file]"
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/tests/test_basic_agent.py</span>
    <span class="card-oneliner">The two key tests.</span>
    <span class="card-badge">Diff</span>
  </summary>

Exact delta vs `/source` for `activation/tests/test_basic_agent.py`:

```diff
--- a/activation/tests/test_basic_agent.py
+++ b/activation/tests/test_basic_agent.py
@@ -1,22 +1,140 @@
+"""
+Two end-to-end agent rollouts on a GPU node.
+
+    uv run sky exec --sync <node> -- uv run pytest activation/tests/test_basic_agent.py --gpu --slow -s
+
+The reporters write under the synced folder, so IB/TMP/SYNC/AGENT_TEST/{primes,dapo}/report.tressoir.html
+morphs here every 15 s while the node runs.
+"""
+import json
+import os
+import random
+from dataclasses import replace
+
+import pytest
+
+from activation.agent import AgentConfig, RolloutReporter
+from activation.common.data_syncing import resolve_path
+from activation.dataset.loaders import DapoMathDataset
+from activation.harness import FREE_DEVICE, HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+MODEL_NAME = "qwen3.5-9b"
+MODEL_ID = "Qwen/Qwen3.5-9B"
+
 BASE_AGENT_CONFIG = AgentConfig(
-    # AgentConfig with just the default tools.
-    # Based on Qwen/Qwen3.5-9B.
-    # Limit to 10 steps. No thinking.
+    # Default tools only (shell, python, parallel_tool_call, submit_answer). Thinking is off by the harness default.
+    system_prompt=(
+        "You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
+        "rather than guessing. When you are done, call submit_answer exactly once with the final answer."
+    ),
+    model_name=MODEL_NAME,
+    call_kwargs={"sampling_params": {"max_tokens": 2048, "temperature": 0.7}},
+    max_turns=10,
+    max_tool_errors=3,
+    max_duration=300,
 )
 
+
+def _make_harness() -> HarnessRuntime:
+    return HarnessRuntime(HarnessRuntimeConfig(
+        model_configs={MODEL_NAME: ModelConfig(MODEL_NAME, MODEL_ID)},
+        agent_max_concurrent=8,
+    ))
+
+
+def _print_rollout(result) -> None:
+    print(
+        f"\n=== rollout seed={result.seed} finish={result.finish_reason} turns={result.num_turns} "
+        f"tokens in/cached/out={result.num_input_tokens}/{result.num_cached_input_tokens}/{result.num_output_tokens} "
+        f"duration={result.duration:.1f}s score={result.score:.2f} answer={result.answer!r}"
+    )
+    for step in result.trajectory:
+        if step["role"] == "assistant":
+            calls = ", ".join(f"{call['name']}({json.dumps(call['arguments'])[:80]})" for call in step["tool_calls"])
+            print(f"  assistant: {step['content'][:200]!r} -> {calls}")
+        else:
+            for output in step["tool_call_results"]:
+                print(f"  tool: {output[:200]!r}")
+
+
+@pytest.mark.gpu
+@pytest.mark.slow
 def test_basic_agent_primes():
-    # @AI: key test: give it to me in full.
-    ... # Simple task to compute the sum of the 140th, 141st and 142nd prime numbers.
-    ... # rollout group size 2, with reporting. No attached dataset.
+    """A self-contained task, no dataset: a group of 2 rollouts with the live report."""
+    harness = _make_harness()
+    config = replace(
+        BASE_AGENT_CONFIG,
+        user_prompt="Compute the sum of the 140th, 141st and 142nd prime numbers (2 is the 1st prime).",
+    )
+    reporter = RolloutReporter(
+        str(resolve_path("AGENT_TEST/primes")),
+        title="Agent test: primes",
+        description=f"{MODEL_ID}, group of 2, {config.max_turns} turns max, default tools.",
+    )
+    try:
+        groups = harness.rollout_manager.perform_grouped_rollouts(
+            [config], group_count=2, base_seed=0, perform_scoring=False, reporter=reporter,
+        )
+    finally:
+        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)
+
+    assert len(groups) == 1 and len(groups[0]) == 2
+    for result in groups[0]:
+        _print_rollout(result)
+        assert result.finish_reason == "submitted", result.finish_reason
+        assert any(
+            call["name"] in ("python", "shell", "parallel_tool_call")
+            for step in result.trajectory for call in step.get("tool_calls", [])
+        ), "no tool was used"
+        assert str(result.answer).replace(",", "").strip().lstrip("-").isdigit(), result.answer
+    answers = [int(str(result.answer).replace(",", "").strip()) for result in groups[0]]
+    print(f"answers {answers}; expected 2441 (809 + 811 + 821)")
+    assert 2441 in answers, answers  # at least one of the two rollouts gets it (drop this line to print only)
+    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
+    assert os.path.exists(os.path.join(reporter.report_folder, "report_data.json"))
+    print("\n=== harness stats ===")
+    print(json.dumps(harness.harness_stats.summarize(), indent=2))
 
+
+@pytest.mark.gpu
+@pytest.mark.slow
 def test_basic_agent_dapo():
-    # @AI: key test: give it to me in full.
-    harness = ...
-    dapo_math_dataset = ...
-    dataset_tasks = ... # Shuffle, then select 2
-    rollout_manager = harness.rollout_manager
-    rollout_reporter = ... # Allow me to live view the rollouts.
-    rollout_manager.perform_grouped_rollouts(
-        ..., # group size 2
+    """Two DAPO-Math tasks, a group of 2 each, scored; then the same call served from the cache."""
+    harness = _make_harness()
+    dataset = DapoMathDataset.load(harness, max_examples=64)
+    harness.dataset_manager.register_dataset(dataset)
+    tasks = list(dataset.scorable_tasks.values())
+    random.Random(0).shuffle(tasks)
+    tasks = tasks[:2]
+    configs = [replace(BASE_AGENT_CONFIG, user_prompt=task.agent_prompt, dataset_task=task) for task in tasks]
+    reporter = RolloutReporter(
+        str(resolve_path("AGENT_TEST/dapo")),
+        title="Agent test: DAPO-Math",
+        description=f"{MODEL_ID}, 2 tasks x group of 2, {BASE_AGENT_CONFIG.max_turns} turns max, default tools.",
     )
-    # Some display. Problem need not be solved.
\ No newline at end of file
+    try:
+        groups = harness.rollout_manager.perform_grouped_rollouts(
+            configs, group_count=2, base_seed=0, perform_scoring=True, caching_id="agent_test_dapo", reporter=reporter,
+        )
+        loads_before = len(harness.harness_stats.model_loading_times)
+        # The same request again: every rollout comes from the cache, the engine is not touched.
+        cached = harness.rollout_manager.perform_grouped_rollouts(
+            configs, group_count=2, base_seed=0, perform_scoring=True, caching_id="agent_test_dapo",
+        )
+    finally:
+        harness.loaded_models[MODEL_NAME].engine_to_device(FREE_DEVICE)
+
+    assert len(groups) == 2 and all(len(group) == 2 for group in groups)
+    for task, group in zip(tasks, groups):
+        print(f"\n### task {task.task_id}: gold {task.gold_answer}\n{task.agent_prompt[:300]}")
+        for result in group:
+            _print_rollout(result)  # the problem need not be solved
+            assert result.finish_reason in ("submitted", "max_turns", "max_tool_errors", "max_duration", "no_tool_call")
+            assert result.score in (0.0, 1.0)
+            assert result.agent_config.dataset_task.task_id == task.task_id
+    print(f"scores {[[result.score for result in group] for group in groups]}")
+    assert len(harness.harness_stats.model_loading_times) == loads_before
+    assert [[r.answer for r in group] for group in cached] == [[r.answer for r in group] for group in groups]
+    assert os.path.exists(os.path.join(reporter.report_folder, "report.tressoir.html"))
+    print("\n=== harness stats ===")
+    print(json.dumps(harness.harness_stats.summarize(), indent=2))
```

</details>

<details class="card" data-tressoir-markdown open>
  <summary>
    <span class="card-title">activation/bench/agent_probes/dapo_bench.py</span>
    <span class="card-oneliner">DAPO success-rate probe: N problems x group G, summary JSON with the mixed-outcome count.</span>
    <span class="card-badge">Diff</span>
  </summary>

New file vs `/source` for `activation/bench/agent_probes/dapo_bench.py`:

```diff
new file mode 100644
--- /dev/null
+++ b/activation/bench/agent_probes/dapo_bench.py
@@ -0,0 +1,103 @@
+"""
+DAPO-Math success-rate probe: N problems x a group of G rollouts with one model, scored exactly.
+Answers two questions: how often does the agent solve a problem, and is there a learning signal (groups
+with mixed outcomes are what group-relative training learns from). The live page and the uncut
+trajectories go to the synced folder (`DAPO_BENCH/<tag>`), the rollouts to the cache
+(`ROLLOUTS/dapo_bench_<tag>`), and a summary JSON next to the page.
+
+  uv run sky exec --sync <node> -- uv run python -m activation.bench.agent_probes.dapo_bench --model Qwen/Qwen3.5-9B --problems 100 --group 2
+"""
+import argparse
+import collections
+import json
+import random
+import time
+from dataclasses import replace
+
+from activation.agent import AgentConfig, RolloutReporter
+from activation.common.data_syncing import resolve_path
+from activation.dataset.loaders import DapoMathDataset
+from activation.harness import HarnessRuntime, HarnessRuntimeConfig, ModelConfig
+
+SYSTEM_PROMPT = (
+    "You are a careful problem solver working in a sandbox. Use the python or shell tools to compute "
+    "rather than guessing. When you are done, call submit_answer exactly once with the final answer."
+)
+
+
+def main() -> None:
+    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
+    parser.add_argument("--model", default="Qwen/Qwen3.5-9B", help="HF model id")
+    parser.add_argument("--problems", type=int, default=100, help="distinct problems")
+    parser.add_argument("--group", type=int, default=2, help="rollouts per problem")
+    parser.add_argument("--pool", type=int, default=1000, help="unique problems loaded before sampling")
+    parser.add_argument("--seed", type=int, default=0, help="problem sampling seed and rollout base seed")
+    parser.add_argument("--max-turns", type=int, default=10)
+    parser.add_argument("--max-tokens", type=int, default=2048, help="per turn")
+    parser.add_argument("--temperature", type=float, default=0.7)
+    parser.add_argument("--concurrent", type=int, default=64, help="agents in flight")
+    parser.add_argument("--tag", default=None, help="report / cache tag (default: from the model id)")
+    args = parser.parse_args()
+
+    tag = args.tag or args.model.split("/")[-1].lower()
+    model_name = tag
+    harness = HarnessRuntime(HarnessRuntimeConfig(
+        model_configs={model_name: ModelConfig(model_name, args.model)},
+        agent_max_concurrent=args.concurrent,
+    ))
+    dataset = DapoMathDataset.load(harness, max_examples=args.pool)
+    harness.dataset_manager.register_dataset(dataset)
+    tasks = list(dataset.scorable_tasks.values())
+    random.Random(args.seed).shuffle(tasks)
+    tasks = tasks[:args.problems]
+
+    base = AgentConfig(
+        system_prompt=SYSTEM_PROMPT, model_name=model_name,
+        call_kwargs={"sampling_params": {"max_tokens": args.max_tokens, "temperature": args.temperature}},
+        max_turns=args.max_turns, max_tool_errors=3, max_duration=600,
+    )
+    configs = [replace(base, user_prompt=task.agent_prompt, dataset_task=task) for task in tasks]
+    report_folder = resolve_path(f"DAPO_BENCH/{tag}")
+    reporter = RolloutReporter(
+        str(report_folder), title=f"DAPO-Math probe: {args.model}",
+        description=f"{len(tasks)} problems x group of {args.group}, {args.max_turns} turns max, temperature {args.temperature}, default tools.",
+    )
+    started = time.time()
+    groups = harness.rollout_manager.perform_grouped_rollouts(
+        configs, group_count=args.group, base_seed=args.seed, perform_scoring=True,
+        caching_id=f"dapo_bench_{tag}", reporter=reporter,
+    )
+    elapsed = time.time() - started
+
+    # ------------------------------------------------------------------------------- summary
+    rollouts = [result for group in groups for result in group]
+    per_problem = [sum(result.score for result in group) / len(group) for group in groups]
+    solved_any = sum(1 for rate in per_problem if rate > 0)
+    solved_all = sum(1 for rate in per_problem if rate == 1.0)
+    mixed = sum(1 for rate in per_problem if 0 < rate < 1)
+    finish = collections.Counter(result.finish_reason for result in rollouts)
+    summary = {
+        "model": args.model, "problems": len(tasks), "group": args.group, "rollouts": len(rollouts),
+        "accuracy": sum(result.score for result in rollouts) / max(1, len(rollouts)),
+        "problems_solved_by_any_rollout": solved_any, "problems_solved_by_every_rollout": solved_all,
+        "problems_with_mixed_outcomes": mixed,   # the learning signal for group-relative training
+        "problems_never_solved": len(tasks) - solved_any,
+        "finish_reasons": dict(finish),
+        "mean_turns": sum(result.num_turns for result in rollouts) / max(1, len(rollouts)),
+        "mean_output_tokens": sum(result.num_output_tokens for result in rollouts) / max(1, len(rollouts)),
+        "mean_input_tokens": sum(result.num_input_tokens for result in rollouts) / max(1, len(rollouts)),
+        "mean_duration_s": sum(result.duration for result in rollouts) / max(1, len(rollouts)),
+        "wall_clock_s": elapsed,
+        "per_problem": [
+            {"task_id": task.task_id, "gold": task.gold_answer, "scores": [result.score for result in group],
+             "answers": [result.answer for result in group], "finish": [result.finish_reason for result in group]}
+            for task, group in zip(tasks, groups)
+        ],
+    }
+    (report_folder / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
+    print(json.dumps({key: value for key, value in summary.items() if key != "per_problem"}, indent=1))
+    print(f"summary: {report_folder / 'summary.json'}")
+
+
+if __name__ == "__main__":
+    main()
```

</details>
