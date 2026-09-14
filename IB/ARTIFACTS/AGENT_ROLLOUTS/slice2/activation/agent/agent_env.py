"""
Simple, podman-based agent environment: one `--rm` container per env, every operation a `podman exec`
with the code or script on stdin. Rootless podman on a workstation (see README), rootful podman inside a
privileged Sky node container. Truncation of outputs is the tool layer's job, not the env's.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import uuid
from pathlib import Path

DEFAULT_IMAGE = "docker.io/library/python:3.12-slim"
OUTPUT_CAPTURE_LIMIT_CHARS = 4_000_000   # a command's combined output beyond this keeps its head and tail (a runaway print once produced 2 GB)
DEFAULT_MEMORY_LIMIT_MB = 8192   # sandbox memory: `podman run --memory` where cgroups exist (hard kill), ulimit -v at 85% per process (MemoryError first)
SOFT_LIMIT_FRACTION = 0.85
EXEC_GRACE_SECONDS = 10   # how long the client waits past the in-container `timeout` before killing itself


def podman_available() -> bool:
    return shutil.which("podman") is not None


class AgentEnv:
    """
    Simple, podman-based environment.
    """
    hard_limits_available = True   # class-wide: cleared the first time `podman run --memory` is refused for lack of cgroups

    def __init__(self, dockerfile_path: str | None = None, env_args: dict[str, str] | None = None,
                 default_image: str = DEFAULT_IMAGE, memory_limit_mb: int | None = DEFAULT_MEMORY_LIMIT_MB):
        if not podman_available():
            raise RuntimeError("podman is not installed; see README.md (rootless podman) or the Sky image.")
        self.env_id = uuid.uuid4().hex[:12]
        self.memory_limit_mb = memory_limit_mb   # None disables both limits
        self.image = self._build_image(dockerfile_path) if dockerfile_path else default_image
        run_args = []
        for key, value in (env_args or {}).items():
            run_args.append(key)
            if value not in (None, ""):
                run_args.append(str(value))
        # Always --rm so a lost handle still leaves nothing behind; no network unless env_args ask for it.
        argv = ["podman", "run", "-d", "--rm", "--name", f"agent-env-{self.env_id}"]
        if not any(arg.startswith("--network") or arg == "--net" for arg in run_args):
            argv += ["--network", "none"]
        # The hard limit is a container cgroup; inside Sky nodes podman has no cgroups, so the first failure
        # mentioning them turns the flag off for every later env in this process and the soft limit carries.
        hard_limit = []
        if self.memory_limit_mb and AgentEnv.hard_limits_available and not any(arg.startswith("--memory") or arg == "-m" for arg in run_args):
            hard_limit = ["--memory", f"{int(self.memory_limit_mb)}m"]
        tail = run_args + [self.image, "sleep", "infinity"]
        completed = subprocess.run(argv + hard_limit + tail, capture_output=True, text=True)
        if completed.returncode != 0 and hard_limit and "cgroup" in completed.stderr.lower():
            AgentEnv.hard_limits_available = False
            completed = subprocess.run(argv + tail, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"podman run failed: {completed.stderr.strip()}")
        self.container_id: str | None = completed.stdout.strip()

    @staticmethod
    def _build_image(dockerfile_path: str) -> str:
        """Built once per Dockerfile content; the tag is the content hash."""
        path = Path(dockerfile_path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        tag = f"localhost/activation-env:{digest}"
        exists = subprocess.run(["podman", "image", "exists", tag], capture_output=True)
        if exists.returncode != 0:
            print(f"AgentEnv - Building {tag} from {path}.", flush=True)
            build = subprocess.run(["podman", "build", "-t", tag, "-f", str(path), str(path.parent)], capture_output=True, text=True)
            if build.returncode != 0:
                raise RuntimeError(f"podman build failed: {build.stderr.strip()[-2000:]}")
        return tag

    # ----------------------------------------------------------------------------- operations
    def _exec(self, argv: list[str], stdin: str | None, timeout: int | None) -> tuple[str, bool]:
        """
        Run inside the container. Combined stdout/stderr and ok. On timeout the in-container `timeout`
        kills the command first (so partial output is still read), then the client gives up.
        """
        assert self.container_id, "environment was shut down"
        command = ["podman", "exec", "-i", self.container_id]
        inner: list[str] = []
        if timeout is not None:
            inner += ["timeout", "-s", "KILL", str(int(timeout))]
        inner += argv
        if self.memory_limit_mb:
            # Soft limit at 85% of the sandbox memory: the process gets MemoryError (which the model reads)
            # before the container's hard limit, where there is one, kills it without a word.
            soft_kb = int(self.memory_limit_mb * SOFT_LIMIT_FRACTION) * 1024
            command += ["sh", "-c", f'ulimit -v {soft_kb} && exec "$@"', "sh"]
        command += inner
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            output, _ = process.communicate(stdin, timeout=None if timeout is None else timeout + EXEC_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
            return _cap_output(output or "") + f"\n[timed out after {timeout} s]", False
        output = _cap_output(output or "")
        if process.returncode == 137 and timeout is not None:
            return output + f"\n[timed out after {timeout} s]", False
        return output, process.returncode == 0

    def run_python_code(self, code: str, timeout: int | None = None) -> tuple[str, bool]:
        """Runs python code (no session persistence). Returns combined stdout/stderr and ok; partial output on timeout."""
        return self._exec(["python3", "-"], code, timeout)

    def run_shell(self, script: str, timeout: int | None = None) -> tuple[str, bool]:
        """Runs a bash script. Returns combined stdout/stderr and ok; partial output on timeout."""
        return self._exec(["bash", "-s"], script, timeout)

    def write_file(self, path: str, content: str) -> bool:
        _, ok = self._exec(["sh", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "sh", path], content, 60)
        return ok

    def read_file(self, path: str) -> tuple[str, bool]:
        return self._exec(["cat", path], None, 60)

    def shutdown(self) -> None:
        """Idempotent; the container is --rm so `rm -f` both stops and removes it."""
        container_id, self.container_id = self.container_id, None
        if container_id:
            subprocess.run(["podman", "rm", "-f", container_id], capture_output=True)

    def __del__(self):
        """Auto-delete container"""
        try:
            self.shutdown()
        except Exception:
            pass


def _cap_output(text: str, limit: int = OUTPUT_CAPTURE_LIMIT_CHARS) -> str:
    """Head and tail of a runaway output; nothing downstream (activation context, cache, reports) has to carry more."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [output capped: {len(text) - limit} chars dropped by the sandbox] ...\n{text[-half:]}"
