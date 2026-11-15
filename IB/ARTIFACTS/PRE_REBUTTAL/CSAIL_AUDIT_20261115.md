# CSAIL check before new jobs (2026-11-15, after TIG reopened slurm-login)

TIG asked users to check every file an agent or shell might read or execute before running jobs. Checked over the guarded
connection (`IB/CANON/CANON_ARTIFACTS/cluster_access/csail_ssh.sh`, via `jobs/cluster_lib.sh`):

- **AFS home** `/afs/csail.mit.edu/u/n/ngom`: no shell startup files (`.bashrc`, `.profile`, … absent); `.ssh` unreadable without
  an AFS token (non-interactive); `crontab` not permitted for the user. No Slurm jobs of ours queued or running.
- **Our scratch** (`/data/scratch-fast/ngom`, `/data/scratch/ngom`): nothing changed after 2026-11-11 03:00 UTC (both by
  modification time and by inode change time, which userspace cannot set) except the outputs of our own runs
  `S_kv_fr90k_s1` / `S_kv_fr90k_s2` (jobs 3021417/3021418) and their parent folders. The recent code-like files are our own
  uploads of 2026-11-09/11 (`sec35/code/sec35_code_v1`, `sec35/jobs`, `sec45/jobs/generated_v3`).
- **Python environment** `/data/scratch/ngom/activation/venv` (uv 0.12.16, CPython 3.14 from `.../uv_python`): no file modified
  in the last 30 days; its `.pth` files are the standard four (`_virtualenv`, `distutils-precedence`, `nvidia_cutlass_dsl_packages`,
  and the editable install pointing at `/data/scratch/ngom/activation/ws`).

No sign of tampering. Still, because the incident window is not known to us, new CSAIL jobs will **not** execute anything that
was already there:
1. a fresh Python environment built from the repo's `uv.lock` with a freshly uploaded `uv` and a new uv cache and Python;
2. code bundles and job scripts uploaded fresh and checked by sha256 after upload;
3. a fresh Hugging Face cache (weights re-downloaded);
4. no CSAIL-produced `.pt` pairs or pickles loaded (standing rule); starting pairs come from S3 (EC2-trained).
