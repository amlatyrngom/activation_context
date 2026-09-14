# Activation Context
TBD

# Pre-requisite Installs
Installing `uv`.
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Installing `docker`.
```bash
# Installation
curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
sudo sh /tmp/get-docker.sh

# User Permissions
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world
```

Installing `node` and `npm`.
```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.bashrc
nvm install --lts
nvm alias default 'lts/*'
node --version
npm --version
```

Cloning from GitHub.
```bash
(type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& sudo mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y



# Configuring git
sed -n 's/^GITHUB_TOKEN=//p' .env |
  head -n 1 |
  env -u GITHUB_TOKEN -u GH_TOKEN \
  gh auth login --hostname github.com --git-protocol https --with-token
gh auth setup-git
gh auth status

# Cloning
git clone https://github.com/amlatyrngom/activation_context.git
```


# SkyPilot GPU nodes

Make sure `~/.aws/credentials` is populated, or set `AWS_PROFILE` for a custom
profile. SkyPilot manages the instance and SSH keys; no `.pem` file is needed.

The runtime is a project image based on NVIDIA's CUDA development image. It
contains `nvcc` and the dependencies locked by `uv.lock`. `sky image` builds the
lock-tagged image with local Docker and pushes it to the project's private ECR
repository. SkyPilot then pulls that image on the remote VM; it does not upload
local Docker image bytes directly.

Set `ACTIVATION_SKY_IMAGE` to a full `docker:REGISTRY/REPOSITORY:TAG`
reference when using another AWS account or registry.

```bash
uv run sky image
uv run sky setup --name ac-test --gpu l40s --gpu-count 1
```

`--gpu-count N` requests N GPUs on one node; the harness then runs one vLLM
engine per visible GPU (`VLLMWrapper`, no extra flag). On AWS, `--gpu
rtx6000pro --gpu-count 2` lands on `g7e.12xlarge` and `--gpu-count 4` on
`g7e.24xlarge`; the L40S 4× shape is `g6e.12xlarge`. Multi-GPU capacity is
scarcer than single-GPU, so expect more region shopping during setup.

`sky exec` always performs an exact local-to-remote source upload before it
submits the command. A failed upload prevents execution. A stopped managed
cluster is started with `--retry-until-up`, so transient AWS capacity errors
keep retrying instead of aborting the command. Restart stays in the cluster's
existing zone; press Ctrl-C to stop waiting, or use teardown and a fresh setup
when relocation is required.

If a project-root `.env` exists, `sky exec` passes it to SkyPilot with
`--secret-file`. Values are injected into the trusted remote job environment;
the file remains excluded from the exact rsync mirror and is not written to
`~/sky_workdir`. SkyPilot protects transport and its normal secret surfaces,
but job code can still print its environment, so do not echo secrets or write
them into artifacts. When `.env` is absent, execution proceeds without secret
injection.

Remote commands also receive `VLLM_WORKER_MULTIPROC_METHOD=spawn`. This avoids
CUDA re-initialization failures when the project has inspected CUDA before
vLLM starts its engine worker.

Use the explicit upload command to preview or perform that mirror without
running a job:

```bash
uv run sky upload ac-test --dry-run
uv run sky upload ac-test
uv run sky exec ac-test uv run pytest activation/tests/test_basic_engine.py --gpu
```

`~/sky_workdir` is disposable and exactly owned by the local repository.
Remote-only state belongs outside it. Model/JIT caches use `/root/.cache`, and
checkpoints or results should be written beneath `~/activation_artifacts`.

Watch a run's artifact folder while it writes it. `sky watch` rsyncs the remote folder into a
local `IB/TMP` path every few seconds (changed files replaced, nothing deleted locally), until
Ctrl-C, until a named file appears (the file a run writes last), or until a time limit. A
`.tressoir.html` in the folder morphs in the editor as each sync lands, so a live report on the
node is a live report here. Run it in a second terminal or in the background next to `sky exec`.

```bash
uv run sky watch ac-test '~/activation_artifacts/run-1/' IB/TMP/run-1/ \
  --interval 15 --until-file training_stats.json
```

Download artifacts explicitly. Quote a remote path beginning with `~` so the
local shell does not expand it. Downloads never delete local files and keep
existing files unless `--overwrite` is passed. A destination inside the source
repository must be under `IB/TMP`; destinations outside the repository are also
allowed.

```bash
uv run sky download ac-test \
  '~/activation_artifacts/run-1/' \
  IB/TMP/run-1/
```

Pausing retains the node disk, container, caches, and artifacts. Teardown
deletes them; upload important checkpoints to durable object storage before
teardown.

```bash
uv run sky pause ac-test
uv run sky teardown ac-test
uv run sky help
```
