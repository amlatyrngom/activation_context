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
Make sure you populate `~/.aws/credentials` or a custom `AWS_PROFILE`.
```bash
uv run sky help
```
