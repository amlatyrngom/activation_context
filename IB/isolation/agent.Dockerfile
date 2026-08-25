FROM mcr.microsoft.com/devcontainers/javascript-node:24-bookworm

USER root

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv and its managed CPython independently of the development user's
# home so the toolchain is already available when either container starts.
RUN curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-installer.sh \
    && UV_UNMANAGED_INSTALL=/usr/local/bin sh /tmp/uv-installer.sh \
    && rm /tmp/uv-installer.sh

ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python

RUN uv python install 3.14 \
    && ln -sf "$(uv python find 3.14)" /usr/local/bin/python3.14 \
    && ln -sf /usr/local/bin/python3.14 /usr/local/bin/python

# These are the configured agent harnesses, baked into the image rather than
# fetched each time a workspace is created.
RUN npm install --global --no-audit --no-fund \
      @anthropic-ai/claude-code@latest \
      @openai/codex@latest \
      @earendil-works/pi-coding-agent@latest \
    && npm cache clean --force

USER node
