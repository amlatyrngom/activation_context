#!/usr/bin/env bash

set -euo pipefail

fail() {
  printf 'Tressoir isolation: %s\n' "$1" >&2
  exit 1
}

prepare_development_user() {
  [ "$(id -u)" -ne 0 ] || fail "the development user must not be root"
  sudo -n true >/dev/null 2>&1 ||
    fail "the development user needs passwordless sudo"

  sudo chown -R "$(id -u):$(id -g)" "$HOME" ||
    fail "cannot repair ownership of $HOME"
  install -d \
    "$HOME/.cache" \
    "$HOME/.config" \
    "$HOME/.local/bin" \
    "$HOME/.local/share" \
    "$HOME/.local/state" ||
    fail "cannot prepare development-user directories"

  [ -w "$HOME" ] || fail "$HOME is not writable"
  [ -w "${XDG_CACHE_HOME:-$HOME/.cache}" ] ||
    fail "the development-user cache is not writable"
}

install_selected_harnesses() {
  local target
  local -a packages commands
  packages=()
  commands=()

  if [ -n "${NVM_DIR:-}" ] && [ -f "$NVM_DIR/nvm.sh" ]; then
    # Dev Container lifecycle commands are non-interactive, so nvm is not
    # automatically sourced even though the Node feature has installed it.
    # shellcheck disable=SC1090
    . "$NVM_DIR/nvm.sh"
  fi
  command -v node >/dev/null 2>&1 || fail "Node.js is not available"
  command -v npm >/dev/null 2>&1 || fail "npm is not available"

  for target in claude codex pi; do
    case ",${TRESSOIR_AGENT_TARGETS:-}," in
      *",$target,"*)
        case "$target" in
          claude)
            packages+=("@anthropic-ai/claude-code@latest")
            commands+=("claude")
            ;;
          codex)
            packages+=("@openai/codex@latest")
            commands+=("codex")
            ;;
          pi)
            packages+=("@earendil-works/pi-coding-agent@latest")
            commands+=("pi")
            ;;
        esac
        ;;
    esac
  done

  if [ "${#packages[@]}" -eq 0 ]; then
    printf '%s\n' "Tressoir isolation: no agent CLIs selected"
    return
  fi

  npm install --global --no-audit --no-fund "${packages[@]}" ||
    fail "could not install the selected agent CLIs"
  for target in "${commands[@]}"; do
    command -v "$target" >/dev/null 2>&1 ||
      fail "the selected $target CLI is not on PATH"
  done
}

prepare_agent_workspace() {
  local source_options
  source_options=$(findmnt -no OPTIONS --target /source) ||
    fail "cannot inspect /source"

  case ",$source_options," in
    *,ro,*) ;;
    *) fail "/source is not read-only" ;;
  esac

  [ -w /workspace/IB ] || fail "/workspace/IB is not writable"

  sudo install -d \
    -o "$(id -un)" \
    -g "$(id -gn)" \
    /workspace

  tar -C /source --exclude='./IB' -cf - . |
    tar -C /workspace -xf -
  [ -w /workspace ] || fail "/workspace is not writable"
}

append_isolation_notice() {
  local file
  file="$1"

  if grep -Fq '<!-- TRESSOIR ISOLATION: BEGIN -->' "$file"; then
    return
  fi

  cat >> "$file" <<'EOF'

<!-- TRESSOIR ISOLATION: BEGIN -->
## Tressoir isolation

You are running in an agent container.

- `/source` is the user’s read-only source repository.
- `/workspace` is your container-local working copy.
- `/workspace/IB` is shared with the user.
- Changes outside `IB` do not reach the user.
<!-- TRESSOIR ISOLATION: END -->
EOF
}

setup_agent_guidance() {
  if [ -f /workspace/AGENTS.md ]; then
    append_isolation_notice /workspace/AGENTS.md
  fi

  if [ -f /workspace/CLAUDE.md ]; then
    append_isolation_notice /workspace/CLAUDE.md
  fi
}

case "${1:-}" in
  --user)
    prepare_development_user
    [ -w "$PWD" ] || fail "the user workspace is not writable"
    install_selected_harnesses
    ;;
  --agent)
    prepare_development_user
    prepare_agent_workspace
    setup_agent_guidance
    install_selected_harnesses
    ;;
  *)
    fail "expected --user or --agent"
    ;;
esac
