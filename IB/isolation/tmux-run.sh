#!/usr/bin/env bash

set -Eeuo pipefail

readonly PROGRAM_NAME="${0##*/}"
readonly SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}")
readonly HOLD_COMMAND_FLAG="--tmux-run-hold-command"
readonly HISTORY_LIMIT=50000

usage() {
  cat <<EOF
Usage:
  ${PROGRAM_NAME} SESSION -- COMMAND [ARG ...]

Create SESSION, run COMMAND inside it, and attach to it. If SESSION already
exists, attach to that session instead; the supplied command is not run again.
When COMMAND exits, its output and status remain visible until you press Enter.
Mouse-wheel scrolling is enabled, with up to ${HISTORY_LIMIT} lines of scrollback.

tmux keeps the command alive when its terminal or VS Code window disconnects,
provided this Dev Container remains running. It cannot keep work running while
a local laptop is asleep, or across a container stop, rebuild, or deletion.

SESSION may contain letters, numbers, underscores, and hyphens.
EOF
}

fail() {
  printf '%s: %s\n' "$PROGRAM_NAME" "$*" >&2
  printf 'Run %s --help for usage.\n' "$PROGRAM_NAME" >&2
  exit 2
}

hold_command_result() {
  local command_name=${1##*/}
  local command_status

  set +e
  "$@"
  command_status=$?
  set -e

  stty sane 2>/dev/null || true
  printf '\n%s: command %q exited with status %d.\n' \
    "$PROGRAM_NAME" "$command_name" "$command_status" >&2
  printf 'Press Enter to close this tmux session.\n' >&2
  IFS= read -r _ || true
  exit "$command_status"
}

attach_to_session() {
  local target_name=$1

  if [[ -n ${TMUX:-} ]]; then
    tmux switch-client -t "$target_name"
  else
    exec tmux attach-session -t "$target_name"
  fi
}

configure_session() {
  local target_name=$1

  tmux set-option -t "$target_name" mouse on
  tmux set-window-option -t "${target_name}:" history-limit "$HISTORY_LIMIT"
}

if [[ ${1:-} == "$HOLD_COMMAND_FLAG" ]]; then
  shift
  (( $# > 0 )) || fail "internal command wrapper received no command"
  hold_command_result "$@"
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

(( $# >= 3 )) || fail "expected SESSION -- COMMAND [ARG ...]"

readonly session_name=$1
shift

[[ $session_name =~ ^[[:alnum:]_-]+$ ]] \
  || fail "invalid session name ${session_name@Q}"
[[ $1 == "--" ]] || fail "expected -- after the session name"
shift
(( $# > 0 )) || fail "expected a command after --"

command -v tmux >/dev/null 2>&1 \
  || fail "tmux is not installed; rebuild the Dev Container from the current image"

if tmux has-session -t "$session_name" 2>/dev/null; then
  printf '%s: session %q already exists; attaching without rerunning the command\n' \
    "$PROGRAM_NAME" "$session_name" >&2
  configure_session "$session_name"
  attach_to_session "$session_name"
  exit 0
fi

printf '%s: creating session %q in %q\n' \
  "$PROGRAM_NAME" "$session_name" "$PWD" >&2

tmux new-session -d -s "$session_name" -c "$PWD" -- \
  "$SCRIPT_PATH" "$HOLD_COMMAND_FLAG" "$@"
configure_session "$session_name"
attach_to_session "$session_name"
