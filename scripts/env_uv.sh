#!/usr/bin/env bash

# Configure uv to use project-local writable directories.
# Usage:
#   source scripts/env_uv.sh

_env_uv_script_path=""
_env_uv_is_sourced=0

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _env_uv_script_path="${BASH_SOURCE[0]}"
  [[ "${BASH_SOURCE[0]}" != "${0}" ]] && _env_uv_is_sourced=1
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _env_uv_script_path="${(%):-%N}"
  case "${ZSH_EVAL_CONTEXT:-}" in
    *:file) _env_uv_is_sourced=1 ;;
  esac
fi

if [[ -z "${_env_uv_script_path}" ]]; then
  echo "could not determine script path"
  return 1 2>/dev/null || exit 1
fi

if [[ "${_env_uv_is_sourced}" -ne 1 ]]; then
  echo "source this script instead of executing it:"
  echo "  source scripts/env_uv.sh"
  exit 1
fi

_env_uv_root="$(cd "$(dirname "${_env_uv_script_path}")/.." && pwd)"

export HOME="${_env_uv_root}/.uv-home"
export XDG_CACHE_HOME="${_env_uv_root}/.uv-cache"
export XDG_DATA_HOME="${_env_uv_root}/.uv-data"
export XDG_STATE_HOME="${_env_uv_root}/.uv-state"
export UV_CACHE_DIR="${XDG_CACHE_HOME}"

mkdir -p \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${XDG_DATA_HOME}" \
  "${XDG_STATE_HOME}"

echo "uv environment configured"
echo "  HOME=${HOME}"
echo "  XDG_CACHE_HOME=${XDG_CACHE_HOME}"
echo "  XDG_DATA_HOME=${XDG_DATA_HOME}"
echo "  XDG_STATE_HOME=${XDG_STATE_HOME}"
echo "  UV_CACHE_DIR=${UV_CACHE_DIR}"

if command -v uv >/dev/null 2>&1; then
  echo "  uv=$(command -v uv)"
else
  echo "  uv not found in PATH"
fi

unset _env_uv_root
unset _env_uv_script_path
unset _env_uv_is_sourced
