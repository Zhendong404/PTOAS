#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/scripts/ptoas_env.sh"

llvm_lit_bin="${LLVM_LIT_BIN:-$(command -v llvm-lit || true)}"
if [[ -z "${llvm_lit_bin}" ]]; then
  echo "error: missing llvm-lit in PATH; source scripts/ptoas_env.sh first" >&2
  exit 1
fi

exec "${llvm_lit_bin}" -sv "${repo_root}/test/phase2" "$@"
