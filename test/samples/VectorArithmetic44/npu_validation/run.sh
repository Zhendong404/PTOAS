#!/usr/bin/env bash
set -euo pipefail

RUN_MODE="${RUN_MODE:-npu}"
SOC_VERSION="${SOC_VERSION:-Ascend950}"
BUILD_DIR="${BUILD_DIR:-build}"
GENERATOR="${GENERATOR:-Ninja}"
RUN_EXEC="${RUN_EXEC:-1}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_KERNEL_CPP="${PTO_KERNEL_CPP:-${ROOT_DIR}/../../../../build/output/VectorArithmetic44/vector_arith_44_cases-pto.cpp}"
WORKSPACE_DIR_DEFAULT="$(cd "${ROOT_DIR}/../../../../../" && pwd)"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-${WORKSPACE_DIR:-${WORKSPACE_DIR_DEFAULT}}/pto-isa}"
ASCEND_RUNTIME_DIR="${ASCEND_RUNTIME_DIR:-${ROOT_DIR}/.ascend_runtime}"
ASCEND_PROCESS_LOG_PATH="${ASCEND_PROCESS_LOG_PATH:-${ASCEND_RUNTIME_DIR}/log}"

if [[ ! -f "${PTO_KERNEL_CPP}" ]]; then
  echo "[ERROR] generated kernel source not found: ${PTO_KERNEL_CPP}" >&2
  echo "[ERROR] please generate build/output/VectorArithmetic44/vector_arith_44_cases-pto.cpp first" >&2
  exit 1
fi

cd "${ROOT_DIR}"
python3 "${ROOT_DIR}/golden.py"

mkdir -p "${ASCEND_PROCESS_LOG_PATH}"
export ASCEND_PROCESS_LOG_PATH

rm -rf "${ROOT_DIR:?}/${BUILD_DIR}"
mkdir -p "${ROOT_DIR}/${BUILD_DIR}"

cd "${ROOT_DIR}/${BUILD_DIR}"
cmake -G "${GENERATOR}" \
  -DRUN_MODE="${RUN_MODE}" \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DPTO_KERNEL_CPP="${PTO_KERNEL_CPP}" \
  -DPTO_ISA_ROOT="${PTO_ISA_ROOT}" \
  ..
cmake --build . -j

echo "[INFO] build passed: ${ROOT_DIR}/${BUILD_DIR}/vector_arith_44_cases"

if [[ "${RUN_EXEC}" != "1" ]]; then
  echo "[INFO] skip execution and compare (set RUN_EXEC=1 to enable)"
  exit 0
fi

cd "${ROOT_DIR}"
"${ROOT_DIR}/${BUILD_DIR}/vector_arith_44_cases"

python3 "${ROOT_DIR}/compare.py"
