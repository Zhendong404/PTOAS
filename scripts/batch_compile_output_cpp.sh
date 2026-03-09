#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

SRC_ROOT="${REPO_ROOT}/output"
BUILD_ROOT="${REPO_ROOT}/build/output_obj"
LOG_DIR=""

COMPILER=""
PTO_ISA_PATH=""
EXTRA_ARGS=()

print_usage() {
  cat <<'EOF'
批量编译 output 目录下所有 .cpp 文件并汇总结果。

用法:
  scripts/batch_compile_output_cpp.sh \
    --compiler <编译器路径> \
    --pto-isa-path <PTO-ISA路径> \
    [--compile-arg <单个参数>]... \
    [--src-root <源码目录>] \
    [--build-root <产物目录>] \
    [--log-dir <日志目录>]

参数说明:
  --compiler, -c       编译器路径，例如: /usr/bin/g++ 或 /path/to/ccec
  --pto-isa-path, -p   PTO-ISA 根路径。脚本会自动检测以下 include 目录:
                       1) <PTO-ISA>/include
                       2) <PTO-ISA>
  --compile-arg        额外编译参数。可重复传入，例如:
                       --compile-arg "-std=c++17" --compile-arg "-O2"
  --src-root           要扫描的 .cpp 根目录，默认: <repo>/output
  --build-root         .o 产物目录，默认: <repo>/build/output_obj
  --log-dir            编译日志目录，默认: <build-root>/logs
  --help, -h           显示帮助

示例:
  scripts/batch_compile_output_cpp.sh \
    -c /usr/bin/g++ \
    -p /path/to/pto-isa \
    --compile-arg "-std=c++17" \
    --compile-arg "-O2"
EOF
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --compiler|-c)
      [[ $# -ge 2 ]] || die "--compiler 缺少参数"
      COMPILER="$2"
      shift 2
      ;;
    --pto-isa-path|-p)
      [[ $# -ge 2 ]] || die "--pto-isa-path 缺少参数"
      PTO_ISA_PATH="$2"
      shift 2
      ;;
    --compile-arg)
      [[ $# -ge 2 ]] || die "--compile-arg 缺少参数"
      EXTRA_ARGS+=("$2")
      shift 2
      ;;
    --src-root)
      [[ $# -ge 2 ]] || die "--src-root 缺少参数"
      SRC_ROOT="$2"
      shift 2
      ;;
    --build-root)
      [[ $# -ge 2 ]] || die "--build-root 缺少参数"
      BUILD_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      [[ $# -ge 2 ]] || die "--log-dir 缺少参数"
      LOG_DIR="$2"
      shift 2
      ;;
    --help|-h)
      print_usage
      exit 0
      ;;
    *)
      die "未知参数: $1 (使用 --help 查看用法)"
      ;;
  esac
done

[[ -n "${COMPILER}" ]] || die "必须指定 --compiler"
[[ -n "${PTO_ISA_PATH}" ]] || die "必须指定 --pto-isa-path"
[[ -x "${COMPILER}" ]] || die "编译器不可执行: ${COMPILER}"
[[ -d "${SRC_ROOT}" ]] || die "源码目录不存在: ${SRC_ROOT}"
[[ -d "${PTO_ISA_PATH}" ]] || die "PTO-ISA 路径不存在: ${PTO_ISA_PATH}"

if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${BUILD_ROOT}/logs"
fi

mkdir -p "${BUILD_ROOT}" "${LOG_DIR}" || die "创建目录失败"

INCLUDE_DIRS=()
if [[ -f "${PTO_ISA_PATH}/include/pto/pto-inst.hpp" ]]; then
  INCLUDE_DIRS+=("${PTO_ISA_PATH}/include")
fi
if [[ -f "${PTO_ISA_PATH}/pto/pto-inst.hpp" ]]; then
  INCLUDE_DIRS+=("${PTO_ISA_PATH}")
fi
[[ ${#INCLUDE_DIRS[@]} -gt 0 ]] || die "未找到 pto/pto-inst.hpp，请检查 --pto-isa-path"

declare -a CPP_FILES=()
while IFS= read -r -d '' file; do
  CPP_FILES+=("${file}")
done < <(find "${SRC_ROOT}" -type f -name "*.cpp" -print0 | sort -z)

TOTAL_COUNT=${#CPP_FILES[@]}
[[ ${TOTAL_COUNT} -gt 0 ]] || die "未在 ${SRC_ROOT} 下找到 .cpp 文件"

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_FILES=()

START_TIME="$(date +%s)"

echo "[INFO] 编译器: ${COMPILER}"
echo "[INFO] 源目录: ${SRC_ROOT}"
echo "[INFO] 产物目录: ${BUILD_ROOT}"
echo "[INFO] 日志目录: ${LOG_DIR}"
echo "[INFO] PTO-ISA: ${PTO_ISA_PATH}"
echo "[INFO] include: ${INCLUDE_DIRS[*]}"
echo "[INFO] 文件总数: ${TOTAL_COUNT}"
echo

for src in "${CPP_FILES[@]}"; do
  rel_path="${src#"${SRC_ROOT}/"}"
  obj_path="${BUILD_ROOT}/${rel_path%.cpp}.o"
  log_path="${LOG_DIR}/${rel_path%.cpp}.log"

  mkdir -p "$(dirname -- "${obj_path}")" "$(dirname -- "${log_path}")" || die "创建子目录失败"

  cmd=("${COMPILER}")
  for arg in "${EXTRA_ARGS[@]}"; do
    cmd+=("${arg}")
  done
  for inc in "${INCLUDE_DIRS[@]}"; do
    cmd+=("-I${inc}")
  done
  cmd+=("-c" "${src}" "-o" "${obj_path}")

  echo "[BUILD] ${rel_path}"
  if "${cmd[@]}" >"${log_path}" 2>&1; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    FAIL_COUNT=$((FAIL_COUNT + 1))
    FAILED_FILES+=("${rel_path}")
  fi
done

END_TIME="$(date +%s)"
ELAPSED="$((END_TIME - START_TIME))"

echo
echo "========== 编译汇总 =========="
echo "总文件数 : ${TOTAL_COUNT}"
echo "成功数   : ${SUCCESS_COUNT}"
echo "失败数   : ${FAIL_COUNT}"
echo "耗时(秒) : ${ELAPSED}"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
  echo
  echo "失败文件列表:"
  for f in "${FAILED_FILES[@]}"; do
    echo "  - ${f} (log: ${LOG_DIR}/${f%.cpp}.log)"
  done
  exit 1
fi

echo "[INFO] 全部编译成功"
exit 0
