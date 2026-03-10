#!/usr/bin/env bash
set -uo pipefail   # 注意：去掉 -e，避免失败直接退出整个脚本

BASE_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "${BASE_DIR}/../.." && pwd)"
WORKSPACE_DIR="$(cd -- "${REPO_DIR}/.." && pwd)"

# Allow overriding tool/python explicitly:
#   PTOAS_BIN=/path/to/ptoas PYTHON_BIN=/path/to/python ./runop.sh all
PTOAS_BIN="${PTOAS_BIN:-}"
PTOBC_BIN="${PTOBC_BIN:-}"
PYTHON_BIN="${PYTHON_BIN:-}"
PTOAS_OUT_DIR="${PTOAS_OUT_DIR:-}"
PTOAS_ENABLE_INSERT_SYNC="${PTOAS_ENABLE_INSERT_SYNC:-1}"
PTOAS_FLAGS="${PTOAS_FLAGS:-}"
PTO_PTO_DIRS="${PTO_PTO_DIRS:-InjectSync}"
PTOAS_GEN_NPU_VALIDATION="${PTOAS_GEN_NPU_VALIDATION:-0}"
NPU_VALIDATION_RUN_MODE="${NPU_VALIDATION_RUN_MODE:-npu}"          # sim|npu
NPU_VALIDATION_SOC_VERSION="${NPU_VALIDATION_SOC_VERSION:-Ascend910B1}"
NPU_VALIDATION_OUTPUT_ROOT="${NPU_VALIDATION_OUTPUT_ROOT:-}"      # optional: passed to generate_testcase.py --output-root
NPU_VALIDATION_AICORE_ARCH="${NPU_VALIDATION_AICORE_ARCH:-}"      # optional: passed to generate_testcase.py --aicore-arch
ENABLE_BC=0
RUNTIME_ENV_STATUS=0
RUNTIME_ENV_MSG=""
RUNTIME_ENV_PRINTED_BOOTSTRAP=0

usage() {
  cat <<EOF
Usage:
  $0 [--enablebc] -t <name>   # e.g. -t Shls  -> run all .py in folder Shls
  $0 [--enablebc] all         # traverse every subfolder, run all .py under each
  $0 --enablebc               # alias for: $0 --enablebc all

Env:
  PTOAS_BIN   # path to ptoas executable (optional)
  PTOBC_BIN   # path to ptobc executable (optional)
  PYTHON_BIN  # python executable to run samples (optional)
  PTOAS_OUT_DIR  # where generated *.mlir/*.cpp go (optional; defaults to a temp dir)
  PTOAS_FLAGS  # extra flags passed to ptoas (e.g. --enable-insert-sync)
  PTOAS_ENABLE_INSERT_SYNC  # 1 to append --enable-insert-sync to PTOAS_FLAGS (default: 1)
  PTO_PTO_DIRS  # space-separated dirs to run .pto directly (default: InjectSync)
  PTOAS_GEN_NPU_VALIDATION  # 1 to auto-generate NPU validation testcases for each generated .cpp (default: 0)
  NPU_VALIDATION_RUN_MODE   # sim|npu passed to generate_testcase.py (default: npu)
  NPU_VALIDATION_SOC_VERSION  # e.g. Ascend910B1 (default: Ascend910B1)
  NPU_VALIDATION_OUTPUT_ROOT  # optional output root for generated testcases
  NPU_VALIDATION_AICORE_ARCH  # optional override passed to bisheng (e.g. dav-c310-vec)

Flags:
  --enablebc  # enable: python -> .pto -> ptobc -> .pto -> ptoas
  --gen-npu-validation  # generate NPU validation testcases + write \${PTOAS_OUT_DIR}/run_all_npu_validation.sh
EOF
  exit 1
}

print_env_vars() {
  local title="$1"
  echo "========== ${title} =========="
  printf "%-28s %s\n" "BASE_DIR" "${BASE_DIR}"
  printf "%-28s %s\n" "REPO_DIR" "${REPO_DIR}"
  printf "%-28s %s\n" "WORKSPACE_DIR" "${WORKSPACE_DIR}"
  printf "%-28s %s\n" "PTOAS_BIN" "${PTOAS_BIN}"
  printf "%-28s %s\n" "PTOBC_BIN" "${PTOBC_BIN}"
  printf "%-28s %s\n" "PYTHON_BIN" "${PYTHON_BIN}"
  printf "%-28s %s\n" "PTOAS_OUT_DIR" "${PTOAS_OUT_DIR}"
  printf "%-28s %s\n" "PTOAS_ENABLE_INSERT_SYNC" "${PTOAS_ENABLE_INSERT_SYNC}"
  printf "%-28s %s\n" "PTOAS_FLAGS" "${PTOAS_FLAGS}"
  printf "%-28s %s\n" "PTO_PTO_DIRS" "${PTO_PTO_DIRS}"
  printf "%-28s %s\n" "PTOAS_GEN_NPU_VALIDATION" "${PTOAS_GEN_NPU_VALIDATION}"
  printf "%-28s %s\n" "NPU_VALIDATION_RUN_MODE" "${NPU_VALIDATION_RUN_MODE}"
  printf "%-28s %s\n" "NPU_VALIDATION_SOC_VERSION" "${NPU_VALIDATION_SOC_VERSION}"
  printf "%-28s %s\n" "NPU_VALIDATION_OUTPUT_ROOT" "${NPU_VALIDATION_OUTPUT_ROOT}"
  printf "%-28s %s\n" "NPU_VALIDATION_AICORE_ARCH" "${NPU_VALIDATION_AICORE_ARCH}"
  printf "%-28s %s\n" "ENABLE_BC" "${ENABLE_BC}"
  printf "%-28s %s\n" "LLVM_BUILD_DIR" "${LLVM_BUILD_DIR:-}"
  printf "%-28s %s\n" "MLIR_PYTHON_ROOT" "${MLIR_PYTHON_ROOT:-}"
  printf "%-28s %s\n" "PTO_PYTHON_ROOT" "${PTO_PYTHON_ROOT:-}"
  printf "%-28s %s\n" "PTO_PYTHON_BUILD_ROOT" "${PTO_PYTHON_BUILD_ROOT:-}"
  printf "%-28s %s\n" "PYTHONPATH" "${PYTHONPATH:-}"
  printf "%-28s %s\n" "LD_LIBRARY_PATH" "${LD_LIBRARY_PATH:-}"
  echo "=============================="
}

ucfirst() {
  local s="$1"
  local first="${s:0:1}"
  local rest="${s:1}"
  printf '%s%s\n' "$(printf '%s' "$first" | tr '[:lower:]' '[:upper:]')" "$rest"
}

lcfirst() {
  local s="$1"
  local first="${s:0:1}"
  local rest="${s:1}"
  printf '%s%s\n' "$(printf '%s' "$first" | tr '[:upper:]' '[:lower:]')" "$rest"
}

prepend_path_if_exists() {
  local var_name="$1"
  local value="$2"
  local current="${!var_name:-}"
  [[ -n "${value}" ]] || return 0
  [[ -e "${value}" ]] || return 0
  if [[ ":${current}:" == *":${value}:"* ]]; then
    return 0
  fi
  if [[ -z "${current}" ]]; then
    printf -v "${var_name}" '%s' "${value}"
  else
    printf -v "${var_name}" '%s:%s' "${value}" "${current}"
  fi
  export "${var_name}"
}

ensure_runtime_env_once() {
  local python="$1"

  # 1=already ready, 2=already failed
  if [[ "${RUNTIME_ENV_STATUS}" == "1" ]]; then
    return 0
  fi
  if [[ "${RUNTIME_ENV_STATUS}" == "2" ]]; then
    return 1
  fi

  if "$python" -c "import mlir.ir" >/dev/null 2>&1; then
    RUNTIME_ENV_STATUS=1
    return 0
  fi

  local llvm_build_dir="${LLVM_BUILD_DIR:-${WORKSPACE_DIR}/llvm-project/build-shared}"
  local mlir_python_root="${MLIR_PYTHON_ROOT:-${llvm_build_dir}/tools/mlir/python_packages/mlir_core}"
  local pto_python_root="${PTO_PYTHON_ROOT:-${REPO_DIR}/install}"
  local pto_python_build_root="${PTO_PYTHON_BUILD_ROOT:-${REPO_DIR}/build/python}"

  prepend_path_if_exists PYTHONPATH "${mlir_python_root}"
  prepend_path_if_exists PYTHONPATH "${pto_python_root}"
  prepend_path_if_exists PYTHONPATH "${pto_python_build_root}"

  prepend_path_if_exists LD_LIBRARY_PATH "${llvm_build_dir}/lib"
  prepend_path_if_exists LD_LIBRARY_PATH "${REPO_DIR}/install/lib"
  prepend_path_if_exists LD_LIBRARY_PATH "${REPO_DIR}/build/lib"

  if "$python" -c "import mlir.ir" >/dev/null 2>&1; then
    RUNTIME_ENV_STATUS=1
    if [[ "${RUNTIME_ENV_PRINTED_BOOTSTRAP}" == "0" ]]; then
      RUNTIME_ENV_PRINTED_BOOTSTRAP=1
      print_env_vars "Runtime Env (after auto-bootstrap)"
    fi
    return 0
  fi

  RUNTIME_ENV_STATUS=2
  RUNTIME_ENV_MSG="python cannot import mlir.ir (tried auto-bootstrap from ${mlir_python_root}, ${pto_python_root}, ${pto_python_build_root})"
  return 1
}

resolve_ptoas_bin() {
  if [[ -n "${PTOAS_BIN}" ]]; then
    echo "${PTOAS_BIN}"
    return 0
  fi

  # Common locations:
  # - out-of-tree build in repo: PTOAS/build/tools/ptoas/ptoas
  # - legacy layout: build/bin/ptoas
  local cand
  cand="${BASE_DIR}/../../build/tools/ptoas/ptoas"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="${BASE_DIR}/../../../../build/bin/ptoas"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="$(command -v ptoas 2>/dev/null || true)"
  [[ -n "$cand" && -x "$cand" ]] && { echo "$cand"; return 0; }

  echo ""
  return 1
}

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    echo "${PYTHON_BIN}"
    return 0
  fi
  local cand
  cand="$(command -v python 2>/dev/null || true)"
  [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  cand="$(command -v python3 2>/dev/null || true)"
  [[ -n "$cand" ]] && { echo "$cand"; return 0; }
  echo ""
  return 1
}

resolve_ptobc_bin() {
  if [[ -n "${PTOBC_BIN}" ]]; then
    echo "${PTOBC_BIN}"
    return 0
  fi

  local cand
  cand="${BASE_DIR}/../../build/tools/ptobc/ptobc"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="${BASE_DIR}/../../build/bin/ptobc"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="${BASE_DIR}/../../../../build/bin/ptobc"
  [[ -x "$cand" ]] && { echo "$cand"; return 0; }
  cand="$(command -v ptobc 2>/dev/null || true)"
  [[ -n "$cand" && -x "$cand" ]] && { echo "$cand"; return 0; }

  echo ""
  return 1
}

write_npu_validation_runner() {
  local out_dir="$1"
  local runner="${out_dir}/run_all_npu_validation.sh"
  cat >"${runner}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/npu_validation_logs}"
STAGE="${STAGE:-all}"   # all|build|run
BUILD_JOBS="${BUILD_JOBS:-}"

usage() {
  cat <<USAGE
Usage:
  $0 [--list] [--build|--run|--all]

Env:
  STAGE=all|build|run   # default: all
  LOG_DIR=/path         # default: \${ROOT_DIR}/npu_validation_logs
  BUILD_JOBS=N          # build parallelism (default: nproc)

Notes:
  - Discovers run.sh under: \${ROOT_DIR}/*/npu_validation/**/run.sh
  - Separate build/run:
      STAGE=build $0
      STAGE=run   $0
USAGE
  exit 1
}

want_list=0
for a in "$@"; do
  case "$a" in
    --list) want_list=1 ;;
    --build) STAGE="build" ;;
    --run) STAGE="run" ;;
    --all) STAGE="all" ;;
    -h|--help) usage ;;
    *) echo "[ERROR] Unknown arg: $a" >&2; usage ;;
  esac
done

case "${STAGE}" in
  all|build|run) ;;
  *) echo "[ERROR] Unknown STAGE=${STAGE} (expected: all|build|run)" >&2; exit 2 ;;
esac

if [[ -z "${BUILD_JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    BUILD_JOBS="$(nproc)"
  else
    BUILD_JOBS="4"
  fi
fi

mkdir -p "${LOG_DIR}"
summary_tsv="${LOG_DIR}/summary.tsv"
printf "case\tstage\tstatus\texit_code\telapsed_s\tlog\n" >"${summary_tsv}"

  mapfile -t RUNS < <(
  find "${ROOT_DIR}" -type f -name run.sh \
    \( -path "*/npu_validation/run.sh" -o -path "*/npu_validation/*/run.sh" \) \
    -print | sort
)

if [[ ${#RUNS[@]} -eq 0 ]]; then
  echo "[WARN] No npu_validation run.sh found under ${ROOT_DIR}"
  exit 0
fi

if [[ "${want_list}" == "1" ]]; then
  printf "%s\n" "${RUNS[@]}"
  exit 0
fi

ok=0
fail=0

if [[ "${STAGE}" == "all" || "${STAGE}" == "build" ]]; then
  echo "[INFO] build parallelism: ${BUILD_JOBS}"
  printf "%s\0" "${RUNS[@]}" | xargs -0 -n 1 -P "${BUILD_JOBS}" -I {} bash -c 'cd "$(dirname "$1")" && STAGE="build" bash ./run.sh' _ {}
  build_rc=$?
  if [[ "${STAGE}" == "build" ]]; then
    echo "========== SUMMARY =========="
    echo "OK=0  FAIL=0"
    echo "summary: ${summary_tsv}"
    exit $build_rc
  fi
  if [[ $build_rc -ne 0 ]]; then
    echo "[ERROR] build failed"
    exit $build_rc
  fi
fi

for run_sh in "${RUNS[@]}"; do
  case_dir="$(cd -- "$(dirname -- "${run_sh}")" && pwd)"

  # Layout A (new generator): <root>/<Sample>/npu_validation/<testcase>/run.sh
  # Layout B (legacy/custom): <root>/<Sample>/npu_validation/run.sh
  case_name=""
  sample_name=""
  if [[ "$(basename "${case_dir}")" == "npu_validation" ]]; then
    sample_name="$(basename "$(dirname "${case_dir}")")"
    case_name="default"
  else
    npu_dir="$(dirname "${case_dir}")"
    if [[ "$(basename "${npu_dir}")" == "npu_validation" ]]; then
      sample_name="$(basename "$(dirname "${npu_dir}")")"
      case_name="$(basename "${case_dir}")"
    else
      sample_name="$(basename "$(dirname "${case_dir}")")"
      case_name="$(basename "${case_dir}")"
    fi
  fi

  case_id="${sample_name}/${case_name}"
  log_path="${LOG_DIR}/${sample_name}__${case_name}.log"

  start_ts="$(date +%s)"
  set +e
  (cd "${case_dir}" && STAGE="${STAGE}" bash ./run.sh) >"${log_path}" 2>&1
  rc=$?
  set -e
  end_ts="$(date +%s)"
  elapsed="$(( end_ts - start_ts ))"

  if [[ $rc -eq 0 ]]; then
    echo -e "${case_id}\tOK\t${log_path}"
    printf "%s\t%s\tOK\t%d\t%d\t%s\n" "${case_id}" "${STAGE}" "${rc}" "${elapsed}" "${log_path}" >>"${summary_tsv}"
    ok=$((ok+1))
  else
    echo -e "${case_id}\tFAIL(rc=${rc})\t${log_path}"
    printf "%s\t%s\tFAIL\t%d\t%d\t%s\n" "${case_id}" "${STAGE}" "${rc}" "${elapsed}" "${log_path}" >>"${summary_tsv}"
    fail=$((fail+1))
  fi
done

echo "========== SUMMARY =========="
echo "OK=${ok}  FAIL=${fail}"
echo "summary: ${summary_tsv}"
exit $([[ $fail -eq 0 ]] && echo 0 || echo 1)
EOF
  chmod +x "${runner}"
}

gen_npu_validation_for_cpp() {
  local python="$1"
  local cpp="$2"
  local script="${REPO_DIR}/test/npu_validation/scripts/generate_testcase.py"
  local cpp_dir
  cpp_dir="$(cd -- "$(dirname -- "$cpp")" && pwd)"
  local stem
  stem="$(basename "$cpp" .cpp)"
  local log_path="${cpp_dir}/npu_validation_gen_${stem}.log"
  local -a cmd=("$python" "$script" --input "$cpp" --run-mode "${NPU_VALIDATION_RUN_MODE}" --soc-version "${NPU_VALIDATION_SOC_VERSION}")
  if [[ -n "${NPU_VALIDATION_OUTPUT_ROOT}" ]]; then
    cmd+=(--output-root "${NPU_VALIDATION_OUTPUT_ROOT}")
  fi
  if [[ -n "${NPU_VALIDATION_AICORE_ARCH}" ]]; then
    cmd+=(--aicore-arch "${NPU_VALIDATION_AICORE_ARCH}")
  fi
  if "${cmd[@]}" >"${log_path}" 2>&1; then
    rm -f "${log_path}"
    return 0
  fi
  return 1
}

process_one_dir() {
  local A="$1" # folder name (e.g. Abs)
  local out_dir="$2"
  local dir ptoas ptobc python out_subdir
  dir="${BASE_DIR}/${A}"
  out_subdir="${out_dir}/${A}"
  mkdir -p "${out_subdir}"

  ptoas="$(resolve_ptoas_bin)"
  ptobc="$(resolve_ptobc_bin)"
  python="$(resolve_python_bin)"
  local use_ptobc_roundtrip=0
  if [[ "${ENABLE_BC}" == "1" ]]; then
    use_ptobc_roundtrip=1
  fi
  local -a ptoas_flags=()
  if [[ -n "${PTOAS_FLAGS}" ]]; then
    # shellcheck disable=SC2206
    ptoas_flags=(${PTOAS_FLAGS})
  fi
  if [[ "${PTOAS_ENABLE_INSERT_SYNC}" == "1" ]]; then
    local has_insync=0
    if ((${#ptoas_flags[@]})); then
      for f in "${ptoas_flags[@]}"; do
        if [[ "$f" == "--enable-insert-sync" ]]; then
          has_insync=1
          break
        fi
      done
    fi
    [[ $has_insync -eq 1 ]] || ptoas_flags+=(--enable-insert-sync)
  fi
  local -a ptoas_cmd_base=("$ptoas")
  if (( ${#ptoas_flags[@]} )); then
    ptoas_cmd_base+=("${ptoas_flags[@]}")
  fi

  if [[ -z "$ptoas" || ! -x "$ptoas" ]]; then
    echo -e "${A}\tFAIL\tMissing executable: PTOAS_BIN (searched common paths)"
    return 0
  fi
  if [[ -z "$python" || ! -x "$python" ]]; then
    echo -e "${A}\tFAIL\tMissing python: PYTHON_BIN (python/python3 not found)"
    return 0
  fi
  if ! ensure_runtime_env_once "$python"; then
    echo -e "${A}\tFAIL\t${RUNTIME_ENV_MSG}; please source scripts/ptoas_env.sh or export PYTHONPATH"
    return 0
  fi
  if [[ $use_ptobc_roundtrip -eq 1 ]] && [[ -z "$ptobc" || ! -x "$ptobc" ]]; then
    echo -e "${A}\tFAIL\tMissing executable: PTOBC_BIN (searched common paths)"
    return 0
  fi
  if [[ ! -d "$dir" ]]; then
    echo -e "${A}\tSKIP\tMissing dir: $dir"
    return 0
  fi

  # Run every .py file in this directory (no requirement that name matches folder).
  local f mlir ptobc_file decoded_pto cpp base overall=0
  for f in "$dir"/*.py; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f" .py)"
    local expect_fail=0
    case "$base" in
      *_invalid|*_xfail) expect_fail=1 ;;
    esac
    # Some samples are expected to fail depending on the selected ptoas flags.
    #
    # alloc_tile_addr.py uses `pto.alloc_tile addr=...`, which is only accepted
    # by the ptoas tool when assembling at Level-3.
    if [[ "$base" == "alloc_tile_addr" ]]; then
      local has_level3=0
      if ((${#ptoas_flags[@]})); then
        for ((i=0; i<${#ptoas_flags[@]}; i++)); do
          if [[ "${ptoas_flags[$i]}" == "--pto-level=level3" ]]; then
            has_level3=1
            break
          fi
          if [[ "${ptoas_flags[$i]}" == "--pto-level" ]]; then
            if (( i + 1 < ${#ptoas_flags[@]} )) && [[ "${ptoas_flags[$((i+1))]}" == "level3" ]]; then
              has_level3=1
              break
            fi
          fi
        done
      fi
      [[ $has_level3 -eq 1 ]] || expect_fail=1
    fi
    mlir="${out_subdir}/${base}-pto-ir.pto"
    cpp="${out_subdir}/${base}-pto.cpp"

    if ! "$python" "$f" > "$mlir"; then
      if [[ $expect_fail -eq 1 ]]; then
        echo -e "${A}(${base}.py)\tXFAIL\tpython failed as expected"
        continue
      fi
      echo -e "${A}(${base}.py)\tFAIL\tpython failed: ${base}.py"
      overall=1
      continue
    fi

    local pto_input="$mlir"
    ptobc_file="${out_subdir}/${base}.ptobc"
    decoded_pto="${out_subdir}/${base}-roundtrip.pto"
    if [[ $use_ptobc_roundtrip -eq 1 ]]; then
      if ! "$ptobc" encode "$mlir" -o "$ptobc_file" >/dev/null 2>&1; then
        if [[ $expect_fail -eq 1 ]]; then
          echo -e "${A}(${base}.py)\tXFAIL\tptobc encode failed as expected"
          continue
        fi
        echo -e "${A}(${base}.py)\tFAIL\tptobc encode failed: $(basename "$mlir")"
        overall=1
        continue
      fi
      if ! "$ptobc" decode "$ptobc_file" -o "$decoded_pto" >/dev/null 2>&1; then
        if [[ $expect_fail -eq 1 ]]; then
          echo -e "${A}(${base}.py)\tXFAIL\tptobc decode failed as expected"
          continue
        fi
        echo -e "${A}(${base}.py)\tFAIL\tptobc decode failed: $(basename "$ptobc_file")"
        overall=1
        continue
      fi
      pto_input="$decoded_pto"
    fi

    # Write output via -o to avoid mixing debug prints with generated C++.
    local -a ptoas_cmd=("${ptoas_cmd_base[@]}" "$pto_input" -o "$cpp")
    if ! "${ptoas_cmd[@]}" >/dev/null 2>&1; then
      if [[ $expect_fail -eq 1 ]]; then
        echo -e "${A}(${base}.py)\tXFAIL\tptoas failed as expected"
        continue
      fi
      echo -e "${A}(${base}.py)\tFAIL\tptoas failed: $(basename "$mlir")"
      overall=1
      continue
    fi

    if [[ $expect_fail -eq 1 ]]; then
      echo -e "${A}(${base}.py)\tFAIL\texpected failure but succeeded"
      overall=1
      continue
    fi

    # Regression guard: SubsetOp valid-shape inference must not produce 0.
    # This breaks downstream NPU compilation (e.g. vadd_pto_pingpong workspace ping/pong).
    if [[ "$base" == "vadd_pto_pingpong" ]]; then
      if grep -Fq ", 0, SLayout" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tgenerated tile has valid dim 0 (subset valid-shape bug)"
        overall=1
        continue
      fi
    fi

    # Regression guard for Issue #112:
    # `--enable-insert-sync` must not push PIPE_M -> PIPE_FIX into high event IDs
    # for the autosync tmatmulk sample, otherwise it may deadlock on Ascend NPU.
    if [[ "$base" == "tmatmulk_autosync" ]]; then
      if grep -Eq "set_flag\\(PIPE_M,[[:space:]]*PIPE_FIX,[[:space:]]*EVENT_ID[3-7]\\)" "$cpp" || \
         grep -Eq "wait_flag\\(PIPE_M,[[:space:]]*PIPE_FIX,[[:space:]]*EVENT_ID[3-7]\\)" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tdeadlock signature: PIPE_M->PIPE_FIX uses EVENT_ID[3-7]"
        overall=1
        continue
      fi
    fi

    # Regression guard for issue #117: vector mask must be reset for each
    # `pto.section.vector` region to avoid cross-kernel state leakage.
    # Use an existing sample (Complex/cv_region.py) that contains a vector section.
    if [[ "$base" == "cv_region" ]]; then
      if ! grep -Fq "#if defined(__DAV_VEC__)" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tmissing __DAV_VEC__ guard"
        overall=1
        continue
      fi
      if ! grep -Fq "set_mask_norm();" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tmissing set_mask_norm() reset"
        overall=1
        continue
      fi
      if ! grep -Fq "set_vector_mask(-1, -1);" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tmissing set_vector_mask(-1, -1) reset"
        overall=1
        continue
      fi
    fi

    # Regression guard: bf16 tiles must lower to `bfloat16_t` in Tile<> / GlobalTensor<> templates.
    if [[ "$base" == "bf16_tile" ]]; then
      if ! grep -Fq "GlobalTensor<bfloat16_t" "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tbf16 GlobalTensor element type is not bfloat16_t"
        overall=1
        continue
      fi
      if ! grep -Eq "Tile<[^>]*, bfloat16_t," "$cpp"; then
        echo -e "${A}(${base}.py)\tFAIL\tbf16 Tile element type is not bfloat16_t"
        overall=1
        continue
      fi
    fi

    local extra_msg=""
    if [[ "${PTOAS_GEN_NPU_VALIDATION}" == "1" ]]; then
      if gen_npu_validation_for_cpp "$python" "$cpp"; then
        extra_msg="; npu_validation: OK"
      else
        echo -e "${A}(${base}.py)\tFAIL\tnpu_validation generation failed: $(basename "$cpp")"
        overall=1
        continue
      fi
    fi

    echo -e "${A}(${base}.py)\tOK\tgenerated: $(basename "$cpp")${extra_msg}"
  done

  # Run .pto files only for allowed dirs (default: InjectSync) to avoid legacy IR.
  local allow_pto=0
  for d in ${PTO_PTO_DIRS}; do
    if [[ "$A" == "$d" ]]; then
      allow_pto=1
      break
    fi
  done

  if [[ $allow_pto -eq 1 ]]; then
    for f in "$dir"/*.pto; do
      [[ -f "$f" ]] || continue
      case "$f" in
        *-pto-ir.pto) continue ;;
      esac
      base="$(basename "$f" .pto)"
      local pto_input="$f"
      ptobc_file="${out_subdir}/${base}.ptobc"
      decoded_pto="${out_subdir}/${base}-roundtrip.pto"
      cpp="${out_subdir}/${base}.cpp"

      if [[ $use_ptobc_roundtrip -eq 1 ]]; then
        if ! "$ptobc" encode "$f" -o "$ptobc_file" >/dev/null 2>&1; then
          echo -e "${A}(${base}.pto)\tFAIL\tptobc encode failed: $(basename "$f")"
          overall=1
          continue
        fi
        if ! "$ptobc" decode "$ptobc_file" -o "$decoded_pto" >/dev/null 2>&1; then
          echo -e "${A}(${base}.pto)\tFAIL\tptobc decode failed: $(basename "$ptobc_file")"
          overall=1
          continue
        fi
        pto_input="$decoded_pto"
      fi

      local -a ptoas_cmd=("${ptoas_cmd_base[@]}" "$pto_input" -o "$cpp")
      if ! "${ptoas_cmd[@]}" >/dev/null 2>&1; then
        echo -e "${A}(${base}.pto)\tFAIL\tptoas failed: $(basename "$f")"
        overall=1
        continue
      fi

      # Regression guard: dynamic valid_shape must be preserved through lowering.
      # If `valid_col` is dynamic, PTOToEmitC must construct the Tile with a
      # runtime argument (i.e. emit `= Tile<...>(...)` instead of `Tile<...>;`).
      if [[ "$base" == "test_dynamic_valid_shape" ]]; then
        if ! grep -Fq "= Tile<TileType::Vec, float" "$cpp"; then
          echo -e "${A}(${base}.pto)\tFAIL\tmissing dynamic Tile constructor (valid_col likely dropped)"
          overall=1
          continue
        fi
      fi

      local extra_msg=""
      if [[ "${PTOAS_GEN_NPU_VALIDATION}" == "1" ]]; then
        if gen_npu_validation_for_cpp "$python" "$cpp"; then
          extra_msg="; npu_validation: OK"
        else
          echo -e "${A}(${base}.pto)\tFAIL\tnpu_validation generation failed: $(basename "$cpp")"
          overall=1
          continue
        fi
      fi

      echo -e "${A}(${base}.pto)\tOK\tgenerated: $(basename "$cpp")${extra_msg}"
    done
  fi

  return $overall
}

run_all() {
  local results tmp out_dir
  out_dir="${PTOAS_OUT_DIR}"
  if [[ -z "${out_dir}" ]]; then
    out_dir="$(mktemp -d -t ptoas.samples.XXXXXX)"
  else
    mkdir -p "${out_dir}"
  fi

  if [[ "${PTOAS_GEN_NPU_VALIDATION}" == "1" ]]; then
    write_npu_validation_runner "${out_dir}"
    echo "NPU validation runner: ${out_dir}/run_all_npu_validation.sh"
  fi

  tmp="$(mktemp -t ptoas.runop.XXXXXX)"
  for d in "${BASE_DIR}"/*/; do
    [[ -d "$d" ]] || continue
    process_one_dir "$(basename "$d")" "$out_dir" >>"$tmp"
  done

  echo "========== SUMMARY =========="
  sort "$tmp" | awk -F'\t' '
    BEGIN { ok=0; fail=0; skip=0; }
    {
      printf "%-12s %-4s %s\n", $1, $2, $3;
      if ($2=="OK") ok++;
      else if ($2=="FAIL") fail++;
      else if ($2=="SKIP") skip++;
    }
    END {
      print "-----------------------------";
      printf "OK=%d  FAIL=%d  SKIP=%d\n", ok, fail, skip;
      print "=============================";
      exit (fail==0 ? 0 : 1);
    }'
}

# -----------------------------------------------------------------------------
# CLI flags
# -----------------------------------------------------------------------------
positional_args=()
for arg in "$@"; do
  case "$arg" in
    --enablebc) ENABLE_BC=1 ;;
    --gen-npu-validation) PTOAS_GEN_NPU_VALIDATION=1 ;;
    -h|--help) usage ;;
    *) positional_args+=("$arg") ;;
  esac
done
set -- "${positional_args[@]}"

if [[ "${ENABLE_BC}" == "1" ]] && [[ $# -eq 0 ]]; then
  set -- all
fi

print_env_vars "Runtime Env (effective)"

if [[ $# -eq 1 && "$1" == "all" ]]; then
  run_all
elif [[ $# -eq 2 && "$1" == "-t" ]]; then
  A="$(ucfirst "$2")"
  out_dir="${PTOAS_OUT_DIR}"
  if [[ -z "${out_dir}" ]]; then
    out_dir="$(mktemp -d -t ptoas.samples.XXXXXX)"
  else
    mkdir -p "${out_dir}"
  fi
  echo ""
  echo "========== SUMMARY =========="
  if [[ "${PTOAS_GEN_NPU_VALIDATION}" == "1" ]]; then
    write_npu_validation_runner "${out_dir}"
    echo "NPU validation runner: ${out_dir}/run_all_npu_validation.sh"
  fi
  process_one_dir "$A" "$out_dir" | awk -F'\t' '{ printf "%-12s %-4s %s\n", $1, $2, $3 }'
else
  usage
fi
