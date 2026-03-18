#!/usr/bin/env bash
set -euo pipefail

ptoas_bin="./build/tools/ptoas/ptoas"

if [[ ! -x "${ptoas_bin}" ]]; then
  echo "error: missing ./build/tools/ptoas/ptoas" >&2
  exit 1
fi

echo "phase2 check: tload_contract_trace.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/tload_contract_trace.mlir -o /dev/null 2>&1 | \
  FileCheck test/phase2/tload_contract_trace.mlir

echo "phase2 check: tstore_branch_shape.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/tstore_branch_shape.mlir -o /dev/null 2>&1 | \
  FileCheck test/phase2/tstore_branch_shape.mlir

echo "phase2 check: tabs_precheck.mlir"
"${ptoas_bin}" --pto-backend=a5vm test/phase2/tabs_precheck.mlir -o /dev/null 2>&1 | \
  FileCheck test/phase2/tabs_precheck.mlir

echo "phase2 check: unary_template_shape.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/unary_template_shape.mlir -o /dev/null 2>&1 | \
  FileCheck test/phase2/unary_template_shape.mlir

echo "phase2 check: ctest"
ctest --test-dir build --output-on-failure
