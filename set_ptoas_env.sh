#!/usr/bin/env bash
# after `quick_install.sh`, run `source set_ptoas_env.sh` in a new shell to find the lib
export PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"
export PATH="${PTO_SOURCE_DIR}/build/tools/ptoas:${PATH}"
export LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"

(cd "${PTO_SOURCE_DIR}/test/samples/MatMul" && python ./tmatmulk.py > ./tmatmulk.pto && ptoas ./tmatmulk.pto -o ./tmatmulk.cpp)
(cd "${PTO_SOURCE_DIR}/test/samples/Abs" && python ./abs.py > ./abs.pto && ptoas --enable-insert-sync ./abs.pto -o ./abs.cpp)

echo "test set_env: OK"
