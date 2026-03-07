// RUN: ! ptoas %S/softmax_chain.pto -o %t.no_dir.cpp > %t.no_dir.log 2>&1
// RUN: FileCheck %s --check-prefix=NO-DIR < %t.no_dir.log
// RUN: ! ptoas %S/softmax_chain.pto --enable-op-fusion --disable-oplib-lowering --op-lib-dir=%S/oplib -o %t.dep.cpp > %t.dep.log 2>&1
// RUN: FileCheck %s --check-prefix=REMOVED < %t.dep.log
// RUN: rm -rf %t.empty && mkdir -p %t.empty
// RUN: cp %S/resources/bad_empty_simd_template.txt %t.empty/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.empty -o %t.empty.cpp > %t.empty.log 2>&1
// RUN: FileCheck %s --check-prefix=EMPTY-SIMD < %t.empty.log
// RUN: rm -rf %t.core && mkdir -p %t.core
// RUN: cp %S/resources/bad_simd_core_slot_template.txt %t.core/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.core -o %t.core.cpp > %t.core.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-CORE < %t.core.log
// RUN: rm -rf %t.lanes && mkdir -p %t.lanes
// RUN: cp %S/resources/bad_simd_lanes_template.txt %t.lanes/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.lanes -o %t.lanes.cpp > %t.lanes.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-LANES < %t.lanes.log

// NO-DIR: Error: --op-lib-dir is required.
// REMOVED: Unknown command line argument '--disable-oplib-lowering'
// EMPTY-SIMD: E_OPLIB_EMPTY_BODY_FOR_SIMD
// BAD-CORE: E_OPLIB_SIMD_INVALID_CORE_SLOT
// BAD-LANES: E_OPLIB_SIMD_LANES_MISMATCH
