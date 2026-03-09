// RUN: { ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 -o - 2>&1; } | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NOT: pto.simd.tile_to_memref
