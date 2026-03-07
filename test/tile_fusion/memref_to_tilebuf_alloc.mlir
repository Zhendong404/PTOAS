// RUN: ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --dump-ir-after-memref-to-tilebuf -o - | FileCheck %s

// CHECK: func.func @flash_attention_softmax_block
// CHECK: %[[A:[0-9]+]] = pto.alloc_tile addr = %{{[A-Za-z0-9_]+}} : !pto.tile_buf
// CHECK: %[[B:[0-9]+]] = pto.alloc_tile addr = %{{[A-Za-z0-9_]+}} : !pto.tile_buf
// CHECK: pto.tmul ins(%[[A]], %[[A]] : !pto.tile_buf
// CHECK: outs(%[[B]] : !pto.tile_buf
// CHECK-NOT: builtin.unrealized_conversion_cast
