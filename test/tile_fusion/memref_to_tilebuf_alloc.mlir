// RUN: { ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --print-ir-after-all --print-ir-after-all-func-filter=flash_attention_softmax_block -o /dev/null 2>&1 || true; } | FileCheck %s

// CHECK-LABEL: IR Dump After {{.*}}PTOMemrefToTileBufPass
// CHECK: func.func @flash_attention_softmax_block
// CHECK-DAG: pto.tmuls ins(%{{[0-9]+}}, %{{[A-Za-z0-9_]+}} : !pto.tile_buf{{.*}}, f32) outs(%{{[0-9]+}} : !pto.tile_buf{{.*}})
// CHECK-DAG: pto.tadds ins(%{{[0-9]+}}, %{{[A-Za-z0-9_]+}} : !pto.tile_buf{{.*}}, f32) outs(%{{[0-9]+}} : !pto.tile_buf{{.*}})
// CHECK-DAG: pto.tdivs ins(%{{[0-9]+}}, %{{[A-Za-z0-9_]+}} : !pto.tile_buf{{.*}}, f32) outs(%{{[0-9]+}} : !pto.tile_buf{{.*}})
// CHECK-DAG: %[[A:[0-9]+]] = pto.alloc_tile addr = %{{[A-Za-z0-9_]+}} : !pto.tile_buf
// CHECK-DAG: %[[B:[0-9]+]] = pto.alloc_tile addr = %{{[A-Za-z0-9_]+}} : !pto.tile_buf
// CHECK-DAG: pto.tmul ins(%[[A]], %[[A]] : !pto.tile_buf
// CHECK-DAG: outs(%[[B]] : !pto.tile_buf
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: IR Dump After PTOCreateFusionGroups
