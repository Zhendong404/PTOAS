// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionLoadStoreElision/) exit; print}' | FileCheck %s

// VPTO-focused regression:
// - keep yielded frontier materialization (`%9` / `%17` stores remain)
// - after vecscope became a region, store-elision must still forward SSA
//   values inside each vecscope leaf body instead of reloading from the
//   just-materialized intermediate tiles.
// - normalize fusion-region yielded frontier to the underlying memrefs at
//   store-elision time, while re-binding region results for downstream users.

// CHECK-LABEL: IR Dump After PTOFusionLoadStoreElision
// CHECK-LABEL: func.func @kernel_online_update(
// CHECK: %[[REGION0:[0-9]+]]:4 = pto.fusion_region {
// CHECK: %[[STRAIGHT_INDEX:[0-9]+]] = arith.addi %{{[0-9]+}}, %{{[0-9]+}} : index
// CHECK: %[[STRAIGHT_SRC:[0-9]+]] = pto.vlds %4[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_SRC0:[0-9]+]] = pto.vlds %0[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_MAX:[0-9]+]] = pto.vmax %[[STRAIGHT_SRC]], %[[STRAIGHT_SRC0]], %mask
// CHECK: pto.vsts %[[STRAIGHT_MAX]], %9[%[[STRAIGHT_INDEX]]], %mask
// CHECK-NOT: pto.vlds %9[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_SRC1:[0-9]+]] = pto.vlds %4[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_SUB:[0-9]+]] = pto.vsub %[[STRAIGHT_SRC1]], %[[STRAIGHT_MAX]], %mask
// CHECK: %[[STRAIGHT_EXP0:[0-9]+]] = pto.vexp %[[STRAIGHT_SUB]], %mask
// CHECK: pto.vsts %[[STRAIGHT_EXP0]], %12[%[[STRAIGHT_INDEX]]], %mask
// CHECK: %[[STRAIGHT_SRC2:[0-9]+]] = pto.vlds %0[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_SUB1:[0-9]+]] = pto.vsub %[[STRAIGHT_SRC2]], %[[STRAIGHT_MAX]], %mask
// CHECK: %[[STRAIGHT_EXP1:[0-9]+]] = pto.vexp %[[STRAIGHT_SUB1]], %mask
// CHECK: pto.vsts %[[STRAIGHT_EXP1]], %14[%[[STRAIGHT_INDEX]]], %mask
// CHECK-NOT: pto.vlds %11[%[[STRAIGHT_INDEX]]]
// CHECK-NOT: pto.vlds %12[%[[STRAIGHT_INDEX]]]
// CHECK-NOT: pto.vlds %14[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_LOAD5:[0-9]+]] = pto.vlds %5[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_MUL0:[0-9]+]] = pto.vmul %[[STRAIGHT_EXP0]], %[[STRAIGHT_LOAD5]], %mask
// CHECK: %[[STRAIGHT_LOAD1:[0-9]+]] = pto.vlds %1[%[[STRAIGHT_INDEX]]]
// CHECK: %[[STRAIGHT_MUL1:[0-9]+]] = pto.vmul %[[STRAIGHT_EXP1]], %[[STRAIGHT_LOAD1]], %mask
// CHECK: %[[STRAIGHT_ADD:[0-9]+]] = pto.vadd %[[STRAIGHT_MUL0]], %[[STRAIGHT_MUL1]], %mask
// CHECK: pto.vsts %[[STRAIGHT_ADD]], %17[%[[STRAIGHT_INDEX]]], %mask
// CHECK: pto.yield(%9, %12, %14, %17) : (memref<1x16xf32
// CHECK: } {pto.fusion.group_id = 0 : i64} : memref<1x16xf32
// CHECK: %[[REGION0_OUT3:[0-9]+]] = pto.bind_tile %[[REGION0]]#3, %c1, %c16
// CHECK: %[[REGION1:[0-9]+]]:2 = pto.fusion_region {
// CHECK: %[[G1_LOAD0:[0-9]+]] = pto.vlds %6[%{{[^]]+}}]
// CHECK: %[[G1_MUL0:[0-9]+]] = pto.vmul %[[G1_LOAD0]], %{{[0-9]+}}, %mask
// CHECK: %[[G1_LOAD1:[0-9]+]] = pto.vlds %2[%{{[^]]+}}]
// CHECK: %[[G1_MUL1:[0-9]+]] = pto.vmul %[[G1_LOAD1]], %{{[0-9]+}}, %mask
// CHECK: pto.vsts %[[G1_MUL1]], %6[%{{[^]]+}}], %mask
// CHECK-NOT: pto.vlds %6[%{{[^]]+}}]
// CHECK: %[[G1_ADD:[0-9]+]] = pto.vadd %[[G1_MUL0]], %[[G1_MUL1]], %mask
// CHECK: pto.vsts %[[G1_ADD]], %2[%{{[^]]+}}], %mask
// CHECK: pto.yield(%6, %2) : (memref<16x128xf32
