// RUN: ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s

// This test guards CreateFusionGroups via the materializer's group discovery log.
// CHECK: [op-fusion] found 2 group(s) in @flash_attention_softmax_block
// CHECK-DAG: func.func private @__pto_fused_group_0_0
// CHECK-DAG: func.func private @__pto_fused_group_1_1
