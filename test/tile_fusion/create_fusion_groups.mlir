// RUN: ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s

// This test guards grouping + outlining ordering before OP-Lib lowering.
// CHECK: [op-fusion] found 1 group(s) in @flash_attention_softmax_block
// CHECK-NEXT: [op-fusion] outlined group_id=0 into @__pto_fused_group_0_0
// CHECK: [op-fusion] imported OP-Lib entry: role=variant op=tmax
// CHECK: func.func private @__pto_fused_group_0_0
