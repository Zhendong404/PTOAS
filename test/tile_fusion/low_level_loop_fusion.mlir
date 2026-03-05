// RUN: ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s

// CHECK: [op-fusion] low-level loop fusion changed 2 fused function(s)
// CHECK: func.func private @__pto_fused_group_0_0
