// RUN: { ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --print-ir-after-all --print-ir-after-all-func-filter=flash_attention_softmax_block -o /dev/null 2>&1 || true; } | FileCheck %s

// CHECK: IR Dump After PTOLoweringSyncToPipe
// CHECK: func.func @flash_attention_softmax_block
// CHECK: IR Dump After PTOCreateFusionGroups
// CHECK: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-NOT: ('builtin.module' operation)
// CHECK-NOT: func.func private @__pto_oplib_inst_
