// RUN: { ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --print-ir-after-all -o /dev/null 2>&1 || true; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK-SAME: attributes {pto.fusion.group_id = 0 : i64}
// CHECK: call @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: call @__pto_fused_group_0_0(
