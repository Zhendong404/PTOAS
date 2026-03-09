// RUN: { ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK-SAME: attributes {pto.fusion.group_id = 0 : i64}
// CHECK: call @__pto_oplib_inst___seed__seed_vec_bin_core__tadd__f32(
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: call @__pto_fused_group_0_0(
