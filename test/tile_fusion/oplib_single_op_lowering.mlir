// RUN: ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --dump-ir-after-oplib-lowering -o - | FileCheck %s

// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmul__f32(
// CHECK-LABEL: func.func private @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK: call @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: call @__pto_fused_group_0_0(
// CHECK-NOT: scf.for
