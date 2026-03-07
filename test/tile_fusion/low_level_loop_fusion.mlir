// RUN: ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --dump-ir-after-op-fusion -o - | FileCheck %s

// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK: scf.for
// CHECK-COUNT-3: arith.addf
// CHECK-NOT: call @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-NOT: pto.simd.
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: call @__pto_fused_group_0_0(
