// RUN: { ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK: scf.for
// CHECK-COUNT-3: arith.mulf
// CHECK: arith.addf
// CHECK-NOT: call @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-NOT: pto.simd.load
// CHECK-NOT: pto.simd.store
// CHECK-NOT: pto.simd.predicate
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: call @__pto_fused_group_0_0(
