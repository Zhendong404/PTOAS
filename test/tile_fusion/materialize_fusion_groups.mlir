// RUN: { ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmul__f32(
// CHECK-SAME: pto.oplib.instance.from_seed = true
// CHECK-SAME: pto.oplib.instance.op = "tmul"
// CHECK-LABEL: func.func private @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-SAME: pto.oplib.instance.from_seed = false
// CHECK-SAME: pto.oplib.instance.op = "tadd"
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK-COUNT-3: call @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK-COUNT-3: call @__pto_oplib_inst___seed__seed_vec_bin_core__tmul__f32(
