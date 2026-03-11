// RUN: { ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul___seed__seed_l3_float_binary_elementwise_core__tmul__f32(
// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd___seed__seed_l3_float_binary_elementwise_core__tadd__f32(
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK: call @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul___seed__seed_l3_float_binary_elementwise_core__tmul__f32(
// CHECK-LABEL: func.func @flash_attention_softmax_block(
// CHECK: call @__pto_fused_group_0_0(
