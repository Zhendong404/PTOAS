// RUN: { ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd___seed__seed_l3_float_binary_elementwise_core__tadd__f32(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tmul___seed__seed_l3_float_binary_elementwise_core__tmul__f32(
// CHECK-NOT: @__pto_oplib_inst_l3_float_binary_elementwise_template_{{.*}}f16
