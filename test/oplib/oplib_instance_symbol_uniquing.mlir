// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --enable-insert-sync --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// Same seed-rewritten `tadd` template is instantiated for both 1x16 and
// 16x128 memref signatures in this kernel. Their symbol bases collide, so the
// second instance must be uniqued instead of redefining the first symbol.

// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd___seed__seed_l3_float_binary_elementwise_core__tadd__f32(
// CHECK-DAG: func.func private @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd___seed__seed_l3_float_binary_elementwise_core__tadd__f32_1(
// CHECK: func.call @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd___seed__seed_l3_float_binary_elementwise_core__tadd__f32(
// CHECK: func.call @__pto_oplib_inst_l3_float_binary_elementwise_template_tadd___seed__seed_l3_float_binary_elementwise_core__tadd__f32_1(
