// RUN: { ptoas %S/generic_shape_64x32_f32.pto --op-lib-dir=%S/oplib --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmul__f32(
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tadd__f32(
// CHECK-LABEL: func.func @binary_chain_f32_64x32(
// CHECK: call @__pto_oplib_inst___seed__seed_vec_bin_core__tadd__f32(
// CHECK: call @__pto_oplib_inst___seed__seed_vec_bin_core__tmul__f32(
// CHECK: memref<64x32xf32{{.*}}#pto.address_space<vec>>
