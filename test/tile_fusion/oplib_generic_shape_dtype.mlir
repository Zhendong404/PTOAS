// RUN: { ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core_f16__tadd__f16(
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core_f16__tmin__f16(
// CHECK-LABEL: func.func private @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmul__f32(
