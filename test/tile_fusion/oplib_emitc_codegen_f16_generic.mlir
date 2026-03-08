// RUN: ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 -o %t.generic.cpp
// RUN: FileCheck %s < %t.generic.cpp

// CHECK-DAG: __global__ AICORE void __pto_oplib_inst___seed__seed_vec_bin_core_f16__tadd__f16(
// CHECK-DAG: __global__ AICORE void __pto_oplib_inst___seed__seed_vec_bin_core_f16__tmin__f16(
// CHECK-DAG: vadd(
// CHECK-DAG: vmin(
// CHECK-DAG: CreatePredicate<half>(
// CHECK-NOT: PTOAS__OPLIB_
