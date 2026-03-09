// RUN: ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 -o %t.generic.cpp
// RUN: FileCheck %s < %t.generic.cpp

// CHECK-DAG: __global__ AICORE void binary_chain_f16_64x64(
// CHECK-DAG: __global__ AICORE void binary_chain_f32_48x48(
// CHECK-DAG: TADD(
// CHECK-DAG: TMIN(
// CHECK-DAG: vadd(
// CHECK: uint32_t [[ACTIVE_COUNT:v[0-9]+]] = 0;
// CHECK-NEXT: [[ACTIVE_COUNT]] = (uint32_t) ([[TAIL_COUNT:v[0-9]+]] < [[VEC_WIDTH:v[0-9]+]] ? [[TAIL_COUNT]] : [[VEC_WIDTH]]);
// CHECK-NEXT: MaskReg [[PRED:v[0-9]+]] = CreatePredicate<float>([[ACTIVE_COUNT]]);
// CHECK-NOT: PTOAS__OPLIB_
