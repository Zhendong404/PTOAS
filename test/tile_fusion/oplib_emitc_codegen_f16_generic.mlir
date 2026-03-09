// RUN: ptoas %S/generic_shape_dtype_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 -o %t.generic.cpp
// RUN: FileCheck %s < %t.generic.cpp

// CHECK-DAG: __global__ AICORE void binary_chain_f16_64x64(
// CHECK-DAG: __global__ AICORE void binary_chain_f32_48x48(
// CHECK-DAG: TADD(
// CHECK-DAG: TMIN(
// CHECK-DAG: vadd(
// CHECK: uint32_t [[ACTIVE_COUNT:v[0-9]+]] = 0;
// CHECK: [[ACTIVE_COUNT]] = [[TAIL_COUNT:v[0-9]+]];
// CHECK: MaskReg [[PRED:v[0-9]+]] = CreatePredicate<float>([[ACTIVE_COUNT]]);
// CHECK-NOT: PTOAS__OPLIB_
