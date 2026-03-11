// RUN: python3 %S/../samples/VectorArithmetic44/vector_arith_44_cases.py --mode emitc > %t.emitc.pto
// RUN: ptoas %t.emitc.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.emitc.cpp
// RUN: FileCheck %s --input-file=%t.emitc.cpp

// CHECK: __global__ AICORE void vector_arith_44_emitc(
// CHECK: RegTensor<float>
// CHECK: __builtin_fmod(
// CHECK: vsel(
// CHECK: vexp(
// CHECK: vsqrt(
// CHECK: MODE_ZEROING
