// RUN: ptoas %S/../oplib/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.cpp
// RUN: FileCheck %s < %t.cpp

// CHECK-LABEL: [aicore] inline __attribute__((always_inline)) void __pto_fused_group_1_1(
// CHECK: __VEC_SCOPE__ {
// CHECK-COUNT-4: vlds(
// CHECK: vadd(
// CHECK: vdiv(
// CHECK-COUNT-1: vsts(
// CHECK-NOT: vsel(
// CHECK-LABEL: [aicore] inline __attribute__((always_inline)) void __pto_fused_group_0_0(
