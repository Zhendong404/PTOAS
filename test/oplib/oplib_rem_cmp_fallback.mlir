// RUN: ptoas %S/rem_cmp_fallback.pto --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.fallback.cpp
// RUN: FileCheck %s < %t.fallback.cpp

// CHECK: TREM(
// CHECK: TREMS(
// CHECK: TCMP(
// CHECK: TCMPS(
// CHECK: TPRELU(
// CHECK: TLRELU(
// CHECK-NOT: __pto_oplib_inst_
// CHECK-NOT: PTOAS__OPLIB_
