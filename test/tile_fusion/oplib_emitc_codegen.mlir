// RUN: ptoas %S/binary_max_min_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 -o %t.bin6.cpp
// RUN: FileCheck %s --check-prefix=BIN6 < %t.bin6.cpp
// RUN: ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --pto-arch=a5 -o %t.softmax.cpp
// RUN: FileCheck %s --check-prefix=SOFTMAX < %t.softmax.cpp

// BIN6-DAG: vlds(
// BIN6-DAG: vsts(
// BIN6-DAG: vadd(
// BIN6-DAG: vsub(
// BIN6-DAG: vmul(
// BIN6-DAG: vdiv(
// BIN6-DAG: vmax(
// BIN6-DAG: vmin(
// BIN6-DAG: CreatePredicate<float>(
// BIN6: __VEC_SCOPE__ {
// BIN6-NOT: vlds_mask(
// BIN6-NOT: PTOAS__OPLIB_

// SOFTMAX: vlds(
// SOFTMAX: vsts(
// SOFTMAX: vadd(
// SOFTMAX: CreatePredicate<float>(
// SOFTMAX: __VEC_SCOPE__ {
// SOFTMAX-NOT: vlds_mask(
// SOFTMAX-NOT: PTOAS__OPLIB_
