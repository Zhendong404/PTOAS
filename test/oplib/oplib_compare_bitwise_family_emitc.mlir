// RUN: { ptoas %S/compare_bitwise_family_emitc.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o %t.cpp 2>&1; } | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=EMITC < %t.cpp

// IR-DAG: pto.oplib.instance.op = "tcmp"
// IR-DAG: pto.oplib.instance.variant_id = "lt"
// IR-DAG: pto.oplib.instance.op = "tsel"
// IR-DAG: pto.oplib.instance.op = "tand"
// IR-DAG: pto.oplib.instance.op = "tshrs"
// IR-DAG: pto.oplib.instance.op = "tnot"

// EMITC: __global__ AICORE void compare_bitwise_family_emitc(
// EMITC-DAG: __VEC_SCOPE__ {
// EMITC-DAG: CreatePredicate<
// EMITC-DAG: vlds(
// EMITC-DAG: vsts(
// EMITC-DAG: vcmp(
// EMITC-DAG: LT
// EMITC-DAG: vand(
// EMITC-DAG: vshr(
// EMITC-DAG: vxor(
