// RUN: { ptoas %S/reduction_broadcast_family_emitc.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o %t.cpp 2>&1; } | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s --check-prefix=EMITC < %t.cpp

// IR-DAG: pto.oplib.instance.op = "trowsum"
// IR-DAG: pto.oplib.instance.op = "tcolsum"
// IR-DAG: pto.oplib.instance.variant_id = "binary_tree"
// IR-DAG: pto.oplib.instance.op = "trowexpandmul"
// IR-DAG: pto.oplib.instance.op = "texpands"

// EMITC: __global__ AICORE void reduction_broadcast_family_emitc_f32(
// EMITC-DAG: __VEC_SCOPE__ {
// EMITC-DAG: vlds(
// EMITC-DAG: vsts(
// EMITC-DAG: vadd(
// EMITC-DAG: vmul(
// EMITC-DAG: vdup(
// EMITC-DAG: CreatePredicate<float>(
