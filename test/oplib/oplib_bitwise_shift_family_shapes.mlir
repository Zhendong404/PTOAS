// RUN: { ptoas %S/bitwise_shift_family_static.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=STATIC
// RUN: { ptoas %S/bitwise_shift_family_dynamic_vshape.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=DYNAMIC
// RUN: { ptoas %S/bitwise_shift_family_i16_smoke.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s --check-prefix=I16

// STATIC-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// STATIC-DAG: pto.oplib.instance.op = "tand"
// STATIC-DAG: pto.oplib.instance.op = "tor"
// STATIC-DAG: pto.oplib.instance.op = "txor"
// STATIC-DAG: pto.oplib.instance.op = "tshl"
// STATIC-DAG: pto.oplib.instance.op = "tshr"
// STATIC-DAG: pto.oplib.instance.op = "tands"
// STATIC-DAG: pto.oplib.instance.op = "tors"
// STATIC-DAG: pto.oplib.instance.op = "txors"
// STATIC-DAG: pto.oplib.instance.op = "tshls"
// STATIC-DAG: pto.oplib.instance.op = "tshrs"
// STATIC-DAG: pto.oplib.instance.op = "tnot"

// DYNAMIC-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// DYNAMIC-DAG: pto.oplib.instance.op = "tand"
// DYNAMIC-DAG: pto.oplib.instance.op = "tor"
// DYNAMIC-DAG: pto.oplib.instance.op = "txor"
// DYNAMIC-DAG: pto.oplib.instance.op = "tshl"
// DYNAMIC-DAG: pto.oplib.instance.op = "tshr"
// DYNAMIC-DAG: pto.oplib.instance.op = "tands"
// DYNAMIC-DAG: pto.oplib.instance.op = "tors"
// DYNAMIC-DAG: pto.oplib.instance.op = "txors"
// DYNAMIC-DAG: pto.oplib.instance.op = "tshls"
// DYNAMIC-DAG: pto.oplib.instance.op = "tshrs"
// DYNAMIC-DAG: pto.oplib.instance.op = "tnot"

// I16-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// I16-DAG: pto.oplib.instance.op = "tand"
// I16-DAG: pto.oplib.instance.op = "tshrs"
// I16-DAG: pto.oplib.instance.op = "tnot"
