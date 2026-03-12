// RUN: { ptoas %S/compare_select_family.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: pto.oplib.instance.op = "tsel"
// CHECK-DAG: pto.oplib.instance.variant_id = "mask"
// CHECK-DAG: pto.oplib.instance.op = "tsels"
// CHECK-DAG: pto.oplib.instance.variant_id = "scalar_mode"
// CHECK: call @__pto_oplib_inst_l3_select_mask_template_tsel_mask(
// CHECK: call @__pto_oplib_inst_l3_select_scalar_template_tsels_scalar_mode(
