// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: cp %S/../../oplib/level3/cmp_tile_tile_templates.mlir %t.dir/
// RUN: { ptoas %S/compare_family.pto --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: pto.oplib.instance.kind = "l3_cmp_tile_tile_template"
// CHECK-DAG: pto.oplib.instance.op = "tcmp"
// CHECK-DAG: pto.oplib.instance.variant_id = "lt"
// CHECK: call @__pto_oplib_inst_l3_cmp_tile_tile_template_tcmp_lt(
