// RUN: { ptoas %S/reduction_broadcast_family_static.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_reduce_row_template_trowsum___seed__seed_l3_reduce_row_core__trowsum__f32(
// CHECK: pto.oplib.instance.from_seed = true
// CHECK: pto.oplib.instance.op = "trowsum"
// CHECK: pto.oplib.instance.seed_id = "seed_l3_reduce_row_core"

// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_reduce_row_template_trowmax___seed__seed_l3_reduce_row_core__trowmax__f32(
// CHECK: pto.oplib.instance.from_seed = true
// CHECK: pto.oplib.instance.op = "trowmax"
// CHECK: pto.oplib.instance.seed_id = "seed_l3_reduce_row_core"

// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_reduce_row_template_trowmin___seed__seed_l3_reduce_row_core__trowmin__f32(
// CHECK: pto.oplib.instance.from_seed = true
// CHECK: pto.oplib.instance.op = "trowmin"
// CHECK: pto.oplib.instance.seed_id = "seed_l3_reduce_row_core"

// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_reduce_col_template_tcolmax___seed__seed_l3_reduce_col_core__tcolmax__f32(
// CHECK: pto.oplib.instance.from_seed = true
// CHECK: pto.oplib.instance.op = "tcolmax"
// CHECK: pto.oplib.instance.seed_id = "seed_l3_reduce_col_core"

// CHECK-LABEL: func.func private @__pto_oplib_inst_l3_reduce_col_template_tcolmin___seed__seed_l3_reduce_col_core__tcolmin__f32(
// CHECK: pto.oplib.instance.from_seed = true
// CHECK: pto.oplib.instance.op = "tcolmin"
// CHECK: pto.oplib.instance.seed_id = "seed_l3_reduce_col_core"
