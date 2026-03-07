// RUN: { ptoas %S/binary_max_min_chain.pto --op-lib-dir=%S/oplib --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmin__f32(
// CHECK-SAME: pto.oplib.instance.from_seed = true
// CHECK-SAME: pto.oplib.instance.op = "tmin"
// CHECK-LABEL: func.func private @__pto_oplib_inst_v_tmax_f32_fast(
// CHECK-SAME: pto.oplib.instance.from_seed = false
// CHECK-SAME: pto.oplib.instance.op = "tmax"
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK: arith.maximumf
// CHECK: arith.minimumf
