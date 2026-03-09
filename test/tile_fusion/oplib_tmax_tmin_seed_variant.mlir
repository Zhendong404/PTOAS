// RUN: { ptoas %S/binary_max_min_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOLowLevelLoopFusion
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmin__f32(
// CHECK-SAME: pto.oplib.instance.from_seed = true
// CHECK-SAME: pto.oplib.instance.op = "tmin"
// CHECK-LABEL: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmax__f32(
// CHECK-SAME: pto.oplib.instance.from_seed = true
// CHECK-SAME: pto.oplib.instance.op = "tmax"
// CHECK-LABEL: func.func private @__pto_fused_group_0_0(
// CHECK: arith.maximumf
// CHECK: arith.minimumf
