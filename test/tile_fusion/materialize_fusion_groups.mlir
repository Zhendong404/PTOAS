// RUN: ptoas %S/softmax_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s

// CHECK: [op-fusion] outlined group_id=0 into @__pto_fused_group_0_0
// CHECK-DAG: [op-fusion] imported OP-Lib entry: role=variant op=tmax
// CHECK-DAG: [op-fusion] imported OP-Lib entry: role=variant op=tadd
// CHECK-DAG: [op-fusion] imported OP-Lib entry: role=seed seed_id=seed_vec_bin_core
// CHECK: [op-fusion] selected variant: op=tadd dtype=f32 variant_id=v_tadd_f32_fast
// CHECK: [op-fusion] selected variant: op=tmul dtype=f32 variant_id=__seed__seed_vec_bin_core__tmul__f32
// CHECK: [op-fusion] inline-libcall touched 2 function(s), inlined 6 call(s)
