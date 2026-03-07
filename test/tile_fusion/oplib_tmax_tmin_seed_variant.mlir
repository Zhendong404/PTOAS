// RUN: ptoas %S/binary_max_min_chain.pto --enable-op-fusion --op-lib-dir=%S/oplib --op-fusion-debug --dump-ir-after-op-fusion 2>&1 | FileCheck %s

// CHECK: [op-fusion] found 1 group(s) in @binary_chain_all6
// CHECK: [op-fusion] outlined group_id=0 into @__pto_fused_group_0_0
// CHECK: [op-fusion] selected variant: op=tmax dtype=f32 variant_id=v_tmax_f32_fast cost=1 priority=10 source=variant
// CHECK: [op-fusion] selected variant: op=tmin dtype=f32 variant_id=__seed__seed_vec_bin_core__tmin__f32 cost=10 priority=0 source=seed
// CHECK: [op-fusion] inline-libcall touched 2 function(s), inlined 6 call(s)
