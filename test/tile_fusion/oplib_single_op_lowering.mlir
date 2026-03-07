// RUN: ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --op-fusion-debug --dump-ir-after-oplib-lowering 2>&1 | FileCheck %s

// CHECK: [op-fusion] found 1 group(s) in @flash_attention_softmax_block
// CHECK: [op-fusion] outlined group_id=0 into @__pto_fused_group_0_0
// CHECK-DAG: [op-fusion] selected variant: op=tmul dtype=f32 variant_id=__seed__seed_vec_bin_core__tmul__f32
// CHECK-DAG: [op-fusion] selected variant: op=tadd dtype=f32 variant_id=v_tadd_f32_fast
// CHECK: func.func private @__pto_oplib_inst___seed__seed_vec_bin_core__tmul__f32(!pto.tile_buf
// CHECK: func.func private @__pto_oplib_inst_v_tadd_f32_fast(!pto.tile_buf
// CHECK: call @__pto_fused_group_0_0(
// CHECK-NOT: [op-fusion] inline-libcall touched
// CHECK-NOT: [op-fusion] low-level loop fusion changed
