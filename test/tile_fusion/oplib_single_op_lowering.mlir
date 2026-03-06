// RUN: ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --op-fusion-debug --dump-ir-after-oplib-lowering 2>&1 | FileCheck %s

// CHECK: [op-fusion] selected variant: op=tmul dtype=f32 variant_id=__seed__seed_vec_bin_core__tmul__f32
// CHECK: [op-fusion] materialized single op=tmul in @flash_attention_softmax_block
// CHECK: [op-fusion] selected variant: op=tadd dtype=f32 variant_id=v_tadd_f32_fast
// CHECK: [op-fusion] materialized single op=tadd in @flash_attention_softmax_block
// CHECK: [op-fusion] instantiate+inline touched 1 function(s), inlined 6 call(s)
// CHECK-NOT: [op-fusion] low-level loop fusion changed
