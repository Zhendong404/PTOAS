// RUN: { ptoas %S/generic_shape_64x32_f32_dynamic_vshape.pto --op-lib-dir=%S/oplib --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-LABEL: func.func private @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK: pto.simd.tile_to_memref {{.*}} to memref<?x?xf32, strided<[32, 1]>, #pto.address_space<vec>>

// CHECK-LABEL: IR Dump After {anonymous}::PTOTileBufToMemrefPass
// CHECK-LABEL: func.func private @__pto_oplib_inst_v_tadd_f32_fast(
// CHECK-SAME: memref<?x?xf32, strided<[32, 1]>, #pto.address_space<vec>>
