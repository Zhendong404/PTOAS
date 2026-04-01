// RUN: bash -lc 'set +e; ptoas --pto-backend=vpto --emit-vpto %s -o - 2>&1; echo EXIT:$?' | FileCheck %s

// CHECK: error: 'pto.pstu' op result #1 must be pointer-like buffer type
// CHECK: EXIT:1
module {
  func.func @pstu_memref_base_out_should_fail(
      %src: !pto.ptr<f32, ub>, %base: !pto.ptr<f32, ub>, %index: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %align = pto.vldas %src : !pto.ptr<f32, ub> -> !pto.align
      %next_align, %next_base = pto.pstu %align, %mask, %base : !pto.align, !pto.mask<b32>, !pto.ptr<f32, ub> -> !pto.align, memref<256xf32, #pto.address_space<vec>>
    }
    return
  }
}
