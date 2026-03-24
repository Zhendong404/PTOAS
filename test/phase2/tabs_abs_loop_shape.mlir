// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @tabs_abs_loop_shape
// CHECK: scf.for %[[DUMMY:[^ ]+]] = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} {
// CHECK: scf.for %[[CHUNK:[^ ]+]] = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}}
// CHECK: a5vm.vlds
// CHECK: a5vm.vabs
// CHECK: a5vm.vsts
// CHECK-NOT: a5vm.scope = "__VEC_SCOPE__"
// CHECK-NOT: dist = "__VEC_SCOPE__"
// CHECK: } {
// CHECK-NEXT: } {llvm.loop.aivector_scope}
// CHECK-NOT: emitc.call_opaque "TABS"

module {
  func.func @tabs_abs_loop_shape() {
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tabs ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}

// The chosen lowered loop carries explicit AIV vec-scope semantics through
// llvm.loop.aivector_scope on the carrier loop.
// This contract therefore locks both loop ownership and the ordered
// a5vm.vlds -> a5vm.vabs -> a5vm.vsts vector primitive sequence.
