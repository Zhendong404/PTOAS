// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @abs_tload_contract
// CHECK: a5vm.load
// CHECK-SAME: layout = "nd"
// CHECK-SAME: src_shape = [32, 32]
// CHECK-SAME: src_strides = [32, 1]
// CHECK-SAME: tile_layout = "row_major"
// CHECK-SAME: tile_domain = "vec"
// CHECK-SAME: valid_rows = 32
// CHECK-SAME: valid_cols = 32
// CHECK-SAME: pad_mode = "none"
// CHECK-SAME: has_pad_value = false
// CHECK-SAME: left_padding_present = false
// CHECK-SAME: right_padding_present = false
// CHECK-SAME: init_out_buffer = false
// CHECK-SAME: has_init_condition = false
// CHECK-SAME: trace_offsets = [0, 0]
// CHECK-SAME: trace_sizes = [32, 32]

module {
  func.func @abs_tload_contract(%src: !pto.ptr<f32>, %index: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %tv = pto.make_tensor_view %src, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %slice = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%slice : !pto.partition_tensor_view<32x32xf32>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
