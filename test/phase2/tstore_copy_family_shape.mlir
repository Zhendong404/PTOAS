// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @tstore_copy_family_shape
// CHECK-DAG: %[[ZERO_I64:.*]] = arith.constant 0 : i64
// CHECK-DAG: %[[STRIDE_BYTES:.*]] = arith.constant 128 : i64
// CHECK-DAG: %[[LOOP_STRIDE:.*]] = arith.constant 4096 : i64
// CHECK-DAG: %[[NBURST:.*]] = arith.constant 32 : i64
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[GM_BASE_BYTES:.*]] = pto.castptr %arg0 : !pto.ptr<f32, gm> -> !pto.ptr<i8, gm>
// CHECK: %[[GM_OFFSET_PTR:.*]] = pto.addptr %[[GM_BASE_BYTES]], %c0 : <i8, gm> -> <i8, gm>
// CHECK: pto.set_loop_size_ubtoout %[[C1_I64]], %[[C1_I64]]
// CHECK: pto.set_loop1_stride_ubtoout %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: pto.set_loop2_stride_ubtoout %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: %[[GM_TYPED_PTR:.*]] = pto.castptr %[[GM_OFFSET_PTR]] : !pto.ptr<i8, gm> -> !pto.ptr<f32, gm>
// CHECK: pto.copy_ubuf_to_gm %{{.*}}, %[[GM_TYPED_PTR]], %[[ZERO_I64]], %[[NBURST]], %[[STRIDE_BYTES]], %[[ZERO_I64]], %[[STRIDE_BYTES]], %[[STRIDE_BYTES]]
// CHECK-SAME: : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
// CHECK-NOT: g_shape =
// CHECK-NOT: g_strides =
// CHECK-NOT: dst_strides =
// CHECK-NOT: trace_offsets =
// CHECK-NOT: trace_sizes =

module {
  func.func @tstore_copy_family_shape(%dst: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %tv = pto.make_tensor_view %dst, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %slice = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%slice : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
