// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: mkdir -p %t.dir/families
// RUN: cp %S/resources/good_select_scalar_template.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: { ptoas %s --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: pto.oplib.instance.kind = "l3_select_scalar_template"
// CHECK-DAG: pto.oplib.instance.op = "tsels"
// CHECK-DAG: pto.oplib.instance.dtype = "f32"
// CHECK-DAG: pto.oplib.instance.variant_id = "scalar_mode"
// CHECK: call @__pto_oplib_inst_l3_select_scalar_template_tsels_scalar_mode(

module {
  func.func @select_scalar_family_f32(
      %in0: !pto.ptr<f32>,
      %in1: !pto.ptr<f32>,
      %out0: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %select_mode = arith.constant 1 : i32

    %in0_tv = pto.make_tensor_view %in0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %in1_tv = pto.make_tensor_view %in1, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>
    %out0_tv = pto.make_tensor_view %out0, shape = [%c32, %c32], strides = [%c32, %c1] : !pto.tensor_view<?x?xf32>

    %in0_pt = pto.partition_view %in0_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %in1_pt = pto.partition_view %in1_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %out0_pt = pto.partition_view %out0_tv, offsets = [%c0, %c0], sizes = [%c32, %c32] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %a = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %b = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %d0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%in0_pt : !pto.partition_tensor_view<32x32xf32>) outs(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%in1_pt : !pto.partition_tensor_view<32x32xf32>) outs(%b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.tsels ins(%a, %b, %select_mode : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, i32) outs(%d0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    pto.tstore ins(%d0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%out0_pt : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
