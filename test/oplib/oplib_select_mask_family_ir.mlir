// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: mkdir -p %t.dir/families
// RUN: cp %S/../../oplib/level3/select_mask_templates.mlir %t.dir/
// RUN: cp %S/../../oplib/level3/families/a5_oplib_v1_manifest.yaml %t.dir/families/
// RUN: { ptoas %s --enable-op-fusion --op-lib-dir=%t.dir --pto-arch=a5 --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK-LABEL: IR Dump After PTOInstantiateAndLowerToLibCall
// CHECK-DAG: pto.oplib.instance.kind = "l3_select_mask_template"
// CHECK-DAG: pto.oplib.instance.op = "tsel"
// CHECK-DAG: pto.oplib.instance.dtype = "f32"
// CHECK-DAG: pto.oplib.instance.variant_id = "mask"
// CHECK: call @__pto_oplib_inst_l3_select_mask_template_tsel_mask(

module {
  func.func @select_mask_family_f32() {
    %mask = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %src0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %src1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tsel ins(%mask, %src0, %src1 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
