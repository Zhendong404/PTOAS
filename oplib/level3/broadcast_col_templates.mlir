// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: broadcast_col
// Source template: skeletons/broadcast_col.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from broadcast-column skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = broadcast_col
  // axes = dtype=f32, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_tcolexpand_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_col_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolexpand",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %md, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %lhs = vector.maskedload %m0[%c0, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
        scf.for %r = %c0 to %rows step %c1 {
          %result = arith.addf %lhs, %passive : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }
}
