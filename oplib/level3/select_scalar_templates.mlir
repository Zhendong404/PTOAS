module {
  func.func private @__pto_oplib_variant_tsels_scalar_mode_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %selectMode: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_select_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tsels",
        pto.oplib.variant_id = "scalar_mode",
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
        pto.oplib.match.arg3.rows = -1 : i64,
        pto.oplib.match.arg3.cols = -1 : i64,
        pto.oplib.match.arg3.blayout = "row_major",
        pto.oplib.match.arg3.slayout = "any",
        pto.oplib.match.arg3.fractal = -1 : i64,
        pto.oplib.match.scalar_pos = 2 : i64,
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %modeZero = arith.constant 0 : i32
    %takeSrc0 = arith.cmpi ne, %selectMode, %modeZero : i32
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %zeroF = arith.constant dense<0.0> : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %a = vector.maskedload %m0[%r, %cidx], %mask, %zeroF {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %b = vector.maskedload %m1[%r, %cidx], %mask, %zeroF {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %d = arith.select %takeSrc0, %a, %b : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %d {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }
}
