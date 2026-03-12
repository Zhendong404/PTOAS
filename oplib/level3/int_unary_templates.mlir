module {
  func.func private @__pto_oplib_variant_tnot_i32(
      %src: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_int_unary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tnot",
        pto.oplib.variant_id = "tile",
        pto.oplib.match.dtype = "i32",
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
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %zero = arith.constant dense<0> : vector<32xi32>
      %allOnes = arith.constant dense<-1> : vector<32xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<32xi1>
          %a = vector.maskedload %m0[%r, %cidx], %mask, %zero {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xi32> into vector<32xi32>
          %d = arith.xori %a, %allOnes : vector<32xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %d {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xi32>
        }
      }
    }
    return
  }
}
