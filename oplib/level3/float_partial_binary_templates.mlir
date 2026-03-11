module {
  func.func private @__pto_oplib_seed_l3_float_partial_binary_core(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "seed",
        pto.oplib.seed_id = "seed_l3_float_partial_binary_core",
        pto.oplib.seed_dtype = "f32",
        pto.oplib.seed.support_dtypes = ["f32"],
        pto.oplib.seed.support_ops = ["tpartadd", "tpartmax", "tpartmin"],
        pto.oplib.seed.core_slot = "binary_ewise_core",
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
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rowsd = memref.dim %md, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %colsd = memref.dim %md, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %rows01Lt = arith.cmpi slt, %rows0, %rows1 : index
    %rows01 = arith.select %rows01Lt, %rows0, %rows1 : index
    %rowsMinLt = arith.cmpi slt, %rows01, %rowsd : index
    %rows = arith.select %rowsMinLt, %rows01, %rowsd : index

    %cols01Lt = arith.cmpi slt, %cols0, %cols1 : index
    %cols01 = arith.select %cols01Lt, %cols0, %cols1 : index
    %colsMinLt = arith.cmpi slt, %cols01, %colsd : index
    %cols = arith.select %colsMinLt, %cols01, %colsd : index

    %zero = arith.constant dense<0.0> : vector<32xf32>
    pto.simd.vec_scope {
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<32xi1>
          %a = vector.maskedload %m0[%r, %cidx], %mask, %zero {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %b = vector.maskedload %m1[%r, %cidx], %mask, %zero {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %d = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core", pto.simd.exec_mode = "MODE_ZEROING"} : vector<32xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %d {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32>
        }
      }
    }
    return
  }
}
