module {
  func.func private @__pto_oplib_variant_tands_i16_scalar(
      %src: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tands",
        pto.oplib.variant_id = "tile_scalar_i16_scalar",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    scf.for %r = %c0 to %rows step %c1 {
      scf.for %cidx = %c0 to %cols step %c1 {
        %a = memref.load %m0[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
        %d = arith.andi %a, %scalar : i16
        memref.store %d, %md[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func private @__pto_oplib_variant_tors_i16_scalar(
      %src: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tors",
        pto.oplib.variant_id = "tile_scalar_i16_scalar",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    scf.for %r = %c0 to %rows step %c1 {
      scf.for %cidx = %c0 to %cols step %c1 {
        %a = memref.load %m0[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
        %d = arith.ori %a, %scalar : i16
        memref.store %d, %md[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func private @__pto_oplib_variant_txors_i16_scalar(
      %src: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "txors",
        pto.oplib.variant_id = "tile_scalar_i16_scalar",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    scf.for %r = %c0 to %rows step %c1 {
      scf.for %cidx = %c0 to %cols step %c1 {
        %a = memref.load %m0[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
        %d = arith.xori %a, %scalar : i16
        memref.store %d, %md[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func private @__pto_oplib_variant_tshls_i16_scalar(
      %src: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshls",
        pto.oplib.variant_id = "tile_scalar_i16_scalar",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    scf.for %r = %c0 to %rows step %c1 {
      scf.for %cidx = %c0 to %cols step %c1 {
        %a = memref.load %m0[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
        %d = arith.shli %a, %scalar : i16
        memref.store %d, %md[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func private @__pto_oplib_variant_tshrs_i16_scalar(
      %src: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshrs",
        pto.oplib.variant_id = "tile_scalar_i16_scalar",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    scf.for %r = %c0 to %rows step %c1 {
      scf.for %cidx = %c0 to %cols step %c1 {
        %a = memref.load %m0[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
        %d = arith.shrui %a, %scalar : i16
        memref.store %d, %md[%r, %cidx] : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
      }
    }
    return
  }

}
