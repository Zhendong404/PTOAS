module {
  func.func private @__pto_oplib_variant_tmax_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmax",
        pto.oplib.variant_id = "v_tmax_f32_fast",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "row_major",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 10 : i64,
        pto.simd.level = "binary_ewise_v1",
        pto.simd.lanes = 64 : i64
      } {
    %m0 = builtin.unrealized_conversion_cast %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
    %m1 = builtin.unrealized_conversion_cast %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
    %md = builtin.unrealized_conversion_cast %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>

    %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c1024 step %c64 {
      %remain = arith.subi %c1024, %i : index
      %mask = pto.simd.predicate %remain : index -> vector<64xi1>
      %a = pto.simd.load %flat0, %i, %mask : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
      %b = pto.simd.load %flat1, %i, %mask : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
      %c = arith.maximumf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>
      pto.simd.store %c, %flatd, %i, %mask : vector<64xf32>, memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1>
    }
    return
  }

  func.func private @__pto_oplib_variant_tadd_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tadd",
        pto.oplib.variant_id = "v_tadd_f32_fast",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "row_major",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 2 : i64,
        pto.oplib.priority = 5 : i64,
        pto.simd.level = "binary_ewise_v1",
        pto.simd.lanes = 64 : i64
      } {
    %m0 = builtin.unrealized_conversion_cast %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
    %m1 = builtin.unrealized_conversion_cast %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
    %md = builtin.unrealized_conversion_cast %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>

    %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c1024 step %c64 {
      %remain = arith.subi %c1024, %i : index
      %mask = pto.simd.predicate %remain : index -> vector<64xi1>
      %a = pto.simd.load %flat0, %i, %mask : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
      %b = pto.simd.load %flat1, %i, %mask : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>
      %c = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>
      pto.simd.store %c, %flatd, %i, %mask : vector<64xf32>, memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1>
    }
    return
  }

  func.func private @__pto_oplib_seed_vec_bin_core(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "seed",
        pto.oplib.seed_id = "seed_vec_bin_core",
        pto.oplib.seed_dtype = "f32",
        pto.oplib.seed.support_dtypes = ["f16", "f32"],
        pto.oplib.seed.support_ops = ["tadd", "tsub", "tmul", "tdiv", "tmax", "tmin"],
        pto.oplib.seed.core_slot = "binary_ewise_core",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "row_major",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64,
        pto.simd.level = "binary_ewise_v1",
        pto.simd.lanes = 64 : i64
      } {
    %m0 = builtin.unrealized_conversion_cast %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
    %m1 = builtin.unrealized_conversion_cast %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>
    %md = builtin.unrealized_conversion_cast %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>>

    %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [1024], strides: [1] : memref<32x32xf32, strided<[?, ?], offset: ?>, #pto.address_space<vec>> to memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    scf.for %i = %c0 to %c1024 step %c64 {
      %remain = arith.subi %c1024, %i : index
      %mask = pto.simd.predicate %remain : index -> vector<64xi1>
      %a, %off0 = pto.simd.load_pu %flat0, %i, %mask {step = 64 : i64} : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>, index
      %b, %off1 = pto.simd.load_pu %flat1, %i, %mask {step = 64 : i64} : memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> vector<64xf32>, index
      %c = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>
      %off2 = pto.simd.store_pu %c, %flatd, %i, %mask {step = 64 : i64} : vector<64xf32>, memref<1024xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, index, vector<64xi1> -> index
      %use0 = arith.addi %off0, %off1 : index
      %use1 = arith.addi %use0, %off2 : index
      %_ = arith.addi %use1, %i : index
    }
    return
  }
}
