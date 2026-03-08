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
        pto.oplib.priority = 10 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %total = arith.constant 1024 : index

    %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

    %zero = arith.constant dense<0.0> : vector<64xf32>
    pto.simd.vec_scope {
      scf.for %i = %c0 to %total step %c64 {
        %remain = arith.subi %total, %i : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %a = vector.maskedload %flat0[%i], %mask, %zero : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
        %b = vector.maskedload %flat1[%i], %mask, %zero : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
        %c = arith.maximumf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>
        vector.maskedstore %flatd[%i], %mask, %c : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
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
        pto.oplib.priority = 5 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %total = arith.constant 1024 : index

    %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

    %zero = arith.constant dense<0.0> : vector<64xf32>
    pto.simd.vec_scope {
      scf.for %i = %c0 to %total step %c64 {
        %remain = arith.subi %total, %i : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %a = vector.maskedload %flat0[%i], %mask, %zero : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
        %b = vector.maskedload %flat1[%i], %mask, %zero : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
        %c = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>
        vector.maskedstore %flatd[%i], %mask, %c : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
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
        pto.oplib.seed.support_dtypes = ["f32"],
        pto.oplib.seed.support_ops = ["tadd", "tsub", "tmul", "tdiv", "tmax", "tmin"],
        pto.oplib.seed.core_slot = "binary_ewise_core",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "row_major",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %total = arith.constant 1024 : index

    %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [%total], strides: [1] : memref<32x32xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>

    %zero = arith.constant dense<0.0> : vector<64xf32>
    pto.simd.vec_scope {
      scf.for %i = %c0 to %total step %c64 {
        %remain = arith.subi %total, %i : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %a = vector.maskedload %flat0[%i], %mask, %zero : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
        %b = vector.maskedload %flat1[%i], %mask, %zero : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
        %c = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<64xf32>
        vector.maskedstore %flatd[%i], %mask, %c : memref<?xf32, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }

  func.func private @__pto_oplib_seed_vec_bin_core_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "seed",
        pto.oplib.seed_id = "seed_vec_bin_core_f16",
        pto.oplib.seed_dtype = "f16",
        pto.oplib.seed.support_dtypes = ["f16"],
        pto.oplib.seed.support_ops = ["tadd", "tsub", "tmul", "tdiv", "tmax", "tmin"],
        pto.oplib.seed.core_slot = "binary_ewise_core",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "row_major",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<64x64xf16, strided<[64, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<64x64xf16, strided<[64, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<64x64xf16, strided<[64, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %total = arith.constant 4096 : index

    %flat0 = memref.reinterpret_cast %m0 to offset: [0], sizes: [%total], strides: [1] : memref<64x64xf16, strided<[64, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf16, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flat1 = memref.reinterpret_cast %m1 to offset: [0], sizes: [%total], strides: [1] : memref<64x64xf16, strided<[64, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf16, strided<[1], offset: ?>, #pto.address_space<vec>>
    %flatd = memref.reinterpret_cast %md to offset: [0], sizes: [%total], strides: [1] : memref<64x64xf16, strided<[64, 1], offset: 0>, #pto.address_space<vec>> to memref<?xf16, strided<[1], offset: ?>, #pto.address_space<vec>>

    %zero = arith.constant dense<0.0> : vector<128xf16>
    pto.simd.vec_scope {
      scf.for %i = %c0 to %total step %c128 {
        %remain = arith.subi %total, %i : index
        %lt = arith.cmpi slt, %remain, %c128 : index
        %active = arith.select %lt, %remain, %c128 : index
        %mask = vector.create_mask %active : vector<128xi1>
        %a = vector.maskedload %flat0[%i], %mask, %zero : memref<?xf16, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
        %b = vector.maskedload %flat1[%i], %mask, %zero : memref<?xf16, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
        %c = arith.addf %a, %b {pto.simd.core_slot = "binary_ewise_core"} : vector<128xf16>
        vector.maskedstore %flatd[%i], %mask, %c : memref<?xf16, strided<[1], offset: ?>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
      }
    }
    return
  }
}
