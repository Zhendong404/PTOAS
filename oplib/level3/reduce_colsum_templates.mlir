// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: reduce_colsum
// Source template: skeletons/reduce_colsum.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from reduction column-sum skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = reduce_colsum
  // axes = dtype=bf16, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_bf16(
      %src0: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "bf16",
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
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xbf16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xbf16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xbf16> into vector<64xbf16>
          %result = arith.addf %lhs, %acc {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xbf16>
          scf.yield %result : vector<64xbf16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xbf16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=bf16, core_op=arith.addf, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_bf16(
      %src0: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
        pto.oplib.match.dtype = "bf16",
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
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=bf16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xbf16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xbf16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xbf16> into vector<64xbf16>
          %result = arith.addf %lhs, %acc {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xbf16>
          scf.yield %result : vector<64xbf16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xbf16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=f16, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "f16",
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
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xf16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %result = arith.addf %lhs, %acc {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf16>
          scf.yield %result : vector<64xf16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=f16, core_op=arith.addf, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
        pto.oplib.match.dtype = "f16",
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
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xf16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %result = arith.addf %lhs, %acc {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf16>
          scf.yield %result : vector<64xf16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=f32, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
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
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xf32>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = arith.addf %lhs, %acc {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          scf.yield %result : vector<64xf32>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=f32, core_op=arith.addf, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
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
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xf32>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = arith.addf %lhs, %acc {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          scf.yield %result : vector<64xf32>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=i16, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "i16",
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
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %result = arith.addi %lhs, %acc : vector<64xi16>
          scf.yield %result : vector<64xi16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=i16, core_op=arith.addi, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
        pto.oplib.match.dtype = "i16",
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
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %result = arith.addi %lhs, %acc : vector<64xi16>
          scf.yield %result : vector<64xi16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=i32, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
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
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi32>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %result = arith.addi %lhs, %acc : vector<64xi32>
          scf.yield %result : vector<64xi32>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=i32, core_op=arith.addi, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
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
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi32>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %result = arith.addi %lhs, %acc : vector<64xi32>
          scf.yield %result : vector<64xi32>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=i8, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "i8",
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
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi8>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %result = arith.addi %lhs, %acc : vector<64xi8>
          scf.yield %result : vector<64xi8>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=i8, core_op=arith.addi, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
        pto.oplib.match.dtype = "i8",
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
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi8>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %result = arith.addi %lhs, %acc : vector<64xi8>
          scf.yield %result : vector<64xi8>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=u16, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "u16",
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
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %result = arith.addi %lhs, %acc : vector<64xi16>
          scf.yield %result : vector<64xi16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=u16, core_op=arith.addi, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
        pto.oplib.match.dtype = "u16",
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
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi16>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %result = arith.addi %lhs, %acc : vector<64xi16>
          scf.yield %result : vector<64xi16>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=u32, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "u32",
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
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi32>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %result = arith.addi %lhs, %acc : vector<64xi32>
          scf.yield %result : vector<64xi32>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=u32, core_op=arith.addi, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
        pto.oplib.match.dtype = "u32",
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
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi32>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %result = arith.addi %lhs, %acc : vector<64xi32>
          scf.yield %result : vector<64xi32>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=u8, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_tcolsum_linear_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "linear",
        pto.oplib.match.dtype = "u8",
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
        pto.oplib.match.is_binary = false,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi8>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %result = arith.addi %lhs, %acc : vector<64xi8>
          scf.yield %result : vector<64xi8>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
      }
    }
    return
  }

  // family_id = reduce_colsum
  // axes = dtype=u8, core_op=arith.addi, variant_id=binary
  func.func private @__pto_oplib_variant_tcolsum_binary_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_colsum_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcolsum",
        pto.oplib.variant_id = "binary",
        pto.oplib.match.dtype = "u8",
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
        pto.oplib.match.is_binary = true,

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      scf.for %cidx = %c0 to %cols step %c64 {
        %remain = arith.subi %cols, %cidx : index
        %lt = arith.cmpi slt, %remain, %c64 : index
        %active = arith.select %lt, %remain, %c64 : index
        %mask = vector.create_mask %active : vector<64xi1>
        %reduced = scf.for %r = %c0 to %rows step %c1 iter_args(%acc = %passive) -> (vector<64xi8>) {
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %result = arith.addi %lhs, %acc : vector<64xi8>
          scf.yield %result : vector<64xi8>
        }
        vector.maskedstore %md[%c0, %cidx], %mask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
      }
    }
    return
  }
}
