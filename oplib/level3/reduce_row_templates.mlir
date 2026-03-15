// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: reduce_row
// Source template: skeletons/reduce_row.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from reduction row skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = reduce_row
  // axes = dtype=f32, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_trowsum_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowsum",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %init = arith.constant 0.0 : f32
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      %outMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %reduced = scf.for %cidx = %c0 to %cols step %c64 iter_args(%acc = %init) -> (f32) {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = vector.reduction <add>, %lhs, %acc : vector<64xf32> into f32
          scf.yield %result : f32
        }
        %outVec = vector.splat %reduced : vector<64xf32>
        vector.maskedstore %md[%r, %c0], %outMask, %outVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }

  // family_id = reduce_row
  // axes = dtype=f32, core_op=arith.maximumf, variant_id=linear
  func.func private @__pto_oplib_variant_trowmax_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowmax",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %init = arith.constant -3.40282347E+38 : f32
    pto.simd.vec_scope {
      %passive = arith.constant dense<-3.40282347E+38> : vector<64xf32>
      %outMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %reduced = scf.for %cidx = %c0 to %cols step %c64 iter_args(%acc = %init) -> (f32) {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = vector.reduction <maximumf>, %lhs, %acc : vector<64xf32> into f32
          scf.yield %result : f32
        }
        %outVec = vector.splat %reduced : vector<64xf32>
        vector.maskedstore %md[%r, %c0], %outMask, %outVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }

  // family_id = reduce_row
  // axes = dtype=f32, core_op=arith.minimumf, variant_id=linear
  func.func private @__pto_oplib_variant_trowmin_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_reduce_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowmin",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %init = arith.constant 3.40282347E+38 : f32
    pto.simd.vec_scope {
      %passive = arith.constant dense<3.40282347E+38> : vector<64xf32>
      %outMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %reduced = scf.for %cidx = %c0 to %cols step %c64 iter_args(%acc = %init) -> (f32) {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = vector.reduction <minimumf>, %lhs, %acc : vector<64xf32> into f32
          scf.yield %result : f32
        }
        %outVec = vector.splat %reduced : vector<64xf32>
        vector.maskedstore %md[%r, %c0], %outMask, %outVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
      }
    }
    return
  }
}
