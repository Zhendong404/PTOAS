// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: compare
// Source template: skeletons/compare.instance.tmpl.mlir
// Axes: dtype, condition, variant_id
// Output role: importer-active concrete templates synchronized from compare skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = cmp_tile_scalar
  // axes = dtype=f16, condition=EQ, core_op=arith.cmpf, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7E00> : vector<64xf16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %cmp = arith.cmpf oeq, %lhs, %scalarVec : vector<64xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f16, condition=NE, core_op=arith.cmpf, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7E00> : vector<64xf16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %cmp = arith.cmpf one, %lhs, %scalarVec : vector<64xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f16, condition=LT, core_op=arith.cmpf, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7E00> : vector<64xf16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %cmp = arith.cmpf olt, %lhs, %scalarVec : vector<64xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f16, condition=LE, core_op=arith.cmpf, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7E00> : vector<64xf16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %cmp = arith.cmpf ole, %lhs, %scalarVec : vector<64xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f16, condition=GT, core_op=arith.cmpf, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7E00> : vector<64xf16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %cmp = arith.cmpf ogt, %lhs, %scalarVec : vector<64xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f16, condition=GE, core_op=arith.cmpf, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7E00> : vector<64xf16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf16> into vector<64xf16>
          %cmp = arith.cmpf oge, %lhs, %scalarVec : vector<64xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f32, condition=EQ, core_op=arith.cmpf, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7FC00000> : vector<64xf32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %cmp = arith.cmpf oeq, %lhs, %scalarVec : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f32, condition=NE, core_op=arith.cmpf, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7FC00000> : vector<64xf32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %cmp = arith.cmpf one, %lhs, %scalarVec : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f32, condition=LT, core_op=arith.cmpf, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7FC00000> : vector<64xf32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %cmp = arith.cmpf olt, %lhs, %scalarVec : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f32, condition=LE, core_op=arith.cmpf, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7FC00000> : vector<64xf32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %cmp = arith.cmpf ole, %lhs, %scalarVec : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f32, condition=GT, core_op=arith.cmpf, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7FC00000> : vector<64xf32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %cmp = arith.cmpf ogt, %lhs, %scalarVec : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=f32, condition=GE, core_op=arith.cmpf, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: f32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0x7FC00000> : vector<64xf32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %cmp = arith.cmpf oge, %lhs, %scalarVec : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i16, condition=EQ, core_op=arith.cmpi, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi eq, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i16, condition=NE, core_op=arith.cmpi, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi ne, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i16, condition=LT, core_op=arith.cmpi, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi slt, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i16, condition=LE, core_op=arith.cmpi, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi sle, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i16, condition=GT, core_op=arith.cmpi, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi sgt, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i16, condition=GE, core_op=arith.cmpi, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi sge, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i32, condition=EQ, core_op=arith.cmpi, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi eq, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i32, condition=NE, core_op=arith.cmpi, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi ne, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i32, condition=LT, core_op=arith.cmpi, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi slt, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i32, condition=LE, core_op=arith.cmpi, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi sle, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i32, condition=GT, core_op=arith.cmpi, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi sgt, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i32, condition=GE, core_op=arith.cmpi, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi sge, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i8, condition=EQ, core_op=arith.cmpi, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi eq, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i8, condition=NE, core_op=arith.cmpi, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi ne, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i8, condition=LT, core_op=arith.cmpi, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi slt, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i8, condition=LE, core_op=arith.cmpi, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi sle, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i8, condition=GT, core_op=arith.cmpi, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi sgt, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=i8, condition=GE, core_op=arith.cmpi, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi sge, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u16, condition=EQ, core_op=arith.cmpi, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi eq, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u16, condition=NE, core_op=arith.cmpi, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi ne, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u16, condition=LT, core_op=arith.cmpi, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi ult, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u16, condition=LE, core_op=arith.cmpi, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi ule, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u16, condition=GT, core_op=arith.cmpi, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi ugt, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u16, condition=GE, core_op=arith.cmpi, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i16,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi16>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi16> into vector<64xi16>
          %cmp = arith.cmpi uge, %lhs, %scalarVec : vector<64xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u32, condition=EQ, core_op=arith.cmpi, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi eq, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u32, condition=NE, core_op=arith.cmpi, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi ne, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u32, condition=LT, core_op=arith.cmpi, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi ult, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u32, condition=LE, core_op=arith.cmpi, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi ule, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u32, condition=GT, core_op=arith.cmpi, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi ugt, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u32, condition=GE, core_op=arith.cmpi, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i32,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %cmp = arith.cmpi uge, %lhs, %scalarVec : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u8, condition=EQ, core_op=arith.cmpi, variant_id=eq
  func.func private @__pto_oplib_variant_tcmps_eq_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "eq",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.cmp_mode = "EQ",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi eq, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u8, condition=NE, core_op=arith.cmpi, variant_id=ne
  func.func private @__pto_oplib_variant_tcmps_ne_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ne",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.cmp_mode = "NE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi ne, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u8, condition=LT, core_op=arith.cmpi, variant_id=lt
  func.func private @__pto_oplib_variant_tcmps_lt_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "lt",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.cmp_mode = "LT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi ult, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u8, condition=LE, core_op=arith.cmpi, variant_id=le
  func.func private @__pto_oplib_variant_tcmps_le_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "le",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.cmp_mode = "LE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi ule, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u8, condition=GT, core_op=arith.cmpi, variant_id=gt
  func.func private @__pto_oplib_variant_tcmps_gt_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "gt",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.cmp_mode = "GT",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi ugt, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }

  // family_id = cmp_tile_scalar
  // axes = dtype=u8, condition=GE, core_op=arith.cmpi, variant_id=ge
  func.func private @__pto_oplib_variant_tcmps_ge_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_cmp_tile_scalar_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tcmps",
        pto.oplib.variant_id = "ge",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.cmp_mode = "GE",
        pto.oplib.match.scalar_pos = 1 : i64,
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

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi8>
      %zeroI8 = arith.constant dense<0> : vector<64xi8>
      %oneI8 = arith.constant dense<1> : vector<64xi8>
      %scalarVec = vector.splat %scalar : vector<64xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8> into vector<64xi8>
          %cmp = arith.cmpi uge, %lhs, %scalarVec : vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi8>
        }
      }
    }
    return
  }
}
