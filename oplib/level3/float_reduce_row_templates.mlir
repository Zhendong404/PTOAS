module {
  func.func private @__pto_oplib_variant_trowsum_f32(
      %src: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %tmp: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_reduce_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowsum",
        pto.oplib.variant_id = "tile_tmp",
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
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %mt = pto.simd.tile_to_memref %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %zeroVec = arith.constant dense<0.0> : vector<32xf32>
      %zeroScalar = arith.constant 0.0 : f32
      %mask1 = vector.create_mask %c1 : vector<32xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %blkCount = scf.for %cidx = %c0 to %cols step %c64 iter_args(%blk = %c0) -> (index) {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<32xi1>
          %a = vector.maskedload %m0[%r, %cidx], %mask, %zeroVec {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %part = vector.reduction <add>, %a : vector<32xf32> into f32
          %partVec = vector.splat %part : vector<32xf32>
          %tidx = arith.muli %blk, %c64 : index
          vector.maskedstore %mt[%r, %tidx], %mask1, %partVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32>
          %nextBlk = arith.addi %blk, %c1 : index
          scf.yield %nextBlk : index
        }
        %rowAcc = scf.for %bi = %c0 to %blkCount step %c1 iter_args(%acc = %zeroScalar) -> (f32) {
          %tidx = arith.muli %bi, %c64 : index
          %pv = vector.maskedload %mt[%r, %tidx], %mask1, %zeroVec {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %part = vector.reduction <add>, %pv : vector<32xf32> into f32
          %next = arith.addf %acc, %part : f32
          scf.yield %next : f32
        }
        %rowVec = vector.splat %rowAcc : vector<32xf32>
        vector.maskedstore %md[%r, %c0], %mask1, %rowVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32>
      }
    }
    return
  }

  func.func private @__pto_oplib_variant_trowmax_f32(
      %src: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %tmp: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_reduce_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowmax",
        pto.oplib.variant_id = "tile_tmp",
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
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %mt = pto.simd.tile_to_memref %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %negInfVec = arith.constant dense<-3.40282347E+38> : vector<32xf32>
      %negInfScalar = arith.constant -3.40282347E+38 : f32
      %mask1 = vector.create_mask %c1 : vector<32xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %blkCount = scf.for %cidx = %c0 to %cols step %c64 iter_args(%blk = %c0) -> (index) {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<32xi1>
          %a = vector.maskedload %m0[%r, %cidx], %mask, %negInfVec {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %part = vector.reduction <maximumf>, %a : vector<32xf32> into f32
          %partVec = vector.splat %part : vector<32xf32>
          %tidx = arith.muli %blk, %c64 : index
          vector.maskedstore %mt[%r, %tidx], %mask1, %partVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32>
          %nextBlk = arith.addi %blk, %c1 : index
          scf.yield %nextBlk : index
        }
        %rowAcc = scf.for %bi = %c0 to %blkCount step %c1 iter_args(%acc = %negInfScalar) -> (f32) {
          %tidx = arith.muli %bi, %c64 : index
          %pv = vector.maskedload %mt[%r, %tidx], %mask1, %negInfVec {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %part = vector.reduction <maximumf>, %pv : vector<32xf32> into f32
          %next = arith.maximumf %acc, %part : f32
          scf.yield %next : f32
        }
        %rowVec = vector.splat %rowAcc : vector<32xf32>
        vector.maskedstore %md[%r, %c0], %mask1, %rowVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32>
      }
    }
    return
  }

  func.func private @__pto_oplib_variant_trowmin_f32(
      %src: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %tmp: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      attributes {
        pto.oplib.kind = "l3_reduce_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowmin",
        pto.oplib.variant_id = "tile_tmp",
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
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %mt = pto.simd.tile_to_memref %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    pto.simd.vec_scope {
      %posInfVec = arith.constant dense<3.40282347E+38> : vector<32xf32>
      %posInfScalar = arith.constant 3.40282347E+38 : f32
      %mask1 = vector.create_mask %c1 : vector<32xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %blkCount = scf.for %cidx = %c0 to %cols step %c64 iter_args(%blk = %c0) -> (index) {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : vector<32xi1>
          %a = vector.maskedload %m0[%r, %cidx], %mask, %posInfVec {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %part = vector.reduction <minimumf>, %a : vector<32xf32> into f32
          %partVec = vector.splat %part : vector<32xf32>
          %tidx = arith.muli %blk, %c64 : index
          vector.maskedstore %mt[%r, %tidx], %mask1, %partVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32>
          %nextBlk = arith.addi %blk, %c1 : index
          scf.yield %nextBlk : index
        }
        %rowAcc = scf.for %bi = %c0 to %blkCount step %c1 iter_args(%acc = %posInfScalar) -> (f32) {
          %tidx = arith.muli %bi, %c64 : index
          %pv = vector.maskedload %mt[%r, %tidx], %mask1, %posInfVec {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32> into vector<32xf32>
          %part = vector.reduction <minimumf>, %pv : vector<32xf32> into f32
          %next = arith.minimumf %acc, %part : f32
          scf.yield %next : f32
        }
        %rowVec = vector.splat %rowAcc : vector<32xf32>
        vector.maskedstore %md[%r, %c0], %mask1, %rowVec {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<32xi1>, vector<32xf32>
      }
    }
    return
  }
}
