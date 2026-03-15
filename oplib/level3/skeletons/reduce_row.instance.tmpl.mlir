  // family_id = @@FAMILY_ID@@
  // axes = @@AXIS_VALUES@@
  func.func private @@@FUNC_NAME@@(
@@ARG_DECLS@@
      ) attributes {
        pto.oplib.kind = "@@KIND@@",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "@@OP_NAME@@",
        pto.oplib.variant_id = "@@VARIANT_ID@@",
        pto.oplib.match.dtype = "@@MATCH_DTYPE@@",
@@MATCH_ATTRS@@
        pto.oplib.cost = @@COST@@ : i64,
        pto.oplib.priority = @@PRIORITY@@ : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : @@INPUT_TILE_TYPE@@ to @@INPUT_MEMREF_TYPE@@
    %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : @@INPUT_MEMREF_TYPE@@
    %cols = memref.dim %m0, %c1 : @@INPUT_MEMREF_TYPE@@
    %init = arith.constant @@REDUCE_INIT@@ : @@SCALAR_TYPE@@
    pto.simd.vec_scope {
      %passive = arith.constant @@PASSIVE_VECTOR@@ : @@INPUT_VECTOR_TYPE@@
      %outMask = vector.create_mask %c1 : @@MASK_VECTOR_TYPE@@
@@EXTRA_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        %reduced = scf.for %cidx = %c0 to %cols step %c64 iter_args(%acc = %init) -> (@@SCALAR_TYPE@@) {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : @@MASK_VECTOR_TYPE@@
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : @@INPUT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@INPUT_VECTOR_TYPE@@ into @@INPUT_VECTOR_TYPE@@
@@COMPUTE@@          scf.yield %result : @@SCALAR_TYPE@@
        }
        %outVec = vector.splat %reduced : @@INPUT_VECTOR_TYPE@@
        vector.maskedstore %md[%r, %c0], %outMask, %outVec {pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@INPUT_VECTOR_TYPE@@
      }
    }
    return
  }
