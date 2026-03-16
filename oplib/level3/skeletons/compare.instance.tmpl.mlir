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
        pto.oplib.match.cmp_mode = "@@CMP_MODE@@",
@@SCALAR_POS_ATTR@@@@MATCH_ATTRS@@
        pto.oplib.cost = @@COST@@ : i64,
        pto.oplib.priority = @@PRIORITY@@ : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : @@INPUT_TILE_TYPE@@ to @@INPUT_MEMREF_TYPE@@
@@RHS_MEMREF_CAST@@    %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : @@INPUT_MEMREF_TYPE@@
    %cols = memref.dim %m0, %c1 : @@INPUT_MEMREF_TYPE@@
    pto.simd.vec_scope {
      %passive = arith.constant @@PASSIVE_VECTOR@@ : @@INPUT_VECTOR_TYPE@@
      %zeroI8 = arith.constant dense<0> : @@RESULT_VECTOR_TYPE@@
      %oneI8 = arith.constant dense<1> : @@RESULT_VECTOR_TYPE@@
@@RHS_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        scf.for %cidx = %c0 to %cols step %c64 {
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : @@MASK_VECTOR_TYPE@@
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : @@INPUT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@INPUT_VECTOR_TYPE@@ into @@INPUT_VECTOR_TYPE@@
@@RHS_LOAD@@          %cmp = @@CORE_OP@@ @@CMP_PREDICATE@@, %lhs, @@RHS_VALUE@@ : @@INPUT_VECTOR_TYPE@@
          vector.maskedstore %md[%r, %cidx], %mask, %zeroI8 {pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@RESULT_VECTOR_TYPE@@
          vector.maskedstore %md[%r, %cidx], %cmp, %oneI8 {pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@RESULT_VECTOR_TYPE@@
        }
      }
    }
    return
  }
