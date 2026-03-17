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
        pto.oplib.match.scalar_pos = @@SCALAR_POS@@ : i64,
@@MATCH_ATTRS@@
        pto.oplib.cost = @@COST@@ : i64,
        pto.oplib.priority = @@PRIORITY@@ : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : @@SRC0_TILE_TYPE@@ to @@SRC0_MEMREF_TYPE@@
    %m1 = pto.simd.tile_to_memref %src1 : @@SRC1_TILE_TYPE@@ to @@SRC1_MEMREF_TYPE@@
    %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %zeroMode = arith.constant dense<0> : @@SCALAR_ARG_VECTOR_TYPE@@
    %rows = memref.dim %m0, %c0 : @@SRC0_MEMREF_TYPE@@
    %cols = memref.dim %m0, %c1 : @@SRC0_MEMREF_TYPE@@
    %repeatTimes = arith.ceildivsi %cols, %c64 : index
    pto.simd.vec_scope {
      %passive = arith.constant @@PASSIVE_VECTOR@@ : @@RESULT_VECTOR_TYPE@@
      %scalarRaw = vector.splat %scalar : @@SCALAR_ARG_VECTOR_TYPE@@
      %scalarVec = arith.cmpi ne, %scalarRaw, %zeroMode : @@SCALAR_ARG_VECTOR_TYPE@@
@@EXTRA_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %c64 : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %c64 : index
          %active = arith.select %lt, %remain, %c64 : index
          %mask = vector.create_mask %active : @@MASK_VECTOR_TYPE@@
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : @@SRC0_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@SRC0_VECTOR_TYPE@@ into @@SRC0_VECTOR_TYPE@@
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : @@SRC1_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@SRC1_VECTOR_TYPE@@ into @@SRC1_VECTOR_TYPE@@
@@COMPUTE@@          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@RESULT_VECTOR_TYPE@@
        }
      }
    }
    return
  }
