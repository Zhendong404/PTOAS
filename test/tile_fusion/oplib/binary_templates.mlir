module {
  func.func @__pto_oplib_variant_tmax_f32(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmax",
        pto.oplib.variant_id = "v_tmax_f32_fast",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "any",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 10 : i64
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
    %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %v = arith.maximumf %a, %b : f32
        memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func @__pto_oplib_variant_tadd_f32(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.kind = "l3_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tadd",
        pto.oplib.variant_id = "v_tadd_f32_fast",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.rows = -1 : i64,
        pto.oplib.match.cols = -1 : i64,
        pto.oplib.match.blayout = "any",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 2 : i64,
        pto.oplib.priority = 5 : i64
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
    %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %v = arith.addf %a, %b : f32
        memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func @__pto_oplib_seed_vec_bin_core(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
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
        pto.oplib.match.blayout = "any",
        pto.oplib.match.slayout = "any",
        pto.oplib.match.fractal = -1 : i64,
        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
    %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %v = arith.addf %a, %b {pto.oplib.core_slot = "binary_ewise_core"} : f32
        memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      }
    }
    return
  }
}
