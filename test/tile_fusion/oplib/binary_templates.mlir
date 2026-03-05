module {
  func.func @__pto_oplib_tadd_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tadd",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
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

  func.func @__pto_oplib_tsub_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tsub",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
    %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %v = arith.subf %a, %b : f32
        memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func @__pto_oplib_tmul_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tmul",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
    %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %v = arith.mulf %a, %b : f32
        memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      }
    }
    return
  }

  func.func @__pto_oplib_tdiv_template(
      %src0: memref<?x?xf32, #pto.address_space<vec>>,
      %src1: memref<?x?xf32, #pto.address_space<vec>>,
      %dst: memref<?x?xf32, #pto.address_space<vec>>)
      attributes {
        pto.oplib.op = "tdiv",
        pto.oplib.kind = "binary_elementwise_template",
        pto.oplib.rank = 2 : i64,
        pto.oplib.seed_dtype = "f32"
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m = memref.dim %dst, %c0 : memref<?x?xf32, #pto.address_space<vec>>
    %n = memref.dim %dst, %c1 : memref<?x?xf32, #pto.address_space<vec>>
    scf.for %i = %c0 to %m step %c1 {
      scf.for %j = %c0 to %n step %c1 {
        %a = memref.load %src0[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %b = memref.load %src1[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
        %v = arith.divf %a, %b : f32
        memref.store %v, %dst[%i, %j] : memref<?x?xf32, #pto.address_space<vec>>
      }
    }
    return
  }
}
