// RUN: FileCheck %s --check-prefix=CMP-TILE < %S/../../oplib/level3/cmp_tile_tile_templates.mlir
// RUN: FileCheck %s --check-prefix=CMP-SCALAR < %S/../../oplib/level3/cmp_tile_scalar_templates.mlir
// RUN: FileCheck %s --check-prefix=SEL-MASK < %S/../../oplib/level3/select_mask_templates.mlir

// CMP-TILE: %zeroI8 = arith.constant dense<0> : vector<64xi8>
// CMP-TILE: %oneI8 = arith.constant dense<1> : vector<64xi8>
// CMP-TILE: %mask = vector.create_mask %active : vector<64xi1>
// CMP-TILE: vector.maskedstore %md[%r, %cidx], %mask, %zeroI8
// CMP-TILE: vector.maskedstore %md[%r, %cidx], %cmp, %oneI8

// CMP-SCALAR: %zeroI8 = arith.constant dense<0> : vector<64xi8>
// CMP-SCALAR: %oneI8 = arith.constant dense<1> : vector<64xi8>
// CMP-SCALAR: %mask = vector.create_mask %active : vector<64xi1>
// CMP-SCALAR: vector.maskedstore %md[%r, %cidx], %mask, %zeroI8
// CMP-SCALAR: vector.maskedstore %md[%r, %cidx], %cmp, %oneI8

// SEL-MASK: %passiveMask = arith.constant dense<0> : vector<64xi8>
// SEL-MASK: %zeroMask = arith.constant dense<0> : vector<64xi8>
// SEL-MASK: %laneMask = vector.create_mask %active : vector<64xi1>
// SEL-MASK: %maskBytes = vector.maskedload %m0[%r, %cidx], %laneMask, %passiveMask
// SEL-MASK: %maskVec = arith.cmpi ne, %maskBytes, %zeroMask : vector<64xi8>
// SEL-MASK: %result = arith.select %maskVec, %lhs, %rhs
