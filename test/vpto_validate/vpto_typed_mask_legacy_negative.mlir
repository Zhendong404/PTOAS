// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: expected '<'
// CHECK: Error: Failed to parse MLIR.

module {
  func.func @legacy_mask(%arg0: !pto.mask) -> !pto.mask
      attributes {pto.version_selection_applied} {
    return %arg0 : !pto.mask
  }
}
