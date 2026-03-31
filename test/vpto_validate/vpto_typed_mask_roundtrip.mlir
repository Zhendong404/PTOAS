// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @mask_b8(%arg0: !pto.mask<b8>) -> !pto.mask<b8>
// CHECK-LABEL: func.func @mask_b16(%arg0: !pto.mask<b16>) -> !pto.mask<b16>
// CHECK-LABEL: func.func @mask_b32(%arg0: !pto.mask<b32>) -> !pto.mask<b32>

module {
  func.func @mask_b8(%arg0: !pto.mask<b8>) -> !pto.mask<b8>
      attributes {pto.version_selection_applied} {
    return %arg0 : !pto.mask<b8>
  }

  func.func @mask_b16(%arg0: !pto.mask<b16>) -> !pto.mask<b16>
      attributes {pto.version_selection_applied} {
    return %arg0 : !pto.mask<b16>
  }

  func.func @mask_b32(%arg0: !pto.mask<b32>) -> !pto.mask<b32>
      attributes {pto.version_selection_applied} {
    return %arg0 : !pto.mask<b32>
  }
}
