// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'arith.addi' op using value defined outside the region
// CHECK: pto.strict_vecscope

module {
  func.func @strict_vecscope_rejects_implicit_capture(%arg0: i32) {
    pto.strict_vecscope(%arg0) {
    ^bb0(%captured: i32):
      %0 = arith.addi %captured, %arg0 : i32
    } : (i32) -> ()
    return
  }
}
