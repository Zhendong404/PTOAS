// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @strict_vecscope_explicit_capture
// CHECK: pto.strict_vecscope(%[[ARG:[^ ]+]]) {
// CHECK: ^bb0(%[[CAP:[^ :]+]]: i32):
// CHECK: } : (i32) -> ()

module {
  func.func @strict_vecscope_explicit_capture(%arg0: i32) {
    pto.strict_vecscope(%arg0) {
    ^bb0(%captured: i32):
      %0 = arith.addi %captured, %captured : i32
    } : (i32) -> ()
    return
  }
}
