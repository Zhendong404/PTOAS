// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - | FileCheck --check-prefix=CHECK-POS %s

// CHECK-POS-LABEL: @copy_ubuf_to_gm
// CHECK-POS: pto.copy_ubuf_to_gm %arg0, %arg1, %[[OFFSET:[^,]+]], %[[COLS:[^,]+]], %[[BURST_LEN:[^,]+]], %[[SRC_GAP:[^,]+]], %[[DST_STRIDE:[^,]+]], %[[SRC_STRIDE:[^ ]+]]
// CHECK-POS-SAME: : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
module {
  func.func @copy_ubuf_to_gm(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, gm>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    pto.copy_ubuf_to_gm %src, %dst, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c4_i64, %c128_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
    return
  }
}
