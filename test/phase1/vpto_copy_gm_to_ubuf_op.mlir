// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - | FileCheck --check-prefix=CHECK-POS %s

// CHECK-POS-LABEL: @copy_gm_to_ubuf
// CHECK-POS: %[[FALSE:.*]] = arith.constant false
// CHECK-POS: pto.copy_gm_to_ubuf %arg0, %arg1, %[[OFFSET:[^,]+]], %[[COLS:[^,]+]], %[[BURST_LEN:[^,]+]], %[[SRC_GAP:[^,]+]], %[[DST_GAP:[^,]+]], %[[FALSE]], %[[PAD:[^,]+]], %[[GMSTRIDE:[^,]+]], %[[UBSTRIDE:[^ ]+]]
// CHECK-POS-SAME: : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
module {
  func.func @copy_gm_to_ubuf(%src: !pto.ptr<f32, gm>, %dst: !pto.ptr<f32, ub>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cfalse = arith.constant false
    pto.copy_gm_to_ubuf %src, %dst, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %cfalse, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
    return
  }
}
