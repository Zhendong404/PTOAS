// RUN: ! ptoas %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.copy_ubuf_to_gm' op requires UB source and GM destination
module {
  func.func @copy_ubuf_to_gm_wrong_direction(%src: !pto.ptr<f32, gm>, %dst: !pto.ptr<f32, ub>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    pto.copy_ubuf_to_gm %src, %dst, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c4_i64, %c128_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64
    return
  }
}
