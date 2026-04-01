// RUN: ! ptoas %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.copy_gm_to_ubuf' op requires source and destination element byte widths to match
module {
  func.func @copy_gm_to_ubuf_width_mismatch(%src: !pto.ptr<i8, gm>, %dst: !pto.ptr<f32, ub>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %cfalse = arith.constant false
    pto.copy_gm_to_ubuf %src, %dst, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %cfalse, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<i8, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
    return
  }
}
