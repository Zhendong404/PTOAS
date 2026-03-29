// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: define{{.*}} @vcvt_f32_to_bf16(
// CHECK: %[[MASK32_PACK:.+]] = call { <256 x i1>, i32 } @llvm.hivm.plt.b32.v300(i32 64)
// CHECK: %[[MASK32:.+]] = extractvalue { <256 x i1>, i32 } %[[MASK32_PACK]], 0
// CHECK: call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(<64 x float> %[[SRC32:.+]], <256 x i1> %[[MASK32]], i32 1, i32 0, i32 1)

// CHECK-LABEL: define{{.*}} @vcvt_bf16_to_f32(
// CHECK: %[[MASK16_PACK:.+]] = call { <256 x i1>, i32 } @llvm.hivm.plt.b16.v300(i32 128)
// CHECK: %[[MASK16:.+]] = extractvalue { <256 x i1>, i32 } %[[MASK16_PACK]], 0
// CHECK: call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(<128 x bfloat> %[[SRC16:.+]], <256 x i1> %[[MASK16]], i32 0)

module {
  func.func @vcvt_f32_to_bf16(%src: !pto.vec<64xf32>) -> !pto.vec<128xbf16> attributes {pto.version_selection_applied} {
    %0 = pto.vcvt %src {part = "PART_ODD", round_mode = "ROUND_A", sat = "RS_ENABLE"} : !pto.vec<64xf32> -> !pto.vec<128xbf16>
    return %0 : !pto.vec<128xbf16>
  }

  func.func @vcvt_bf16_to_f32(%src: !pto.vec<128xbf16>) -> !pto.vec<64xf32> attributes {pto.version_selection_applied} {
    %0 = pto.vcvt %src {part = "PART_EVEN"} : !pto.vec<128xbf16> -> !pto.vec<64xf32>
    return %0 : !pto.vec<64xf32>
  }
}
