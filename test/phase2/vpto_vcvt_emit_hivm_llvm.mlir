// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm %s -o - 2>/dev/null | FileCheck %s

// CHECK-LABEL: define{{.*}} @vcvt_f32_to_bf16(
// CHECK: call <128 x bfloat> @llvm.hivm.vcvtff.f322bf16.x(
// CHECK: call void @llvm.hivm.vstsx1{{.*}}(

// CHECK-LABEL: define{{.*}} @vcvt_bf16_to_f32(
// CHECK: call <64 x float> @llvm.hivm.vcvtff.bf162f32.x(
// CHECK: call void @llvm.hivm.vstsx1{{.*}}(

module {
  func.func @vcvt_f32_to_bf16(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<bf16, ub>, %idx: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b16 "PAT_ALL" : !pto.mask<b16>
      %vec = pto.vlds %src[%idx] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %cvt = pto.vcvt %vec {part = "PART_ODD", round_mode = "ROUND_A", sat = "RS_ENABLE"} : !pto.vreg<64xf32> -> !pto.vreg<128xbf16>
      pto.vsts %cvt, %dst[%idx], %mask : !pto.vreg<128xbf16>, !pto.ptr<bf16, ub>, !pto.mask<b16>
    }
    return
  }

  func.func @vcvt_bf16_to_f32(%src: !pto.ptr<bf16, ub>, %dst: !pto.ptr<f32, ub>, %idx: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %vec = pto.vlds %src[%idx] : !pto.ptr<bf16, ub> -> !pto.vreg<128xbf16>
      %cvt = pto.vcvt %vec {part = "PART_EVEN"} : !pto.vreg<128xbf16> -> !pto.vreg<64xf32>
      pto.vsts %cvt, %dst[%idx], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    }
    return
  }
}
