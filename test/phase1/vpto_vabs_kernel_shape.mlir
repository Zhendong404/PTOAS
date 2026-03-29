// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-POS-LABEL: @vabs_kernel
// CHECK-POS: %[[MASK:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask
// CHECK-POS: %[[LOAD:.+]] = pto.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.vec<64xf32>
// CHECK-POS: %[[ABS:.+]] = pto.vabs %[[LOAD]], %[[MASK]] : !pto.vec<64xf32>, !pto.mask -> !pto.vec<64xf32>
// CHECK-POS: pto.vsts %[[ABS]], %arg1[%arg2], %[[MASK]] : !pto.vec<64xf32>, !pto.ptr<f32, ub>, !pto.mask
module {
  func.func @vabs_kernel(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %index: index) {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %tile = pto.vlds %src[%index] : !pto.ptr<f32, ub> -> !pto.vec<64xf32>
    %abs = pto.vabs %tile, %mask : !pto.vec<64xf32>, !pto.mask -> !pto.vec<64xf32>
    pto.vsts %abs, %dst[%index], %mask : !pto.vec<64xf32>, !pto.ptr<f32, ub>, !pto.mask
    return
  }
}

// CHECK-ERR: error: 'pto.vabs' op requires matching register vector shape
module {
  func.func @vabs_shape_mismatch(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %index: index) {
    %mask = pto.pset_b32 "PAT_ALL" : !pto.mask
    %tile = pto.vlds %src[%index] : !pto.ptr<f32, ub> -> !pto.vec<64xf32>
    %abs = pto.vabs %tile, %mask : !pto.vec<64xf32>, !pto.mask -> !pto.vec<128xi16>
    pto.vsts %abs, %dst[%index], %mask : !pto.vec<128xi16>, !pto.ptr<f32, ub>, !pto.mask
    return
  }
}
