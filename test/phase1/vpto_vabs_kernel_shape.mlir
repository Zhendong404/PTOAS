// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - | FileCheck --check-prefix=CHECK-POS %s

// CHECK-POS-LABEL: @vabs_kernel
// CHECK-POS: %[[MASK:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
// CHECK-POS: %[[LOAD:.+]] = pto.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// CHECK-POS: %[[ABS:.+]] = pto.vabs %[[LOAD]], %[[MASK]] : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// CHECK-POS: pto.vsts %[[ABS]], %arg1[%arg2], %[[MASK]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
module {
  func.func @vabs_kernel(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %index: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %tile = pto.vlds %src[%index] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %abs = pto.vabs %tile, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      pto.vsts %abs, %dst[%index], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    }
    return
  }
}
