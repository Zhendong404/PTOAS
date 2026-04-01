// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'pto.vabs' op requires matching register vector shape
module {
  func.func @vabs_shape_mismatch(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>, %index: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %tile = pto.vlds %src[%index] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %abs = pto.vabs %tile, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xi16>
      pto.vsts %abs, %dst[%index], %mask : !pto.vreg<128xi16>, !pto.ptr<f32, ub>, !pto.mask<b32>
    }
    return
  }
}
