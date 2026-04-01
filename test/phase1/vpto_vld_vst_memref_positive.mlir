// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - | FileCheck %s

// CHECK-LABEL: @vld_vst_memref_positive
// CHECK: %[[MASK_ALL:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
// CHECK: %[[ALIGN:.+]] = pto.vldas %arg0 : !pto.ptr<f32, ub> -> !pto.align
// CHECK: %[[V0:.+]] = pto.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// CHECK: %[[V1:.+]], %[[NEXT_ALIGN:.+]], %[[NEXT_SRC:.+]] = pto.vldus %arg0, %[[ALIGN]] : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
// CHECK: %[[PMASK:.+]] = pto.plds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.mask<b32>
// CHECK: pto.vsts %[[V0]], %arg1[%arg2], %[[MASK_ALL]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
// CHECK: pto.vsts %[[V1]], %arg1[%arg2], %[[PMASK]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
// CHECK: pto.psts %[[MASK_ALL]], %arg1[%arg2] : !pto.mask<b32>, !pto.ptr<f32, ub>
module {
  func.func @vld_vst_memref_positive(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>,
                                     %index: index) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %mask_all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %align = pto.vldas %src : !pto.ptr<f32, ub> -> !pto.align
      %v0 = pto.vlds %src[%index] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %v1, %next_align, %next_src = pto.vldus %src, %align : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
      %pmask = pto.plds %src[%index] : !pto.ptr<f32, ub> -> !pto.mask<b32>
      pto.vsts %v0, %dst[%index], %mask_all : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
      pto.vsts %v1, %dst[%index], %pmask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
      pto.psts %mask_all, %dst[%index] : !pto.mask<b32>, !pto.ptr<f32, ub>
    }
    return
  }
}
