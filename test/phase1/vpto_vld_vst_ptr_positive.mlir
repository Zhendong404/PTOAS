// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - | FileCheck %s

// CHECK-LABEL: @vld_vst_ptr_positive
// CHECK: %[[SRC:.+]] = pto.castptr %arg0 : i64 -> !pto.ptr<f32, ub>
// CHECK: %[[DST:.+]] = pto.castptr %arg1 : i64 -> !pto.ptr<f32, ub>
// CHECK: %[[MASK_ALL:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
// CHECK: %[[ALIGN:.+]] = pto.vldas %[[SRC]] : !pto.ptr<f32, ub> -> !pto.align
// CHECK: %[[V0:.+]] = pto.vlds %[[SRC]][%arg2] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// CHECK: %[[V1:.+]], %[[NEXT_ALIGN:.+]], %[[NEXT_SRC:.+]] = pto.vldus %[[SRC]], %[[ALIGN]] : !pto.ptr<f32, ub>, !pto.align -> !pto.vreg<64xf32>, !pto.align, !pto.ptr<f32, ub>
// CHECK: %[[PMASK:.+]] = pto.plds %[[SRC]][%arg2] : !pto.ptr<f32, ub> -> !pto.mask<b32>
// CHECK: pto.vsts %[[V0]], %[[DST]][%arg2], %[[MASK_ALL]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
// CHECK: pto.vsts %[[V1]], %[[DST]][%arg2], %[[PMASK]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
// CHECK: pto.psts %[[MASK_ALL]], %[[DST]][%arg2] : !pto.mask<b32>, !pto.ptr<f32, ub>
module {
  func.func @vld_vst_ptr_positive(%src_addr: i64, %dst_addr: i64,
                                  %index: index) attributes {pto.version_selection_applied} {
    %src = pto.castptr %src_addr : i64 -> !pto.ptr<f32, ub>
    %dst = pto.castptr %dst_addr : i64 -> !pto.ptr<f32, ub>
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
