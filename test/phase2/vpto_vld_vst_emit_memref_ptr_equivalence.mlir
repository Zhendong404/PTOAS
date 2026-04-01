// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-text %s -o - 2>/dev/null | FileCheck --check-prefix=TEXT %s
// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm %s -o - 2>/dev/null | FileCheck --check-prefix=LLVM %s

// TEXT-LABEL: define void @memref_form(
// TEXT: call <64 x float> @llvm.hivm.vldsx1{{.*}}(ptr addrspace(6)
// TEXT: call void @llvm.hivm.vstsx1{{.*}}(<64 x float>
// TEXT-LABEL: define void @ptr_form(
// TEXT: call <64 x float> @llvm.hivm.vldsx1{{.*}}(ptr addrspace(6)
// TEXT: call void @llvm.hivm.vstsx1{{.*}}(<64 x float>

// LLVM-LABEL: define{{.*}} @memref_form(
// LLVM: call <64 x float> @llvm.hivm.vldsx1{{.*}}(ptr addrspace(6)
// LLVM: call void @llvm.hivm.vstsx1{{.*}}(<64 x float>
// LLVM-LABEL: define{{.*}} @ptr_form(
// LLVM: call <64 x float> @llvm.hivm.vldsx1{{.*}}(ptr addrspace(6)
// LLVM: call void @llvm.hivm.vstsx1{{.*}}(<64 x float>

module {
  func.func @memref_form(
      %src: memref<256xf32, #pto.address_space<vec>>,
      %dst: memref<256xf32, #pto.address_space<vec>>,
      %offset: index, %mask: !pto.mask<b32>) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %v = pto.vlds %src[%offset] : memref<256xf32, #pto.address_space<vec>> -> !pto.vreg<64xf32>
      pto.vsts %v, %dst[%offset], %mask : !pto.vreg<64xf32>, memref<256xf32, #pto.address_space<vec>>, !pto.mask<b32>
    }
    return
  }

  func.func @ptr_form(
      %src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>,
      %offset: index, %mask: !pto.mask<b32>) attributes {pto.version_selection_applied} {
    pto.vecscope {
      %v = pto.vlds %src[%offset] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      pto.vsts %v, %dst[%offset], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    }
    return
  }
}
