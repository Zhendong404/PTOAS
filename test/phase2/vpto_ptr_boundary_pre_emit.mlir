// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s

// CHECK: func.func @memref_boundary_kernel(%arg0: !pto.ptr<f32, ub>, %arg1: !pto.ptr<f32, ub>, %arg2: index, %arg3: !pto.mask<b32>)
// CHECK-NOT: pto.castptr %arg0
// CHECK-NOT: pto.castptr %arg1
// CHECK: pto.vecscope {
// CHECK: %[[LOAD:.+]] = pto.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// CHECK: pto.vsts %[[LOAD]], %arg1[%arg2], %arg3 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

module {
  func.func @memref_boundary_kernel(
      %src: memref<256xf32, #pto.address_space<vec>>,
      %dst: memref<256xf32, #pto.address_space<vec>>,
      %offset: index, %mask: !pto.mask<b32>) {
    pto.vecscope {
      %v = pto.vlds %src[%offset] : memref<256xf32, #pto.address_space<vec>> -> !pto.vreg<64xf32>
      pto.vsts %v, %dst[%offset], %mask : !pto.vreg<64xf32>, memref<256xf32, #pto.address_space<vec>>, !pto.mask<b32>
    }
    return
  }
}
