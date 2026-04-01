// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - | FileCheck %s

// CHECK-LABEL: func.func @abs_kernel_2d
// CHECK: pto.copy_gm_to_ubuf
// CHECK: %[[MASK:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
// CHECK: %[[LOAD:.+]] = pto.vlds %{{.+}} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// CHECK: %[[ABS:.+]] = pto.vabs %[[LOAD]], %[[MASK]] : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// CHECK: pto.vsts %[[ABS]], %{{.+}}, %[[MASK]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
// CHECK: pto.copy_ubuf_to_gm
// CHECK-NOT: llvm.hivm
module {
  func.func @abs_kernel_2d(%base: !pto.ptr<f32, gm>, %ubuf: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, gm>, %index: index) attributes {pto.version_selection_applied} {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    %cfalse = arith.constant false
    pto.copy_gm_to_ubuf %base, %ubuf, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %cfalse, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
    pto.vecscope {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %loaded = pto.vlds %ubuf[%index] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %abs = pto.vabs %loaded, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      pto.vsts %abs, %ubuf[%index], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    }
    pto.copy_ubuf_to_gm %ubuf, %out, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c4_i64, %c128_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64
    return
  }
}
