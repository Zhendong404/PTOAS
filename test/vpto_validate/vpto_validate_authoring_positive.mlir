// RUN: ptoas --pto-backend=vpto --emit-vpto %s -o - 2>/dev/null | FileCheck %s --check-prefix=IR
// RUN: ptoas --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=direct_input_ok %s -o /dev/null > %t 2>&1
// RUN: FileCheck %s --input-file=%t --check-prefix=VALIDATE

// IR-LABEL: func.func @direct_input_ok(
// IR: %[[MASK:.+]] = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
// IR: %[[LOAD:.+]] = pto.vlds %arg0[%arg2] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
// IR: %[[ABS:.+]] = pto.vabs %[[LOAD]], %[[MASK]] : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
// IR: pto.vsts %[[ABS]], %arg1[%arg2], %[[MASK]] : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// VALIDATE: pto-validate-vpto-ir
// VALIDATE-LABEL: func.func @direct_input_ok(
// VALIDATE: !pto.mask<b32>
// VALIDATE-NOT: PTOToVPTO

module {
  func.func @direct_input_ok(%src: !pto.ptr<f32, ub>, %dst: !pto.ptr<f32, ub>,
                             %idx: index) attributes {pto.version_selection_applied} {
    %c1 = arith.constant 1 : index
    scf.for %i = %idx to %c1 step %c1 {
      %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
      %v = pto.vlds %src[%idx] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
      %abs = pto.vabs %v, %mask : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
      pto.vsts %abs, %dst[%idx], %mask : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
    } {llvm.loop.aivector_scope}
    return
  }
}
