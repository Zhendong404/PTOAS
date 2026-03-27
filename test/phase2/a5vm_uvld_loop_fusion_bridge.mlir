// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOLowLevelLoopFusion/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=FUSED
// RUN: awk '/IR Dump After PTOA5VMExpandBridgeOps/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=EXPAND

// FUSED-LABEL: IR Dump After PTOLowLevelLoopFusion
// FUSED: scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %c1{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// FUSED: scf.for %[[ROW:.+]] = %c0{{(_[0-9]+)?}} to %c16{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// FUSED: %[[UVLD0:.+]] = a5vm.uvld %{{.+}}#1[%{{.+}}] : memref<1x16xf32
// FUSED: %[[DUP0:.+]] = a5vm.vdup %[[UVLD0]]
// FUSED: %[[UVLD1:.+]] = a5vm.uvld %{{.+}}#2[%{{.+}}] : memref<1x16xf32
// FUSED: %[[DUP1:.+]] = a5vm.vdup %[[UVLD1]]
// FUSED: %[[CHUNK:.+]]:3 = scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %{{.+}} step %c1{{(_[0-9]+)?}} iter_args(%{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}, %{{.+}} = %{{.+}}) -> (i32, i32, i32) {
// FUSED: a5vm.vmul %{{.+}}, %[[DUP0]], %{{.+}}
// FUSED: a5vm.vmul %{{.+}}, %[[DUP1]], %{{.+}}
// FUSED: a5vm.vadd

// EXPAND-LABEL: IR Dump After PTOA5VMExpandBridgeOps
// EXPAND-NOT: a5vm.uvld
// EXPAND: scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %c1{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// EXPAND: scf.for %{{.*}} = %c0{{(_[0-9]+)?}} to %c16{{(_[0-9]+)?}} step %c1{{(_[0-9]+)?}} {
// EXPAND: %[[PTR0:.+]] = llvm.getelementptr %{{.+}}[%{{.+}}] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
// EXPAND: %[[ALIGN0:.+]] = a5vm.vldas %[[PTR0]] : !llvm.ptr<6> -> !a5vm.align
// EXPAND: %[[LOAD0:.+]], %{{.+}}, %{{.+}} = a5vm.vldus %[[PTR0]], %[[ALIGN0]] : !llvm.ptr<6>, !a5vm.align -> !a5vm.vec<64xf32>, !a5vm.align, !llvm.ptr<6>
// EXPAND: a5vm.vdup %[[LOAD0]]
// EXPAND: %[[PTR1:.+]] = llvm.getelementptr %{{.+}}[%{{.+}}] : (!llvm.ptr<6>, i64) -> !llvm.ptr<6>, f32
// EXPAND: %[[ALIGN1:.+]] = a5vm.vldas %[[PTR1]] : !llvm.ptr<6> -> !a5vm.align
// EXPAND: %[[LOAD1:.+]], %{{.+}}, %{{.+}} = a5vm.vldus %[[PTR1]], %[[ALIGN1]] : !llvm.ptr<6>, !a5vm.align -> !a5vm.vec<64xf32>, !a5vm.align, !llvm.ptr<6>
// EXPAND: a5vm.vdup %[[LOAD1]]
