// RUN: { ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --op-lib-dir=%S/../../oplib/level3 --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 || true; } | awk '/IR Dump After PTOFusionLoadStoreElision/{found=1} found{if ($0 ~ /^\/\/ -----\/\/ IR Dump After / && $0 !~ /PTOFusionLoadStoreElision/) exit; print}' | FileCheck %s

// A5VM-focused regression:
// - keep yielded frontier materialization (`%9` / `%17` stores remain)
// - eliminate round-trip reloads and dead intermediate stores where forwarding is
//   provably mask-compatible in the lowered loop body.

// CHECK-LABEL: IR Dump After PTOFusionLoadStoreElision
// CHECK-LABEL: func.func @kernel_online_update(
// CHECK: %[[MAXV:[0-9]+]] = a5vm.vmax %{{[0-9]+}}, %{{[0-9]+}}, %mask
// CHECK: a5vm.vsts %[[MAXV]], %9[%arg10], %mask
// CHECK: %[[SRC4:[0-9]+]] = a5vm.vlds %4[%arg10]
// CHECK-NOT: a5vm.vlds %9[%arg10]
// CHECK: %[[SUB0:[0-9]+]] = a5vm.vsub %[[SRC4]], %[[MAXV]], %mask_252
// CHECK: a5vm.vsts %[[SUB0]], %11[%arg10], %mask_252
// CHECK: %mask_260
// CHECK-NOT: a5vm.vsts %{{[0-9]+}}, %11[%arg10], %mask_260
// CHECK: %mask_262
// CHECK-NOT: a5vm.vlds %16[%arg10]
// CHECK: %mask_264
// CHECK: %[[ADDV:[0-9]+]] = a5vm.vadd %{{[0-9]+}}, %{{[0-9]+}}, %mask_264
// CHECK: a5vm.vsts %[[ADDV]], %17[%arg10], %mask_264
