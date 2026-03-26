// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOLowLevelLoopFusion/{after_low=1} after_low && /IR Dump After CSE/{found=1} found{if (found > 1 && /IR Dump After PTOFusionPredicateElision/) exit; print; found=2}' %t | FileCheck %s --check-prefix=PRE
// RUN: awk '/IR Dump After PTOFusionPredicateElision/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=POST

// PRE-LABEL: IR Dump After CSE
// PRE: %[[LOOPRES:[^:]+]]:2 = scf.for %{{[^ ]+}} = %c0 to %{{[^ ]+}} step %c64 iter_args(%[[ARG11:[^ ]+]] = %{{[^ ]+}}, %[[ARG12:[^ ]+]] = %{{[^ ]+}}) -> (i32, i32) {
// PRE: %[[MASK0:[^,]+]], %[[OUT0:[^ ]+]] = a5vm.plt_b32 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// PRE: %[[MASK1:[^,]+]], %[[OUT1:[^ ]+]] = a5vm.plt_b32 %[[ARG11]] : i32 -> !a5vm.mask, i32
// PRE: %[[MASK2:[^,]+]], %[[OUT2:[^ ]+]] = a5vm.plt_b32 %[[ARG12]] : i32 -> !a5vm.mask, i32
// PRE: scf.yield %[[OUT1]], %[[OUT2]] : i32, i32

// POST-LABEL: IR Dump After PTOFusionPredicateElision
// POST: %[[LOOPRES:[^:]+]]:2 = scf.for %{{[^ ]+}} = %c0 to %{{[^ ]+}} step %c64 iter_args(%[[ARG11:[^ ]+]] = %{{[^ ]+}}, %[[ARG12:[^ ]+]] = %{{[^ ]+}}) -> (i32, i32) {
// POST: %[[MASK0:[^,]+]], %[[OUT0:[^ ]+]] = a5vm.plt_b32 %{{[^ ]+}} : i32 -> !a5vm.mask, i32
// POST: %[[MASK1:[^,]+]], %[[OUT1:[^ ]+]] = a5vm.plt_b32 %[[ARG11]] : i32 -> !a5vm.mask, i32
// POST-NOT: = a5vm.plt_b32 %[[ARG12]]
// POST: scf.yield %[[OUT1]], %[[OUT1]] : i32, i32
