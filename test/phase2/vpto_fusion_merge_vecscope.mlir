// RUN: ptoas %S/../samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=vpto --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null > %t 2>&1
// RUN: awk '/IR Dump After PTOFusionMergeVecScope/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=MERGED
// RUN: awk '/IR Dump After PTOLowLevelLoopFusion/{found=1} found{if (found > 1 && /IR Dump After /) exit; print; found=2}' %t | FileCheck %s --check-prefix=FUSED

// MERGED-LABEL: IR Dump After PTOFusionMergeVecScope
// MERGED: pto.vecscope {
// MERGED: pto.vmax
// MERGED-NOT: pto.vecscope {
// MERGED: pto.vsub
// MERGED-NOT: pto.vecscope {
// MERGED: pto.vexp
// MERGED-NOT: pto.vecscope {
// MERGED: pto.vmul
// MERGED-NOT: pto.vecscope {
// MERGED: pto.vadd

// FUSED-LABEL: IR Dump After PTOLowLevelLoopFusion
// FUSED: pto.vecscope {
// FUSED: pto.vmax
// FUSED-NOT: pto.vecscope {
// FUSED: pto.vsub
// FUSED-NOT: pto.vecscope {
// FUSED: pto.vexp
// FUSED-NOT: pto.vecscope {
// FUSED: pto.vmul
// FUSED-NOT: pto.vecscope {
// FUSED: pto.vadd

module {
}
