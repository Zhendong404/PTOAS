// RUN: ./build/tools/ptoas/ptoas test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 | awk '/IR Dump After PTOLowLevelLoopFusion/{found=1} found{if ($0 ~ /^\\/\\/ -----\\/\\/ IR Dump After / && $0 !~ /PTOLowLevelLoopFusion/) exit; print}' | FileCheck %s --check-prefix=LOW
// RUN: ./build/tools/ptoas/ptoas test/samples/PyPTOIRParser/paged_attention_example_kernel_online_update.pto --enable-op-fusion --pto-arch=a5 --pto-backend=a5vm --print-ir-after-all --print-ir-after-all-func-filter=kernel_online_update -o /dev/null 2>&1 | awk '/IR Dump After PTOFlattenFusionRegion/{found=1} found{if ($0 ~ /^\\/\\/ -----\\/\\/ IR Dump After / && $0 !~ /PTOFlattenFusionRegion/) exit; print}' | FileCheck %s --check-prefix=FLAT

// LOW-LABEL: IR Dump After PTOLowLevelLoopFusion
// LOW: pto.fusion_region {
// LOW: a5vm.plt_b32
// LOW: pto.yield(

// FLAT-LABEL: IR Dump After PTOFlattenFusionRegion
// FLAT: a5vm.plt_b32
// FLAT-NOT: pto.fusion_region
// FLAT-NOT: pto.yield
