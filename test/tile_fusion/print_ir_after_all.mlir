// RUN: { ptoas %S/softmax_chain.pto --op-lib-dir=%S/oplib --print-ir-after-all -o /dev/null 2>&1; } | FileCheck %s

// CHECK: IR Dump After PTOLoweringSyncToPipe
// CHECK: IR Dump After {{.*}}PTOViewToMemrefPass
// CHECK: IR Dump After PlanMemory
// CHECK: IR Dump After PTOInstantiateAndLowerToLibCall
