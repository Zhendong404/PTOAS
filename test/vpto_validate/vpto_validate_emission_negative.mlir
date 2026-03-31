// RUN: ! ptoas --pto-backend=vpto --emit-vpto %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ! ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-text %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ! ptoas --pto-arch=a5 --pto-backend=vpto --vpto-emit-hivm-llvm %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: 'memref.subview' op must be eliminated before emission-stage VPTO validation
// CHECK: VPTO emission preparation failed: emission-stage legality verification failed

module {
  func.func @bad_scaffold(%idx: index)
      attributes {pto.version_selection_applied} {
    %buf = memref.alloc() : memref<1x16xf32, #pto.address_space<vec>>
    %sub = memref.subview %buf[0, 0] [1, 16] [1, 1]
      : memref<1x16xf32, #pto.address_space<vec>>
         to memref<1x16xf32, strided<[16, 1], offset: 0>, #pto.address_space<vec>>
    %d = memref.dim %sub, %idx
      : memref<1x16xf32, strided<[16, 1], offset: 0>, #pto.address_space<vec>>
    return
  }
}
