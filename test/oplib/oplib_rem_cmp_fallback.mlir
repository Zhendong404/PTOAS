// RUN: { ptoas %S/rem_cmp_fallback.pto --enable-op-fusion --op-lib-dir=%S/../../oplib/level3 --pto-arch=a5 -o %t.fallback.cpp 2>&1 || true; } | FileCheck %s

// CHECK: error: 'pto.tcmp' op dst element type must be i32 mask type
