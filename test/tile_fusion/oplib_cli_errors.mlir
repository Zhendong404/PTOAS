// RUN: ptoas %S/softmax_chain.pto -o %t.no_dir.cpp 2>&1 | FileCheck %s --check-prefix=NO-DIR
// RUN: ptoas %S/softmax_chain.pto --enable-op-fusion --disable-oplib-lowering --op-lib-dir=%S/oplib -o %t.dep.cpp 2>&1 | FileCheck %s --check-prefix=DEP

// NO-DIR: Error: --op-lib-dir is required when OP-LIB lowering is enabled.
// DEP: Error: --enable-op-fusion requires OP-LIB lowering; remove --disable-oplib-lowering.
