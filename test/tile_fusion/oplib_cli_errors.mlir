// RUN: ! ptoas %S/softmax_chain.pto -o %t.no_dir.cpp > %t.no_dir.log 2>&1
// RUN: FileCheck %s --check-prefix=NO-DIR < %t.no_dir.log
// RUN: ! ptoas %S/softmax_chain.pto --enable-op-fusion --disable-oplib-lowering --op-lib-dir=%S/oplib -o %t.dep.cpp > %t.dep.log 2>&1
// RUN: FileCheck %s --check-prefix=DEP < %t.dep.log

// NO-DIR: Error: --op-lib-dir is required when OP-LIB lowering is enabled
// DEP: Error: --enable-op-fusion requires OP-LIB lowering
