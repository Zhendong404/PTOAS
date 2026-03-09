// RUN: ! ptoas %S/softmax_chain.pto -o %t.no_dir.cpp > %t.no_dir.log 2>&1
// RUN: FileCheck %s --check-prefix=NO-DIR < %t.no_dir.log
// RUN: ! ptoas %S/softmax_chain.pto --enable-op-fusion --disable-oplib-lowering --op-lib-dir=%S/oplib -o %t.dep.cpp > %t.dep.log 2>&1
// RUN: FileCheck %s --check-prefix=REMOVED < %t.dep.log
// RUN: rm -rf %t.empty && mkdir -p %t.empty
// RUN: cp %S/resources/bad_empty_simd_template.txt %t.empty/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.empty -o %t.empty.cpp > %t.empty.log 2>&1
// RUN: FileCheck %s --check-prefix=EMPTY-SIMD < %t.empty.log
// RUN: rm -rf %t.core && mkdir -p %t.core
// RUN: cp %S/resources/bad_simd_core_slot_template.txt %t.core/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.core -o %t.core.cpp > %t.core.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-CORE < %t.core.log
// RUN: rm -rf %t.lanes && mkdir -p %t.lanes
// RUN: cp %S/resources/bad_simd_lanes_template.txt %t.lanes/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.lanes -o %t.lanes.cpp > %t.lanes.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-LANES < %t.lanes.log
// RUN: rm -rf %t.simd_attr && mkdir -p %t.simd_attr
// RUN: cp %S/resources/bad_simd_missing_attrs_template.txt %t.simd_attr/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.simd_attr -o %t.simd_attr.cpp > %t.simd_attr.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-SIMD-ATTR < %t.simd_attr.log
// RUN: rm -rf %t.disallowed && mkdir -p %t.disallowed
// RUN: cp %S/resources/bad_disallowed_ir_template.txt %t.disallowed/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.disallowed -o %t.disallowed.cpp > %t.disallowed.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-IR < %t.disallowed.log
// RUN: rm -rf %t.bad_vec && mkdir -p %t.bad_vec
// RUN: cp %S/resources/bad_vector_unsupported_template.txt %t.bad_vec/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_vec --pto-arch=a5 -o %t.bad_vec.cpp > %t.bad_vec.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-VEC < %t.bad_vec.log
// RUN: rm -rf %t.legacy && mkdir -p %t.legacy
// RUN: cp %S/resources/bad_legacy_unrealized_cast_template.txt %t.legacy/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.legacy -o %t.legacy.cpp > %t.legacy.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-LEGACY-CAST < %t.legacy.log
// RUN: rm -rf %t.missing_knobs && mkdir -p %t.missing_knobs
// RUN: cp %S/resources/bad_vector_missing_knobs_template.txt %t.missing_knobs/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.missing_knobs --pto-arch=a5 -o %t.missing_knobs.cpp > %t.missing_knobs.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MISSING-KNOBS < %t.missing_knobs.log
// RUN: rm -rf %t.bad_prefix && mkdir -p %t.bad_prefix
// RUN: cp %S/resources/bad_vector_bad_prefix_template.txt %t.bad_prefix/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_prefix --pto-arch=a5 -o %t.bad_prefix.cpp > %t.bad_prefix.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-PREFIX < %t.bad_prefix.log
// RUN: rm -rf %t.missing_vld && mkdir -p %t.missing_vld
// RUN: cp %S/resources/bad_vector_missing_vld_template.txt %t.missing_vld/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.missing_vld --pto-arch=a5 -o %t.missing_vld.cpp > %t.missing_vld.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MISSING-VLD < %t.missing_vld.log
// RUN: rm -rf %t.missing_vst && mkdir -p %t.missing_vst
// RUN: cp %S/resources/bad_vector_missing_vst_template.txt %t.missing_vst/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.missing_vst --pto-arch=a5 -o %t.missing_vst.cpp > %t.missing_vst.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-MISSING-VST < %t.missing_vst.log
// RUN: rm -rf %t.bad_vst_prefix && mkdir -p %t.bad_vst_prefix
// RUN: cp %S/resources/bad_vector_bad_vst_prefix_template.txt %t.bad_vst_prefix/bad.mlir
// RUN: ! ptoas %S/softmax_chain.pto --op-lib-dir=%t.bad_vst_prefix --pto-arch=a5 -o %t.bad_vst_prefix.cpp > %t.bad_vst_prefix.log 2>&1
// RUN: FileCheck %s --check-prefix=BAD-VST-PREFIX < %t.bad_vst_prefix.log

// NO-DIR: Error: --op-lib-dir is required.
// REMOVED: Unknown command line argument '--disable-oplib-lowering'
// EMPTY-SIMD: E_OPLIB_EMPTY_BODY_FOR_SIMD
// BAD-CORE: E_OPLIB_SIMD_INVALID_CORE_SLOT
// BAD-LANES: E_OPLIB_SIMD_LANES_MISMATCH
// BAD-SIMD-ATTR: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-IR: E_OPLIB_BODY_DISALLOWED_IR
// BAD-VEC: A5 OP-Lib vector lowering unsupported
// BAD-LEGACY-CAST: E_OPLIB_BODY_DISALLOWED_IR
// BAD-MISSING-KNOBS: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-MISSING-KNOBS: pto.simd.exec_mode
// BAD-PREFIX: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-PREFIX: must start with 'MODE_'
// BAD-MISSING-VLD: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-MISSING-VLD: pto.simd.vld_dist
// BAD-MISSING-VST: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-MISSING-VST: pto.simd.vst_dist
// BAD-VST-PREFIX: E_OPLIB_SIMD_ATTR_REQUIRED
// BAD-VST-PREFIX: must start with 'DIST_'
