// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

// The VPTO backend is emitted from tools/ptoas rather than a TableGen pass;
// these registrations continue to describe the shared pre-backend pipeline.

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

// Define PTOInsertSync Pass
def PTOInsertSync : Pass<"pto-insert-sync", "func::FuncOp"> {
  let summary = "Insert synchronization flags for PTO dialect";
  let description = [{
    Analyzes data dependencies between Cube, Vector and MTE pipelines
    and inserts explicit synchronization instructions.
  }];
 
  // Declare the constructor function name
  let constructor = "mlir::pto::createPTOInsertSyncPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::arith::ArithDialect" 
  ];
}

def InferPTOLayout : Pass<"pto-infer-layout", "func::FuncOp"> {
  let summary = "Infer GlobalTensor layout (ND/DN/NZ) for make_tensor_view";
  let description = [{
    For each pto.make_tensor_view with fully static shape/stride, compute layout
    (ND/DN/NZ) per design rules and attach the optional `layout` attribute.
    Dynamic stride/shape are left unset.
  }];
  let constructor = "mlir::pto::createInferPTOLayoutPass()";
  let dependentDialects = ["pto::PTODialect", "arith::ArithDialect"];
}

def PTOA5NormalizeTMov : Pass<"pto-a5-normalize-tmov", "func::FuncOp"> {
  let summary = "Normalize risky A5 vec->vec col_major TMOV into row-major reinterpret + TMOV";
  let description = [{
    Rewrites A5 `pto.tmov` patterns that use vec->vec col_major/none_box tiles:
      src(col_major) -> pto.treshape(row_major) -> pto.tmov(row_major->row_major)
      dst(col_major) -> pto.treshape(row_major)
    This avoids unsupported A5 PTO-ISA vec->vec col_major TMOV paths while
    preserving alias semantics via SSA treshape (no real data movement).
  }];
  let constructor = "mlir::pto::createPTOA5NormalizeTMovPass()";
  let dependentDialects = ["pto::PTODialect", "func::FuncDialect"];
}


def PlanMemory : Pass<"pto-plan-memory", "ModuleOp"> {
  let summary = "Plan memory for PTO Ops";
  let constructor = "mlir::pto::createPlanMemoryPass()";
  let dependentDialects = ["pto::PTODialect", ];
  let options = [
    Option<"memMode", "mem-plan-mode", "pto::MemPlanMode",
           "pto::MemPlanMode::LOCAL_MEM_PLAN",
           "plan mem mode (default is LOCAL_MEM_PLAN)",
           [{::llvm::cl::values(
               clEnumValN(pto::MemPlanMode::LOCAL_MEM_PLAN, "local-mem-plan",
                          "plan mem mode is for local tile buffers"),
               clEnumValN(
                   pto::MemPlanMode::GLOBAL_WORKSPACE_PLAN,
                   "global-work-space-plan",
                   "plan mem mode is for global workspace allocation"))}]>,
    Option<"enableGlobalReuse", "enable-global-workspace-reuse", "bool",
           /*default=*/"false",
           "Enable global workspace reuse ,default : false">,
    Option<"enablePrintMemoryAllocatedSize", "enable-print-memory-allocated-size", "bool",
           /*default=*/"false",
           "Enable print memory allocated size, default : false">,
    Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
           /*default=*/"false",
           "restrict memory inplace as isa, default : false">,
  ];
}
def PTOLoweringSyncToPipe : Pass<"pto-lowering-sync-to-pipe", "func::FuncOp"> {
  let summary = "Lower high-level sync ops to low-level pipe ops";
  let description = [{
    Converts high-level `pto.record_event` and `pto.wait_event` operations
    into low-level `pto.set_flag` and `pto.wait_flag` operations based on
    hardware pipe mapping logic.
  }];

  let constructor = "mlir::pto::createLoweringSyncToPipePass()";
  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PreFusionAnalysis : Pass<"pto-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Analyze block-local tile-fusion candidates before planning";
  let description = [{
    Builds reusable pre-planning analysis state for tile fusion in tile-native
    PTO IR. This pass is analysis-only and does not emit grouping or
    scheduling decisions.
  }];
  let constructor = "mlir::pto::createPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PrintPreFusionAnalysis
    : Pass<"pto-print-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Print pre-fusion analysis results for testing and debugging";
  let description = [{
    Computes the pre-fusion analysis result and prints a stable textual dump of
    block-local compute nodes, edges, liveness, iteration-domain classes, and
    local boundaries.
  }];
  let constructor = "mlir::pto::createPrintPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def FusionPlan : Pass<"pto-fusion-plan", "func::FuncOp"> {
  let summary = "Build conservative tile-fusion planning groups from block-local analysis";
  let description = [{
    Consumes PreFusionAnalysis results, forms conservative block-local fusion
    groups for currently supported tile-native compute ops, and annotates
    accepted group members with planning metadata for later scheduling.
  }];
  let constructor = "mlir::pto::createFusionPlanPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def OpScheduling : Pass<"pto-op-scheduling", "func::FuncOp"> {
  let summary = "Compact planned fusion groups into block-local contiguous spans";
  let description = [{
    Consumes fusion planning metadata emitted by FusionPlanPass and performs
    block-local instruction scheduling to make each accepted fusion group a
    contiguous span without redefining group membership.
  }];
  let constructor = "mlir::pto::createOpSchedulingPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOFusionRegionGen : Pass<"pto-fusion-region-gen", "func::FuncOp"> {
  let summary = "Wrap one scheduled fusion span into one pto.fusion_region";
  let description = [{
    Consumes scheduled tile-fusion metadata and encapsulates one contiguous
    fusion group span into one pto.fusion_region without redefining group
    membership or physical order.
  }];
  let constructor = "mlir::pto::createPTOFusionRegionGenPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOWrapFunctionsInSections : Pass<"pto-wrap-functions-in-sections", "func::FuncOp"> {
  let summary = "Wrap attributed single-core functions in PTO cube/vector sections";
  let description = [{
    Rewrites frontend-marked functions into the existing PTO section form:
    - `pto.kernel_kind = #pto.kernel_kind<cube>` -> wrap the function body in
      `pto.section.cube`
    - `pto.kernel_kind = #pto.kernel_kind<vector>` -> wrap the function body
      in `pto.section.vector`
  }];

  let constructor = "mlir::pto::createPTOWrapFunctionsInSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOLowerFrontendPipeOps : Pass<"pto-lower-frontend-pipe-ops", "func::FuncOp"> {
  let summary = "Lower frontend TPUSH/TPOP pipe ops to internal pipe ops";
  let description = [{
    Converts frontend `aic/aiv_initialize_pipe`, `tpush_to_*`,
    `tpop_from_*`, and `tfree_from_*` ops into the internal unified pipe IR:
    `initialize_l2l_pipe` / `initialize_l2g2l_pipe`, `tpush`,
    `declare_tile`, `tpop`, and `tfree`.
  }];

  let constructor = "mlir::pto::createPTOLowerFrontendPipeOpsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOAssignDefaultFrontendPipeId : Pass<"pto-assign-default-frontend-pipe-id", "func::FuncOp"> {
  let summary = "Materialize default id = 0 on frontend TPUSH/TPOP pipe ops";
  let description = [{
    Preserves backward compatibility with older frontend pipe syntax that
    omitted `id`. This pass rewrites missing `id` attrs on
    `aic/aiv_initialize_pipe`, `tpush_to_*`, `tpop_from_*`, and
    `tfree_from_*` to an explicit `id = 0` before frontend pipe lowering.
  }];

  let constructor = "mlir::pto::createPTOAssignDefaultFrontendPipeIdPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOInferValidatePipeInit : Pass<"pto-infer-validate-pipe-init", "ModuleOp"> {
  let summary = "Infer and validate internal pipe init nosplit configuration";
  let description = [{
    Runs after frontend pipe lowering and before memory planning. For each
    logical pipe, this pass:
    - validates that downstream `pto.tpush` / `pto.tpop` / `pto.tfree` users
      do not mix `split = 0` with `split = 1/2`
    - preserves explicit `nosplit` attrs on internal pipe init ops and rejects
      conflicts across peer pipe init pairs
    - infers missing `nosplit` attrs from downstream split usage for backward
      compatibility with older IR that omitted init-level `nosplit`
    - propagates the resolved `nosplit` value across peer pipe init pairs so
      both ends of one logical pipe agree before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOInferValidatePipeInitPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOResolveReservedBuffers : Pass<"pto-resolve-reserved-buffers", "ModuleOp"> {
  let summary = "Resolve reserved local buffer addresses and peer pipe flag bases";
  let description = [{
    Runs after `pto-plan-memory`. Assumes `pto.reserve_buffer` base addresses
    have already been planned, then:
    - aligns missing `flag_base` attrs for peer internal pipe init ops
    - rejects internal pipe init ops without explicit `flag_base` when their
      `local_addr` cannot be traced back to `pto.reserve_buffer` /
      `pto.import_reserved_buffer`
    - replaces `pto.reserve_buffer` / `pto.import_reserved_buffer` results with
      resolved constant local addresses
    - erases the frontend-only reserve/import ops before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOResolveReservedBuffersPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def ExpandTileOp : Pass<"pto-expand-tile-op", "ModuleOp"> {
  let summary =
      "Expand tilebuf-native PTO ops into VPTO TileLang helper calls";
  let description = [{
    Expands tile-level operations (pto.tadd, pto.tsub, etc.) by invoking the
    TileLang Python DSL to instantiate template libraries. This pass is the
    PTO IR -> VPTO IR boundary: local tile operands must reach the pass as
    native !pto.tile_buf values, tensor views must remain in tensor_view /
    partition_tensor_view form, and memref operands from the legacy bridge are
    rejected.

    Each tile op is replaced by a func.call to the generated template function,
    with tile_buf operands passed directly. View/tensor-view operands may still
    use helper-call bridging after the boundary, because the generated VPTO
    helper bodies are allowed to materialize memref/ptr/index values.

    After this pass, the Inline pass inlines template bodies, and
    FoldTileBufIntrinsics resolves tile_buf_addr / tile_valid_rows /
    tile_valid_cols to VPTO-side memref/ptr/constant values.
  }];
  let constructor = "mlir::pto::createExpandTileOpPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect"
  ];
  let options = [
    Option<"tilelangPath", "tilelang-path", "std::string",
           /*default=*/"\"\"",
           "Path to directory of .py tilelang DSL template files">,
    Option<"tilelangPkgPath", "tilelang-pkg-path", "std::string",
           /*default=*/"\"\"",
           "PYTHONPATH for tilelang_dsl package (added to env)">,
    Option<"pythonExe", "python-exe", "std::string",
           /*default=*/"\"python3\"",
           "Python executable for tilelang DSL invocation">
  ];
}

def FoldTileBufIntrinsics : Pass<"pto-fold-tile-buf-intrinsics", "mlir::func::FuncOp"> {
  let summary = "Fold structured-view intrinsics after template inlining";
  let description = [{
    After TileLang DSL template functions are inlined, the IR contains
    structured-view intrinsics whose operands are now bound to concrete values.

    This pass resolves them:
    - pto.tile_buf_addr is resolved from native tile_buf metadata carried by
      producers such as pto.pointer_cast, explicit-address pto.alloc_tile,
      tilebuf-to-tilebuf pto.bind_tile, pto.subview, pto.bitcast, and
      pto.treshape, then materialized as the requested memref or !pto.ptr.
    - pto.tile_valid_rows is folded to arith.constant when v_row is static, or
      replaced with the traced dynamic native valid-row SSA value.
    - pto.tile_valid_cols is folded to arith.constant when v_col is static, or
      replaced with the traced dynamic native valid-col SSA value.

    tensor_view family:
    - pto.tensor_view_addr is resolved from native pto.make_tensor_view and
      pto.partition_view metadata, then materialized as the requested memref or
      !pto.ptr.
    - pto.get_tensor_view_dim is folded to arith.constant for static tensor
      dimensions, or to the native shape/size SSA operand for dynamic dims.
    - pto.get_tensor_view_stride is folded to the native make_tensor_view
      stride operand, with partition views inheriting their source strides.

    Bridge-shaped values produced by memref-sourced
    builtin.unrealized_conversion_cast are rejected instead of being traced
    through.
  }];
  let constructor = "mlir::pto::createFoldTileBufIntrinsicsPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOInlineLibCall : Pass<"pto-inline-libcall", "ModuleOp"> {
  let summary = "Materialize OP-Lib instance bodies and inline OP-Lib calls";
  let description = [{
    Resolves OP-Lib instance declarations generated by OP-Lib lowering,
    materializes instance bodies, and inlines OP-Lib calls into caller/fused
    helper functions. Function signatures stay in !pto.tile_buf form.
  }];
  let constructor = "mlir::pto::createPTOInlineLibCallPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for OP-Lib instantiation/inlining">];
}

def PTOVerifyTFree : Pass<"pto-verify-tfree", "func::FuncOp"> {
  let summary = "Verify explicit matching pto.tfree placement for pto.tpop";
  let description = [{
    For each `pto.tpop` in section.cube / section.vector, verify there is a
    matching later `pto.tfree(pipe_handle)` in the same block, the borrowed
    tile is not used after the free, and the same pipe does not accumulate
    multiple outstanding pops before the matching free.
  }];

  let constructor = "mlir::pto::createPTOVerifyTFreePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

def PTOInferVPTOVecScope
    : Pass<"pto-infer-vpto-vecscope", "func::FuncOp"> {
  let summary =
      "Infer missing pto.vecscope regions for VPTO vector operation clusters";
  let description = [{
    Runs near the VPTO emission boundary after inlining, canonicalization,
    CSE, pointer normalization, and bridge-op expansion have exposed the final
    SSA shape. The pass greedily clusters contiguous VPTO vector operations
    into `pto.vecscope` regions while preserving explicit vector-scope carriers
    and treating DMA/copy/sync, unresolved calls, terminators, and forbidden
    operations as boundaries.

    The inferred `pto.vecscope` form remains resultless. Values whose type is
    `!pto.vreg`, `!pto.mask`, or `!pto.align` must not escape the inferred
    scope.
  }];
  let constructor = "mlir::pto::createPTOInferVPTOVecScopePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVPTOIR : Pass<"pto-validate-vpto-ir", "ModuleOp"> {
  let summary =
      "Validate authoring-stage VPTO legality before ptr-boundary canonicalization";
  let description = [{
    Runs the authoring-stage VPTO legality verifier on post-mainline VPTO IR.
    This stage keeps the memref-first authoring surface legal, while checking
    the shared structural contracts that must hold before emission-boundary
    canonicalization.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVPTOEmissionIR
    : Pass<"pto-validate-vpto-emission-ir", "ModuleOp"> {
  let summary =
      "Validate emission-stage VPTO legality after ptr-boundary canonicalization";
  let description = [{
    Runs the emission-stage VPTO legality verifier on ptr-form VPTO IR after
    `PTOVPTOPtrBoundary`. This stage re-checks the shared authoring contracts
    and confirms the final emission surface no longer carries memref boundary
    state or residual bridge scaffold.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOEmissionIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOVPTOIfCanonicalize
    : Pass<"pto-vpto-if-canonicalize", "func::FuncOp"> {
  let summary =
      "Canonicalize only scf.if cleanup patterns before A5 low-level fusion";
  let description = [{
    Applies the builtin `scf.if` canonicalization patterns to pre-existing
    `scf.if` operations without running the global Canonicalizer. This keeps
    the pre-loop-fusion cleanup focused on removing residual constant-control-
    flow structure while preserving surrounding `scf.for` loop headers.
  }];
  let constructor = "mlir::pto::createPTOVPTOIfCanonicalizePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOVPTOExpandBridgeOps
    : Pass<"pto-vpto-expand-bridge-ops", "func::FuncOp"> {
  let summary =
      "Expand temporary VPTO bridge ops back to emission-ready VPTO IR";
  let description = [{
    Low-level fusion may keep temporary bridge ops in VPTO IR so legality and
    alias analysis can still see memref-form operands. This pass expands those
    bridge ops back to the existing emission-ready pointer-level VPTO forms
    before backend emission.
  }];
  let constructor = "mlir::pto::createPTOVPTOExpandBridgeOpsPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::pto::PTODialect"];
}

def PTOVPTOPtrBoundary
    : Pass<"pto-vpto-ptr-boundary", "ModuleOp"> {
  let summary =
      "Canonicalize the final VPTO emission boundary from memref-first IR to ptr ABI";
  let description = [{
    Runs the final emission-boundary ptr canonicalization after the backend
    mainline has finished its memref-first optimization pipeline. This pass
    rewrites eligible memref function arguments to same-space `!pto.ptr`,
    rejects memref function results, canonicalizes supported body-level VPTO
    buffer-like ops to ptr-form, and drops dead boundary/view scaffold such as
    trivial `pto.castptr`, `pto.bind_tile`, `memref.subview`,
    `memref.reinterpret_cast`, and `memref.memory_space_cast` once they become
    unused.
  }];
  let constructor = "mlir::pto::createPTOVPTOPtrBoundaryPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect"];
}

def PTOLowLevelLoopFusion : Pass<"pto-low-level-loop-fusion", "ModuleOp"> {
  let summary =
      "Fuse adjacent VPTO post-lowering loop nests inside pto.fusion_region";
  let description = [{
    Conservatively fuses adjacent post-lowering loop stages for already-planned
    fusion groups after the tile-native frontend fusion core. The formal input
    contract for this pass is low-level `scf.for + pto.*` structure that
    remains inside `pto.fusion_region` until explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOLowLevelLoopFusionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::vector::VectorDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for low-level loop fusion">];
}

def PTOFusionPredicateElision
    : Pass<"pto-fusion-predicate-elision", "func::FuncOp"> {
  let summary =
      "Elide redundant fusion-local VPTO plt predicate materialization";
  let description = [{
    Scans `pto.fusion_region` bodies after VPTO post-lowering CSE and prepares
    fusion-local redundant `pto.plt_*` predicate elimination before
    load/store cleanup and explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOFusionPredicateElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOFusionLoadStoreElision
    : Pass<"pto-fusion-load-store-elision", "func::FuncOp"> {
  let summary =
      "Elide fusion-local VPTO load/store round trips before flattening";
  let description = [{
    Removes local store materialization inside canonical fused loop bodies
    before explicit flatten. It also performs conservative frontier-aware
    cleanup for non-escaping fusion-region internal buffers.
  }];
  let constructor = "mlir::pto::createPTOFusionLoadStoreElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOFlattenFusionRegion : Pass<"pto-flatten-fusion-region", "func::FuncOp"> {
  let summary = "Flatten pto.fusion_region wrappers back into the parent block";
  let description = [{
    Eliminates residual `pto.fusion_region` wrappers after region-local
    lowering is complete by moving body operations back to the parent block,
    replacing region results with `pto.yield` operands, and erasing both
    `pto.yield` and the wrapper op.
  }];
  let constructor = "mlir::pto::createPTOFlattenFusionRegionPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def VPTOPtrNormalize
    : Pass<"vpto-ptr-normalize", "ModuleOp"> {
  let summary =
      "Normalize VPTO ptr-like values and users into a uniform !pto.ptr form";
  let description = [{
    Uses MLIR's conversion framework to normalize VPTO ptr-related forms before
    the existing VPTO ptr-boundary canonicalization runs. This pass rewrites
    supported tile-buffer and memref view producers such as `pto.tile_buf_addr`
    and `memref.subview`, and updates VPTO memory ops to consume the normalized
    ptr-form consistently.
  }];
  let constructor = "mlir::pto::createVPTOPtrNormalizePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect"];
}

def VPTOPtrCastCleanup
    : Pass<"vpto-ptr-cast-cleanup", "ModuleOp"> {
  let summary = "Collapse transient ptr-memref-ptr bridge casts after VPTO ptr normalization";
  let description = [{
    Eliminates bridge chains such as
    `!pto.ptr -> builtin.unrealized_conversion_cast -> memref.cast ->
     builtin.unrealized_conversion_cast -> !pto.ptr`
    when the outer ptr types already match.
  }];
  let constructor = "mlir::pto::createVPTOPtrCastCleanupPass()";
  let dependentDialects = ["mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect"];
}

#endif // MLIR_DIALECT_PTO_PASSES
