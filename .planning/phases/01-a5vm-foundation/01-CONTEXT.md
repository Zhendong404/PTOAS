# Phase 1: A5VM Foundation - Context

**Gathered:** 2026-03-19
**Status:** Ready for replanning

<domain>
## Phase Boundary

Introduce the corrected backend boundary and the minimum hardware-facing `a5vm` IR model needed for the `Abs` path. This phase establishes the corrected dialect identity, namespace, primitive op surface, and backend switch shape, but does not yet implement PTO `TLOAD` / `TABS` / `TSTORE` behavior directly.

</domain>

<decisions>
## Implementation Decisions

### A5VM identity and namespace
- `a5vm` remains a first-class dialect under the project directory tree, but its C++ namespace must be `mlir::a5vm`, not `mlir::pto::a5vm`.
- `a5vm` should keep the normalized vector type spelling such as `!a5vm.vec<64xf32>`.
- The already-landed namespace and op-surface choices from the first Phase 1 attempt should be treated as superseded by this corrected context.

### Primitive-op direction
- `a5vm` ops should stay as close as practical to CCE builtin naming rather than freezing PTO-interface-shaped pseudo-ops such as `a5vm.load` and `a5vm.store`.
- Phase 1 should define the minimum primitive surface needed by the real PTO-library-aligned lowering shape, including vector-memory and vector-compute primitives such as `vld`, `vabs`, and `vst`, plus any loop-scope or copy-family primitives that later `TLOAD` / `TSTORE` lowering will need.
- Phase 1 must not encode PTO interface semantics directly into A5VM ops when the real hardware-facing primitive is lower-level and differently named.
- General control flow and scalar arithmetic should remain in shared dialects when they are not hardware-facing.

### Backend switching and output seam
- Keep dual backend paths during the correction pass.
- Select backend through an explicit CLI flag rather than a hidden or hardwired mode switch.
- Default CLI behavior should remain compatible with current usage, but new backend selection must be available for developers.
- The final textual HIVM emission seam still belongs at the current `emitc::translateToCpp` boundary, but Phase 1 should not over-specify final intrinsic names before the primitive A5VM surface is corrected.

### Failure and placeholder policy
- When something can be emitted as legal textual IR, do that even if some details remain provisional.
- When a case cannot be emitted legally, output should include explicit unresolved markers or comments instead of silently guessing.
- Placeholder handling should preserve enough context that required intrinsic mappings can be reviewed and confirmed later.

### Replanning Notes
- The previous assumption that the `Abs` path should be represented primarily with `a5vm.load` / `a5vm.abs` / `a5vm.store` is incorrect and must not guide new planning.
- Downstream planning must use the A5-side PTO library implementation plus CCE builtin families as the semantic source of truth, not the currently landed A5VM pseudo-load/store design.
- Replanning does not need to preserve or mirror `a2a3` implementation details; only the A5 PTO path matters for this effort.

### Claude's Discretion
- Exact file split for the corrected A5VM dialect implementation under the existing PTO directory tree
- Exact choice of which primitive ops land in Phase 1 versus wait for Phase 2, as long as they align to the PTO-library-backed lowering shape
- Exact flag names for backend selection and debug controls

</decisions>

<specifics>
## Specific Ideas

- Dual backend mode should remain explicit and developer-controlled via a backend-selection flag.
- The old `emitc` path should remain available during the correction pass rather than being deleted immediately.
- `a5vm` type syntax should stay simple and normalized, for example `!a5vm.vec<64xf32>`.
- The dialect namespace must be `mlir::a5vm`.
- A5VM op names should be pulled toward CCE builtin naming, not toward PTO-interface naming.
- The corrected A5VM layer should provide primitives that a PTO-library-faithful `TLOAD` / `TABS` / `TSTORE` lowering can compose later.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and phase scope
- `.planning/PROJECT.md` — project goal, compatibility constraints, and v1 scope
- `.planning/REQUIREMENTS.md` — corrected phase-mapped requirements and acceptance boundaries
- `.planning/ROADMAP.md` — corrected phase boundary and success criteria for Phase 1
- `.planning/STATE.md` — current milestone position and open correction items

### Existing backend integration points
- `tools/ptoas/ptoas.cpp` — current pass pipeline, target-arch handling, and final `emitc::translateToCpp` emission point
- `include/PTO/Transforms/Passes.h` — existing pass creation APIs and likely registration surface for the corrected backend pass
- `lib/PTO/Transforms/PTOToEmitC.cpp` — existing backend pass shape and current lowering boundary being replaced in stages

### PTO and sample semantics
- `test/samples/Abs/abs.py` — initial acceptance sample and exact PTO op path exercised in v1
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp` — public PTO instruction template behavior for `TLOAD`, `TABS`, and `TSTORE`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp` — A5-side TLOAD implementation behavior and layout / transfer structure
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp` — A5-side TABS implementation path and unary-op behavior
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp` — A5-side TSTORE implementation behavior and tile-domain branching
- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` — CCE builtin wrappers that should inform corrected A5VM primitive naming

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/ptoas/ptoas.cpp`: already contains the pass pipeline, target-arch flag flow, and final output file handling that the corrected backend should reuse.
- `include/PTO/Transforms/Passes.h`: provides the natural place to declare the corrected backend pass entrypoints.
- `lib/PTO/Transforms/PTOToEmitC.cpp`: useful as the current reference for where PTO backend lowering begins and how op patterns are organized, but the new A5VM primitive surface must move away from the currently landed pseudo-load/store approach.

### Established Patterns
- CLI/tool orchestration is centralized in `tools/ptoas/ptoas.cpp`.
- New compiler passes are declared in `include/PTO/Transforms/Passes.h` and implemented under `lib/PTO/Transforms/`.
- Dialect definitions and types belong in `include/PTO/IR/` and `lib/PTO/IR/`, which is still the correct home for the `a5vm` dialect even though its C++ namespace should be `mlir::a5vm`.

### Integration Points
- Backend selection should integrate where `ptoas` currently decides between A3/A5 codegen and invokes `createEmitPTOManualPass(...)`.
- Textual HIVM output should still integrate where `ptoas` currently calls `emitc::translateToCpp(...)`, but only after the corrected primitive A5VM surface is in place.
- Phase 1 replanning should treat the corrected dialect module and the new final emitter seam as first-class components, not temporary helper code inside the old EmitC translation path.

</code_context>

<deferred>
## Deferred Ideas

- Exact PTO `TLOAD` / `TABS` / `TSTORE` lowering structure belongs to Phase 2, but Phase 1 must stop making assumptions that contradict the PTO library.
- Exact final HIVM intrinsic names remain deferred.

</deferred>

---
*Phase: 01-a5vm-foundation*
*Context gathered: 2026-03-19*
