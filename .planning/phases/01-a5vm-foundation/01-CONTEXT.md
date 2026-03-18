# Phase 1: A5VM Foundation - Context

**Gathered:** 2026-03-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Introduce the new backend boundary and the minimum `a5vm` IR model needed for the `Abs` path. This phase establishes the new dialect, backend switch shape, and the thinnest textual HIVM emission path, but does not yet need to finish full PTO semantic lowering beyond what is necessary to stand up the backend skeleton.

</domain>

<decisions>
## Implementation Decisions

### Backend switching
- Keep dual backend paths during Phase 1.
- Select backend through an explicit CLI flag rather than a hidden or hardwired mode switch.
- Default CLI behavior should remain compatible with current usage, but new backend selection must be available for developers.

### Output format
- The new backend should target output that is as close as possible to final consumer-facing LLVM `.ll` style text.
- The output should avoid retaining extra intermediate semantic noise unless needed for unresolved cases.
- The textual HIVM emitter should be connected in `ptoas` at the current final emission point, directly replacing `emitc::translateToCpp` when the `a5vm` backend is selected.

### A5VM abstraction boundary
- `a5vm` lives in the existing PTO directory and namespace structure as a new dialect module, not as an ad hoc extension inside the old EmitC pass.
- `a5vm` should use a simple unified vector type spelling such as `!a5vm.vec<64xf32>`.
- `a5vm` is responsible for the hardware-facing abstraction layer, while the textual emitter is responsible for final HIVM intrinsic name synthesis.
- Phase 1 should already connect the thinnest usable textual HIVM emitter path rather than stopping at raw `a5vm` IR.

### Failure and placeholder policy
- When something can be emitted as legal textual IR, do that even if some details remain provisional.
- When a case cannot be emitted legally, output should include explicit unresolved markers or comments instead of silently guessing.
- Placeholder handling should preserve enough context that required intrinsic mappings can be reviewed and confirmed later.

### Developer diagnostics
- Provide explicit developer-facing debug flags for:
  - printing backend intermediate IR
  - printing intrinsic-selection decisions
  - continuing through unresolved mappings and producing a summary list
- Error messages for the new backend should be developer-oriented and include operation/type/mapping context rather than only end-user summaries.
- Unresolved intrinsic or emission gaps should be reported both in terminal/log output and in a separate artifact or file.

### Claude's Discretion
- Exact flag names for backend selection and debug controls
- Exact organization of the new dialect source files within the PTO directory hierarchy
- Exact mechanics for how unresolved intrinsic summaries are accumulated and written

</decisions>

<specifics>
## Specific Ideas

- Dual backend mode should be explicit and developer-controlled via a backend-selection flag.
- The old `emitc` path should remain available during Phase 1 rather than being deleted immediately.
- The final artifact for the new path should read like LLVM `.ll` text, not like a debug dump of a custom IR.
- `a5vm` type syntax should stay simple and normalized, for example `!a5vm.vec<64xf32>`.
- Name construction for final HIVM intrinsics should happen in the emitter, not be frozen into `a5vm` op names.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and phase scope
- `.planning/PROJECT.md` — project goal, compatibility constraints, and v1 scope
- `.planning/REQUIREMENTS.md` — phase-mapped requirements and acceptance boundaries
- `.planning/ROADMAP.md` — fixed phase boundary and success criteria for Phase 1
- `.planning/STATE.md` — current milestone position and open questions

### Existing backend integration points
- `tools/ptoas/ptoas.cpp` — current pass pipeline, target-arch handling, and final `emitc::translateToCpp` emission point
- `include/PTO/Transforms/Passes.h` — existing pass creation APIs and likely registration surface for the new backend pass
- `lib/PTO/Transforms/PTOToEmitC.cpp` — existing backend pass shape and current lowering boundary being replaced in stages

### PTO and sample semantics
- `test/samples/Abs/abs.py` — initial acceptance sample and exact PTO op path exercised in v1
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp` — public PTO instruction template behavior for `TLOAD`, `TABS`, and `TSTORE`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp` — TLOAD implementation behavior and layout constraints
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp` — TABS implementation path and unary-op behavior
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp` — TSTORE implementation behavior and tile-domain branching

### Intrinsic wrapper references
- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` — CCE builtin wrapper families and naming patterns for eventual HIVM intrinsic synthesis

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/ptoas/ptoas.cpp`: already contains the pass pipeline, target-arch flag flow, and final output file handling that the new backend should reuse.
- `include/PTO/Transforms/Passes.h`: provides the natural place to declare the new backend pass entrypoints.
- `lib/PTO/Transforms/PTOToEmitC.cpp`: useful as the current reference for where PTO backend lowering begins and how op patterns are organized, even though Phase 1 should not keep the new backend embedded inside EmitC long term.
- `.planning/codebase/CONVENTIONS.md` and `.planning/codebase/STRUCTURE.md`: establish that new compiler-facing modules should live under `include/PTO/**` and `lib/PTO/**` using the existing PascalCase / MLIR-style organization.

### Established Patterns
- CLI/tool orchestration is centralized in `tools/ptoas/ptoas.cpp`.
- New compiler passes are declared in `include/PTO/Transforms/Passes.h` and implemented under `lib/PTO/Transforms/`.
- Dialect definitions and types belong in `include/PTO/IR/` and `lib/PTO/IR/`, which is the correct home for the new `a5vm` dialect.
- Developer-facing compiler diagnostics in this repo are generally explicit and technical rather than product-polished, which aligns with the requested error style.

### Integration Points
- Backend selection should integrate where `ptoas` currently decides between A3/A5 codegen and invokes `createEmitPTOManualPass(...)`.
- Textual HIVM output should integrate where `ptoas` currently calls `emitc::translateToCpp(...)`.
- Phase 1 planning should treat the new dialect module and the new final emitter as first-class components, not temporary helper code inside the old EmitC translation path.

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---
*Phase: 01-a5vm-foundation*
*Context gathered: 2026-03-18*
