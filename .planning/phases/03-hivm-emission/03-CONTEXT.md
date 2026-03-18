# Phase 3: HIVM Emission - Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Emit textual LLVM HIVM intrinsic IR from `a5vm` and replace the current `emitc` final output dependency for the implemented subset. This phase is about the textual emission boundary, unresolved intrinsic handling, naming responsibility, and developer observability around the emitter. It does not redefine PTO semantics or expand the implemented operation subset beyond what earlier phases already established.

</domain>

<decisions>
## Implementation Decisions

### LLVM-like output skeleton
- The default output should aim for a complete LLVM `.ll`-style module, not only a function-body fragment.
- The textual emitter should use LLVM-style types directly in the emitted text, such as `<64 x float>`, rather than carrying A5VM type syntax into the final artifact.
- Control flow and ordinary scalar/vector arithmetic that appear on the implemented path should also be emitted in LLVM-style textual form rather than left in mixed-dialect form.
- When there is a tradeoff between strict local conservatism and closeness to the final downstream-consumed HIVM text shape, Phase 3 should favor the final consumption-oriented form.

### Unresolved intrinsic handling
- When an intrinsic is not yet user-confirmed but the backend still needs to preserve the emitted structure, the main output should use placeholder intrinsic names rather than dropping the call site.
- When only the operation family is known, the emitter should still produce a structured placeholder call whose argument and result structure preserve as much real information as possible.
- Unresolved or provisional emission cases should produce a separate unresolved report artifact in addition to what appears in the main textual output.
- The main emitted text should still read as close as possible to real HIVM/LLVM IR even when placeholders are present; the report artifact is the primary place for explaining missing mappings.

### Intrinsic naming boundary
- Intrinsic-name synthesis should be driven by op kind, element type, vector shape, and variant-selection fields such as load/store or `sx1`-style suffix decisions.
- The information needed for name synthesis may be split between explicit `a5vm` op attributes and emitter-side derivation from operand/type shape; the design does not require every decision input to be stored redundantly on the op itself.
- When name synthesis fails, diagnostics and unresolved reporting should include the operation, the participating naming fields, and which fields are missing or unsupported.
- Naming logic should be organized as shared family-specific builders, with at least separate `load/store` and unary-style paths reusing common string-fragment assembly helpers.

### Developer observability
- By default, normal tool output should contain only the final HIVM textual artifact; debugging and reporting features must be explicitly enabled by developer-facing flags.
- The main HIVM text should continue to use the normal output destination, while debug/trace information should go to `stderr` by default so it does not pollute the emitted IR.
- An explicit `--dump-a5vm-ir` style flag should be supported as part of the Phase 3 developer workflow.
- The unresolved report should include, at minimum, the source op, candidate or placeholder intrinsic name, the naming fields that participated, the fields that were missing, and a useful source location.

### Claude's Discretion
- Exact textual module boilerplate details, so long as the default output remains LLVM `.ll`-like and suitable for downstream verification.
- Exact placeholder intrinsic spelling convention, so long as it is machine-distinguishable and structurally preserves the original call shape.
- Exact split of responsibility between `a5vm` attributes and emitter-side derivation for each individual op family.
- Exact CLI spelling for the new emitter/debug/report flags beyond the requirement that they are explicit and developer-facing.

</decisions>

<specifics>
## Specific Ideas

- The final emitted artifact should look like something close to a real LLVM `.ll` file, not like a custom debug dump with A5VM syntax still showing through.
- Even unresolved cases should keep the call structure in the main output so the user can inspect roughly what the downstream compiler would consume.
- Placeholder and unresolved handling should preserve enough typed structure that the later intrinsic inventory work can be driven from real emitted sites rather than reverse-engineering logs.
- Phase 3 should keep the emitter abstraction boundary clean: `a5vm` remains the hardware-facing IR, and the textual emitter is where final HIVM spelling is assembled.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and prior phase context
- `.planning/PROJECT.md` — project goal, compatibility constraints, and v1 scope
- `.planning/REQUIREMENTS.md` — phase-mapped requirements and acceptance boundaries, including `HIVM-01` to `HIVM-03`
- `.planning/ROADMAP.md` — fixed Phase 3 boundary and success criteria
- `.planning/STATE.md` — current milestone position
- `.planning/phases/01-a5vm-foundation/01-CONTEXT.md` — locked backend-boundary, output-shape, and diagnostic decisions from Phase 1
- `.planning/phases/02-pto-lowering/02-CONTEXT.md` — locked PTO-to-A5VM semantic-preservation boundary that Phase 3 must not erode

### Current tool integration points
- `tools/ptoas/ptoas.cpp` — current pass pipeline, backend arch selection, `emitc::createFormExpressionsPass()`, and final `emitc::translateToCpp(...)` boundary being replaced
- `include/PTO/Transforms/Passes.h` — pass declaration surface for any new A5VM-to-HIVM lowering or emission entrypoints
- `include/PTO/Transforms/Passes.td` — pass registration definitions if new passes/options need to be introduced

### Existing backend reference
- `lib/PTO/Transforms/PTOToEmitC.cpp` — current output path and current places where backend-specific naming or operation flattening occur
- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` — CCE wrapper naming families that help infer the eventual HIVM intrinsic naming dimensions

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `tools/ptoas/ptoas.cpp`: already owns the final output file selection, pass pipeline execution, and the exact point where `emitc` translation is invoked today.
- `include/PTO/Transforms/Passes.h`: provides the natural declaration point for any A5VM-to-HIVM transformation or textual emission pass wiring added in this phase.
- `lib/PTO/Transforms/PTOToEmitC.cpp`: serves as the current backend reference for op-family-specific naming and emission decisions, even though the final output format is changing.

### Established Patterns
- The tool keeps backend orchestration centralized in `ptoas`, so Phase 3 should attach the new textual emission path at the existing final emission boundary rather than inventing a parallel standalone tool.
- New compiler functionality is usually surfaced through explicit pass factory declarations under `include/PTO/Transforms/` with implementations in `lib/PTO/Transforms/`.
- Earlier phases already established that the new backend should coexist with the old path during bring-up and should expose developer-oriented diagnostics rather than silently guessing.

### Integration Points
- The new emitter must replace the current `emitc::translateToCpp(...)` path when the A5VM backend is selected, while preserving the rest of the pass pipeline shape unless a new backend-local final lowering pass is required.
- Any unresolved-report generation must integrate with the same end-of-tool emission flow that currently writes the final output file.
- The debug flags for A5VM dump and unresolved reporting should connect at the tool boundary so planning can keep normal and debug channels separated.

</code_context>

<deferred>
## Deferred Ideas

- Exact user-confirmed HIVM intrinsic names for the implemented subset are deferred to the later inventory-validation phase once emitted sites can be enumerated concretely.
- Broader debug-flag surface beyond the minimum explicit A5VM dump and unresolved reporting controls can be expanded later if the first implementation shows gaps.

</deferred>

---
*Phase: 03-hivm-emission*
*Context gathered: 2026-03-19*
